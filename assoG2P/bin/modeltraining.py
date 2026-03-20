#!/usr/bin/env python3
"""
Model training module (适配preprocess元数据衔接)
支持多模型训练、结果可视化、预测等功能
- GWAS特征筛选和LD过滤已前移到preprocess模块
- 训练阶段仅使用preprocess已筛选的特征
"""

import os
import sys
import json
import logging
import subprocess
import time
import shutil
import atexit
import signal
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

# SHAP 支持（可选）
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ======================== 绘图质量固定开关（standardized） ========================
# 固定实现：不再通过命令行或函数参数切换；所有训练/可视化产物统一使用期刊质量配置。
PUB_QUALITY_MODE: bool = True

# ======================== 调试控制开关 ========================
# 设置为 False 时：保留模型训练过程中产生的所有临时文件和目录，方便排查问题
# 设置为 True  时：恢复原来的自动清理临时文件逻辑
CLEANUP_TEMP_FILES = True

# ======================== 全局临时文件管理器 ========================
class TempFileManager:
    """
    全局临时文件管理器，用于注册和清理临时文件/目录
    支持程序退出时自动清理（包括正常退出、异常退出、信号中断）
    """
    def __init__(self):
        self.temp_files: List[Path] = []
        self.temp_dirs: List[Path] = []
        self.preprocess_tmp_dirs: List[Path] = []  # preprocess阶段的临时目录（tmp_p，不清理）
        self._cleanup_registered = False
        self._cleaned = False  # 标记是否已清理，防止重复清理
        self._register_exit_handlers()
    
    def _register_exit_handlers(self):
        """注册退出处理函数"""
        if self._cleanup_registered:
            return
        
        # 注册 atexit 处理函数（正常退出时调用）
        atexit.register(self.cleanup_on_exit)
        
        # 注册信号处理函数（SIGINT: Ctrl+C, SIGTERM: 终止信号）
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            # 在某些环境下（如线程中）可能无法注册信号处理器，忽略错误
            pass
        
        self._cleanup_registered = True
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.warning(f"Received signal {signum}, cleaning up temporary files...")
        self.cleanup_on_exit()
        # 重新发送信号，让程序正常退出
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    def register_file(self, file_path: Path) -> Path:
        """注册临时文件"""
        file_path = Path(file_path).absolute()
        if file_path not in self.temp_files:
            self.temp_files.append(file_path)
        return file_path
    
    def register_dir(self, dir_path: Path) -> Path:
        """注册临时目录"""
        dir_path = Path(dir_path).absolute()
        dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path not in self.temp_dirs:
            self.temp_dirs.append(dir_path)
        return dir_path
    
    def register_preprocess_tmp_dir(self, dir_path: Path) -> Path:
        """注册preprocess阶段的临时目录（tmp_p，不清理，preprocess模块执行完毕不删除）"""
        dir_path = Path(dir_path).absolute()
        if dir_path not in self.preprocess_tmp_dirs:
            self.preprocess_tmp_dirs.append(dir_path)
        return dir_path
    
    def cleanup_on_exit(self, cleanup_preprocess: bool = False):
        """
        清理所有注册的临时文件和目录
        
        :param cleanup_preprocess: 是否清理preprocess阶段的临时目录（默认False，preprocess模块执行完毕不删除tmp_p目录）
        """
        if not CLEANUP_TEMP_FILES:
            return
        
        # 防止重复清理：如果已经清理过临时文件/目录，且本次不清理preprocess，则跳过
        if self._cleaned and not cleanup_preprocess:
            return
        
        try:
            # 1. 清理临时文件（如果尚未清理）
            if not self._cleaned:
                for fp in self.temp_files:
                    try:
                        if fp.exists():
                            fp.unlink()
                    except Exception as e:
                        pass
                
                # 2. 清理临时目录（逆序删除，确保子目录先删除）
                for dp in reversed(self.temp_dirs):
                    try:
                        if dp.exists():
                            shutil.rmtree(dp, ignore_errors=True)
                    except Exception as e:
                        pass
                
                # 清空列表
                self.temp_files.clear()
                self.temp_dirs.clear()
                self._cleaned = True
            
            # 3. 清理preprocess阶段的临时目录（默认不清理，preprocess模块执行完毕不删除tmp_p目录）
            if cleanup_preprocess:
                for dp in reversed(self.preprocess_tmp_dirs):
                    try:
                        if dp.exists():
                            shutil.rmtree(dp, ignore_errors=True)
                    except Exception as e:
                        pass
                self.preprocess_tmp_dirs.clear()
                
        except Exception as e:
            logger.warning(f"Error during temporary file cleanup (ignored): {e}")
    
    def clear(self):
        """清空所有注册的临时文件/目录（不删除）"""
        self.temp_files.clear()
        self.temp_dirs.clear()
        self.preprocess_tmp_dirs.clear()
        self._cleaned = False

# 创建全局临时文件管理器实例
_temp_file_manager = TempFileManager()

# ======================== 日志配置 ========================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

# ======================== 1. 工具函数：元数据读取 & 输入解析 ========================
def load_preprocess_metadata(metadata_file: str) -> Dict:
    """读取preprocess生成的JSON元数据文件"""
    if not Path(metadata_file).exists():
        raise FileNotFoundError(f"预处理元数据文件不存在: {metadata_file}")
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"元数据文件格式错误（非合法JSON）: {metadata_file}, 错误: {str(e)}")
    
    # 验证核心字段（gwas_genotype_prefix可以为None，表示文件不存在）
    required_fields = ["valid_samples", "output_train_file"]
    missing_fields = [f for f in required_fields if f not in metadata]
    if missing_fields:
        raise ValueError(f"元数据缺失关键信息: {', '.join(missing_fields)}")
    
    # 以前这里会验证 gwas_genotype_prefix 对应的 PLINK 文件是否存在，并在缺失时输出 warning。
    # 目前 model training 阶段不再依赖这些中间 PLINK 文件，为避免干扰用户，这里不再做存在性检查。
    # 如需使用 GWAS/LD 相关的 PLINK 输出，请在对应模块中单独验证路径。
    
    return metadata

def parse_train_input_path(input_path: str) -> Dict:
    """解析训练输入路径，支持元数据文件/训练文件两种模式"""
    input_path = Path(input_path).absolute().as_posix()
    result = {
        "train_file": None,
        "metadata": None,
        "gwas_genotype_prefix": None,
        "task_type": None,  # 从元数据读取的task_type
        "preprocess_tmp_dir": None,  # 从元数据读取的preprocess临时目录
        "_metadata_file_path": None  # 元数据文件路径，用于路径解析
    }

    # 模式1：输入是元数据文件
    if input_path.endswith("_metadata.json"):
        metadata = load_preprocess_metadata(input_path)
        result["metadata"] = metadata
        result["train_file"] = metadata["output_train_file"]
        result["gwas_genotype_prefix"] = metadata.get("gwas_genotype_prefix")
        result["task_type"] = metadata.get("task_type")  # 从元数据读取task_type
        result["preprocess_tmp_dir"] = metadata.get("preprocess_tmp_dir")  # 从元数据读取preprocess临时目录
        result["_metadata_file_path"] = input_path  # 保存元数据文件路径
    # 模式2：输入是训练文件，尝试关联元数据
    else:
        result["train_file"] = input_path
        # 尝试多种可能的元数据文件路径
        train_file_path = Path(input_path)
        possible_metadata_paths = [
            f"{os.path.splitext(input_path)[0]}_metadata.json",  # train_data.txt -> train_data_metadata.json
            str(train_file_path.parent / f"{train_file_path.stem}_metadata.json"),  # 同上，但使用Path
            str(train_file_path.parent / f"{train_file_path.parent.name}_metadata.json"),  # 如果输出目录名作为前缀
        ]
        # 也尝试查找目录下所有metadata.json文件
        if train_file_path.parent.exists():
            for metadata_file in train_file_path.parent.glob("*_metadata.json"):
                possible_metadata_paths.append(str(metadata_file))
        
        metadata_found = False
        for metadata_path in possible_metadata_paths:
            if Path(metadata_path).exists():
                try:
                    metadata = load_preprocess_metadata(metadata_path)
                    result["metadata"] = metadata
                    result["gwas_genotype_prefix"] = metadata.get("gwas_genotype_prefix")
                    result["task_type"] = metadata.get("task_type")  # 从元数据读取task_type
                    result["preprocess_tmp_dir"] = metadata.get("preprocess_tmp_dir")
                    result["_metadata_file_path"] = metadata_path  # 保存元数据文件路径
                    metadata_found = True
                    break
                except Exception as e:
                    continue  # 尝试下一个路径
        
        if not metadata_found:
            pass
    
    # 验证训练文件存在性
    if not Path(result["train_file"]).exists():
        raise FileNotFoundError(f"Training file does not exist: {result['train_file']}")
    
    return result

# ======================== 2. 数据加载 & 处理 ========================
def clean_feature_names(feature_names: List[str]) -> List[str]:
    """
    清理特征名称，移除LightGBM不支持的特殊JSON字符
    
    LightGBM不支持的特征名称字符：{ } [ ] , : " ' 等JSON特殊字符
    """
    import re
    # 定义需要替换的特殊字符（JSON特殊字符）
    special_chars = r'[{}\[\]\,"\':]'
    
    cleaned_names = []
    for name in feature_names:
        # 替换特殊字符为下划线
        cleaned = re.sub(special_chars, '_', str(name))
        # 移除连续的下划线
        cleaned = re.sub(r'_+', '_', cleaned)
        # 移除开头和结尾的下划线
        cleaned = cleaned.strip('_')
        # 如果清理后为空，使用原始名称的哈希值
        if not cleaned:
            cleaned = f"feature_{hash(name) % 1000000}"
        cleaned_names.append(cleaned)
    
    return cleaned_names

def load_training_data(train_file: str, valid_samples: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """
    加载训练数据，复用preprocess的有效样本
    
    Returns:
        X: 特征DataFrame（列名已清理）
        y: 目标变量Series
        snp_name_mapping: 原始SNP名称到清理后名称的映射字典
    """
    
    # 检查文件格式
    train_file_path = Path(train_file)
    if not train_file_path.exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    
    file_ext = train_file_path.suffix.lower()
    if file_ext in ['.vcf', '.vcf.gz']:
        raise ValueError(
            f"VCF file detected: {train_file}\n"
            f"Error: Model training/prediction requires preprocessed training data format (tab-separated .txt file with 'sample' column as index).\n"
            f"VCF files must be preprocessed first using the 'preprocess' command.\n"
            f"Please run: assog2p preprocess -g {train_file} -p <phenotype_file> -o <output_dir>\n"
            f"Then use the preprocessed output file for training/prediction."
        )
    
    # 检测是否是压缩文件
    is_compressed = file_ext == '.gz' or train_file.endswith('.gz')
    
    try:
        # 尝试自动检测分隔符
        # 先读取前几行来检测分隔符
        import csv
        import gzip
        
        # 根据是否压缩选择打开方式
        # 注意：gzip 文件不支持 seek(0)，所以需要重新打开文件
        if is_compressed:
            with gzip.open(train_file, 'rt', encoding='utf-8') as f:
                # 读取前10行来检测分隔符（跳过空行）
                sample_lines = []
                for _ in range(10):
                    line = f.readline()
                    if line.strip():  # 跳过空行
                        sample_lines.append(line)
                    if not line:  # 文件结束
                        break
        else:
            with open(train_file, 'r', encoding='utf-8') as f:
                # 读取前10行来检测分隔符（跳过空行）
                sample_lines = []
                for _ in range(10):
                    line = f.readline()
                    if line.strip():  # 跳过空行
                        sample_lines.append(line)
                    if not line:  # 文件结束
                        break
                f.seek(0)  # 重置文件指针（仅对非压缩文件有效）
        
        # 检查是否读取到数据
        if not sample_lines:
            raise ValueError(f"File {train_file} appears to be empty or contains only empty lines")
        
        # 检测最常见的分隔符
        # 使用csv.Sniffer来检测分隔符（更准确）
        delimiter = "\t"  # 默认使用制表符
        try:
            # 尝试使用csv.Sniffer自动检测
            sample_text = "".join(sample_lines[:5])  # 使用前5行
            sniffer = csv.Sniffer()
            detected_delimiter = sniffer.sniff(sample_text, delimiters="\t, ").delimiter
            if detected_delimiter:
                delimiter = detected_delimiter
        except Exception as e:
            # 手动检测：统计每种分隔符出现的次数
            delimiter_counts = {"\t": [], ",": [], " ": []}
            
            for line in sample_lines:
                if not line.strip():
                    continue
                delimiter_counts["\t"].append(line.count("\t"))
                delimiter_counts[","].append(line.count(","))
                delimiter_counts[" "].append(line.count(" "))
            
            # 计算每种分隔符的平均出现次数（排除0值）
            delimiter_avg = {}
            for sep, counts in delimiter_counts.items():
                non_zero_counts = [c for c in counts if c > 0]
                if non_zero_counts:
                    delimiter_avg[sep] = sum(non_zero_counts) / len(non_zero_counts)
                else:
                    delimiter_avg[sep] = 0
            
            # 选择平均出现次数最多的分隔符（但至少要有2个字段）
            max_avg = max(delimiter_avg.values())
            if max_avg > 0:
                for sep, avg_count in delimiter_avg.items():
                    if avg_count == max_avg and avg_count > 0:
                        delimiter = sep
                        break
            
        
        # 验证检测到的分隔符：检查第一行使用该分隔符后有多少字段
        if sample_lines:
            first_line = sample_lines[0].strip()
            field_count = len(first_line.split(delimiter))
            
            # 如果字段数异常（太多或太少），尝试其他分隔符
            if field_count > 1000 or field_count < 2:
                logger.warning(f"Detected field count ({field_count}) seems abnormal, trying alternative delimiters")
                for alt_sep in ["\t", ",", " "]:
                    if alt_sep != delimiter:
                        alt_count = len(first_line.split(alt_sep))
                        if 2 <= alt_count <= 1000:
                            logger.info(f"Switching to delimiter {repr(alt_sep)} (field count: {alt_count})")
                            delimiter = alt_sep
                            break
        
        # 读取数据（sample列为索引，最后一列为表型）
        # 使用更健壮的读取方式，处理格式不一致的情况
        # pandas 的 read_csv 会自动处理 gzip 压缩文件（如果文件名以 .gz 结尾）
        df = None
        read_errors = []
        
        # 策略1: 尝试使用检测到的分隔符，跳过错误行
        try:
            df = pd.read_csv(
                train_file,
                sep=delimiter,
                index_col="sample",
                na_filter=False,  # 禁用缺失值过滤（preprocess已处理）
                engine='python',  # 使用Python引擎，更健壮但稍慢，支持压缩文件
                on_bad_lines='skip',  # 跳过格式错误的行（pandas >= 1.3.0）
                compression='gzip' if is_compressed else 'infer'  # 明确指定压缩格式
            )
        except (TypeError, ValueError) as e:
            read_errors.append(f"Strategy 1 (on_bad_lines='skip'): {str(e)}")
            # 策略2: 对于旧版本的pandas，使用不同的参数
            try:
                df = pd.read_csv(
                    train_file,
                    sep=delimiter,
                    index_col="sample",
                    na_filter=False,
                    engine='python',
                    error_bad_lines=False,  # pandas < 1.3.0
                    warn_bad_lines=True,
                    compression='gzip' if is_compressed else 'infer'
                )
            except (TypeError, ValueError) as e2:
                read_errors.append(f"Strategy 2 (error_bad_lines=False): {str(e2)}")
                # 策略3: 使用最基本的读取方式，不跳过错误行
                try:
                    df = pd.read_csv(
                        train_file,
                        sep=delimiter,
                        index_col="sample",
                        na_filter=False,
                        engine='python',
                        compression='gzip' if is_compressed else 'infer'
                    )
                except Exception as e3:
                    read_errors.append(f"Strategy 3 (basic mode): {str(e3)}")
                    # 策略4: 尝试使用C引擎（更快，但可能不够健壮，且可能不支持压缩）
                    if not is_compressed:  # C引擎可能不支持压缩文件
                        try:
                            df = pd.read_csv(
                                train_file,
                                sep=delimiter,
                                index_col="sample",
                                na_filter=False,
                                engine='c',
                                low_memory=False
                            )
                        except Exception as e4:
                            read_errors.append(f"Strategy 4 (C engine): {str(e4)}")
                    else:
                        read_errors.append(f"Strategy 4 (C engine): Skipped (compressed file, C engine may not support)")
                        # 如果所有策略都失败，检查是否是缺少 "sample" 列的问题
                        error_msg = f"Failed to read file {train_file} with delimiter {repr(delimiter)}. Tried strategies:\n"
                        error_msg += "\n".join(f"  - {err}" for err in read_errors)
                        
                        # 检查是否是缺少 "sample" 列的问题
                        if any("Index sample invalid" in err or "sample" in err.lower() for err in read_errors):
                            # 尝试读取第一行来检查列名
                            try:
                                with open(train_file, 'r', encoding='utf-8') as f:
                                    header_line = f.readline().strip()
                                    if delimiter:
                                        columns = header_line.split(delimiter)
                                        if 'sample' not in columns:
                                            error_msg += f"\n\nError: File does not contain 'sample' column as required.\n"
                                            error_msg += f"Found columns: {columns[:10]}{'...' if len(columns) > 10 else ''}\n"
                                            error_msg += f"This file format is not compatible with model training/prediction.\n"
                                            error_msg += f"Expected format: Tab-separated file with 'sample' column as index (from preprocess module output).\n"
                                            if file_ext in ['.vcf', '.vcf.gz']:
                                                error_msg += f"Note: VCF files must be preprocessed first using: assog2p preprocess -g <vcf_file> -p <phenotype_file> -o <output_dir>"
                            except Exception:
                                pass
                        
                        error_msg += f"\n\nPlease check the file format. Expected delimiter: {repr(delimiter)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e4
        
        if df is None or df.empty:
            raise ValueError(f"Failed to read data from {train_file} or file is empty")
        
        # 验证文件格式：确保有 "sample" 索引
        if df.index.name != "sample" and "sample" not in str(df.index.name).lower():
            # 检查是否有 sample 列
            if "sample" in df.columns:
                logger.warning("Found 'sample' column but not as index. Attempting to set it as index...")
                df = df.set_index("sample")
            else:
                raise ValueError(
                    f"File {train_file} does not have 'sample' column/index as required.\n"
                    f"Current index name: {df.index.name}\n"
                    f"Current columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}\n"
                    f"This file format is not compatible with model training/prediction.\n"
                    f"Expected format: Tab-separated file with 'sample' column as index (from preprocess module output)."
                )
        
        # 复用preprocess的有效样本，减少数据量
        if valid_samples:
            df = df.loc[df.index.isin(valid_samples)]
        
        # 拆分特征和目标变量
        # 确保排除phenotype列（无论它在哪个位置）
        if 'phenotype' in df.columns:
            X = df.drop(columns=['phenotype'])
            y = df['phenotype']
        else:
            # 如果没有phenotype列名，假设最后一列是表型
            X = df.iloc[:, :-1]  # 前n-1列：SNP特征
            y = df.iloc[:, -1]   # 最后一列：表型（分类/回归）
        
        # 进一步过滤：排除任何包含"phenotype"的列名
        X = filter_phenotype_from_dataframe(X)

        # ======================== 关键修改：统一SNP特征命名 ========================
        # 目的：让训练和预测阶段的特征命名保持一致（chr_pos），
        # 避免出现训练使用 1_4392528 而预测使用 1_4392528_A 这种不匹配情况。
        import re
        pattern_colon = re.compile(r'^([^:]+):([^_]+)_[^_]+$')       # 1:223238_A -> (1, 223238)
        pattern_under = re.compile(r'^([^_]+)_([^_]+)_[^_]+$')       # 1_223238_A -> (1, 223238)

        renamed_cols: List[str] = []
        for col in X.columns:
            col_str = str(col)
            chr_part = pos_part = None

            m = pattern_colon.match(col_str)
            if m:
                chr_part, pos_part = m.group(1), m.group(2)
            else:
                m2 = pattern_under.match(col_str)
                if m2:
                    chr_part, pos_part = m2.group(1), m2.group(2)

            if chr_part and pos_part:
                renamed_cols.append(f"{chr_part}_{pos_part}")
            else:
                renamed_cols.append(col_str)

        X.columns = renamed_cols
        
        # 清理特征名称（移除LightGBM不支持的特殊字符）
        original_cols = X.columns.tolist()
        cleaned_cols = clean_feature_names(original_cols)
        
        # 创建SNP名称映射（原始名称 -> 清理后名称）
        snp_name_mapping = {}
        
        # 检查是否有重复的特征名称（清理后可能产生重复）
        if len(cleaned_cols) != len(set(cleaned_cols)):
            from collections import Counter, defaultdict
            col_counts = Counter(cleaned_cols)
            col_indices = defaultdict(int)
            final_cols = []
            
            for i, cleaned_col in enumerate(cleaned_cols):
                if col_counts[cleaned_col] > 1:
                    # 如果有重复，添加索引后缀
                    col_indices[cleaned_col] += 1
                    final_col = f"{cleaned_col}_{col_indices[cleaned_col]}"
                else:
                    final_col = cleaned_col
                
                final_cols.append(final_col)
                snp_name_mapping[original_cols[i]] = final_col
            
            X.columns = final_cols
        else:
            X.columns = cleaned_cols
            # 创建映射
            for orig, cleaned in zip(original_cols, cleaned_cols):
                snp_name_mapping[orig] = cleaned
        
        # 数据合法性校验
        if X.empty or y.empty:
            raise ValueError("Loaded training data is empty")
        if X.isnull().any().any():
            X = X.fillna(-1)
        
        return X, y, snp_name_mapping
    except Exception as e:
        logger.error(f"Failed to load training data: {str(e)}")
        raise

# ======================== 3. 模型训练 & 评估 ========================
def init_model(model_type: str, task_type: str, random_state: int = 42) -> Any:
    """初始化模型（分类/回归）"""
    common_params = {"random_state": random_state}
    
    # 模型类映射（减少重复代码）
    model_classes = {
        "classification": {
            "LightGBM": lgb.LGBMClassifier,
            "RandomForest": RandomForestClassifier,
            "XGBoost": xgb.XGBClassifier,
            "SVM": SVC,
            "CatBoost": CatBoostClassifier,
            "Logistic": LogisticRegression
        },
        "regression": {
            "LightGBM": lgb.LGBMRegressor,
            "RandomForest": RandomForestRegressor,
            "XGBoost": xgb.XGBRegressor,
            "SVM": SVR,
            "CatBoost": CatBoostRegressor,
            "Logistic": LinearRegression
        }
    }
    
    # 模型参数配置
    model_params = {
        "LightGBM": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "verbose": -1,
            **common_params
        },
        "RandomForest": {
            "n_estimators": 100,
            "max_depth": None,
            "n_jobs": -1,
            **common_params
        },
        "XGBoost": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "verbosity": 0,
            **common_params
        },
        "SVM": {
            "kernel": "rbf",
            **({"probability": True} if task_type == "classification" else {})
            # 注意：SVM不支持random_state参数
        },
        "CatBoost": {
            "iterations": 100,  # 使用iterations而不是n_estimators，与参数网格保持一致
            "learning_rate": 0.1,
            "verbose": 0,
            # [CATBOOST_CLEAN] 禁止 CatBoost 在当前工作目录创建 catboost_info/
            "allow_writing_files": False,
            **common_params
        },
        "Logistic": {
            "n_jobs": -1,
            **({"max_iter": 1000} if task_type == "classification" else {}),
            **({} if task_type == "regression" else common_params)
            # 注意：LinearRegression不支持random_state参数
        }
    }
    
    if task_type not in model_classes:
        raise ValueError(f"不支持的任务类型: {task_type}，可选: {list(model_classes.keys())}")
    
    if model_type not in model_classes[task_type]:
        raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(model_classes[task_type].keys())}")
    
    # 获取模型类和参数
    model_class = model_classes[task_type][model_type]
    params = model_params[model_type]
    
    model = model_class(**params)
    # [CATBOOST_CLEAN_FALLBACK] 兜底：个别版本/配置下仍可能生成 catboost_info，
    # 这里尽量关闭写文件（若不支持该属性则忽略）
    try:
        if model_type == "CatBoost" and hasattr(model, "set_params"):
            model.set_params(allow_writing_files=False)
    except Exception:
        pass
    return model

def get_param_grid(model_type: str, task_type: str) -> Dict:
    """
    获取模型的超参数网格（用于网格搜索）
    
    :param model_type: 模型类型
    :param task_type: 任务类型（classification/regression）
    :return: 参数字典
    """
    # 公共参数网格（分类和回归相同）
    common_param_grids = {
        "LightGBM": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
            'max_depth': [3, 5, 7, -1],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        "RandomForest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        "XGBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        },
        "CatBoost": {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
            'border_count': [32, 64, 128]
        }
    }
    
    # 任务特定的参数网格
    if task_type == "classification":
        task_specific = {
            "SVM": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            "Logistic": {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'lbfgs', 'saga'],
                'max_iter': [500, 1000, 2000]
            }
        }
    else:  # regression
        task_specific = {
            "SVM": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'epsilon': [0.01, 0.1, 0.2]
            },
            "Logistic": {
                'fit_intercept': [True, False]
            }
        }
    
    # 合并公共和任务特定的参数网格
    param_grids = {**common_param_grids, **task_specific}
    
    if model_type not in param_grids:
        raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(param_grids.keys())}")
    
    return param_grids[model_type]

def perform_grid_search(
    model_type: str,
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
    scoring: Optional[str] = None
) -> Any:
    """
    执行超参数网格搜索
    
    :param model_type: 模型类型
    :param task_type: 任务类型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param param_grid: 参数网格
    :param n_iter: RandomizedSearchCV的迭代次数
    :param cv: 交叉验证折数（用于网格搜索）
    :param random_state: 随机种子
    :param scoring: 评分指标（None则使用默认）
    :return: 最佳模型
    """
    # 初始化基础模型
    base_model = init_model(model_type, task_type, random_state=random_state)
    
    # 设置默认评分指标
    if scoring is None:
        if task_type == "classification":
            scoring = 'roc_auc' if len(y_train.unique()) == 2 else 'f1_weighted'
        else:
            scoring = 'neg_mean_squared_error'
    
    # 使用RandomizedSearchCV（比GridSearchCV更快）
    # 限制搜索次数以避免过长时间
    # 注意：如果在ProcessPoolExecutor的子进程中调用，n_jobs应设为1以避免嵌套并行和资源泄漏
    import os
    n_jobs_value = -1  # 默认使用所有核心
    
    # 检查是否在multiprocessing的子进程中
    try:
        from multiprocessing import current_process
        process_name = current_process().name
        # 如果进程名不是'MainProcess'，说明在子进程中，使用n_jobs=1避免嵌套并行
        if process_name != 'MainProcess':
            n_jobs_value = 1
    except Exception:
        # 如果无法检测，保守地使用n_jobs=1（避免资源泄漏）
        # 但为了性能，只在明确检测到子进程时才使用n_jobs=1
        pass
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs_value,
        random_state=random_state,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    
    return search.best_estimator_

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, task_type: str) -> Dict:
    """模型评估（分类/回归）"""
    metrics = {}
    if task_type == "classification":
        # 分类指标：AUC、召回率、F1得分、准确率
        try:
            # 确保y_true和y_pred是numpy数组且长度一致
            y_true_array = _ensure_numpy_array(y_true)
            y_pred_array = _ensure_numpy_array(y_pred)
            
            if len(y_true_array) == 0 or len(y_pred_array) == 0:
                logger.warning("  True values or predictions are empty, cannot calculate metrics")
                metrics["accuracy"] = "N/A"
                metrics["recall"] = "N/A"
                metrics["f1"] = "N/A"
            elif len(y_true_array) != len(y_pred_array):
                logger.warning(f"  True values and predictions length mismatch: {len(y_true_array)} vs {len(y_pred_array)}")
                metrics["accuracy"] = "N/A"
                metrics["recall"] = "N/A"
                metrics["f1"] = "N/A"
            else:
                metrics["accuracy"] = round(accuracy_score(y_true_array, y_pred_array), 4)
                metrics["recall"] = round(recall_score(y_true_array, y_pred_array, average="weighted"), 4)
                metrics["f1"] = round(f1_score(y_true_array, y_pred_array, average="weighted"), 4)
        except Exception as e:
            logger.warning(f"  Failed to calculate classification metrics: {str(e)}")
            metrics["accuracy"] = "N/A"
            metrics["recall"] = "N/A"
            metrics["f1"] = "N/A"
        try:
            # 处理多分类和二分类的AUC计算
            if y_prob is not None and len(y_prob) > 0:
                from sklearn.metrics import roc_auc_score
                
                # 确保y_true和y_prob是numpy数组
                y_true_array = _ensure_numpy_array(y_true)
                y_prob_array = _ensure_numpy_array(y_prob)
                
                # 检查y_prob的形状
                if len(y_prob_array.shape) == 1:
                    y_prob_array = y_prob_array.reshape(-1, 1)
                
                # 检查数据长度是否一致
                if len(y_true_array) != len(y_prob_array):
                    logger.warning(f"  True values and probabilities length mismatch: {len(y_true_array)} vs {len(y_prob_array)}")
                    metrics["auc"] = "N/A"
                else:
                    # 检查唯一标签数量
                    unique_labels = np.unique(y_true_array)
                    n_unique = len(unique_labels)
                    
                    if n_unique < 2:
                        # 只有一个类别，无法计算AUC
                        metrics["auc"] = "N/A"
                    elif n_unique == 2:
                        # 二分类
                        try:
                            if y_prob_array.shape[1] >= 2:
                                metrics["auc"] = round(roc_auc_score(y_true_array, y_prob_array[:, 1]), 4)
                            elif y_prob_array.shape[1] == 1:
                                metrics["auc"] = round(roc_auc_score(y_true_array, y_prob_array[:, 0]), 4)
                            else:
                                metrics["auc"] = "N/A"
                        except Exception as e:
                            logger.warning(f"  Binary classification AUC calculation failed: {str(e)}")
                            metrics["auc"] = "N/A"
                    else:
                        # 多分类：使用macro平均
                        try:
                            from sklearn.preprocessing import label_binarize
                            n_classes = y_prob_array.shape[1]
                            y_true_binarized = label_binarize(y_true_array, classes=range(n_classes))
                            
                            if y_true_binarized.shape[1] == 1:
                                metrics["auc"] = round(roc_auc_score(y_true_array, y_prob_array[:, 1] if y_prob_array.shape[1] > 1 else y_prob_array[:, 0]), 4)
                            else:
                                metrics["auc"] = round(roc_auc_score(y_true_binarized, y_prob_array, average="macro", multi_class="ovr"), 4)
                        except Exception as e:
                            logger.warning(f"  Multi-class AUC calculation failed: {str(e)}")
                            metrics["auc"] = "N/A"
            else:
                metrics["auc"] = "N/A"
                if task_type == "classification":
                    pass
        except Exception as e:
            logger.warning(f"  Failed to calculate AUC: {str(e)}")
            metrics["auc"] = "N/A"
    else:
        # 回归指标：只保留皮尔逊相关系数和p值
        try:
            from scipy.stats import pearsonr
            # 修复：确保y_true和y_pred都是numpy数组，且长度一致
            y_true_array = _ensure_numpy_array(y_true)
            y_pred_array = _ensure_numpy_array(y_pred)
            
            if len(y_true_array) == 0 or len(y_pred_array) == 0:
                logger.warning("  True values or predictions are empty, cannot calculate Pearson correlation")
                metrics["pearson_correlation"] = "N/A"
                metrics["pearson_pvalue"] = "N/A"
            elif len(y_true_array) != len(y_pred_array):
                logger.warning(f"  True values and predictions length mismatch: {len(y_true_array)} vs {len(y_pred_array)}")
                metrics["pearson_correlation"] = "N/A"
                metrics["pearson_pvalue"] = "N/A"
            else:
                pearson_corr, pearson_pvalue = pearsonr(y_true_array, y_pred_array)
                # 相关系数保留4位小数
                metrics["pearson_correlation"] = round(pearson_corr, 4)
                # p值：常规情况下保留6位小数；当极小时用科学计数法表示，避免被四舍五入为0.0
                if pearson_pvalue < 1e-6:
                    metrics["pearson_pvalue"] = float(f"{pearson_pvalue:.2e}")
                else:
                    metrics["pearson_pvalue"] = round(pearson_pvalue, 6)
        except ImportError:
            logger.warning("  scipy not installed, cannot calculate Pearson correlation coefficient and p-value")
            metrics["pearson_correlation"] = "N/A"
            metrics["pearson_pvalue"] = "N/A"
        except Exception as e:
            logger.warning(f"  Failed to calculate Pearson correlation coefficient: {str(e)}")
            metrics["pearson_correlation"] = "N/A"
            metrics["pearson_pvalue"] = "N/A"
    
    # 日志已在evaluate_model内部输出，这里不再重复
    return metrics

def calculate_feature_importance(
    model: Any,
    model_type: str,
    feature_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str
) -> pd.DataFrame:
    """
    计算特征重要性
    
    :param model: 训练好的模型
    :param model_type: 模型类型
    :param feature_names: 特征名称列表
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param task_type: 任务类型（classification/regression）
    :return: 特征重要性DataFrame
    """
    
    try:
        # 不同模型的特征重要性获取方式
        if model_type in ["LightGBM", "XGBoost", "CatBoost"]:
            # 树模型：使用feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # 某些情况下需要先predict
                model.predict(X_train.iloc[:10])
                importances = model.feature_importances_
        
        elif model_type == "RandomForest":
            # 随机森林：使用feature_importances_
            importances = model.feature_importances_
        
        elif model_type == "Logistic":
            # 线性模型：使用系数的绝对值
            if hasattr(model, 'coef_'):
                if task_type == "classification":
                    # 多分类：取所有类别的平均重要性
                    importances = np.abs(model.coef_).mean(axis=0)
                else:
                    # 回归：直接使用系数绝对值
                    importances = np.abs(model.coef_[0])
            else:
                logger.warning(f"  {model_type} model does not support feature importance calculation")
                return pd.DataFrame()
        
        elif model_type == "SVM":
            # SVM：使用支持向量的权重（仅适用于线性核）
            if hasattr(model, 'kernel') and model.kernel == 'linear':
                if hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                else:
                    # 线性核但没有coef_，尝试使用permutation importance
                    logger.info("  SVM linear kernel: using permutation importance as fallback")
                    try:
                        # 使用较小的样本集以加快计算
                        n_samples = min(1000, len(X_train))
                        X_sample = X_train.sample(n=n_samples, random_state=42) if len(X_train) > n_samples else X_train
                        y_sample = y_train.loc[X_sample.index] if len(X_train) > n_samples else y_train
                        
                        scoring = 'roc_auc' if task_type == 'classification' else 'r2'
                        perm_result = permutation_importance(
                            model, X_sample, y_sample, 
                            n_repeats=10, 
                            random_state=42,
                            scoring=scoring,
                            n_jobs=1
                        )
                        importances = perm_result.importances_mean
                    except Exception as e:
                        logger.warning(f"  SVM permutation importance calculation failed: {str(e)}")
                        return pd.DataFrame()
            else:
                # 非线性核SVM：使用permutation importance
                logger.info("  SVM non-linear kernel: using permutation importance")
                try:
                    # 使用较小的样本集以加快计算
                    n_samples = min(1000, len(X_train))
                    X_sample = X_train.sample(n=n_samples, random_state=42) if len(X_train) > n_samples else X_train
                    y_sample = y_train.loc[X_sample.index] if len(X_train) > n_samples else y_train
                    
                    scoring = 'roc_auc' if task_type == 'classification' else 'r2'
                    perm_result = permutation_importance(
                        model, X_sample, y_sample, 
                        n_repeats=10, 
                        random_state=42,
                        scoring=scoring,
                        n_jobs=1
                    )
                    importances = perm_result.importances_mean
                except Exception as e:
                    logger.warning(f"  SVM permutation importance calculation failed: {str(e)}")
                    return pd.DataFrame()
        
        else:
            logger.warning(f"  {model_type} model does not support feature importance calculation")
            return pd.DataFrame()
        
        # 获取原始值（带正负号）用于计算正负效应
        original_values = None
        if model_type == "Logistic":
            # 线性模型：使用原始系数
            if hasattr(model, 'coef_'):
                if task_type == "classification":
                    original_values = model.coef_.mean(axis=0)
                else:
                    original_values = model.coef_[0]
        elif model_type == "SVM":
            # SVM：如果是线性核且有coef_，使用原始系数；否则使用permutation importance（都是正值）
            if hasattr(model, 'kernel') and model.kernel == 'linear' and hasattr(model, 'coef_'):
                if task_type == "classification":
                    original_values = model.coef_.mean(axis=0) if len(model.coef_.shape) > 1 else model.coef_[0]
                else:
                    original_values = model.coef_[0]
            else:
                # 非线性核或使用permutation importance：都是正值，正负效应设为1
                original_values = importances
        elif model_type in ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]:
            # 树模型：特征重要性通常是正数，正负效应设为1（正）
            original_values = importances  # 树模型的重要性都是正数
        
        # 如果无法获取原始值，使用绝对值（正负效应设为1）
        if original_values is None:
            original_values = importances
        
        # 计算绝对值
        abs_values = np.abs(original_values)
        
        # 确定正负效应（1表示正效应，-1表示负效应）
        sign_effect = np.sign(original_values)
        # 将0转换为1（表示正效应）
        sign_effect = np.where(sign_effect == 0, 1, sign_effect)
        
        # 创建特征重要性DataFrame（三列：特征名、绝对值、正负效应）
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_abs': abs_values,
            'effect': sign_effect.astype(int)
        })
        
        # 按绝对值降序排序
        importance_df = importance_df.sort_values('importance_abs', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        logger.info("  Feature importance calculation completed")
        return importance_df
        
    except Exception as e:
        logger.error(f"  Feature importance calculation failed: {str(e)}")
        return pd.DataFrame()

def calculate_shap_values(
    model: Any,
    model_type: str,
    feature_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    # 参与SHAP解释的样本数（默认全量；可用于控制输出/耗时）
    shap_sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    计算SHAP值（格式与feature_importance相同）
    
    :param model: 训练好的模型
    :param model_type: 模型类型
    :param feature_names: 特征名称列表
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param task_type: 任务类型（classification/regression）
    :return: SHAP值DataFrame（三列：feature, shap_abs, effect）
    """
    
    if not SHAP_AVAILABLE:
        logger.warning("  SHAP library not installed, cannot calculate SHAP values. Please run: pip install shap")
        return pd.DataFrame()
    
    try:
        shap_t0 = time.time()
        # 使用全部数据计算SHAP值
        X_shap = X_train
        y_shap = y_train
        X_input = X_shap[feature_names] if feature_names else X_shap

        # 默认：不指定 shap_sample_size 就用全量；SVM/Kernel 路径会再做上限控制
        if shap_sample_size is not None and len(X_input) > shap_sample_size:
            X_input = X_input.sample(n=int(shap_sample_size), random_state=42)
            y_shap = y_shap.loc[X_input.index]
        
        # 根据模型类型选择合适的SHAP解释器
        if model_type in ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]:
            # 树模型：使用TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap[feature_names] if feature_names else X_shap)
        elif model_type == "Logistic":
            # 线性逻辑回归：使用LinearExplainer
            explainer = shap.LinearExplainer(model, X_shap[feature_names] if feature_names else X_shap)
            shap_values = explainer.shap_values(X_shap[feature_names] if feature_names else X_shap)
        elif model_type == "SVM":
            # ======================== SVM: 固定使用线性代理模型（近似） ========================
            # 说明：
            # - 训练模型默认是 rbf 核（见 init_model），KernelExplainer 计算复杂度极高且在 1k+ 特征下非常慢
            # - 这里固定用 LinearSVR / LinearSVC 拟合一个线性代理模型，再用 LinearExplainer 计算SHAP
            from sklearn.svm import LinearSVR, LinearSVC

            # 控制代理模型训练数据量，避免在超大样本时耗时（保持通用性）
            X_sur = X_input
            if len(X_sur) > 5000:
                X_sur = X_sur.sample(n=5000, random_state=42)
            y_sur = y_shap.loc[X_sur.index]

            logger.info(
                f"  [SHAP][SVM] Fixed mode: linear surrogate + LinearExplainer "
                f"(surrogate_samples={len(X_sur):,}, features={X_sur.shape[1]:,}, task={task_type})"
            )

            if task_type == "classification":
                surrogate = LinearSVC(C=1.0, dual=True, max_iter=5000)
            else:
                surrogate = LinearSVR(C=1.0, dual=True, max_iter=5000)

            surrogate.fit(X_sur, y_sur)
            explainer = shap.LinearExplainer(surrogate, X_sur, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_input)
        else:
            # 其他模型：使用KernelExplainer（较慢）
            logger.warning(f"  {model_type} model uses KernelExplainer for SHAP values, may be slow")
            # 使用少量样本作为背景数据
            background_size = min(100, len(X_shap))
            background = X_shap[feature_names].sample(n=background_size, random_state=42) if feature_names else X_shap.sample(n=background_size, random_state=42)
            explainer = shap.KernelExplainer(
                model.predict if task_type == "regression" else lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x),
                background
            )
            shap_values = explainer.shap_values(X_shap[feature_names] if feature_names else X_shap)
        
        # 处理多分类任务的SHAP值（取平均）
        if isinstance(shap_values, list):
            # 多分类：对每个类别的SHAP值取平均
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        elif len(shap_values.shape) > 2:
            # 多维数组：取平均
            shap_values = np.mean(shap_values, axis=0)
        
        # 确保是2D数组
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # 计算每个特征的平均SHAP值（绝对值）
        mean_shap_abs = np.abs(shap_values).mean(axis=0)
        
        # 计算每个特征的平均SHAP值（带符号，用于确定正负效应）
        mean_shap_signed = shap_values.mean(axis=0)
        
        # 确定正负效应（1表示正效应，-1表示负效应）
        sign_effect = np.sign(mean_shap_signed)
        # 将0转换为1（表示正效应）
        sign_effect = np.where(sign_effect == 0, 1, sign_effect)
        
        # 创建SHAP值DataFrame（三列：特征名、绝对值、正负效应）
        # 特征名统一使用当前训练特征矩阵的列名（X_train.columns），
        # 而不是外部传入的原始基因型 SNP ID 列表，保证与整体特征处理逻辑一致。
        # 同时增加鲁棒性检查，确保所有数组长度一致，避免
        # "All arrays must be of the same length" 错误。
        feature_list = X_train.columns.tolist()
        n_features_from_shap = len(mean_shap_abs)
        n_features_from_names = len(feature_list)
        n_features_from_sign = len(sign_effect)

        # 取三者中的最小长度进行对齐
        min_len = min(n_features_from_shap, n_features_from_names, n_features_from_sign)
        if not (n_features_from_shap == n_features_from_names == n_features_from_sign):
            logger.warning(
                " SHAP特征长度不一致，将按最小长度对齐: "
                f"shap={n_features_from_shap}, names={n_features_from_names}, sign={n_features_from_sign}"
            )

        feature_list = feature_list[:min_len]
        mean_shap_abs = mean_shap_abs[:min_len]
        sign_effect = sign_effect[:min_len]

        shap_df = pd.DataFrame({
            'feature': feature_list,
            'shap_abs': mean_shap_abs,
            'effect': sign_effect.astype(int)
        })
        
        # 按绝对值降序排序
        shap_df = shap_df.sort_values('shap_abs', ascending=False)
        shap_df = shap_df.reset_index(drop=True)
        
        logger.info(
            f"  SHAP values calculation completed "
            f"(samples_used={len(X_input):,}, elapsed={time.time()-shap_t0:.2f}s)"
        )
        return shap_df
        
    except Exception as e:
        logger.error(f"  SHAP values calculation failed: {str(e)}")
        import traceback
        return pd.DataFrame()

def _ensure_numpy_array(data: Any) -> np.ndarray:
    """
    确保数据是numpy数组（提取公共代码）
    
    :param data: 输入数据（可能是pd.Series、list或np.ndarray）
    :return: numpy数组
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.values
    else:
        return np.array(data)

def filter_phenotype_columns(columns: List[str]) -> List[str]:
    """
    过滤掉包含phenotype的列名（提取公共代码）
    
    :param columns: 列名列表
    :return: 过滤后的列名列表
    """
    return [col for col in columns if 'phenotype' not in str(col).lower()]

def filter_phenotype_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    从DataFrame中过滤掉包含phenotype的列
    
    :param df: 输入DataFrame
    :return: 过滤后的DataFrame
    """
    phenotype_cols = [col for col in df.columns if 'phenotype' in str(col).lower()]
    # 兜底清理：剔除 preprocess/拼接过程中引入的 index/index.N 伪特征列
    # 典型来源：多染色体拼接时重复列名，read_csv 会自动改名为 index.1/index.2/...
    import re
    index_artifact_cols = [
        col for col in df.columns
        if re.match(r'^index(\.\d+)?$', str(col), flags=re.IGNORECASE)
    ]

    drop_cols = []
    if phenotype_cols:
        drop_cols.extend(phenotype_cols)
    if index_artifact_cols:
        drop_cols.extend(index_artifact_cols)

    if drop_cols:
        # 去重保持顺序
        seen = set()
        drop_cols_unique = []
        for c in drop_cols:
            if c not in seen:
                seen.add(c)
                drop_cols_unique.append(c)
        if index_artifact_cols:
            logger.warning(
                f"Detected and removed index artifact columns from features: {index_artifact_cols[:5]}"
                f"{'...' if len(index_artifact_cols) > 5 else ''}"
            )
        return df.drop(columns=drop_cols_unique, errors='ignore')
    return df

def save_training_results(
    model: Any,
    metrics: Dict,
    selected_snps: List[str],
    output_dir: str,
    model_type: str,
    task_type: str,
    feature_importance_df: Optional[pd.DataFrame] = None,
    shap_df: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    cv_results: Optional[Dict] = None
) -> None:
    """
    保存训练结果（模型文件、评估指标、筛选的SNP列表、特征重要性、SHAP值）
    
    :param model: 训练好的模型
    :param metrics: 评估指标
    :param selected_snps: 筛选的SNP列表
    :param output_dir: 输出目录
    :param model_type: 模型类型
    :param task_type: 任务类型
    :param feature_importance_df: 特征重要性DataFrame
    :param shap_df: SHAP值DataFrame
    :param y_test: 测试集真实值（用于绘制性能曲线）
    :param y_pred: 测试集预测值（用于绘制性能曲线）
    :param y_prob: 测试集预测概率（分类任务需要，用于绘制性能曲线）
    :param cv_results: 交叉验证结果字典（用于绘制交叉验证曲线）
    """
    # 创建模型专属目录
    model_dir = Path(output_dir) / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    # ======================== 输出命名规范化（standardized） ========================
    # 统一输出前缀：{output_dir}/{model_type}/{model_type}
    # 所有产物文件都以该前缀 + 固定后缀命名，避免目录内出现“裸文件名”导致覆盖与歧义
    output_prefix = model_dir / model_type
    
    # 1. 保存模型文件（规范：仍保持“模型名称”命名）
    # - 文件名固定为：{model_type}_model.pkl
    # - 避免把更多前缀信息塞进模型文件名，便于用户直观识别与手动指定
    model_file = model_dir / f"{model_type}_model.pkl"
    import joblib
    joblib.dump(model, model_file)
    
    # 2. 保存评估指标
    metrics_file = Path(f"{output_prefix}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "model_type": model_type,
            "task_type": task_type,
            "metrics": metrics,
            "training_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # 2.5 保存训练阶段使用的特征列表（用于预测阶段特征对齐）
    # 说明：
    # - selected_snps 已经是去除 phenotype 后的所有特征列名（且经过 clean_feature_names 清洗）
    # - 预测阶段将以此列表为基准：
    #   * 训练有、预测也有 → 正常使用；
    #   * 训练有、预测没有 → 在预测矩阵中补一列，整列填0；
    #   * 训练没有、预测有 → 在预测阶段丢弃该列。
    try:
        features_file = Path(f"{output_prefix}_training_features.json")
        with open(features_file, "w") as f:
            json.dump(
                {
                    "model_type": model_type,
                    "task_type": task_type,
                    "feature_names": list(selected_snps) if selected_snps is not None else [],
                    "saved_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )
    except Exception as e:
        # 保存失败不影响训练流程，但会在预测阶段失去自动对齐能力
        logger.warning(f"   Failed to save training feature names (ignored): {e}")
    
    # 3. 保存特征重要性（可选）
    if feature_importance_df is not None and not feature_importance_df.empty:
        # 排除phenotype列
        filtered_importance_df = feature_importance_df[
            feature_importance_df['feature'].astype(str).apply(lambda x: 'phenotype' not in x.lower())
        ]
        importance_file = Path(f"{output_prefix}_feature_importance.txt")
        filtered_importance_df.to_csv(importance_file, sep='\t', index=False)
    
    # 4.5. 保存SHAP值（可选，格式与feature_importance相同）
    if shap_df is not None and not shap_df.empty:
        # 排除phenotype列
        filtered_shap_df = shap_df[
            shap_df['feature'].astype(str).apply(lambda x: 'phenotype' not in x.lower())
        ]
        # 重命名列以匹配feature_importance格式（feature, importance_abs, effect）
        # 但为了区分，我们保持shap_abs列名，或者重命名为importance_abs以兼容可视化
        shap_file = Path(f"{output_prefix}_shap_values.txt")
        # 为了兼容可视化模块，将shap_abs重命名为importance_abs
        shap_output_df = filtered_shap_df.copy()
        shap_output_df = shap_output_df.rename(columns={'shap_abs': 'importance_abs'})
        shap_output_df.to_csv(shap_file, sep='\t', index=False)
        
    # 5. 保存所有绘图数据到一个统一文件（用于visualization模块）
    if y_test is not None and y_pred is not None:
        import pickle
        plotting_data_file = Path(f"{output_prefix}_plotting_data.npz")
        save_dict = {
            # 预测数据（numpy数组）
            'y_test': y_test.values if isinstance(y_test, pd.Series) else y_test,
            'y_pred': y_pred,
            # 元数据（字符串）
            'model_type': np.array([model_type], dtype=object),
            'task_type': np.array([task_type], dtype=object),
            # 兼容字段名（fixed）：publication_quality 固定由内部常量控制
            'publication_quality': np.array([PUB_QUALITY_MODE], dtype=bool)
        }
        # 预测概率（如果存在）
        if y_prob is not None:
            save_dict['y_prob'] = y_prob
        # 交叉验证结果（使用pickle序列化字典）
        if cv_results is not None:
            save_dict['cv_results'] = np.array([pickle.dumps(cv_results)], dtype=object)
        
        np.savez_compressed(plotting_data_file, **save_dict)
    
    logger.info(f"  Results saved successfully: {model_dir}")

# ======================== 5. 核心训练函数 ========================

def _run_single_fold_cv(
    fold_idx: int,
    train_idx,
    val_idx,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    task_type: str,
    random_state: int
):
    """
    进程池中执行的单折训练与评估函数（用于5折交叉验证并行）
    
    修复说明：
    - 使用iloc进行位置索引访问，确保即使DataFrame/Series的索引不是整数也能正确工作
    - train_idx和val_idx是numpy数组，表示位置索引，与iloc兼容
    """
    logger = logging.getLogger(__name__)

    try:
        # 修复：使用iloc进行位置索引访问，确保索引正确
        # train_idx和val_idx是numpy数组，表示位置索引（从0开始）
        # iloc使用位置索引，即使DataFrame/Series的索引是字符串也能正确工作
        X_train_fold = X.iloc[train_idx].copy()  # 添加copy()避免SettingWithCopyWarning
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()

        # 超参数搜索
        param_grid = get_param_grid(model_type, task_type)
        model_fold = perform_grid_search(
            model_type=model_type,
            task_type=task_type,
            X_train=X_train_fold,
            y_train=y_train_fold,
            param_grid=param_grid,
            n_iter=20,  # 每次搜索20个参数组合
            cv=3,       # 网格搜索使用3折交叉验证
            random_state=random_state
        )

        # 预测与概率
        y_pred_fold = model_fold.predict(X_val_fold)
        y_prob_fold = None
        if task_type == "classification":
            try:
                if hasattr(model_fold, "predict_proba"):
                    y_prob_fold = model_fold.predict_proba(X_val_fold)
                elif hasattr(model_fold, "decision_function"):
                    decision_scores = model_fold.decision_function(X_val_fold)
                    from sklearn.utils.extmath import softmax
                    if len(decision_scores.shape) == 1:
                        prob_neg = 1 / (1 + np.exp(decision_scores))
                        prob_pos = 1 - prob_neg
                        y_prob_fold = np.column_stack([prob_neg, prob_pos])
                    else:
                        y_prob_fold = softmax(decision_scores)
            except Exception as e:
                logger.warning(f"[Fold {fold_idx}] Failed to obtain prediction probabilities: {str(e)}")

        fold_metrics = evaluate_model(y_val_fold, y_pred_fold, y_prob_fold, task_type)
        
        # 清理资源：删除模型引用，帮助GC
        del model_fold
        
        # 清理joblib的临时目录（如果存在）
        try:
            import joblib
            import tempfile
            # joblib可能会在tempfile.gettempdir()下创建临时目录
            # 这里我们只是确保资源被释放，实际的清理由resource_tracker处理
            import gc
            gc.collect()
        except Exception:
            pass
        
        return fold_idx, fold_metrics, y_val_fold, y_pred_fold, y_prob_fold
    except Exception as e:
        logger.error(f"[Fold {fold_idx}] Error in worker process: {str(e)}", exc_info=True)
        raise
def run_single_model(
    input_path: str,
    model_type: str,
    output_dir: str,
    task_type: Optional[str] = None,
    n_folds: int = 5,
    random_state: int = 42,
    # 特征重要性计算参数（可选）
    calculate_feature_importance: bool = False,
    # 图表质量固定开关（fixed，已不对外开放）
    pub_quality_mode: bool = True,
    # 预加载的训练数据（可选，用于train-all中避免重复读取文件）
    preloaded_data: Optional[Dict[str, Any]] = None,
) -> int:
    """
    单模型训练主函数
    说明：
    - GWAS / LD / GWAS+LD 综合过滤逻辑已经前移到 preprocess 模块
    - 训练阶段使用 preprocess 已筛选的特征，不再执行额外的特征筛选
    - 本函数支持两种数据获取方式：
        1) 通过 input_path 解析并读取训练文件（默认行为）
        2) 通过 preloaded_data 直接使用预加载的训练数据（用于 train-all，避免重复读盘）
    - 交叉验证流程调整为：先在全数据上进行一次随机搜索得到最佳超参数，再使用固定超参数做 n_folds 折交叉验证评估，不在每一折里重复做超参搜索
    """
    # 强制要求使用 preprocess 生成的 metadata.json 作为训练入口，避免 txt 直读导致样本/任务类型/临时目录等信息不完整
    if not str(input_path).endswith("_metadata.json"):
        raise ValueError(
            "训练输入必须为 preprocess 生成的元数据文件（*_metadata.json）。"
            f"当前输入: {input_path}"
        )
    # 保存对calculate_feature_importance函数的引用，避免与参数名冲突
    _calculate_feature_importance_func = globals()['calculate_feature_importance']
    # 固定实现：忽略外部传入值，统一使用内部常量
    pub_quality_mode = PUB_QUALITY_MODE

    start_time = time.time()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ======================== 临时目录规范化（standardized） ========================
    # 临时文件放在基础输出目录下的 tmp/train 子目录中（与阶段目录同级）
    # 每个阶段在 tmp 下创建自己的子目录，避免不同阶段的临时文件互相干扰
    # 例如：如果 output_dir 是 test/train，则 tmp_root 是 test/tmp/train
    base_output_dir = output_dir_path.parent
    # 重要：tmp 必须按“模型+进程”隔离。
    # train-all 会并行调用 run_single_model；若共享 tmp 或在子进程里删除 base_output_dir/tmp，
    # 会产生竞态：进程互相删对方临时文件，导致随机失败/结果不完整。
    tmp_root = base_output_dir / "tmp" / "train" / model_type
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 使用全局临时文件管理器注册临时目录/文件
    def register_dir(p: Path) -> Path:
        return _temp_file_manager.register_dir(p)

    def register_file(p: Path) -> Path:
        return _temp_file_manager.register_file(p)

    # 使用进程级目录，避免同一模型在不同进程里冲突
    run_tmp_dir = register_dir(tmp_root / f"pid_{os.getpid()}")

    # 预处理阶段的临时目录（从元数据读取，不清理，preprocess模块执行完毕不删除tmp目录）
    preprocess_tmp_dir = None

    try:
        # Step 1: 获取训练数据和任务类型
        if preloaded_data is not None:
            # 来自 train-all 预加载的数据
            X = preloaded_data["X"]
            y = preloaded_data["y"]
            # 预处理阶段的临时目录（如果存在）
            if preloaded_data.get("preprocess_tmp_dir"):
                preprocess_tmp_dir = Path(preloaded_data["preprocess_tmp_dir"]).absolute()
                _temp_file_manager.register_preprocess_tmp_dir(preprocess_tmp_dir)
            # 若未显式指定 task_type，优先使用预加载中的任务类型
            def infer_task_type_from_y(series: pd.Series) -> str:
                values = series.values
                unique_vals = np.unique(values)
                is_all_int = np.all(values == np.round(values))
                if len(unique_vals) <= 10 and is_all_int:
                    return "classification"
                return "regression"

            if task_type is None:
                task_type = preloaded_data.get("task_type") or infer_task_type_from_y(y)
                logger.info(f"Task type: {task_type}")
        else:
            # 通过 input_path 解析元数据并读取训练文件
            input_info = parse_train_input_path(input_path)
            train_file = input_info["train_file"]
            valid_samples = input_info["metadata"]["valid_samples"] if input_info["metadata"] else None

            # 如果元数据里带有 preprocess_tmp_dir（tmp_p目录），则记录（但不清理，preprocess模块执行完毕不删除）
            if input_info["preprocess_tmp_dir"]:
                preprocess_tmp_dir = Path(input_info["preprocess_tmp_dir"]).absolute()
                _temp_file_manager.register_preprocess_tmp_dir(preprocess_tmp_dir)

            # 加载训练数据
            X, y, snp_name_mapping = load_training_data(train_file, valid_samples)

            # 若未提供task_type，则基于表型推断（规则与preprocess一致）
            def infer_task_type_from_y(series: pd.Series) -> str:
                values = series.values
                unique_vals = np.unique(values)
                is_all_int = np.all(values == np.round(values))
                if len(unique_vals) <= 10 and is_all_int:
                    return "classification"
                return "regression"

            if task_type is None:
                task_type = infer_task_type_from_y(y)
                logger.info(f"Task type: {task_type}")

        logger.info(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]:,} features")

        # 使用全部特征（preprocess 已完成所需的特征筛选）
        X_filtered = X

        # Step 2: 先在全数据上进行一次随机搜索，确定最佳超参数（内部自带交叉验证）
        logger.info("Starting hyperparameter search (RandomizedSearchCV)...")
        param_grid = get_param_grid(model_type, task_type)
        # 使用与外层相同的折数进行搜索，避免过度计算
        best_model = perform_grid_search(
            model_type=model_type,
            task_type=task_type,
            X_train=X_filtered,
            y_train=y,
            param_grid=param_grid,
            n_iter=30,      # 适度的搜索次数
            cv=n_folds,     # 与外层评估折数保持一致
            random_state=random_state
        )
        best_params = best_model.get_params()
        logger.info("Hyperparameter search completed, starting cross-validation with fixed best parameters")

        # Step 3: 使用固定超参数做 n_folds 折交叉验证评估（不再在每一折内部重复做随机搜索）
        if task_type == "classification":
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X_filtered, y))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X_filtered, y))

        logger.info(f"Starting {n_folds}-fold cross-validation with fixed hyperparameters")

        cv_fold_metrics = []
        all_y_test = []
        all_y_pred = []
        all_y_prob = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
            train_idx = np.asarray(train_idx)
            val_idx = np.asarray(val_idx)

            X_train_fold = X_filtered.iloc[train_idx].copy()
            X_val_fold = X_filtered.iloc[val_idx].copy()
            y_train_fold = y.iloc[train_idx].copy()
            y_val_fold = y.iloc[val_idx].copy()

            # 使用最佳超参数初始化模型（不做搜索，只做一次拟合）
            model_fold = init_model(model_type, task_type, random_state=random_state)
            try:
                model_fold.set_params(**best_params)
            except ValueError as e:
                # 如果存在不兼容的参数（极少数情况），记录警告并继续使用默认参数
                logger.warning(f"[Fold {fold_idx}] Failed to apply best_params to model: {e}")

            y_pred_fold = None
            y_prob_fold = None
            try:
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred_fold = model_fold.predict(X_val_fold)
                if task_type == "classification":
                    try:
                        if hasattr(model_fold, "predict_proba"):
                            y_prob_fold = model_fold.predict_proba(X_val_fold)
                        elif hasattr(model_fold, "decision_function"):
                            decision_scores = model_fold.decision_function(X_val_fold)
                            from sklearn.utils.extmath import softmax
                            if len(decision_scores.shape) == 1:
                                prob_neg = 1 / (1 + np.exp(decision_scores))
                                prob_pos = 1 - prob_neg
                                y_prob_fold = np.column_stack([prob_neg, prob_pos])
                            else:
                                y_prob_fold = softmax(decision_scores)
                    except Exception as e:
                        logger.warning(f"[Fold {fold_idx}] Failed to obtain prediction probabilities: {str(e)}")
            except Exception as e:
                logger.error(f"[Fold {fold_idx}] Error during training/evaluation: {str(e)}", exc_info=True)
                cv_fold_metrics.append({})
                continue

            fold_metrics = evaluate_model(y_val_fold, y_pred_fold, y_prob_fold, task_type)

            # 日志输出每折关键指标
            if task_type == "regression":
                corr = fold_metrics.get("pearson_correlation", "N/A")
                logger.info(f"[{fold_idx}/{n_folds}] Pearson correlation: {corr}")
            else:
                acc = fold_metrics.get("accuracy", "N/A")
                auc = fold_metrics.get("auc", "N/A")
                logger.info(f"[{fold_idx}/{n_folds}] Accuracy: {acc}, AUC: {auc}")

            cv_fold_metrics.append(fold_metrics)
            all_y_test.append(y_val_fold)
            all_y_pred.append(y_pred_fold)
            if y_prob_fold is not None:
                all_y_prob.append(y_prob_fold)

        # 计算平均指标
        logger.info(f"{n_folds}-fold cross-validation average results:")

        avg_metrics = {}
        if task_type == "regression":
            pearson_corrs = [m.get('pearson_correlation', 0) for m in cv_fold_metrics 
                            if isinstance(m.get('pearson_correlation'), (int, float))]
            pearson_pvalues = [m.get('pearson_pvalue', 1) for m in cv_fold_metrics 
                              if isinstance(m.get('pearson_pvalue'), (int, float))]
            if pearson_corrs:
                avg_metrics['pearson_correlation'] = float(round(np.mean(pearson_corrs), 4))
                avg_metrics['pearson_correlation_std'] = float(round(np.std(pearson_corrs), 4))
            if pearson_pvalues:
                mean_p = float(np.mean(pearson_pvalues))
                std_p = float(np.std(pearson_pvalues))
                # 对极小的平均p值和标准差使用科学计数法表示
                if mean_p < 1e-6:
                    avg_metrics['pearson_pvalue'] = float(f"{mean_p:.2e}")
                else:
                    avg_metrics['pearson_pvalue'] = float(round(mean_p, 6))
                if std_p < 1e-6:
                    avg_metrics['pearson_pvalue_std'] = float(f"{std_p:.2e}")
                else:
                    avg_metrics['pearson_pvalue_std'] = float(round(std_p, 6))
        else:
            accuracies = [m.get('accuracy', 0) for m in cv_fold_metrics 
                         if isinstance(m.get('accuracy'), (int, float))]
            recalls = [m.get('recall', 0) for m in cv_fold_metrics 
                      if isinstance(m.get('recall'), (int, float))]
            f1_scores = [m.get('f1', 0) for m in cv_fold_metrics 
                        if isinstance(m.get('f1'), (int, float))]
            aucs = [m.get('auc', 0) for m in cv_fold_metrics 
                   if isinstance(m.get('auc'), (int, float)) and m.get('auc') != 'N/A']
            
            if accuracies:
                avg_metrics['accuracy'] = float(round(np.mean(accuracies), 4))
                avg_metrics['accuracy_std'] = float(round(np.std(accuracies), 4))
            if recalls:
                avg_metrics['recall'] = float(round(np.mean(recalls), 4))
                avg_metrics['recall_std'] = float(round(np.std(recalls), 4))
            if f1_scores:
                avg_metrics['f1'] = float(round(np.mean(f1_scores), 4))
                avg_metrics['f1_std'] = float(round(np.std(f1_scores), 4))
            if aucs:
                avg_metrics['auc'] = float(round(np.mean(aucs), 4))
                avg_metrics['auc_std'] = float(round(np.std(aucs), 4))
        
        logger.info(f"   {avg_metrics}")
        
        # Step 9: 使用全部数据训练最终模型（用于特征重要性）
        param_grid = get_param_grid(model_type, task_type)
        final_model = perform_grid_search(
            model_type=model_type,
            task_type=task_type,
            X_train=X_filtered,
            y_train=y,
            param_grid=param_grid,
            n_iter=30,  # 最终模型使用更多迭代次数
            cv=5,  # 使用5折交叉验证
            random_state=random_state
        )
        logger.info("Final model training completed")
        
        # 合并所有折的预测结果
        # 修复：确保Series合并时索引唯一，避免重复索引导致的错误
        # 在交叉验证中，每个fold的验证集互不重叠，但为了安全起见，使用ignore_index=False保留原始索引
        # 如果确实有重复索引，使用keys参数区分不同fold
        if all_y_test:
            try:
                y_test_combined = pd.concat(all_y_test, axis=0, ignore_index=False)
                # 检查是否有重复索引（理论上不应该有，因为交叉验证的fold互不重叠）
                if y_test_combined.index.duplicated().any():
                    logger.warning("  Detected duplicate indices in y_test, using keys to distinguish folds")
                    y_test_combined = pd.concat(all_y_test, axis=0, keys=range(len(all_y_test)), names=['fold', None])
                    y_test_combined = y_test_combined.droplevel(0)  # 移除fold级别，保留原始索引
            except Exception as e:
                logger.warning(f"  Failed to concatenate y_test with original indices: {e}, using ignore_index=True")
                y_test_combined = pd.concat(all_y_test, axis=0, ignore_index=True)
        else:
            y_test_combined = None
        
        # 修复：确保numpy数组正确连接
        y_pred_combined = np.concatenate(all_y_pred) if all_y_pred else None
        
        # 修复：确保概率矩阵正确堆叠，处理可能的维度不一致问题
        if all_y_prob and len(all_y_prob) > 0:
            try:
                # 检查所有概率数组的形状是否一致
                shapes = [prob.shape for prob in all_y_prob]
                if len(set(shapes)) > 1:
                    logger.warning(f"  Inconsistent probability shapes: {shapes}, attempting to align")
                    # 如果形状不一致，尝试找到最小维度并截断
                    min_shape = min(shapes, key=lambda x: x[0])
                    all_y_prob = [prob[:min_shape[0]] if prob.shape[0] > min_shape[0] else prob for prob in all_y_prob]
                y_prob_combined = np.vstack(all_y_prob)
            except Exception as e:
                logger.warning(f"  Failed to stack probability arrays: {e}")
                y_prob_combined = None
        else:
            y_prob_combined = None
        
        metrics = avg_metrics  # 使用平均指标作为最终指标

        # Step 10: 计算特征重要性（可选，使用最终模型）
        feature_importance_df = None
        if calculate_feature_importance:
            feature_cols = filter_phenotype_columns(X_filtered.columns.tolist())
            
            feature_importance_df = _calculate_feature_importance_func(
                model=final_model,
                model_type=model_type,
                feature_names=feature_cols,
                X_train=X_filtered[feature_cols] if feature_cols else X_filtered,
                y_train=y,
                task_type=task_type
            )
            logger.info("Feature importance calculation completed")
        
        # Step 10.5: 计算SHAP值（默认计算，使用最终模型）
        # SVM 使用 KernelExplainer（分类任务使用 predict_proba，回归任务使用 predict）
        feature_cols = filter_phenotype_columns(X_filtered.columns.tolist())
        
        shap_df = calculate_shap_values(
            model=final_model,
            model_type=model_type,
            feature_names=feature_cols,
            X_train=X_filtered[feature_cols] if feature_cols else X_filtered,
            y_train=y,
            task_type=task_type
        )
        if not shap_df.empty:
            logger.info("SHAP values calculation completed")
        else:
            # 兜底：空结果可能来自真正失败（如shap未安装或计算错误）
            logger.warning("  SHAP values calculation failed or returned empty results")
            # 如果SHAP值计算失败，使用feature importance作为替代
            if feature_importance_df is not None and not feature_importance_df.empty:
                logger.info("  Using feature importance as alternative to SHAP values")
                # 将feature importance转换为SHAP格式
                shap_df = feature_importance_df.copy()
                shap_df = shap_df.rename(columns={'importance_abs': 'shap_abs'})
                # 确保列顺序正确：feature, shap_abs, effect
                shap_df = shap_df[['feature', 'shap_abs', 'effect']]
                logger.info("  Feature importance converted to SHAP format as fallback")
            else:
                # 如果feature importance也没有计算，尝试计算它
                logger.info("  Attempting to calculate feature importance as fallback for SHAP values")
                feature_importance_fallback = _calculate_feature_importance_func(
                    model=final_model,
                    model_type=model_type,
                    feature_names=feature_cols,
                    X_train=X_filtered[feature_cols] if feature_cols else X_filtered,
                    y_train=y,
                    task_type=task_type
                )
                if not feature_importance_fallback.empty:
                    logger.info("  Feature importance calculated successfully, using as SHAP alternative")
                    # 将feature importance转换为SHAP格式
                    shap_df = feature_importance_fallback.copy()
                    shap_df = shap_df.rename(columns={'importance_abs': 'shap_abs'})
                    # 确保列顺序正确：feature, shap_abs, effect
                    shap_df = shap_df[['feature', 'shap_abs', 'effect']]
                    # 同时更新feature_importance_df，以便后续保存
                    if feature_importance_df is None:
                        feature_importance_df = feature_importance_fallback
                else:
                    logger.warning("  Both SHAP values and feature importance calculation failed")

        # Step 11: 保存结果
        model_dir = Path(output_dir) / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存交叉验证结果
        cv_results = {
            'n_folds': n_folds,
            'fold_metrics': cv_fold_metrics,
            'average_metrics': avg_metrics
        }
        cv_results_file = model_dir / f"{model_type}_cv_results.json"
        with open(cv_results_file, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        
        # 训练阶段不再做GWAS/LD特征筛选，这里将全部特征名（去掉phenotype列）作为selected_snps
        selected_snps = filter_phenotype_columns(X_filtered.columns.tolist())

        save_training_results(
            model=final_model,
            metrics=metrics,
            selected_snps=selected_snps,
            output_dir=output_dir,
            model_type=model_type,
            task_type=task_type,
            feature_importance_df=feature_importance_df,
            shap_df=shap_df,
            y_test=y_test_combined,
            y_pred=y_pred_combined,
            y_prob=y_prob_combined,
            cv_results=cv_results,
        )

        # 最终日志：输出当前模型的训练用时
        total_time = round(time.time() - start_time, 2)
        logger.info(f"Training completed for {model_type} (elapsed time: {total_time} seconds)")
        # 根据调试开关决定是否清理临时目录/文件
        if CLEANUP_TEMP_FILES:
            # 正常结束：仅清理本次运行的临时目录，不清理preprocess的临时目录（preprocess模块不删除tmp目录）
            try:
                # 使用全局管理器清理临时文件/目录（不包括preprocess临时目录）
                _temp_file_manager.cleanup_on_exit(cleanup_preprocess=False)
                # 若 tmp_root 为空，则一并删除 tmp_root（model training 模块自身的 temp_m 根目录）
                # 但需要确保不会删除preprocess的tmp目录
                try:
                    if tmp_root.exists() and not any(tmp_root.iterdir()):
                        # 由于tmp_root是temp_m，preprocess_tmp_dir是tmp_p，名称不同，不会冲突
                        # 但保留检查逻辑作为额外安全措施
                        should_delete = True
                        if preprocess_tmp_dir:
                            preprocess_tmp_dir_abs = Path(preprocess_tmp_dir).absolute()
                            tmp_root_abs = tmp_root.absolute()
                            # 如果tmp_root是preprocess_tmp_dir或其父目录，则不删除（理论上不会发生，因为名称不同）
                            try:
                                # Python 3.9+ 使用 is_relative_to
                                if tmp_root_abs == preprocess_tmp_dir_abs or preprocess_tmp_dir_abs.is_relative_to(tmp_root_abs):
                                    should_delete = False
                            except AttributeError:
                                # Python < 3.9 使用其他方法检查
                                try:
                                    preprocess_tmp_dir_abs.relative_to(tmp_root_abs)
                                    should_delete = False
                                except ValueError:
                                    pass  # 不是相对路径，可以删除
                        if should_delete:
                            shutil.rmtree(tmp_root, ignore_errors=True)
                except Exception:
                    pass
                
                # 正常运行完成后，删除preprocess模块产生的tmp_p目录
                if preprocess_tmp_dir:
                    try:
                        preprocess_tmp_dir_path = Path(preprocess_tmp_dir)
                        if preprocess_tmp_dir_path.exists():
                            shutil.rmtree(preprocess_tmp_dir_path, ignore_errors=True)
                            logger.info(f"Deleted preprocess temporary directory: {preprocess_tmp_dir_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete preprocess temporary directory (ignored): {e}")
                
                # 正常运行完成后，删除GWAS运行产生的output目录
                gwas_output_dir = output_dir_path / "output"
                if gwas_output_dir.exists():
                    try:
                        shutil.rmtree(gwas_output_dir, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete GWAS output directory (ignored): {e}")
            except Exception as cleanup_err:
                pass
        else:
            pass
        
        # ======================== 阶段结束时清理临时目录（standardized） ========================
        # 单模型训练只清理“本进程自己的”临时目录。
        # 注意：train-all 会在父进程统一清理 base_output_dir/tmp，单模型子进程不得删除整个 tmp 根目录。
        if CLEANUP_TEMP_FILES:
            try:
                import shutil
                if run_tmp_dir.exists():
                    shutil.rmtree(run_tmp_dir, ignore_errors=True)
            except Exception as e:
                pass
        
        return 0

    except Exception as e:
        logger.error(f"Single model training failed: {str(e)}", exc_info=True)
        # 根据调试开关决定异常时是否清理临时目录/文件
        if CLEANUP_TEMP_FILES:
            # 异常：只清理本进程自己的临时目录（避免并行冲突）
            try:
                _temp_file_manager.cleanup_on_exit(cleanup_preprocess=False)
                import shutil
                if run_tmp_dir.exists():
                    shutil.rmtree(run_tmp_dir, ignore_errors=True)
            except Exception as cleanup_err:
                pass
        else:
            pass
        return 1


def _run_single_model_silent(*args, **kwargs) -> int:
    """
    子进程包装器：抑制子进程中的冗长日志输出（standardized）
    - 父进程负责输出"正在训练哪个模型 / 耗时"等进度信息
    - 子进程仅保留 WARNING 及以上，避免刷屏
    
    注意：此函数必须在模块级别定义，以便可以被 multiprocessing 的 pickle 序列化
    """
    import logging as _logging
    root = _logging.getLogger()
    old_level = root.level
    try:
        root.setLevel(_logging.WARNING)
        return run_single_model(*args, **kwargs)
    finally:
        root.setLevel(old_level)


def run_all_models(
    input_path: str,
    output_dir: str,
    task_type: Optional[str] = None,
    n_folds: int = 5,
    random_state: int = 42,
    # 特征重要性计算参数（可选）
    calculate_feature_importance: bool = False,
    # 图表质量固定开关（fixed，已不对外开放）
    pub_quality_mode: bool = True
) -> int:
    """训练所有支持的模型，并生成对比报告"""
    # 强制要求使用 preprocess 生成的 metadata.json 作为训练入口
    if not str(input_path).endswith("_metadata.json"):
        raise ValueError(
            "训练输入必须为 preprocess 生成的元数据文件（*_metadata.json）。"
            f"当前输入: {input_path}"
        )
    # 首先解析输入路径，获取task_type默认值，并预先加载训练数据，避免每个模型子进程重复读盘
    input_info = parse_train_input_path(input_path)
    if task_type is None:
        task_type = input_info.get("task_type") or "regression"
        if not input_info.get("task_type"):
            logger.warning(
                "  task_type not specified and no task_type information in metadata, using default: regression"
            )
    else:
        pass

    # 预先加载训练数据（在主进程中执行一次）
    train_file = input_info["train_file"]
    valid_samples = input_info["metadata"]["valid_samples"] if input_info["metadata"] else None
    X, y, snp_name_mapping = load_training_data(train_file, valid_samples)

    preloaded_data = {
        "X": X,
        "y": y,
        "train_file": train_file,
        "valid_samples": valid_samples,
        "metadata": input_info["metadata"],
        "preprocess_tmp_dir": input_info.get("preprocess_tmp_dir"),
        "task_type": task_type,
    }
    
    # 固定实现：忽略外部传入值，统一使用内部常量
    pub_quality_mode = PUB_QUALITY_MODE

    supported_models = ["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"]
    results = {}
    model_metrics_summary = {}  # 记录每个模型的评估指标（用于后续选择最佳模型）  # 新增
    start_time = time.time()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ======================== 临时目录规范化（standardized） ========================
    # 临时文件放在基础输出目录下的 tmp/train 子目录中（与阶段目录同级）
    # 每个阶段在 tmp 下创建自己的子目录，避免不同阶段的临时文件互相干扰
    # 例如：如果 output_dir 是 test/train，则 tmp_dir 是 test/tmp/train
    base_output_dir = output_dir_path.parent
    tmp_dir = base_output_dir / "tmp" / "train"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ======================== 关键改动：使用进程池并行训练所有模型 ========================
        # 说明：
        # - 每个模型类型在一个独立的进程中调用 run_single_model 进行完整训练（含CV、特征重要性、SHAP等）
        # - 进程池大小根据模型个数和CPU核数自动设置，避免资源过载
        # - 父进程输出进度与耗时；子进程通过包装器抑制冗长日志
        max_workers = len(supported_models)
        cpu_count = os.cpu_count() or max_workers
        max_workers = min(max_workers, cpu_count)
        
        from concurrent.futures import ProcessPoolExecutor

        logger.info(
            f"[TRAIN_ALL] 并行训练开始: models={len(supported_models)}, max_workers={max_workers}, "
            f"task_type={task_type}, n_folds={n_folds}"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            submit_ts = {}
            for model_type in supported_models:
                logger.info(f"[TRAIN_ALL] 开始训练模型: {model_type}")
                submit_ts[model_type] = time.time()
                future = executor.submit(
                    _run_single_model_silent,
                    input_path,
                    model_type,
                    output_dir,
                    task_type,
                    n_folds,
                    random_state,
                    calculate_feature_importance,
                    pub_quality_mode,
                    preloaded_data,
                )
                futures[future] = model_type
            
            # 收集各模型的退出码（按完成顺序输出）
            for future in as_completed(futures):
                model_type = futures[future]
                elapsed = time.time() - submit_ts.get(model_type, time.time())
                try:
                    ret_code = future.result()
                except Exception as e:
                    logger.error(f"[TRAIN_ALL] 模型训练异常: model={model_type}, err={e}", exc_info=True)
                    ret_code = 1
                results[model_type] = "成功" if ret_code == 0 else "失败"
                logger.info(
                    f"[TRAIN_ALL] 模型训练结束: model={model_type}, status={results[model_type]}, "
                    f"elapsed={elapsed:.1f}s"
                )

        logger.info(f"[TRAIN_ALL] 并行训练阶段完成, elapsed_total={time.time()-start_time:.1f}s")
        
        # ======================== 阶段结束时清理临时目录（standardized） ========================
        # 在阶段正常结束时，删除整个 tmp 目录（包括目录本身及其所有文件和子目录）
        if CLEANUP_TEMP_FILES:
            try:
                import shutil
                # 删除整个 tmp 目录（output_dir/tmp），而不仅仅是 tmp/train 子目录
                tmp_root_dir = base_output_dir / "tmp"
                if tmp_root_dir.exists():
                    shutil.rmtree(tmp_root_dir, ignore_errors=True)
            except Exception as e:
                pass

        # ======================== 新增：读取各模型评估指标，打印性能并选择最佳模型 ========================
        # 说明：
        # - 每个模型的 metrics.json 由 run_single_model/save_training_results 写入
        # - 分类任务优先使用 AUC 作为选择标准，其次 Accuracy
        # - 回归任务使用 Pearson correlation 作为选择标准
        best_model_type = None
        best_score = None
        best_metric_name = None

        logger.info("Model performance summary (based on *_metrics.json):")
        for model_type in supported_models:
            model_dir = output_dir_path / model_type
            metrics_file = model_dir / f"{model_type}_metrics.json"
            if not metrics_file.exists():
                logger.warning(f"  Metrics file not found for model {model_type}: {metrics_file}")
                continue

            try:
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
            except Exception as e:
                logger.warning(f"  Failed to load metrics for model {model_type}: {e}")
                continue

            metrics = metrics_data.get("metrics", {}) or {}
            model_metrics_summary[model_type] = metrics

            if task_type == "classification":
                # 分类：优先按 AUC 选最佳，其次 Accuracy
                raw_auc = metrics.get("auc")
                auc = raw_auc if isinstance(raw_auc, (int, float)) else None
                raw_acc = metrics.get("accuracy")
                acc = raw_acc if isinstance(raw_acc, (int, float)) else None

                logger.info(
                    f"  {model_type}: accuracy={metrics.get('accuracy')}, "
                    f"recall={metrics.get('recall')}, f1={metrics.get('f1')}, auc={metrics.get('auc')}"
                )

                score = None
                metric_name = None
                if auc is not None:
                    score = auc
                    metric_name = "auc"
                elif acc is not None:
                    score = acc
                    metric_name = "accuracy"
            else:
                # 回归：使用 Pearson correlation 选最佳
                raw_corr = metrics.get("pearson_correlation")
                corr = raw_corr if isinstance(raw_corr, (int, float)) else None
                logger.info(
                    f"  {model_type}: pearson_correlation={metrics.get('pearson_correlation')}, "
                    f"pearson_pvalue={metrics.get('pearson_pvalue')}"
                )
                score = corr
                metric_name = "pearson_correlation"

            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score
                    best_model_type = model_type
                    best_metric_name = metric_name

        # 已按需求：仅保留性能最优模型目录（其内已包含 *_model.pkl 等文件）
        # 因此不再额外复制 best_model.pkl，避免重复保存
        if best_model_type is not None:
            best_model_dir = output_dir_path / best_model_type
            best_model_src = best_model_dir / f"{best_model_type}_model.pkl"
            if best_model_src.exists():
                # 额外保存一个 best_model_info.json，记录最佳模型及所有模型性能
                best_info_file = output_dir_path / "best_model_info.json"
                try:
                    with open(best_info_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "task_type": task_type,
                                "best_model_type": best_model_type,
                                "best_metric_name": best_metric_name,
                                "best_metric_value": best_score,
                                "all_model_metrics": model_metrics_summary,
                                "generated_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                            default=str,
                        )
                    logger.info(
                        f"Best model: {best_model_type} ("
                        f"{best_metric_name}={best_score})"
                    )
                except Exception as e:
                    logger.warning(f"  Failed to save best_model_info.json: {e}")
            else:
                logger.warning(
                    f"  Best model file not found for model {best_model_type}: {best_model_src}"
                )
            
            # 根据用户需求：仅保留性能最优模型，删除其它模型目录
            for model_type in supported_models:
                if model_type == best_model_type:
                    continue
                model_dir = output_dir_path / model_type
                if model_dir.exists():
                    try:
                        import shutil
                        shutil.rmtree(model_dir, ignore_errors=True)
                        logger.info(f"  Removed model directory for non-best model: {model_type}")
                    except Exception as e:
                        logger.warning(f"  Failed to remove model directory {model_dir}: {e}")
        else:
            logger.warning("  No valid metrics found to select the best model.")

        # 生成模型对比报告（保留原有结构，增加整体训练时间信息）
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        report_file = output_dir_path / "model_comparison_report.json"
        with open(report_file, 'w', encoding="utf-8") as f:
            json.dump({
                "task_type": task_type,
                "n_folds": n_folds,
                "random_state": random_state,
                "note": "GWAS/LD特征筛选已在preprocess模块完成",
                "training_results": results,
                "total_training_time": round(time.time() - start_time, 2),
                "generated_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, ensure_ascii=False)

        logger.info("  All models training completed")
        
        # ======================== 阶段结束时清理临时目录（standardized） ========================
        # 在阶段正常结束时，删除整个 tmp 目录（包括目录本身及其所有文件和子目录）
        if CLEANUP_TEMP_FILES:
            try:
                import shutil
                # 删除整个 tmp 目录（output_dir/tmp），而不仅仅是 tmp/train 子目录
                tmp_root_dir = base_output_dir / "tmp"
                if tmp_root_dir.exists():
                    shutil.rmtree(tmp_root_dir, ignore_errors=True)
            except Exception as e:
                pass
        
        # 根据调试开关决定是否清理临时目录/文件（保留原有逻辑作为备用）
        if CLEANUP_TEMP_FILES:
            # 所有模型训练结束后，尝试删除全模型阶段的 tmp 根目录（若为空）
            # 由于tmp_dir是temp_m，preprocess_tmp_dir是tmp_p，名称不同，不会冲突
            # 但保留检查逻辑作为额外安全措施
            try:
                if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                    should_delete = True
                    # 尝试从输入路径获取preprocess_tmp_dir（tmp_p目录）
                    preprocess_tmp_dir = None
                    try:
                        input_info = parse_train_input_path(input_path)
                        if input_info.get("preprocess_tmp_dir"):
                            preprocess_tmp_dir_abs = Path(input_info["preprocess_tmp_dir"]).absolute()
                            preprocess_tmp_dir = preprocess_tmp_dir_abs
                            tmp_dir_abs = tmp_dir.absolute()
                            # 如果tmp_dir是preprocess_tmp_dir或其父目录，则不删除（理论上不会发生，因为名称不同）
                            try:
                                # Python 3.9+ 使用 is_relative_to
                                if tmp_dir_abs == preprocess_tmp_dir_abs or preprocess_tmp_dir_abs.is_relative_to(tmp_dir_abs):
                                    should_delete = False
                            except AttributeError:
                                # Python < 3.9 使用其他方法检查
                                try:
                                    preprocess_tmp_dir_abs.relative_to(tmp_dir_abs)
                                    should_delete = False
                                except ValueError:
                                    pass  # 不是相对路径，可以删除
                    except Exception:
                        pass  # 如果无法获取preprocess_tmp_dir，则按原逻辑删除
                    
                    if should_delete:
                        import shutil
                        shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as cleanup_err:
                logger.warning(f"  Failed to delete all-models training temporary directory (ignored): {cleanup_err}")
            
            # 正常运行完成后，删除preprocess模块产生的tmp_p目录
            try:
                input_info = parse_train_input_path(input_path)
                if input_info.get("preprocess_tmp_dir"):
                    preprocess_tmp_dir_path = Path(input_info["preprocess_tmp_dir"]).absolute()
                    if preprocess_tmp_dir_path.exists():
                        shutil.rmtree(preprocess_tmp_dir_path, ignore_errors=True)
                        logger.info(f"Deleted preprocess temporary directory: {preprocess_tmp_dir_path}")
            except Exception as e:
                logger.warning(f"Failed to delete preprocess temporary directory (ignored): {e}")
            
            # 正常运行完成后，删除GWAS运行产生的output目录
            gwas_output_dir = output_dir_path / "output"
            if gwas_output_dir.exists():
                try:
                    shutil.rmtree(gwas_output_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to delete GWAS output directory (ignored): {e}")
        else:
            pass
        
        return 0

    except Exception as e:
        logger.error(f"  All-models training failed: {str(e)}", exc_info=True)
        # 异常时清理整个 tmp 目录（包括目录本身及其所有文件和子目录）
        if CLEANUP_TEMP_FILES:
            try:
                import shutil
                # 删除整个 tmp 目录（output_dir/tmp），而不仅仅是 tmp/train 子目录
                tmp_root_dir = base_output_dir / "tmp"
                if tmp_root_dir.exists():
                    shutil.rmtree(tmp_root_dir, ignore_errors=True)
            except Exception:
                pass
        return 1

# ======================== 6. 预测函数 ========================
def predict_with_model(
    input_path: str,
    model_path: str,
    output_dir: str,
    task_type: str
) -> int:
    """使用训练好的模型进行预测
    
    支持输入格式：
    - 训练数据格式（.txt，包含sample列作为索引）
    - VCF格式（.vcf/.vcf.gz），会自动转换为训练数据格式
    """
    try:
        # 保存原始输入路径（用于后续清理）
        original_input_path = input_path
        input_path_obj = Path(input_path)
        file_ext = input_path_obj.suffix.lower()
        is_vcf_input = file_ext in ['.vcf', '.gz'] or input_path.endswith('.vcf.gz')
        tmp_dir_to_clean = None
        
        # 如果是VCF文件，先转换为训练数据格式
        if is_vcf_input:
            logger.info(f"Detected VCF file: {input_path}")
            logger.info("Converting VCF to training data format...")
            
            # 导入preprocess模块的函数
            from assoG2P.bin.preprocess import (
                genotype_to_plink,
                plink_to_training_data_optimized,
                auto_detect_chromosomes
            )
            import tempfile
            import shutil
            
            # 创建临时目录
            output_dir_path = Path(output_dir)
            tmp_dir = output_dir_path / "tmp_predict"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir_to_clean = tmp_dir  # 保存临时目录路径用于清理
            
            try:
                # 步骤1: VCF转PLINK
                plink_prefix = str(tmp_dir / "predict_plink")
                logger.info("  Step 1: Converting VCF to PLINK format...")
                plink_prefix = genotype_to_plink(input_path, plink_prefix)
                logger.info(f"  PLINK conversion completed: {plink_prefix}")
                
                # 步骤2: PLINK转训练数据格式
                # 对于预测，我们不需要表型数据，创建一个空的表型DataFrame
                # 但需要确保样本ID匹配
                logger.info("  Step 2: Converting PLINK to training data format...")
                
                # 读取.fam文件获取样本ID
                fam_file = f"{plink_prefix}.fam"
                if not Path(fam_file).exists():
                    raise FileNotFoundError(f"PLINK .fam file not found: {fam_file}")
                
                # 读取样本ID
                fam_df = pd.read_csv(fam_file, sep=r"\s+", header=None, 
                                    names=["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"])
                sample_ids = fam_df["IID"].tolist()
                
                # 创建空的表型DataFrame（所有值设为0，仅用于占位）
                pheno_df = pd.DataFrame({
                    "sample": sample_ids,
                    "phenotype": [0] * len(sample_ids)
                }).set_index("sample")
                
                # 生成临时训练数据文件
                temp_train_file = str(tmp_dir / "predict_train_data.txt")
                
                # 调用plink_to_training_data_optimized
                train_df, actual_train_file = plink_to_training_data_optimized(
                    plink_prefix=plink_prefix,
                    pheno_df=pheno_df,
                    output_file=temp_train_file,
                    tmp_dir=str(tmp_dir),
                    use_cache=True
                )
                
                logger.info(f"  Training data conversion completed: {actual_train_file}")
                
                # 使用转换后的文件作为输入
                input_path = actual_train_file
                
            except Exception as e:
                logger.error(f"  Failed to convert VCF to training data format: {str(e)}")
                raise
        
        # 0. 解析模型路径：仅支持直接指定训练好的模型文件路径（*.pkl）
        model_path_obj = Path(model_path).absolute()
        model_dir: Path
        model_file: Path
        model_type: str = ""

        if not model_path_obj.is_file():
            raise FileNotFoundError(f"模型文件不存在或不是文件路径: {model_path_obj}")

        model_file = model_path_obj
        model_dir = model_file.parent
        name = model_file.stem
        if name.endswith("_model"):
            model_type = name[:-6]
        else:
            model_type = name

        # 1. 加载预测数据（特征矩阵）
        # 说明：
        # - 这里复用 load_training_data 以保证特征清洗规则一致（clean_feature_names 等）；
        # - 对于纯预测数据，如果没有 phenotype 列，则最后一列会被当作 y 丢弃，
        #   因此更推荐采用与训练相同格式的矩阵（包含 phenotype 列），或在上游统一格式。
        X, _, _ = load_training_data(input_path)
        
        # 1.1 加载训练阶段使用的特征列表
        features_file = model_dir / f"{model_type}_training_features.json"
        if features_file.exists():
            try:
                with open(features_file, "r") as f:
                    feature_info = json.load(f)
                train_features = feature_info.get("feature_names") or []
            except Exception as e:
                logger.warning(f"  Failed to load training_features.json, fallback to raw X columns: {e}")
                train_features = list(X.columns)
        else:
            logger.warning(
                "  training_features.json not found, using current X columns as features; "
                "train/predict feature alignment may be unreliable."
            )
            train_features = list(X.columns)
        
        # 1.2 按训练特征集合对齐预测特征：
        # - 训练有、预测也有：直接使用预测中的该列；
        # - 训练有、预测没有：在预测矩阵中补一列，整列填 0；
        # - 训练没有、预测有：不加入模型输入（即丢弃该列）。
        
        # 调试信息：检查特征匹配情况
        
        # 检查特征名称匹配情况
        matched_features = [col for col in train_features if col in X.columns]
        missing_in_predict = [col for col in train_features if col not in X.columns]
        extra_in_predict = [col for col in X.columns if col not in train_features]
        
        logger.info(f"  Feature alignment: {len(matched_features)} matched, {len(missing_in_predict)} missing, {len(extra_in_predict)} extra")
        
        if len(matched_features) == 0:
            logger.warning(
                f"  WARNING: No features matched between training and prediction data!\n"
                f"  This will result in all-zero feature matrix and constant predictions.\n"
                f"  Training features (first 20): {train_features[:20]}\n"
                f"  Prediction features (first 20): {list(X.columns[:20])}\n"
                f"  This may be due to feature name cleaning differences. Please check feature names."
            )
        
        if len(missing_in_predict) > len(matched_features):
            logger.warning(
                f"  WARNING: More features missing ({len(missing_in_predict)}) than matched ({len(matched_features)}).\n"
                f"  This may result in poor prediction quality."
            )
        
        X_aligned = pd.DataFrame(index=X.index)
        # 重新计算 missing_in_predict（避免重复）
        missing_in_predict = [col for col in train_features if col not in X.columns]
        
        for col in train_features:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                X_aligned[col] = 0
        
        # 验证对齐后的特征矩阵
        if X_aligned.empty:
            raise ValueError("Aligned feature matrix is empty!")
        
        # 检查是否所有特征值都是0
        non_zero_counts = (X_aligned != 0).sum(axis=1)
        if (non_zero_counts == 0).all():
            logger.error(
                f"  ERROR: All feature values are zero! This will result in constant predictions.\n"
                f"  Matched features: {len(matched_features)}\n"
                f"  Missing features: {len(missing_in_predict)}\n"
                f"  Please check if feature names match between training and prediction data."
            )
        else:
            pass
        
        if missing_in_predict:
            logger.info(
                f"  {len(missing_in_predict)} feature(s) present in training but missing in predict; "
                f"filled with 0. Example: {missing_in_predict[:5]}"
            )
        
        # 2. 加载训练好的模型
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        import joblib
        model = joblib.load(model_file)
        logger.info("  Loading pre-trained model")
        
        # 3. 执行预测（基于对齐后的特征矩阵）
        # 验证特征矩阵的有效性
        
        # 检查特征矩阵是否有变化
        feature_variance = X_aligned.var()
        zero_variance_features = (feature_variance == 0).sum()
        if zero_variance_features > 0:
            logger.warning(f"  WARNING: {zero_variance_features} features have zero variance (constant values)")
        
        y_pred = model.predict(X_aligned)
        y_prob = model.predict_proba(X_aligned) if task_type == "classification" and hasattr(model, "predict_proba") else None
        
        # 检查预测结果的多样性
        unique_predictions = len(np.unique(y_pred))
        logger.info(f"  Prediction statistics: {unique_predictions} unique values out of {len(y_pred)} samples")
        if unique_predictions == 1:
            logger.error(
                f"  ERROR: All predictions are the same value: {y_pred[0]}\n"
                f"  This usually indicates one of the following issues:\n"
                f"  1. Feature names don't match between training and prediction data\n"
                f"  2. All feature values are zero (check matched_features: {len(matched_features)})\n"
                f"  3. Model is predicting a constant value for all samples\n"
                f"  Please check the feature alignment logs above."
            )
        else:
            logger.info(f"  Prediction range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}], mean: {np.mean(y_pred):.4f}")
        
        # 4. 保存预测结果：仅支持模型文件路径，因此输出固定保存在模型文件所在目录
        pred_output_dir = model_dir
        pred_output_dir.mkdir(parents=True, exist_ok=True)
        # ======================== 输出命名规范化（standardized） ========================
        # 预测结果统一命名为：{model_type}_predictions.tsv
        pred_file = pred_output_dir / f"{model_type}_predictions.tsv"
        
        # 构建结果DataFrame
        result_df = pd.DataFrame({
            "sample": X_aligned.index,
            "prediction": y_pred
        })
        # 分类任务添加概率列
        if task_type == "classification" and y_prob is not None:
            for i in range(y_prob.shape[1]):
                result_df[f"prob_class_{i}"] = y_prob[:, i]
        
        result_df.to_csv(pred_file, sep="\t", index=False)
        logger.info(f"  Prediction completed. Results saved to: {pred_file}")
        
        # 清理临时文件（如果是VCF转换产生的）
        if is_vcf_input and tmp_dir_to_clean and tmp_dir_to_clean.exists():
            try:
                shutil.rmtree(tmp_dir_to_clean, ignore_errors=True)
            except Exception as e:
                pass
        
        return 0
    except Exception as e:
        logger.error(f"  Prediction failed: {str(e)}", exc_info=True)
        return 1
