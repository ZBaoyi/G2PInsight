#!/usr/bin/env python3
"""
Model training module with GWAS integration (适配preprocess元数据衔接)
支持多模型训练、GWAS特征筛选、LD过滤、结果可视化、预测等功能
- 独立控制GWAS特征筛选和LD过滤开关
- 逻辑：先判断LD过滤 → 再判断GWAS（若开启则基于LD过滤后的数据）
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
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
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

# 进度条（可选）
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ======================== 新增：导入子模块 ========================
# 以模块名别名导入，便于后续直接调用 gemma_gwas.run_xxx / plink_ld.run_xxx
from assoG2P.bin import gemma_gwas, plink_ld

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
        logger.warning(f"收到信号 {signum}，正在清理临时文件...")
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
            logger.debug(" 调试模式：CLEANUP_TEMP_FILES=False，跳过临时文件清理")
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
                            logger.debug(f"   删除临时文件: {fp}")
                    except Exception as e:
                        logger.debug(f"   删除临时文件失败 {fp}: {e}")
                
                # 2. 清理临时目录（逆序删除，确保子目录先删除）
                for dp in reversed(self.temp_dirs):
                    try:
                        if dp.exists():
                            shutil.rmtree(dp, ignore_errors=True)
                            logger.debug(f"   删除临时目录: {dp}")
                    except Exception as e:
                        logger.debug(f"   删除临时目录失败 {dp}: {e}")
                
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
                            logger.debug(f"   删除preprocess临时目录: {dp}")
                    except Exception as e:
                        logger.debug(f"   删除preprocess临时目录失败 {dp}: {e}")
                self.preprocess_tmp_dirs.clear()
                
        except Exception as e:
            logger.warning(f" 清理临时文件时出错（已忽略）：{e}")
    
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
    
    # 验证GWAS PLINK文件完整性（不抛出异常，仅记录警告，由调用方处理）
    gwas_prefix = metadata.get("gwas_genotype_prefix")
    if gwas_prefix:
        required_plink = [f"{gwas_prefix}.bed", f"{gwas_prefix}.bim", f"{gwas_prefix}.fam"]
        missing_plink = [f for f in required_plink if not Path(f).exists()]
        if missing_plink:
            logger.warning(f"元数据中的GWAS PLINK文件不存在: {gwas_prefix}")
            logger.warning(f"缺失的文件: {', '.join(missing_plink)}")
            # 将gwas_genotype_prefix设为None，表示文件不存在
            metadata["gwas_genotype_prefix"] = None
            metadata["_gwas_genotype_prefix_missing"] = True
        else:
            logger.debug(f"元数据中的GWAS PLINK文件验证通过: {gwas_prefix}")
    
    logger.debug(f"读取元数据: {len(metadata['valid_samples'])}个样本")
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
                    logger.debug(f"读取元数据文件失败 {metadata_path}: {e}")
                    continue  # 尝试下一个路径
        
        if not metadata_found:
            logger.debug(f"未找到元数据文件，尝试的路径: {possible_metadata_paths}")
    
    # 验证训练文件存在性
    if not Path(result["train_file"]).exists():
        raise FileNotFoundError(f"训练文件不存在: {result['train_file']}")
    
    return result

# ======================== 2. 数据加载 & 预处理 ========================
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
    logger.debug("加载训练数据")
    try:
        # 读取数据（sample列为索引，最后一列为表型）
        df = pd.read_csv(
            train_file,
            sep="\t",
            index_col="sample",
            na_filter=False  # 禁用缺失值过滤（preprocess已处理）
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
            raise ValueError("加载的训练数据为空")
        if X.isnull().any().any():
            X = X.fillna(-1)
        
        return X, y, snp_name_mapping
    except Exception as e:
        logger.error(f" 加载训练数据失败: {str(e)}")
        raise

# ======================== 3. LD过滤 & GWAS特征筛选 ========================
def run_ld_filtering(
    genotype_prefix: str,
    output_prefix: str,
    ld_window_kb: int = 50,
    ld_window: int = 5,
    ld_window_r2: float = 0.2,
    threads: int = 8,
    keep_samples_file: Optional[str] = None,
    extract_snps_file: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    执行LD过滤，返回过滤后的基因型前缀和保留的SNP列表
    
    Args:
        genotype_prefix: 基因型文件前缀
        output_prefix: 输出文件前缀
        ld_window_kb: LD窗口大小（KB）
        ld_window: LD窗口大小（变体数）
        ld_window_r2: LD r²阈值
        threads: 线程数
        keep_samples_file: 样本列表文件（可选，仅对指定样本进行LD过滤）
        extract_snps_file: SNP列表文件（可选，仅对指定SNP进行LD过滤，用于模式4：先GWAS后LD）
    
    Returns:
        (过滤后的基因型前缀, 保留的SNP列表)
    """
    logger.debug(f"LD过滤: 窗口={ld_window_kb}KB, r²={ld_window_r2}")
    
    # 执行LD过滤
    ld_output_prefix = f"{output_prefix}_ld_filtered"
    ld_result = plink_ld.run_ld_filtering(
        input_path=genotype_prefix,
        output_prefix=ld_output_prefix,
        ld_window_kb=ld_window_kb,
        ld_window=ld_window,
        ld_window_r2=ld_window_r2,
        threads=threads,
        keep_intermediate=False,
        keep_samples_file=keep_samples_file,
        extract_snps_file=extract_snps_file
    )
    
    if ld_result != 0:
        raise RuntimeError("LD过滤执行失败")
    
    # 解析LD过滤后的SNP列表（从prune.in文件）
    prune_in_file = f"{ld_output_prefix}.prune.in"
    if not Path(prune_in_file).exists():
        raise FileNotFoundError(f"LD过滤结果文件缺失: {prune_in_file}")
    
    with open(prune_in_file, 'r') as f:
        ld_selected_snps = [line.strip() for line in f if line.strip()]
    
    logger.debug(f"LD过滤完成: {len(ld_selected_snps)}个SNP")
    return ld_output_prefix, ld_selected_snps

def run_gemma_gwas(
    genotype_prefix: str,
    phenotype_file: str,
    output_prefix: str,
    pvalue_threshold: float,
    threads: int = 8,
    model: str = "lmm"
) -> List[str]:
    """
    调用GEMMA执行GWAS，筛选显著SNP（基于过滤后的基因型文件）
    
    注意：此函数依赖于 assoG2P.bin.gemma_gwas 模块中
    run_complete_gwas_pipeline 的实现细节，尤其是：
      - 结果文件由 GEMMA 写入默认的 output/ 目录
      - GWAS结果文件名为 output/{output_prefix}_gwas.assoc.txt
      - 所有GWAS生成的临时文件会在模型训练完成后自动删除
    """
    # 创建输出目录（用于我们自己的派生结果，如significant_snps）
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug("GWAS分析中...")
    try:
        # 调用GWAS核心流程（严格按照 gemma_gwas 模块约定）
        gemma_gwas.run_complete_gwas_pipeline(
            input_plink_prefix=genotype_prefix,
            phenotype_file=phenotype_file,
            output_prefix=output_prefix
        )
    except Exception as e:
        raise RuntimeError(f"GWAS执行失败: {str(e)}")
    
    # 解析GWAS结果，筛选显著SNP
    # gemma_gwas.run_gemma_gwas 内部会对 output_prefix 取 basename 再加后缀 "_gwas" 作为 -o，
    # GEMMA 默认将结果写入当前工作目录下的 output/ 子目录，因此关联文件路径应为：
    #   {当前工作目录}/output/{basename(output_prefix)}_gwas.assoc.txt
    base_prefix = Path(output_prefix).name
    # 使用绝对路径，基于当前工作目录（调用方已切换到用户指定的输出目录）
    gwas_result_file = Path("output") / f"{base_prefix}_gwas.assoc.txt"
    if not gwas_result_file.exists():
        raise FileNotFoundError(f"GWAS结果文件缺失: {gwas_result_file.absolute()}")
    
    # 注册GWAS结果文件以便后续清理
    # 注意：这里需要从全局临时文件管理器注册，但函数签名中没有传递
    # 因此需要在调用run_gemma_gwas的地方注册这些文件
    
    # 读取GWAS结果（兼容不同GEMMA版本列名）
    gwas_df = pd.read_csv(gwas_result_file, sep="\t")
    pvalue_col = None
    for col in ["p_wald", "p_lrt", "P", "pvalue"]:
        if col in gwas_df.columns:
            pvalue_col = col
            break
    if not pvalue_col:
        raise ValueError(f"未识别GWAS结果中的P值列: {gwas_result_file}")
    
    # 筛选显著SNP
    snp_col = "rs" if "rs" in gwas_df.columns else "SNP" if "SNP" in gwas_df.columns else gwas_df.columns[0]
    significant_snps = gwas_df[gwas_df[pvalue_col] < pvalue_threshold][snp_col].tolist()
    
    # 保存显著SNP列表（仍然使用调用方提供的 output_prefix 作为前缀，便于查找）
    # 使用绝对路径，确保无论工作目录如何都能正确保存
    snp_list_file = Path(output_prefix).parent.absolute() / f"{Path(output_prefix).name}_significant_snps.txt"
    with open(snp_list_file, 'w') as f:
        f.write("\n".join(significant_snps))
    
    logger.debug(f"GWAS完成: {len(significant_snps)}个显著SNP")
    return significant_snps

def generate_gwas_phenotype_file(train_ids: List[str], y_train: pd.Series, output_file: str):
    """
    生成GWAS专用表型文件（仅训练集，适配GEMMA与 gemma_gwas 模块）
    
    要求三列格式：FID IID PHENO
      - 该格式是GEMMA官方推荐格式，可直接用于 `gemma -p phenotype_file`
      - 本项目中的 gemma_gwas.merge_phenotype_to_fam 也已支持三列表型文件
    这里将 FID 与 IID 均设置为 sample_id，第三列为表型值（缺失填-9）。
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 写入格式：FID IID PHENO（缺失值填-9）
    with open(output_file, 'w') as f:
        for sample_id, pheno in zip(train_ids, y_train):
            pheno = pheno if not pd.isna(pheno) else -9
            f.write(f"{sample_id}\t{sample_id}\t{pheno}\n")
    
    return output_file

# ======================== 4. 模型训练 & 评估 ========================
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
    
    return model_class(**params)

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
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    
    logger.debug(f"  网格搜索中: {n_iter}次迭代, {cv}折交叉验证...")
    search.fit(X_train, y_train)
    
    logger.debug(f"  最佳参数: {search.best_params_}")
    logger.debug(f"  最佳得分: {search.best_score_:.4f}")
    
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
                logger.warning(" 真实值或预测值为空，无法计算指标")
                metrics["accuracy"] = "N/A"
                metrics["recall"] = "N/A"
                metrics["f1"] = "N/A"
            elif len(y_true_array) != len(y_pred_array):
                logger.warning(f" 真实值和预测值长度不匹配: {len(y_true_array)} vs {len(y_pred_array)}")
                metrics["accuracy"] = "N/A"
                metrics["recall"] = "N/A"
                metrics["f1"] = "N/A"
            else:
                metrics["accuracy"] = round(accuracy_score(y_true_array, y_pred_array), 4)
                metrics["recall"] = round(recall_score(y_true_array, y_pred_array, average="weighted"), 4)
                metrics["f1"] = round(f1_score(y_true_array, y_pred_array, average="weighted"), 4)
        except Exception as e:
            logger.warning(f" 计算分类指标失败: {str(e)}")
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
                    logger.warning(f" 真实值和概率长度不匹配: {len(y_true_array)} vs {len(y_prob_array)}")
                    metrics["auc"] = "N/A"
                else:
                    # 检查唯一标签数量
                    unique_labels = np.unique(y_true_array)
                    n_unique = len(unique_labels)
                    
                    if n_unique < 2:
                        # 只有一个类别，无法计算AUC
                        metrics["auc"] = "N/A"
                        logger.debug(f" 只有一个类别（{unique_labels}），无法计算AUC")
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
                            logger.warning(f" 二分类AUC计算失败: {str(e)}")
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
                            logger.warning(f" 多分类AUC计算失败: {str(e)}")
                            metrics["auc"] = "N/A"
            else:
                metrics["auc"] = "N/A"
                if task_type == "classification":
                    logger.debug(" 预测概率为空，无法计算AUC")
        except Exception as e:
            logger.warning(f" 计算AUC失败: {str(e)}")
            metrics["auc"] = "N/A"
    else:
        # 回归指标：只保留皮尔逊相关系数和p值
        try:
            from scipy.stats import pearsonr
            pearson_corr, pearson_pvalue = pearsonr(y_true, y_pred)
            metrics["pearson_correlation"] = round(pearson_corr, 4)
            metrics["pearson_pvalue"] = round(pearson_pvalue, 6)  # p值通常需要更高精度
        except ImportError:
            logger.warning(" scipy未安装，无法计算皮尔逊相关系数和p值")
            metrics["pearson_correlation"] = "N/A"
            metrics["pearson_pvalue"] = "N/A"
        except Exception as e:
            logger.warning(f" 计算皮尔逊相关系数失败: {str(e)}")
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
    importance_dict = {}
    
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
                logger.warning(f" {model_type}模型不支持特征重要性计算")
                return pd.DataFrame()
        
        elif model_type == "SVM":
            # SVM：使用支持向量的权重（仅适用于线性核）
            if hasattr(model, 'kernel') and model.kernel == 'linear':
                if hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                else:
                    logger.warning(" SVM模型不支持特征重要性计算（非线性核）")
                    return pd.DataFrame()
            else:
                logger.warning(" SVM模型不支持特征重要性计算（非线性核）")
                return pd.DataFrame()
        
        else:
            logger.warning(f" {model_type}模型不支持特征重要性计算")
            return pd.DataFrame()
        
        # 获取原始值（带正负号）用于计算正负效应
        original_values = None
        if model_type in ["Logistic", "SVM"]:
            # 线性模型：使用原始系数
            if hasattr(model, 'coef_'):
                if task_type == "classification":
                    original_values = model.coef_.mean(axis=0)
                else:
                    original_values = model.coef_[0]
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
        
        logger.info(f" 特征重要性计算完成")
        return importance_df
        
    except Exception as e:
        logger.error(f" 特征重要性计算失败: {str(e)}")
        return pd.DataFrame()

# ======================== 辅助函数：减少代码重复 ========================
def _setup_matplotlib() -> Tuple[bool, Any]:
    """
    设置matplotlib环境（提取公共代码）
    
    Returns:
        (is_available, plt_module): matplotlib是否可用和plt模块（如果可用）
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        # 设置字体
        try:
            from assoG2P.bin.font_utils import setup_matplotlib_font
            setup_matplotlib_font()
        except ImportError:
            pass
        return True, plt
    except ImportError:
        logger.warning(" matplotlib未安装，跳过绘图功能")
        return False, None  # type: ignore

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
    if phenotype_cols:
        return df.drop(columns=phenotype_cols)
    return df

# ======================== 可视化函数 ========================
def plot_performance_curves(y_true: pd.Series, y_pred: np.ndarray, y_prob: Optional[np.ndarray], output_dir: Path, model_type: str, task_type: str) -> None:
    """
    绘制性能评估指标变化曲线（分类/回归）
    
    :param y_true: 真实值（pd.Series或np.ndarray）
    :param y_pred: 预测值（np.ndarray）
    :param y_prob: 预测概率（分类任务需要，回归任务为None）
    :param output_dir: 输出目录
    :param model_type: 模型类型
    :param task_type: 任务类型（classification/regression）
    """
    # 设置matplotlib环境
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        return
    
    # 确保y_true是numpy数组
    y_true_values = _ensure_numpy_array(y_true)
    
    y_pred_values = np.array(y_pred)
    
    if task_type == "regression":
        # ========== Regression performance curves ==========
        # 只绘制皮尔逊相关系数和P值
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{model_type} Performance (Regression)', fontsize=16, fontweight='bold')
        
        # 计算皮尔逊相关系数和P值
        try:
            from scipy.stats import pearsonr
            pearson_corr, pearson_pvalue = pearsonr(y_true_values, y_pred_values)
        except ImportError:
            logger.warning("scipy未安装，无法计算皮尔逊相关系数和P值")
            pearson_corr = None
            pearson_pvalue = None
        except Exception as e:
            logger.warning(f"计算皮尔逊相关系数失败: {str(e)}")
            pearson_corr = None
            pearson_pvalue = None
        
        # 1. 皮尔逊相关系数
        ax1.axis('off')
        corr_text = f"Pearson Correlation\n\n"
        if pearson_corr is not None:
            corr_text += f"  r = {pearson_corr:.4f}\n"
            corr_text += f"  R² = {pearson_corr**2:.4f}"
        else:
            corr_text += "  N/A"
        ax1.text(0.5, 0.5, corr_text, fontsize=14, ha='center', va='center', 
                transform=ax1.transAxes, family='monospace', fontweight='bold')
        ax1.set_title('Pearson Correlation', fontsize=14, fontweight='bold')
        
        # 2. P值
        ax2.axis('off')
        pval_text = f"P-value\n\n"
        if pearson_pvalue is not None:
            pval_text += f"  p = {pearson_pvalue:.6f}\n"
            if pearson_pvalue < 0.001:
                pval_text += f"  (p < 0.001)"
            elif pearson_pvalue < 0.01:
                pval_text += f"  (p < 0.01)"
            elif pearson_pvalue < 0.05:
                pval_text += f"  (p < 0.05)"
            else:
                pval_text += f"  (p >= 0.05)"
        else:
            pval_text += "  N/A"
        ax2.text(0.5, 0.5, pval_text, fontsize=14, ha='center', va='center', 
                transform=ax2.transAxes, family='monospace', fontweight='bold')
        ax2.set_title('P-value', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图形
        plot_file = output_dir / "performance_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    else:
        # ========== Classification performance curves ==========
        if y_prob is None:
            logger.warning(" Classification task requires predicted probabilities, skip performance curves")
            return
        
        # 获取类别信息
        unique_classes = np.unique(y_true_values)
        n_classes = len(unique_classes)
        
        # 确定二分类时使用的概率列
        if n_classes == 2:
            prob_col = 1 if y_prob.shape[1] > 1 else 0
            y_prob_binary = y_prob[:, prob_col]
        else:
            # 多分类：使用第一个类别的概率
            prob_col = 0
            y_prob_binary = y_prob[:, prob_col]
        
        # ========== 图1：多个性能曲线合并在一张图中（2x2子图布局）==========
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
            
            fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig1.suptitle(f'{model_type} Performance Curves (Classification)', fontsize=16, fontweight='bold')
            
            if n_classes == 2:
                # 二分类：计算不同阈值下的指标
                thresholds = np.linspace(0, 1, 100)
                accuracies = []
                recalls = []
                f1_scores = []
                
                # 计算ROC曲线
                fpr, tpr, roc_thresholds = roc_curve(y_true_values, y_prob_binary)
                roc_auc = auc(fpr, tpr)
                
                # 计算Precision-Recall曲线
                precision, recall_pr, pr_thresholds = precision_recall_curve(y_true_values, y_prob_binary)
                pr_auc = auc(recall_pr, precision)
                
                # 计算不同阈值下的准确率、召回率和F1得分
                for threshold in thresholds:
                    y_pred_thresh = (y_prob_binary >= threshold).astype(int)
                    accuracies.append(accuracy_score(y_true_values, y_pred_thresh))
                    recalls.append(recall_score(y_true_values, y_pred_thresh, zero_division=0))
                    f1_scores.append(f1_score(y_true_values, y_pred_thresh, zero_division=0))
                
                # 子图1：ROC曲线（AUC曲线）
                ax1 = axes[0, 0]
                ax1.plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess', alpha=0.5)
                ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
                ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
                ax1.set_title('ROC Curve (AUC)', fontsize=13)
                ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])
                
                # 子图2：Precision-Recall曲线（召回率曲线）
                ax2 = axes[0, 1]
                ax2.plot(recall_pr, precision, 'g-', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
                ax2.set_xlabel('Recall', fontsize=12)
                ax2.set_ylabel('Precision', fontsize=12)
                ax2.set_title('Precision-Recall Curve', fontsize=13)
                ax2.legend(loc='lower left')
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                
                # 子图3：准确率曲线
                ax3 = axes[1, 0]
                ax3.plot(thresholds, accuracies, 'r-', lw=2, label='Accuracy')
                ax3.set_xlabel('Threshold', fontsize=12)
                ax3.set_ylabel('Accuracy', fontsize=12)
                ax3.set_title('Accuracy Curve', fontsize=13)
                ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim([0, 1])
                ax3.set_ylim([0, 1])
                
                # 子图4：F1得分曲线
                ax4 = axes[1, 1]
                ax4.plot(thresholds, f1_scores, 'm-', lw=2, label='F1 Score')
                ax4.set_xlabel('Threshold', fontsize=12)
                ax4.set_ylabel('F1 Score', fontsize=12)
                ax4.set_title('F1 Score Curve', fontsize=13)
                ax4.legend(loc='best')
                ax4.grid(True, alpha=0.3)
                ax4.set_xlim([0, 1])
                ax4.set_ylim([0, 1])
                
            else:
                # 多分类：绘制每个类别的ROC曲线
                from sklearn.preprocessing import label_binarize
                y_true_binarized = label_binarize(y_true_values, classes=unique_classes)
                
                # 子图1：ROC曲线
                ax1 = axes[0, 0]
                for i, class_label in enumerate(unique_classes):
                    if y_prob.shape[1] > i:
                        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax1.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.4f})')
                ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
                ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
                ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
                ax1.set_title('ROC Curve (AUC)', fontsize=13)
                ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)
                
                # 其他子图留空或显示提示
                for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'Multi-class metrics\nnot implemented', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            plt.tight_layout()
            plot_file1 = output_dir / "performance_curves.png"
            plt.savefig(plot_file1, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f" Failed to draw performance curves: {str(e)}")
        
        # ========== 图2：预测概率分布单独绘制 ==========
        try:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
            fig2.suptitle(f'{model_type} Predicted Probability Distribution', fontsize=16, fontweight='bold')
            
            if n_classes == 2:
                # 二分类：绘制两个类别的概率分布
                prob_col = 1 if y_prob.shape[1] > 1 else 0
                ax2.hist(y_prob[y_true_values == unique_classes[0], prob_col], bins=30, alpha=0.6, 
                        label=f'True Class={unique_classes[0]}', edgecolor='black', color='blue')
                ax2.hist(y_prob[y_true_values == unique_classes[1], prob_col], bins=30, alpha=0.6, 
                        label=f'True Class={unique_classes[1]}', edgecolor='black', color='red')
            else:
                # 多分类：绘制前3个类别的概率分布
                for i, class_label in enumerate(unique_classes[:3]):
                    if y_prob.shape[1] > i:
                        mask = y_true_values == class_label
                        if mask.sum() > 0:
                            ax2.hist(y_prob[mask, i], bins=30, alpha=0.6, 
                                    label=f'True Class={class_label}', edgecolor='black')
            
            ax2.set_xlabel('Predicted Probability', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Predicted Probability Distribution', fontsize=13)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_file2 = output_dir / "probability_distribution.png"
            plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f" Failed to draw probability distribution: {str(e)}")

def plot_cv_training_curves(cv_results: Dict, output_dir: Path, model_type: str, task_type: str) -> None:
    """
    绘制交叉验证训练过程曲线图
    
    :param cv_results: 交叉验证结果字典，包含每折的指标
    :param output_dir: 输出目录
    :param model_type: 模型类型
    :param task_type: 任务类型
    """
    # 设置matplotlib环境
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        return
    
    n_folds = len(cv_results.get('fold_metrics', []))
    if n_folds == 0:
        logger.warning(" 没有交叉验证结果，跳过训练过程曲线绘制")
        return
    
    # 创建图形
    if task_type == "regression":
        # 只绘制皮尔逊相关系数和P值
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{model_type} Model {n_folds}-Fold Cross-Validation (Regression)', fontsize=16, fontweight='bold')
        
        # 提取指标
        folds = list(range(1, n_folds + 1))
        pearson_corrs = []
        pearson_pvalues = []
        
        for i in range(n_folds):
            fold_metric = cv_results['fold_metrics'][i]
            corr_val = fold_metric.get('pearson_correlation')
            pval_val = fold_metric.get('pearson_pvalue')
            
            if isinstance(corr_val, (int, float)) and not np.isnan(corr_val):
                pearson_corrs.append(corr_val)
            if isinstance(pval_val, (int, float)) and not np.isnan(pval_val) and pval_val > 0:
                pearson_pvalues.append(pval_val)
        
        # 1. 皮尔逊相关系数变化曲线
        ax1.set_xlabel('Fold', fontsize=12)
        ax1.set_ylabel('Pearson Correlation', fontsize=12)
        ax1.set_title('Pearson Correlation Across Folds', fontsize=13)
        ax1.set_xticks(folds)
        ax1.grid(True, alpha=0.3)
        
        if pearson_corrs and len(pearson_corrs) > 0:
            ax1.plot(folds[:len(pearson_corrs)], pearson_corrs, 'o-', linewidth=2, markersize=8, label='Per Fold')
            mean_corr = np.mean(pearson_corrs)
            ax1.axhline(y=mean_corr, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_corr:.4f}')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        
        # 2. p值变化曲线（对数尺度）
        ax2.set_xlabel('Fold', fontsize=12)
        ax2.set_ylabel('P-value (log scale)', fontsize=12)
        ax2.set_title('P-value Across Folds', fontsize=13)
        ax2.set_xticks(folds)
        ax2.grid(True, alpha=0.3)
        
        if pearson_pvalues and len(pearson_pvalues) > 0:
            ax2.semilogy(folds[:len(pearson_pvalues)], pearson_pvalues, 'o-', linewidth=2, markersize=8, label='Per Fold')
            mean_pvalue = np.mean(pearson_pvalues)
            ax2.axhline(y=mean_pvalue, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_pvalue:.6f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        
    else:
        # 分类任务
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # 使用英文标题避免字体问题
        fig.suptitle(f'{model_type} Model {n_folds}-Fold Cross-Validation (Classification)', fontsize=16, fontweight='bold')
        
        # 提取指标
        folds = list(range(1, n_folds + 1))
        accuracies = []
        recalls = []
        f1_scores = []
        aucs = []
        
        for i in range(n_folds):
            fold_metric = cv_results['fold_metrics'][i]
            acc_val = fold_metric.get('accuracy')
            rec_val = fold_metric.get('recall')
            f1_val = fold_metric.get('f1')
            auc_val = fold_metric.get('auc')
            
            if isinstance(acc_val, (int, float)) and not np.isnan(acc_val):
                accuracies.append(acc_val)
            if isinstance(rec_val, (int, float)) and not np.isnan(rec_val):
                recalls.append(rec_val)
            if isinstance(f1_val, (int, float)) and not np.isnan(f1_val):
                f1_scores.append(f1_val)
            if isinstance(auc_val, (int, float)) and not np.isnan(auc_val) and auc_val != 'N/A':
                aucs.append(auc_val)
        
        # 1. 准确率变化曲线
        ax1 = axes[0, 0]
        ax1.set_xlabel('Fold', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy Across Folds', fontsize=13)
        ax1.set_xticks(folds)
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3)
        
        if accuracies and len(accuracies) > 0:
            ax1.plot(folds[:len(accuracies)], accuracies, 'o-', linewidth=2, markersize=8, label='Accuracy')
            mean_acc = np.mean(accuracies)
            ax1.axhline(y=mean_acc, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_acc:.4f}')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        
        # 2. 召回率和F1得分变化曲线
        ax2 = axes[0, 1]
        ax2.set_xlabel('Fold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Recall and F1 Score Across Folds', fontsize=13)
        ax2.set_xticks(folds)
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3)
        
        if recalls and len(recalls) > 0:
            ax2.plot(folds[:len(recalls)], recalls, 'o-', linewidth=2, markersize=8, label='Recall')
            mean_recall = np.mean(recalls)
            ax2.axhline(y=mean_recall, color='r', linestyle='--', linewidth=2, alpha=0.5)
        if f1_scores and len(f1_scores) > 0:
            ax2.plot(folds[:len(f1_scores)], f1_scores, 's-', linewidth=2, markersize=8, label='F1 Score')
            mean_f1 = np.mean(f1_scores)
            ax2.axhline(y=mean_f1, color='g', linestyle='--', linewidth=2, alpha=0.5)
        if recalls or f1_scores:
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. AUC变化曲线
        ax3 = axes[1, 0]
        ax3.set_xlabel('Fold', fontsize=12)
        ax3.set_ylabel('AUC', fontsize=12)
        ax3.set_title('AUC Across Folds', fontsize=13)
        ax3.set_xticks(folds)
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3)
        
        if aucs and len(aucs) > 0:
            ax3.plot(folds[:len(aucs)], aucs, 'o-', linewidth=2, markersize=8, label='AUC', color='purple')
            mean_auc = np.mean(aucs)
            ax3.axhline(y=mean_auc, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_auc:.4f}')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 指标统计摘要
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"{n_folds}-Fold CV Statistics\n\n"
        if accuracies and len(accuracies) > 0:
            stats_text += f"Accuracy:\n"
            stats_text += f"  Mean: {np.mean(accuracies):.4f}\n"
            stats_text += f"  Std: {np.std(accuracies):.4f}\n\n"
        else:
            stats_text += "Accuracy: N/A\n\n"
            
        if recalls and len(recalls) > 0:
            stats_text += f"Recall:\n"
            stats_text += f"  Mean: {np.mean(recalls):.4f}\n"
            stats_text += f"  Std: {np.std(recalls):.4f}\n\n"
        else:
            stats_text += "Recall: N/A\n\n"
            
        if f1_scores and len(f1_scores) > 0:
            stats_text += f"F1 Score:\n"
            stats_text += f"  Mean: {np.mean(f1_scores):.4f}\n"
            stats_text += f"  Std: {np.std(f1_scores):.4f}\n\n"
        else:
            stats_text += "F1 Score: N/A\n\n"
            
        if aucs and len(aucs) > 0:
            stats_text += f"AUC:\n"
            stats_text += f"  Mean: {np.mean(aucs):.4f}\n"
            stats_text += f"  Std: {np.std(aucs):.4f}\n"
        else:
            stats_text += "AUC: N/A\n"
            
        ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    # 保存图形
    plot_file = output_dir / "cv_training_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

def save_training_results(
    model: Any,
    metrics: Dict,
    selected_snps: List[str],
    output_dir: str,
    model_type: str,
    task_type: str,
    feature_importance_df: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None
) -> None:
    """
    保存训练结果（模型文件、评估指标、筛选的SNP列表、特征重要性）
    
    :param model: 训练好的模型
    :param metrics: 评估指标
    :param selected_snps: 筛选的SNP列表
    :param output_dir: 输出目录
    :param model_type: 模型类型
    :param task_type: 任务类型
    :param feature_importance_df: 特征重要性DataFrame
    :param y_test: 测试集真实值（用于绘制性能曲线）
    :param y_pred: 测试集预测值（用于绘制性能曲线）
    :param y_prob: 测试集预测概率（分类任务需要，用于绘制性能曲线）
    """
    # 创建模型专属目录
    model_dir = Path(output_dir) / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存模型文件
    model_file = model_dir / f"{model_type}_model.pkl"
    import joblib
    joblib.dump(model, model_file)
    logger.debug(f"   模型已保存: {model_file.name}")
    
    # 2. 保存评估指标
    metrics_file = model_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "model_type": model_type,
            "task_type": task_type,
            "metrics": metrics,
            "training_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    logger.debug(f"   评估指标已保存: {metrics_file.name}")
    
    # 3. 保存筛选的SNP列表
    snp_file = model_dir / "selected_snps.txt"
    with open(snp_file, 'w') as f:
        if selected_snps:
            # 排除phenotype列
            filtered_snps = filter_phenotype_columns(selected_snps)
            f.write("\n".join(filtered_snps))
        else:
            f.write("No SNPs selected (LD/GWAS skipped or no significant SNPs)")
    logger.debug(f"   SNP列表已保存: {snp_file.name}")
    
    # 4. 保存特征重要性
    if feature_importance_df is not None and not feature_importance_df.empty:
        # 排除phenotype列
        filtered_importance_df = feature_importance_df[
            feature_importance_df['feature'].astype(str).apply(lambda x: 'phenotype' not in x.lower())
        ]
        importance_file = model_dir / "feature_importance.txt"
        filtered_importance_df.to_csv(importance_file, sep='\t', index=False)
        logger.debug(f"   特征重要性已保存: {importance_file.name}")
        
        # 保存Top N特征
        top_n = min(100, len(filtered_importance_df))
        top_features_file = model_dir / "top_features.txt"
        top_features = filtered_importance_df.head(top_n)['feature'].tolist()
        with open(top_features_file, 'w') as f:
            f.write("\n".join(top_features))
        logger.debug(f"   Top {top_n}特征已保存: {top_features_file.name}")
    
    # 5. 绘制性能评估指标变化曲线（分类和回归任务）
    if y_test is not None and y_pred is not None:
        try:
            plot_performance_curves(y_test, y_pred, y_prob, model_dir, model_type, task_type)
            logger.debug(f"   性能评估曲线已保存")
        except Exception as e:
            logger.warning(f" 绘制性能评估曲线失败: {str(e)}")
    
    logger.info(f" 结果保存完成: {model_dir}")

# ======================== 5. 核心训练函数（关键修改：解耦LD和GWAS逻辑） ========================
def run_single_model(
    input_path: str,
    model_type: str,
    output_dir: str,
    feature_selection_mode: int,
    task_type: Optional[str] = None,
    n_folds: int = 5,
    random_state: int = 42,
    gwas_genotype: Optional[str] = None,
    gwas_pvalue: Optional[float] = None,
    # LD过滤参数（当feature_selection_mode需要时使用）
    ld_window_kb: int = 50,
    ld_window: int = 5,
    ld_window_r2: float = 0.2,
    ld_threads: int = 8
) -> int:
    """
    单模型训练主函数
    特征筛选模式（feature_selection_mode）：
    1: 空白对照（不使用GWAS和LD）
    2: GWAS筛选（仅使用GWAS）
    3: LD过滤（仅使用LD）
    4: GWAS和LD综合过滤（先GWAS后LD）
    
    逻辑：
    1. 根据feature_selection_mode确定是否开启LD过滤和GWAS
    2. 先判断是否开启GWAS → 执行GWAS分析，得到显著SNP列表
    3. 再判断是否开启LD过滤 → 对于模式4，仅对GWAS显著SNP执行LD过滤；对于模式3，对所有SNP执行LD过滤
    4. 最终特征筛选：对于模式4，直接使用LD过滤结果（因为LD过滤已经只针对GWAS显著SNP）；若LD过滤失败，则退回到仅使用GWAS显著SNP
    """
    start_time = time.time()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 临时目录根（输出目录下的 temp_m，model training专用，与preprocess的tmp_p区分）
    tmp_root = output_dir_path / "temp_m"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 使用全局临时文件管理器注册临时目录/文件
    def register_dir(p: Path) -> Path:
        return _temp_file_manager.register_dir(p)

    def register_file(p: Path) -> Path:
        return _temp_file_manager.register_file(p)

    # 本次运行直接使用统一的 tmp 目录，不再附加日期/进程号
    run_tmp_dir = register_dir(tmp_root)

    # 预处理阶段的临时目录（从元数据读取，不清理，preprocess模块执行完毕不删除tmp目录）
    preprocess_tmp_dir = None

    try:
        # Step 1: 解析输入路径（核心衔接preprocess）
        input_info = parse_train_input_path(input_path)
        train_file = input_info["train_file"]
        valid_samples = input_info["metadata"]["valid_samples"] if input_info["metadata"] else None
        
        # GWAS路径优先级：优先使用preprocess元数据中的过滤后PLINK前缀，其次才使用用户手动指定
        # 如果元数据中的路径不存在，提示用户手动输入
        if input_info["gwas_genotype_prefix"]:
            # 验证文件是否存在（确保使用绝对路径）
            gwas_prefix = input_info["gwas_genotype_prefix"]
            # 如果是相对路径，尝试转换为绝对路径（相对于元数据文件所在目录）
            if not Path(gwas_prefix).is_absolute() and input_info["metadata"]:
                # 尝试从元数据文件路径推断
                metadata_file = input_info.get("_metadata_file_path")
                if metadata_file:
                    metadata_dir = Path(metadata_file).parent
                    gwas_prefix = str((metadata_dir / gwas_prefix).absolute())
            else:
                # 确保是绝对路径
                gwas_prefix = str(Path(gwas_prefix).absolute())
            
            required_plink = [f"{gwas_prefix}.bed", f"{gwas_prefix}.bim", f"{gwas_prefix}.fam"]
            missing_plink = [f for f in required_plink if not Path(f).exists()]
            if missing_plink:
                logger.warning(f"元数据中的GWAS PLINK文件不存在: {gwas_prefix}")
                logger.warning(f"缺失的文件: {', '.join(missing_plink)}")
                if gwas_genotype:
                    final_gwas_genotype = gwas_genotype
                    logger.info(f"使用命令行提供的GWAS基因型前缀: {final_gwas_genotype}")
                else:
                    final_gwas_genotype = None
                    # 注意：这里不抛出异常，让后续的检查逻辑统一处理错误提示
            else:
                final_gwas_genotype = gwas_prefix
                logger.info(f"使用preprocess元数据中的GWAS基因型前缀: {final_gwas_genotype}")
        elif gwas_genotype:
            final_gwas_genotype = gwas_genotype
            logger.info(f"使用命令行提供的GWAS基因型前缀: {final_gwas_genotype}")
        else:
            final_gwas_genotype = None

        # 如果存在GWAS/LD所需的基因型前缀，记录其所在的 tmp 目录
        # 如果元数据里带有 preprocess_tmp_dir（tmp_p目录），则记录（但不清理，preprocess模块执行完毕不删除）
        if input_info["preprocess_tmp_dir"]:
            preprocess_tmp_dir = Path(input_info["preprocess_tmp_dir"]).absolute()
            _temp_file_manager.register_preprocess_tmp_dir(preprocess_tmp_dir)

        # Step 2: 加载训练数据
        logger.debug("加载训练数据...")
        X, y, snp_name_mapping = load_training_data(train_file, valid_samples)
        logger.info(f"数据集: {X.shape[0]}样本, {X.shape[1]}特征")

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
            logger.info(f"任务类型: {task_type} (自动推断)")
        
        # 初始化交叉验证
        if task_type == "classification":
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y))
        
        logger.info(f"开始{n_folds}折交叉验证")

        # Step 3: 根据feature_selection_mode确定是否启用LD和GWAS
        # 注意：特征筛选在交叉验证之前使用全部数据执行，避免数据泄露
        # 模式说明：
        # 1: 空白对照（不使用GWAS和LD）
        # 2: GWAS筛选（仅使用GWAS）
        # 3: LD过滤（仅使用LD）
        # 4: GWAS和LD综合过滤（先GWAS后LD）
        if feature_selection_mode not in [1, 2, 3, 4]:
            raise ValueError(f" feature_selection_mode必须是1、2、3或4，当前值: {feature_selection_mode}")
        
        enable_ld = feature_selection_mode in [3, 4]  # 模式3或4启用LD
        enable_gwas = feature_selection_mode in [2, 4]  # 模式2或4启用GWAS
        
        mode_names = {
            1: "空白对照（不使用GWAS和LD）",
            2: "GWAS筛选（仅使用GWAS）",
            3: "LD过滤（仅使用LD）",
            4: "GWAS和LD综合过滤（先GWAS后LD）"
        }
        
        # 检查必需的基因型文件
        if (enable_ld or enable_gwas) and not final_gwas_genotype:
            error_msg = (
                f"\n错误：特征筛选模式 {feature_selection_mode} ({mode_names[feature_selection_mode]}) 需要GWAS基因型文件。\n"
                f"请通过以下方式之一提供：\n"
            )
            if input_info["metadata"]:
                error_msg += (
                    f"  1. 检查预处理阶段生成的元数据文件中的 gwas_genotype_prefix 字段\n"
                    f"     如果元数据中的文件路径不存在或文件已被删除，请使用方式2手动指定\n"
                    f"  2. 使用 --gwas_genotype 参数手动指定GWAS基因型文件前缀\n"
                    f"     例如：--gwas_genotype /path/to/your/plink_prefix\n"
                )
            else:
                error_msg += (
                    f"  1. 确保预处理阶段生成的元数据文件（*_metadata.json）存在且包含 gwas_genotype_prefix 字段\n"
                    f"     如果元数据文件不存在，请重新运行预处理步骤\n"
                    f"  2. 使用 --gwas_genotype 参数手动指定GWAS基因型文件前缀\n"
                    f"     例如：--gwas_genotype /path/to/your/plink_prefix\n"
                )
            raise ValueError(error_msg)
        
        # Step 4: 特征筛选（使用全部数据，避免数据泄露）
        # selected_snps / ld_selected_snps 均使用“原始SNP名称”
        selected_snps = None          # GWAS显著SNP列表（原始名称）
        ld_selected_snps = None       # LD过滤保留的SNP列表（原始名称）
        all_sample_ids = X.index.tolist()

        # Step 5: 先执行GWAS特征筛选（如果开启）
        if enable_gwas and final_gwas_genotype:
            logger.debug("GWAS分析中...")
            if gwas_pvalue is None:
                gwas_pvalue = 0.01
            # 生成全部数据的表型文件（用于GWAS），放在 temp_m/gwas
            gwas_dir = register_dir(run_tmp_dir / "gwas")
            pheno_file = gwas_dir / "all_pheno.tsv"
            generate_gwas_phenotype_file(all_sample_ids, y, pheno_file)
            register_file(pheno_file)
            
            # 执行GWAS（直接基于原始/预处理后的GWAS基因型前缀）
            # 注意：GEMMA会将结果写入当前工作目录下的output/目录
            # 为了确保输出到用户指定的目录，需要切换工作目录
            # 所有GWAS生成的临时文件会在模块完成后自动删除
            gwas_output_prefix = gwas_dir / "all_gwas"
            
            # 确保所有文件路径都是绝对路径，避免切换工作目录后找不到文件
            final_gwas_genotype_abs = Path(final_gwas_genotype).absolute().as_posix()
            pheno_file_abs = pheno_file.absolute().as_posix()
            gwas_output_prefix_abs = gwas_output_prefix.absolute().as_posix()
            
            # 保存当前工作目录，然后切换到用户指定的输出目录
            original_cwd = os.getcwd()
            try:
                # 切换到用户指定的输出目录，这样GEMMA会将output/目录创建在这里
                os.chdir(output_dir_path)
                selected_snps = run_gemma_gwas(
                    genotype_prefix=final_gwas_genotype_abs,
                    phenotype_file=pheno_file_abs,
                    output_prefix=gwas_output_prefix_abs,
                    pvalue_threshold=gwas_pvalue,
                    threads=ld_threads
                )
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
            
            logger.info(f"GWAS筛选完成: {len(selected_snps) if selected_snps else 0}个显著SNP")
            
            # 注册GWAS结果文件以便后续清理
            # GEMMA将结果写入用户指定输出目录下的output/目录
            base_prefix = Path(gwas_output_prefix).name
            gwas_result_file = output_dir_path / "output" / f"{base_prefix}_gwas.assoc.txt"
            if gwas_result_file.exists():
                register_file(gwas_result_file)
                logger.debug(f"已注册GWAS结果文件以便清理: {gwas_result_file}")
            
            # 注册GWAS生成的其他文件（kinship矩阵、协变量、中间文件等）
            # GEMMA会在用户指定输出目录下的output/目录中生成多个文件
            gemma_output_dir = output_dir_path / "output"
            if gemma_output_dir.exists():
                # 查找所有与本次GWAS相关的文件
                gwas_patterns = [
                    f"{base_prefix}_gwas*",
                    f"{base_prefix}_kinship*",
                    f"{base_prefix}_covariates*",
                    f"{base_prefix}_clean_geno*",
                    f"{base_prefix}_geno_pca*",
                    f"{base_prefix}_valid_samples*"
                ]
                
                for pattern in gwas_patterns:
                    for file_path in gemma_output_dir.glob(pattern):
                        if file_path.is_file():
                            register_file(file_path)
                            logger.debug(f"已注册GWAS相关文件以便清理: {file_path}")
                
                # 如果output目录下只有本次GWAS生成的文件，也可以考虑注册整个目录
                # 但为了安全起见，只注册文件，不删除整个output目录（可能包含其他运行的结果）
            
            # 注册显著SNP列表文件（run_gemma_gwas函数会在output_prefix所在目录生成）
            # 使用绝对路径，确保无论工作目录如何都能找到文件
            snp_list_file = gwas_output_prefix.parent.absolute() / f"{base_prefix}_significant_snps.txt"
            if snp_list_file.exists():
                register_file(snp_list_file)
                logger.debug(f"已注册GWAS显著SNP列表文件以便清理: {snp_list_file}")
        elif enable_gwas and not final_gwas_genotype:
            logger.warning(" GWAS筛选需要基因型文件，跳过")

        # Step 6: 再执行LD过滤（如果开启）
        if enable_ld and final_gwas_genotype:
            logger.debug("LD过滤中...")
            # 生成有效样本文件（用于LD过滤，使用全部数据）
            ld_dir = register_dir(run_tmp_dir / "ld")
            valid_sample_file = ld_dir / "valid_samples.txt"
            with open(valid_sample_file, 'w') as f:
                for sample_id in all_sample_ids:
                    f.write(f"{sample_id}\t{sample_id}\n")
            register_file(valid_sample_file)
            
            # 模式4：如果已有GWAS显著SNP，只对这些SNP进行LD过滤
            extract_snps_file = None
            if feature_selection_mode == 4 and selected_snps:
                # 创建GWAS显著SNP列表文件，用于LD过滤
                extract_snps_file = ld_dir / "gwas_significant_snps.txt"
                with open(extract_snps_file, 'w') as f:
                    for snp in selected_snps:
                        f.write(f"{snp}\n")
                register_file(extract_snps_file)
                logger.info(f"模式4：仅对{len(selected_snps)}个GWAS显著SNP进行LD过滤")
            
            # 执行LD过滤（所有LD中间文件和结果写入 temp_m/ld）
            # 模式4：只对GWAS显著SNP进行LD过滤
            # 模式3：对所有SNP进行LD过滤
            # 确保所有文件路径都是绝对路径
            final_gwas_genotype_abs = Path(final_gwas_genotype).absolute().as_posix()
            ld_output_prefix_abs = ld_dir.absolute() / "train_ld"
            valid_sample_file_abs = valid_sample_file.absolute().as_posix()
            extract_snps_file_abs = extract_snps_file.absolute().as_posix() if extract_snps_file else None
            
            _, ld_selected_snps = run_ld_filtering(
                genotype_prefix=final_gwas_genotype_abs,
                output_prefix=str(ld_output_prefix_abs),
                ld_window_kb=ld_window_kb,
                ld_window=ld_window,
                ld_window_r2=ld_window_r2,
                threads=ld_threads,
                keep_samples_file=valid_sample_file_abs,
                extract_snps_file=extract_snps_file_abs
            )
            logger.info(f"LD过滤完成: {len(ld_selected_snps)}个SNP")
        elif enable_ld and not final_gwas_genotype:
            logger.warning(" LD过滤需要基因型文件，跳过")

        # Step 7: 应用特征筛选
        # 逻辑说明：
        # - 模式4：先GWAS筛选，再对GWAS显著SNP进行LD过滤，直接使用LD过滤结果（因为LD过滤已经只针对GWAS显著SNP）
        # - 模式2：仅使用GWAS显著SNP
        # - 模式3：仅使用LD过滤结果
        # - 模式1：使用全部特征
        raw_selected_snps = None  # 仍为"原始SNP名称"

        if feature_selection_mode == 4:
            # 模式4：先GWAS后LD，LD过滤结果已经是针对GWAS显著SNP的，直接使用
            if ld_selected_snps:
                raw_selected_snps = ld_selected_snps
                logger.info(f"模式4（先GWAS后LD）: {len(raw_selected_snps)}个SNP（GWAS显著SNP经LD过滤后）")
            elif selected_snps:
                # 如果LD过滤失败或没有结果，退回到GWAS结果
                raw_selected_snps = selected_snps
                logger.warning("模式4：LD过滤无结果，退回仅使用GWAS显著SNP")
            else:
                raw_selected_snps = X.columns.tolist()
                logger.warning("模式4：GWAS和LD均无结果，使用全部特征")
        elif selected_snps:
            # 模式2：仅GWAS
            raw_selected_snps = selected_snps
        elif ld_selected_snps:
            # 模式3：仅LD
            raw_selected_snps = ld_selected_snps
        else:
            # 模式1：全部特征
            raw_selected_snps = X.columns.tolist()

        # 将“原始SNP名称”映射到清理后的列名：
        # - 对于有snp_name_mapping的SNP，用映射后的列名
        # - 对于非SNP特征（不在映射中），直接保留原列名（若在X中存在）
        mapped_snps = []
        for snp in raw_selected_snps:
            cleaned = snp_name_mapping.get(snp, snp)
            if cleaned in X.columns and cleaned not in mapped_snps:
                mapped_snps.append(cleaned)

        if mapped_snps:
            selected_snps = mapped_snps
        else:
            # 理论上不会为空，兜底使用全部特征
            selected_snps = X.columns.tolist()

        # 确保排除phenotype列（如果存在）
        selected_snps = filter_phenotype_columns(selected_snps)
        
        # 应用特征筛选到X
        X_filtered = X[selected_snps]
        logger.info(f"特征筛选完成: {X_filtered.shape[1]}个特征")

        # Step 8: 5折交叉验证训练和评估
        cv_fold_metrics = []
        all_y_test = []
        all_y_pred = []
        all_y_prob = []
        fold_iter = tqdm(splits, total=n_folds, desc="CV folds", unit="fold") if TQDM_AVAILABLE else splits

        for fold_idx, (train_idx, val_idx) in enumerate(fold_iter, 1):
            logger.debug(f"[{fold_idx}/{n_folds}] 训练中...")
            X_train_fold = X_filtered.iloc[train_idx]
            X_val_fold = X_filtered.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # 执行超参数网格搜索
            logger.debug(f"  [{fold_idx}/{n_folds}] 超参数搜索中...")
            param_grid = get_param_grid(model_type, task_type)
            model_fold = perform_grid_search(
                model_type=model_type,
                task_type=task_type,
                X_train=X_train_fold,
                y_train=y_train_fold,
                param_grid=param_grid,
                n_iter=20,  # 每次搜索20个参数组合
                cv=3,  # 网格搜索使用3折交叉验证
                random_state=random_state
            )
            
            # 预测和评估
            y_pred_fold = model_fold.predict(X_val_fold)
            
            # 尝试获取预测概率（分类任务需要）
            y_prob_fold = None
            if task_type == "classification":
                try:
                    if hasattr(model_fold, 'predict_proba'):
                        y_prob_fold = model_fold.predict_proba(X_val_fold)
                    elif hasattr(model_fold, 'decision_function'):
                        # 对于某些SVM模型，如果没有probability=True，使用decision_function
                        decision_scores = model_fold.decision_function(X_val_fold)
                        # 将decision_function转换为概率（简单归一化）
                        if len(decision_scores.shape) == 1:
                            # 二分类
                            from sklearn.utils.extmath import softmax
                            prob_neg = 1 / (1 + np.exp(decision_scores))
                            prob_pos = 1 - prob_neg
                            y_prob_fold = np.column_stack([prob_neg, prob_pos])
                        else:
                            # 多分类：使用softmax
                            from sklearn.utils.extmath import softmax
                            y_prob_fold = softmax(decision_scores)
                    else:
                        logger.warning(f" 模型不支持predict_proba或decision_function，无法计算AUC")
                except Exception as e:
                    logger.warning(f" 获取预测概率失败: {str(e)}")
            
            fold_metrics = evaluate_model(y_val_fold, y_pred_fold, y_prob_fold, task_type)
            cv_fold_metrics.append(fold_metrics)
            
            # 显示当前折的关键指标
            if task_type == "regression":
                corr = fold_metrics.get('pearson_correlation', 'N/A')
                logger.info(f"[{fold_idx}/{n_folds}] 相关系数: {corr}")
            else:
                acc = fold_metrics.get('accuracy', 'N/A')
                auc = fold_metrics.get('auc', 'N/A')
                logger.info(f"[{fold_idx}/{n_folds}] 准确率: {acc}, AUC: {auc}")
            
            # 收集所有折的预测结果（用于最终的性能评估曲线）
            all_y_test.append(y_val_fold)
            all_y_pred.append(y_pred_fold)
            if y_prob_fold is not None:
                all_y_prob.append(y_prob_fold)
            
        
        # 计算平均指标
        logger.info(f"{n_folds}折交叉验证平均结果:")
        
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
                avg_metrics['pearson_pvalue'] = float(round(np.mean(pearson_pvalues), 6))
                avg_metrics['pearson_pvalue_std'] = float(round(np.std(pearson_pvalues), 6))
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
        logger.debug("训练最终模型...")
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
        logger.info("最终模型训练完成")
        
        # 合并所有折的预测结果
        y_test_combined = pd.concat(all_y_test, axis=0) if all_y_test else None
        y_pred_combined = np.concatenate(all_y_pred) if all_y_pred else None
        y_prob_combined = np.vstack(all_y_prob) if all_y_prob and len(all_y_prob) > 0 else None
        
        metrics = avg_metrics  # 使用平均指标作为最终指标

        # Step 10: 计算特征重要性（使用最终模型）
        logger.debug("计算特征重要性...")
        feature_cols = filter_phenotype_columns(X_filtered.columns.tolist())
        
        feature_importance_df = calculate_feature_importance(
            model=final_model,
            model_type=model_type,
            feature_names=feature_cols,
            X_train=X_filtered[feature_cols] if feature_cols else X_filtered,
            y_train=y,
            task_type=task_type
        )
        logger.info("特征重要性计算完成")

        # Step 11: 保存结果
        logger.debug("保存结果...")
        model_dir = Path(output_dir) / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存交叉验证结果
        cv_results = {
            'n_folds': n_folds,
            'fold_metrics': cv_fold_metrics,
            'average_metrics': avg_metrics
        }
        cv_results_file = model_dir / "cv_results.json"
        with open(cv_results_file, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        # 绘制交叉验证训练过程曲线
        try:
            plot_cv_training_curves(cv_results, model_dir, model_type, task_type)
        except Exception as e:
            logger.warning(f" 绘制训练过程曲线失败: {str(e)}")
        
        save_training_results(
            model=final_model,
            metrics=metrics,
            selected_snps=selected_snps,
            output_dir=output_dir,
            model_type=model_type,
            task_type=task_type,
            feature_importance_df=feature_importance_df,
            y_test=y_test_combined,
            y_pred=y_pred_combined,
            y_prob=y_prob_combined
        )

        # 最终日志
        total_time = round(time.time() - start_time, 2)
        logger.info(f"训练完成 (耗时: {total_time}秒)")
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
                                    logger.debug(f"跳过删除tmp_root（与preprocess_tmp_dir重叠）: {tmp_root}")
                            except AttributeError:
                                # Python < 3.9 使用其他方法检查
                                try:
                                    preprocess_tmp_dir_abs.relative_to(tmp_root_abs)
                                    should_delete = False
                                    logger.debug(f"跳过删除tmp_root（与preprocess_tmp_dir重叠）: {tmp_root}")
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
                            logger.info(f"已删除preprocess临时目录: {preprocess_tmp_dir_path}")
                    except Exception as e:
                        logger.warning(f"删除preprocess临时目录失败（已忽略）: {e}")
                
                # 正常运行完成后，删除GWAS运行产生的output目录
                gwas_output_dir = output_dir_path / "output"
                if gwas_output_dir.exists():
                    try:
                        shutil.rmtree(gwas_output_dir, ignore_errors=True)
                        logger.info(f"已删除GWAS output目录: {gwas_output_dir}")
                    except Exception as e:
                        logger.warning(f"删除GWAS output目录失败（已忽略）: {e}")
            except Exception as cleanup_err:
                logger.debug(f"清理临时目录时出错（已忽略）：{cleanup_err}")
        else:
            logger.debug("调试模式：CLEANUP_TEMP_FILES=False，本次运行产生的临时文件和目录将被保留。")
        return 0

    except Exception as e:
        logger.error(f"单模型训练失败: {str(e)}", exc_info=True)
        # 根据调试开关决定异常时是否清理临时目录/文件
        if CLEANUP_TEMP_FILES:
            # 异常：仅清理本次运行登记的临时文件/目录，不触碰preprocess阶段的临时目录
            try:
                _temp_file_manager.cleanup_on_exit(cleanup_preprocess=False)
                # 异常时也检查tmp_root，但需要确保不会删除preprocess的tmp目录
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
                                    logger.debug(f"异常时跳过删除tmp_root（与preprocess_tmp_dir重叠）: {tmp_root}")
                            except AttributeError:
                                # Python < 3.9 使用其他方法检查
                                try:
                                    preprocess_tmp_dir_abs.relative_to(tmp_root_abs)
                                    should_delete = False
                                    logger.debug(f"异常时跳过删除tmp_root（与preprocess_tmp_dir重叠）: {tmp_root}")
                                except ValueError:
                                    pass  # 不是相对路径，可以删除
                        if should_delete:
                            shutil.rmtree(tmp_root, ignore_errors=True)
                except Exception:
                    pass
            except Exception as cleanup_err:
                logger.debug(f"清理临时目录时出错（已忽略）：{cleanup_err}")
        else:
            logger.debug("调试模式：发生异常也保留本次运行产生的临时文件和目录（CLEANUP_TEMP_FILES=False）。")
        return 1
def run_all_models(
    input_path: str,
    output_dir: str,
    feature_selection_mode: int,
    task_type: Optional[str] = None,
    n_folds: int = 5,
    random_state: int = 42,
    gwas_genotype: Optional[str] = None,
    gwas_pvalue: Optional[float] = None,
    # LD过滤参数（当feature_selection_mode需要时使用）
    ld_window_kb: int = 50,
    ld_window: int = 5,
    ld_window_r2: float = 0.2,
    ld_threads: int = 8
) -> int:
    """训练所有支持的模型，并生成对比报告"""
    # 首先解析输入路径，获取task_type默认值
    input_info = parse_train_input_path(input_path)
    if task_type is None:
        task_type = input_info.get("task_type") or "regression"
        if not input_info.get("task_type"):
            logger.warning(f" 未指定task_type且元数据中无task_type信息，使用默认值: regression")
    else:
        pass
    
    supported_models = ["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"]
    results = {}
    start_time = time.time()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 在输出目录下创建统一临时目录 temp_m，用于全模型训练阶段的所有中间文件（model training专用）
    tmp_dir = output_dir_path / "temp_m"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 遍历所有模型训练
        for model_type in supported_models:
            logger.info(f"训练模型: {model_type}")
            ret_code = run_single_model(
                input_path=input_path,
                model_type=model_type,
                output_dir=output_dir,
                feature_selection_mode=feature_selection_mode,
                task_type=task_type,
                n_folds=n_folds,
                random_state=random_state,
                gwas_genotype=gwas_genotype,
                gwas_pvalue=gwas_pvalue,
                ld_window_kb=ld_window_kb,
                ld_window=ld_window,
                ld_window_r2=ld_window_r2,
                ld_threads=ld_threads
            )
            results[model_type] = "成功" if ret_code == 0 else "失败"

        # 生成模型对比报告
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        report_file = output_dir_path / "model_comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                "task_type": task_type,
                "n_folds": n_folds,
                "random_state": random_state,
                "feature_selection_mode": feature_selection_mode,
                "gwas_pvalue": gwas_pvalue if feature_selection_mode in [2, 4] else None,
                "ld_parameters": {
                    "window_kb": ld_window_kb,
                    "window": ld_window,
                    "r2": ld_window_r2
                } if feature_selection_mode in [3, 4] else None,
                "training_results": results,
                "total_training_time": round(time.time() - start_time, 2),
                "generated_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        logger.info(f" 所有模型训练完成")
        
        # 根据调试开关决定是否清理临时目录/文件
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
                                    logger.debug(f"跳过删除tmp_dir（与preprocess_tmp_dir重叠）: {tmp_dir}")
                            except AttributeError:
                                # Python < 3.9 使用其他方法检查
                                try:
                                    preprocess_tmp_dir_abs.relative_to(tmp_dir_abs)
                                    should_delete = False
                                    logger.debug(f"跳过删除tmp_dir（与preprocess_tmp_dir重叠）: {tmp_dir}")
                                except ValueError:
                                    pass  # 不是相对路径，可以删除
                    except Exception:
                        pass  # 如果无法获取preprocess_tmp_dir，则按原逻辑删除
                    
                    if should_delete:
                        import shutil
                        shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as cleanup_err:
                logger.warning(f" 删除全模型训练临时目录失败（已忽略）：{cleanup_err}")
            
            # 正常运行完成后，删除preprocess模块产生的tmp_p目录
            try:
                input_info = parse_train_input_path(input_path)
                if input_info.get("preprocess_tmp_dir"):
                    preprocess_tmp_dir_path = Path(input_info["preprocess_tmp_dir"]).absolute()
                    if preprocess_tmp_dir_path.exists():
                        shutil.rmtree(preprocess_tmp_dir_path, ignore_errors=True)
                        logger.info(f"已删除preprocess临时目录: {preprocess_tmp_dir_path}")
            except Exception as e:
                logger.warning(f"删除preprocess临时目录失败（已忽略）: {e}")
            
            # 正常运行完成后，删除GWAS运行产生的output目录
            gwas_output_dir = output_dir_path / "output"
            if gwas_output_dir.exists():
                try:
                    shutil.rmtree(gwas_output_dir, ignore_errors=True)
                    logger.info(f"已删除GWAS output目录: {gwas_output_dir}")
                except Exception as e:
                    logger.warning(f"删除GWAS output目录失败（已忽略）: {e}")
        else:
            logger.debug("调试模式：CLEANUP_TEMP_FILES=False，本次运行产生的临时文件和目录将被保留。")
        
        return 0

    except Exception as e:
        logger.error(f" 全模型训练失败: {str(e)}", exc_info=True)
        # 异常时也尝试清理空的 tmp 根目录（不会触碰其中仍有内容的情况）
        # 由于tmp_dir是temp_m，preprocess_tmp_dir是tmp_p，名称不同，不会冲突
        # 但保留检查逻辑作为额外安全措施
        try:
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                should_delete = True
                # 尝试从输入路径获取preprocess_tmp_dir（tmp_p目录）
                try:
                    input_info = parse_train_input_path(input_path)
                    if input_info.get("preprocess_tmp_dir"):
                        preprocess_tmp_dir_abs = Path(input_info["preprocess_tmp_dir"]).absolute()
                        tmp_dir_abs = tmp_dir.absolute()
                        # 如果tmp_dir是preprocess_tmp_dir或其父目录，则不删除（理论上不会发生，因为名称不同）
                        try:
                            # Python 3.9+ 使用 is_relative_to
                            if tmp_dir_abs == preprocess_tmp_dir_abs or preprocess_tmp_dir_abs.is_relative_to(tmp_dir_abs):
                                should_delete = False
                                logger.debug(f"异常时跳过删除tmp_dir（与preprocess_tmp_dir重叠）: {tmp_dir}")
                        except AttributeError:
                            # Python < 3.9 使用其他方法检查
                            try:
                                preprocess_tmp_dir_abs.relative_to(tmp_dir_abs)
                                should_delete = False
                                logger.debug(f"异常时跳过删除tmp_dir（与preprocess_tmp_dir重叠）: {tmp_dir}")
                            except ValueError:
                                pass  # 不是相对路径，可以删除
                except Exception:
                    pass  # 如果无法获取preprocess_tmp_dir，则按原逻辑删除
                
                if should_delete:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        return 1

# ======================== 6. 预测函数 ========================
def predict_with_model(
    input_path: str,
    model_type: str,
    output_dir: str,
    task_type: str
) -> int:
    """使用训练好的模型进行预测"""
    try:
        # 1. 加载预测数据
        X, _, _ = load_training_data(input_path)  # 预测数据无需表型和映射，_占位
        
        # 2. 加载训练好的模型
        model_file = Path(output_dir) / model_type / f"{model_type}_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        import joblib
        model = joblib.load(model_file)
        logger.info(f" 加载预训练模型")
        
        # 3. 执行预测
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if task_type == "classification" else None
        
        # 4. 保存预测结果
        output_dir = Path(output_dir) / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_file = output_dir / "predictions.tsv"
        
        # 构建结果DataFrame
        result_df = pd.DataFrame({
            "sample": X.index,
            "prediction": y_pred
        })
        # 分类任务添加概率列
        if task_type == "classification" and y_prob is not None:
            for i in range(y_prob.shape[1]):
                result_df[f"prob_class_{i}"] = y_prob[:, i]
        
        result_df.to_csv(pred_file, sep="\t", index=False)
        logger.info(f" 预测完成")
        
        return 0
    except Exception as e:
        logger.error(f" 预测失败: {str(e)}", exc_info=True)
        return 1

# ======================== 7. 主函数（新增LD过滤命令行参数） ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Training Module (支持独立LD/GWAS控制)")
    subparsers = parser.add_subparsers(dest="command")

    # 单模型训练（新增LD过滤参数）
    train_parser = subparsers.add_parser("train", help="Train single model")
    train_parser.add_argument("-i", "--input", required=True, help="训练文件/元数据文件路径")
    train_parser.add_argument("-m", "--model", required=True, choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"])
    train_parser.add_argument("-o", "--output_dir", required=True, help="输出目录")
    train_parser.add_argument("--task_type", required=False, choices=["classification", "regression"],
                             help="任务类型（可选，如未指定将从元数据读取或默认regression）")
    train_parser.add_argument("--n_folds", type=int, default=5, help="交叉验证折数（默认: 5）")
    train_parser.add_argument("--random_state", type=int, default=42)
    train_parser.add_argument("-f", "--feature_selection_mode", type=int, required=True, choices=[1, 2, 3, 4],
                             help="特征筛选模式(必选): 1=空白对照, 2=GWAS筛选, 3=LD过滤, 4=GWAS和LD综合过滤")
    train_parser.add_argument("--gwas_genotype", help="GWAS基因型文件前缀（模式2或4需要）")
    train_parser.add_argument("--gwas_pvalue", type=float, default=0.01, help="GWAS P值阈值（模式2或4使用，默认5e-8）")
    train_parser.add_argument("--ld_window_kb", type=int, default=50, help="LD窗口大小(KB，模式3或4使用)")
    train_parser.add_argument("--ld_window", type=int, default=5, help="LD窗口变体数（模式3或4使用）")
    train_parser.add_argument("--ld_window_r2", type=float, default=0.2, help="LD r²阈值（模式3或4使用）")
    train_parser.add_argument("--ld_threads", type=int, default=8, help="LD过滤线程数（模式3或4使用）")

    # 全模型训练（新增LD过滤参数）
    train_all_parser = subparsers.add_parser("train-all", help="Train all models")
    train_all_parser.add_argument("-i", "--input", required=True)
    train_all_parser.add_argument("-o", "--output_dir", required=True)
    train_all_parser.add_argument("--task_type", required=False, choices=["classification", "regression"],
                                 help="任务类型（可选，如未指定将从元数据读取或默认regression）")
    train_all_parser.add_argument("--n_folds", type=int, default=5, help="交叉验证折数（默认: 5）")
    train_all_parser.add_argument("--random_state", type=int, default=42)
    train_all_parser.add_argument("-f", "--feature_selection_mode", type=int, required=True, choices=[1, 2, 3, 4],
                                help="特征筛选模式(必选): 1=空白对照, 2=GWAS筛选, 3=LD过滤, 4=GWAS和LD综合过滤")
    train_all_parser.add_argument("--gwas_genotype", help="GWAS基因型文件前缀（模式2或4需要）")
    train_all_parser.add_argument("--gwas_pvalue", type=float, default=0.01, help="GWAS P值阈值（模式2或4使用，默认5e-8）")
    train_all_parser.add_argument("--ld_window_kb", type=int, default=50, help="LD窗口大小(KB，模式3或4使用)")
    train_all_parser.add_argument("--ld_window", type=int, default=5, help="LD窗口变体数（模式3或4使用）")
    train_all_parser.add_argument("--ld_window_r2", type=float, default=0.2, help="LD r²阈值（模式3或4使用）")
    train_all_parser.add_argument("--ld_threads", type=int, default=8, help="LD过滤线程数（模式3或4使用）")

    # 预测
    predict_parser = subparsers.add_parser("predict", help="Predict with trained model")
    predict_parser.add_argument("-i", "--input", required=True, help="预测数据文件")
    predict_parser.add_argument("-m", "--model", required=True, choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"])
    predict_parser.add_argument("-o", "--output_dir", required=True, help="模型输出目录（包含训练好的模型）")
    predict_parser.add_argument("--task_type", required=True, choices=["classification", "regression"])

    args = parser.parse_args()
    if args.command == "train":
        sys.exit(run_single_model(
            input_path=args.input,
            model_type=args.model,
            output_dir=args.output_dir,
            feature_selection_mode=args.feature_selection_mode,
            task_type=args.task_type,
            n_folds=args.n_folds,
            random_state=args.random_state,
            gwas_genotype=args.gwas_genotype,
            gwas_pvalue=args.gwas_pvalue,
            ld_window_kb=args.ld_window_kb,
            ld_window=args.ld_window,
            ld_window_r2=args.ld_window_r2,
            ld_threads=args.ld_threads
        ))
    elif args.command == "train-all":
        sys.exit(run_all_models(
            input_path=args.input,
            output_dir=args.output_dir,
            feature_selection_mode=args.feature_selection_mode,
            task_type=args.task_type,
            n_folds=args.n_folds,
            random_state=args.random_state,
            gwas_genotype=args.gwas_genotype,
            gwas_pvalue=args.gwas_pvalue,
            ld_window_kb=args.ld_window_kb,
            ld_window=args.ld_window,
            ld_window_r2=args.ld_window_r2,
            ld_threads=args.ld_threads
        ))
    elif args.command == "predict":
        sys.exit(predict_with_model(
            input_path=args.input,
            model_type=args.model,
            output_dir=args.output_dir,
            task_type=args.task_type
        ))
    else:
        parser.print_help()
        sys.exit(1)