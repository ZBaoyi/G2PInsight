#!/usr/bin/env python3
"""
preprocess模块 - 分染色体多进程并行预处理
核心特性：
1. 解析.bim文件自动识别染色体（兼容所有PLINK 1.9版本）；
2. 多进程并行处理染色体（进程数≤用户指定值）；
3. PLINK编码使用--recodeA参数；
4. 内存优化+即时删临时文件，解决OOM Killed。
"""

import os
import sys
import json
import gc
import re
import time
import logging
import subprocess
import warnings
import pandas as pd
import math
from pathlib import Path
from typing import Optional, Dict, List, Literal, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

# 忽略警告信息
warnings.filterwarnings('ignore')
# 兼容性处理：BrokenProcessPool 在某些 Python 版本/发行版中可能不可用
# 使用 BrokenExecutor 作为替代，它是 BrokenProcessPool 的基类
try:
    from concurrent.futures import BrokenProcessPool
except ImportError:
    try:
        from concurrent.futures import BrokenExecutor as BrokenProcessPool
    except ImportError:
        # 如果都不可用，定义一个兼容的异常类
        class BrokenProcessPool(Exception):
            """兼容 BrokenProcessPool 异常"""
            pass

# ======================== 1. 基础配置 ========================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

# 尝试导入psutil用于内存监控（可选）
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil未安装，将无法进行内存监控。建议安装：pip install psutil（可选）")

# 进度条（可选）
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 尝试导入pyarrow用于Parquet格式（更高效的存储）
PARQUET_AVAILABLE = False
try:
    import pyarrow  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    # 不显示警告，因为这是可选的优化功能

# 内存优化配置
PLINK_MEM_LIMIT = 40000  # PLINK内存上限（MB）
SNP_DTYPE = pd.Int8Dtype()  # SNP用int8存储
CHUNK_SIZE = 1000  # 分块读取大小
MAX_SAFE_PARALLEL_PROCESSES = 2  # 安全的最大并行进程数（避免OOM，从4减少到2，更保守）
USE_DELAYED_MERGE = True  # 是否使用延迟合并策略（先保存到磁盘，最后合并）
USE_PARQUET = True  # 是否使用Parquet格式（如果可用），比pickle更高效

# ======================== 2. PLINK路径定位 ========================
def get_plink_path() -> str:
    """自动定位PLINK1.9可执行文件"""
    current_file = Path(__file__).absolute()
    software_dir = current_file.parent / "software"
    plink_candidates = [software_dir / "plink"]
    return str(plink_candidates[0])

PLINK_EXECUTABLE = get_plink_path()

# ======================== 2.5. 内存监控工具 ========================
def get_available_memory_gb() -> Optional[float]:
    """获取可用内存（GB），如果psutil不可用则返回None"""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except Exception:
        return None

def get_current_memory_usage_gb() -> Optional[float]:
    """获取当前进程内存使用（GB）"""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    except Exception:
        return None



def save_dataframe_optimized(df: pd.DataFrame, filepath: str, use_parquet: bool = True) -> str:
    """
    高级优化：智能选择存储格式
    - Parquet: 文件小10-50倍，加载快5-10倍，支持列式读取
    - Pickle: 备用方案，兼容性好但效率低
    """
    if use_parquet and PARQUET_AVAILABLE:
        # 使用Parquet格式（高级优化）
        parquet_path = filepath.replace('.pkl', '.parquet')
        try:
            df.to_parquet(
                parquet_path,
                compression='snappy',  # 快速压缩
                index=True,
                engine='pyarrow'
            )
            return parquet_path
        except Exception as e:
            logger.warning(f"Parquet保存失败，回退到pickle：{str(e)}")
            df.to_pickle(filepath)
            return filepath
    else:
        # 使用pickle格式（备用）
        df.to_pickle(filepath)
        return filepath

def load_dataframe_optimized(filepath: str) -> pd.DataFrame:
    """
    高级优化：智能加载，自动识别格式
    """
    # 尝试Parquet格式
    if filepath.endswith('.parquet') or Path(filepath.replace('.pkl', '.parquet')).exists():
        if PARQUET_AVAILABLE:
            parquet_path = filepath.replace('.pkl', '.parquet')
            try:
                return pd.read_parquet(parquet_path, engine='pyarrow')
            except Exception as e:
                logger.warning(f"Parquet加载失败，尝试pickle：{str(e)}")
    
    # 回退到pickle
    return pd.read_pickle(filepath)

def calculate_optimal_chunk_size(base_chunk_size: int, estimated_rows: Optional[int] = None, 
                                  estimated_cols: Optional[int] = None, force_conservative: bool = False) -> int:
    """
    根据可用内存动态计算最优的chunk_size
    :param base_chunk_size: 基础chunk_size
    :param estimated_rows: 预估的行数（样本数）
    :param estimated_cols: 预估的列数（SNP数）
    :param force_conservative: 强制使用保守策略（串行模式或内存受限时）
    :return: 优化后的chunk_size
    """
    if not PSUTIL_AVAILABLE:
        # 如果无法检测内存，使用非常保守的值
        return base_chunk_size if not force_conservative else max(500, base_chunk_size // 2)
    
    available_gb = get_available_memory_gb()
    if available_gb is None:
        return base_chunk_size if not force_conservative else max(500, base_chunk_size // 2)
    
    # 串行模式或强制保守：只使用可用内存的10-15%（非常保守）
    # 并行模式：使用30%
    available_for_chunk_ratio = 0.15 if force_conservative else 0.3
    available_for_chunk_gb = available_gb * available_for_chunk_ratio
    
    # 如果提供了预估维度，可以更精确计算
    if estimated_rows and estimated_cols:
        # 估算：每行每列约1字节（Int8），加上pandas开销约3-4倍（保守估计）
        bytes_per_cell = 4 if force_conservative else 3  # 保守估计
        estimated_memory_per_row_gb = (estimated_cols * bytes_per_cell) / (1024**3)
        if estimated_memory_per_row_gb > 0:
            max_rows_per_chunk = int(available_for_chunk_gb / estimated_memory_per_row_gb)
            # 取基础值和计算值的较小值，避免过大
            if force_conservative:
                # 串行模式：非常保守，最多5000行
                optimal = min(base_chunk_size * 2, max_rows_per_chunk, 5000)
                return max(500, optimal)  # 至少500行
            else:
                optimal = min(base_chunk_size * 10, max_rows_per_chunk, 50000)
                return max(1000, optimal)  # 至少1000行
    
    # 如果没有预估值，基于可用内存简单调整
    if force_conservative:
        # 串行模式：非常保守的策略
        if available_gb < 16:
            return max(500, base_chunk_size // 2)  # 内存紧张：使用很小的chunk
        elif available_gb < 32:
            return base_chunk_size  # 中等内存：使用基础chunk
        else:
            return base_chunk_size * 2  # 较大内存：适度增大
    else:
        # 并行模式：原有逻辑
        if available_gb < 16:
            return max(1000, base_chunk_size)
        elif available_gb < 32:
            return base_chunk_size * 3
        elif available_gb < 64:
            return base_chunk_size * 5
        else:
            return base_chunk_size * 10

# ======================== 3. 染色体名称统一化 ========================
def normalize_chromosome_names(plink_prefix: str) -> None:
    """
    统一化.bim文件中的染色体名称格式为chr1, chr2, ...格式
    如果染色体名称不统一，修改为统一格式
    """
    bim_file = f"{Path(plink_prefix).absolute().as_posix()}.bim"
    if not Path(bim_file).exists():
        raise FileNotFoundError(f"PLINK .bim文件不存在：{bim_file}")
    
    # 读取.bim文件
    bim_df = pd.read_csv(
        bim_file,
        sep="\s+",
        header=None,
        dtype={0: str},
        engine='python'
    )
    
    original_chr_col = bim_df[0].copy()
    
    # 统一化规则：转换为chr1, chr2, ...格式
    def normalize_chr(chr_name: str) -> str:
        """将染色体名称统一化为chr1格式"""
        chr_str = str(chr_name).strip()
        # 如果已经是chr开头，提取数字部分
        if chr_str.lower().startswith('chr'):
            chr_num = chr_str[3:].strip()
            # 如果提取后是数字，返回chr+数字
            if chr_num.isdigit():
                return f"chr{chr_num}"
            # 如果是X, Y, MT等，保留
            return f"chr{chr_num}"
        # 如果直接是数字，添加chr前缀
        elif chr_str.isdigit():
            return f"chr{chr_str}"
        # 如果是X, Y, MT等，添加chr前缀
        elif chr_str.upper() in ['X', 'Y', 'MT', 'M']:
            return f"chr{chr_str.upper()}"
        # 其他情况，尝试提取数字
        else:
            # 尝试提取数字部分
            import re
            match = re.search(r'\d+', chr_str)
            if match:
                return f"chr{match.group()}"
            # 无法处理，返回原值（但添加chr前缀）
            return f"chr{chr_str}"
    
    normalized_chr_col = original_chr_col.apply(normalize_chr)
    
    # 检查是否有变化
    if not normalized_chr_col.equals(original_chr_col):
        
        # 更新.bim文件
        bim_df[0] = normalized_chr_col
        
        # 保存更新后的.bim文件
        bim_df.to_csv(
            bim_file,
            sep="\t",
            header=False,
            index=False
        )
        
        logger.info(f"染色体名称已统一化")

# ======================== 3.5. 自动识别染色体（解析.bim文件） ========================
def auto_detect_chromosomes(plink_prefix: str) -> List[str]:
    """
    兼容所有PLINK版本：通过解析.bim文件自动识别染色体列表
    .bim文件格式：chr(第1列) snp_id genetic_dist physical_pos a1 a2
    """
    bim_file = f"{Path(plink_prefix).absolute().as_posix()}.bim"
    if not Path(bim_file).exists():
        raise FileNotFoundError(f"PLINK .bim文件不存在：{bim_file}")
    
    
    # 分块读取.bim文件，仅提取第一列（染色体）
    chr_series = pd.read_csv(
        bim_file,
        sep="\s+",
        header=None,
        usecols=[0],
        dtype={0: str},
        chunksize=100000,
        engine='python'
    )
    
    # 收集所有唯一的染色体编号
    chr_set = set()
    for chunk in chr_series:
        chr_set.update(chunk[0].unique())
    
    # 排序：数字在前（升序），非数字在后
    chr_list = list(chr_set)
    num_chrs = [c for c in chr_list if c.isdigit()]
    non_num_chrs = [c for c in chr_list if not c.isdigit()]
    num_chrs_sorted = sorted(num_chrs, key=int)
    chr_list_sorted = num_chrs_sorted + non_num_chrs
    
    return chr_list_sorted

# ======================== 4. SNP名称映射 ========================
def get_snp_chr_pos_mapping(bim_file: str, chr_num: str = None) -> dict:
    """
    生成SNP映射（支持指定染色体）
    返回两个映射：
    1. SNP ID -> 新名称（染色体_物理位置）
    2. PLINK列名格式（chr:pos_allele） -> 新名称（染色体_物理位置）
    """
    if not Path(bim_file).exists():
        raise FileNotFoundError(f".bim文件不存在：{bim_file}")

    snp_mapping = {}  # SNP ID -> 新名称
    plink_col_mapping = {}  # PLINK列名格式 -> 新名称
    
    for chunk in pd.read_csv(
        bim_file,
        sep="\s+",
        header=None,
        names=["chr", "snp_id", "genetic_dist", "physical_pos", "a1", "a2"],
        dtype={"chr": str, "snp_id": str, "physical_pos": str, "a1": str, "a2": str},
        chunksize=CHUNK_SIZE
    ):
        if chr_num is not None:
            chunk = chunk[chunk["chr"] == chr_num]
            if chunk.empty:
                continue
        
        # 生成新名称：染色体_物理位置
        chunk["new_snp_name"] = chunk["chr"] + "_" + chunk["physical_pos"].astype(str)
        
        # 映射1：SNP ID -> 新名称
        snp_mapping.update(dict(zip(chunk["snp_id"], chunk["new_snp_name"])))
        
        # 映射2：PLINK列名格式（chr:pos_allele） -> 新名称
        # PLINK的--recodeA生成的列名格式：chr:pos_a1 或 chr:pos_a2
        for idx, row in chunk.iterrows():
            chr_val = str(row["chr"])
            pos_val = str(row["physical_pos"])
            a1_val = str(row["a1"])
            a2_val = str(row["a2"])
            new_name = f"{chr_val}_{pos_val}"
            
            # PLINK可能使用chr:pos_a1或chr:pos_a2格式
            plink_col_mapping[f"{chr_val}:{pos_val}_{a1_val}"] = new_name
            plink_col_mapping[f"{chr_val}:{pos_val}_{a2_val}"] = new_name
            # 也可能使用其他格式，如chr:pos_T等
            plink_col_mapping[f"{chr_val}:{pos_val}"] = new_name

    
    # 返回合并的映射（优先使用PLINK列名映射）
    combined_mapping = {**snp_mapping, **plink_col_mapping}
    return combined_mapping

# ======================== 5. 通用临时文件删除 ========================
def delete_temp_files(prefix: str, exts: list) -> None:
    """批量删除临时文件，强制GC"""
    deleted_files = []
    for ext in exts:
        temp_file = f"{prefix}{ext}"
        if Path(temp_file).exists():
            os.remove(temp_file)
            deleted_files.append(temp_file)
    if deleted_files:
        pass
    gc.collect()

# ======================== 6. 基因型文件格式识别与校验 ========================
def detect_genotype_format(genotype_path: str) -> Literal["vcf", "plink_binary", "plink_text"]:
    """自动识别基因型文件格式"""
    genotype_path = Path(genotype_path).absolute().as_posix()

    if genotype_path.endswith((".vcf", ".vcf.gz")):
        if not Path(genotype_path).exists():
            raise FileNotFoundError(f"VCF文件不存在：{genotype_path}")
        return "vcf"

    plink_bin_files = [f"{genotype_path}.bed", f"{genotype_path}.bim", f"{genotype_path}.fam"]
    if all(Path(f).exists() for f in plink_bin_files):
        return "plink_binary"

    plink_text_files = [f"{genotype_path}.map", f"{genotype_path}.ped"]
    if all(Path(f).exists() for f in plink_text_files):
        return "plink_text"

    raise ValueError(
        f"\n无法识别基因型格式：{genotype_path}\n"
        f"支持格式：VCF(.vcf/.vcf.gz)、PLINK二进制(.bed/.bim/.fam)、PLINK文本(.map/.ped)"
    )

def validate_plink_files(genotype_path: str, fmt: str) -> None:
    """校验PLINK文件完整性"""
    required = {
        "plink_binary": [".bed", ".bim", ".fam"],
        "plink_text": [".map", ".ped"]
    }.get(fmt, [])

    missing = [f"{genotype_path}{ext}" for ext in required if not Path(f"{genotype_path}{ext}").exists()]
    if missing:
        raise FileNotFoundError(f"\nPLINK文件缺失：{', '.join(missing)}")

# ======================== 7. PLINK格式转换 ========================
def plink_text_to_binary(ped_map_prefix: str, output_prefix: str, threads: int = 6) -> str:
    """PLINK文本转二进制"""
    validate_plink_files(ped_map_prefix, "plink_text")
    
    # PLINK不接受threads=0，需要至少为1
    plink_threads = max(1, threads) if threads > 0 else 1

    plink_cmd = [
        PLINK_EXECUTABLE,
        "--file", ped_map_prefix,
        "--make-bed",
        "--out", output_prefix,
        "--threads", str(plink_threads),
        "--allow-extra-chr",
        "--keep-allele-order"
    ]

    logger.info("PLINK文本转二进制...")
    try:
        subprocess.run(plink_cmd, capture_output=True, text=True, check=True)
        delete_temp_files(ped_map_prefix, [".map", ".ped", ".log"])
        return output_prefix
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK执行失败：{e.stderr}")
        raise RuntimeError(f"PLINK文本转二进制失败: {e.stderr}")

def genotype_to_plink(genotype_path: str, output_prefix: str, threads: int = 6) -> str:
    """通用基因型转PLINK二进制"""
    fmt = detect_genotype_format(genotype_path)
    validate_plink_files(genotype_path, fmt)
    
    # PLINK不接受threads=0，需要至少为1
    plink_threads = max(1, threads) if threads > 0 else 1

    if fmt == "vcf":
        plink_cmd = [
            PLINK_EXECUTABLE,
            "--vcf", genotype_path,
            "--make-bed",
            "--out", output_prefix,
            "--threads", str(plink_threads),
            "--allow-extra-chr",
            "--set-missing-var-ids", "@:#",
            "--keep-allele-order"
        ]

        logger.info("VCF转PLINK...")
        try:
            subprocess.run(plink_cmd, capture_output=True, text=True, check=True)
            if "temp_plink" in output_prefix:
                delete_temp_files(genotype_path, [".vcf", ".vcf.gz", ".log"])
            return output_prefix
        except subprocess.CalledProcessError as e:
            logger.error(f"PLINK执行失败：{e.stderr}")
            raise RuntimeError(f"VCF转PLINK失败: {e.stderr}")

    elif fmt == "plink_binary":
        return genotype_path
    elif fmt == "plink_text":
        return plink_text_to_binary(genotype_path, output_prefix, threads)

# ======================== 8. 表型文件加载 ========================
def load_phenotype(pheno_path: str, pheno_col: Optional[str] = None) -> pd.DataFrame:
    """加载表型文件，检查异常值"""
    if pheno_col is not None:
        logger.warning(f"--pheno-col参数无效（表型文件无表头）")

    # 自动识别分隔符
    def detect_separator(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return '\t' if '\t' in first_line else ',' if ',' in first_line else '\s+'

    sep = detect_separator(pheno_path)
    sep_name = "制表符" if sep == '\t' else "逗号" if sep == ',' else "空格"

    # 读取表型（先以字符串读取，便于检查异常值）
    try:
        pheno_df = pd.read_csv(
            pheno_path,
            sep=sep,
            header=None,
            dtype={0: str, 1: str},  # 先读为字符串，便于检查异常值
            engine='python'
        )
    except Exception as e:
        raise RuntimeError(f"读取表型文件失败：{str(e)}")

    if pheno_df.shape[1] != 2:
        raise ValueError(f"\n表型文件列数错误（需2列），当前：{pheno_df.shape[1]}列")

    pheno_df.columns = ["sample", "phenotype"]
    
    # 检查表型值列是否存在异常值
    # 定义所有可能的异常值表示（不区分大小写）
    abnormal_values = ["na", "NA", "nan", "NAN", "NaN", "null", "NULL", "None", "NONE", 
                       "", ".", "-9", "-999", "missing", "MISSING", "NULL", "null"]
    
    # 检查异常值（转换为小写进行比较）
    phenotype_str = pheno_df["phenotype"].astype(str).str.strip().str.lower()
    abnormal_mask = phenotype_str.isin([v.lower() for v in abnormal_values]) | phenotype_str.isna()
    
    if abnormal_mask.any():
        abnormal_count = abnormal_mask.sum()
        abnormal_samples = pheno_df.loc[abnormal_mask, "sample"].tolist()[:10]  # 只显示前10个
        abnormal_pheno_values = pheno_df.loc[abnormal_mask, "phenotype"].tolist()[:10]
        
        error_msg = (
            f"\n表型文件存在异常值！\n"
            f"   - 异常值数量：{abnormal_count}\n"
            f"   - 异常值示例（前10个）：\n"
        )
        for i, (sample, pheno_val) in enumerate(zip(abnormal_samples, abnormal_pheno_values), 1):
            error_msg += f"      {i}. 样本ID: {sample}, 表型值: {pheno_val}\n"
        if abnormal_count > 10:
            error_msg += f"      ... 还有 {abnormal_count - 10} 个异常值\n"
        error_msg += (
            f"\n   请检查表型文件，确保表型值列不包含以下异常值：\n"
            f"   {', '.join(abnormal_values)}\n"
            f"   以及空值、缺失值等。\n"
        )
        logger.error(error_msg)
        sys.exit(1)
    
    # 数据清洗：去重
    pheno_df = pheno_df.drop_duplicates(subset="sample", keep="last")
    
    # 检查样本ID列
    pheno_df = pheno_df[pheno_df["sample"].notna() & (pheno_df["sample"] != "")].reset_index(drop=True)

    if pheno_df.empty:
        raise ValueError("表型文件无有效数据（全为缺失值/空ID）")
    
    # 将表型值转换为数值类型
    try:
        pheno_df["phenotype"] = pd.to_numeric(pheno_df["phenotype"], errors='raise')
    except ValueError as e:
        raise ValueError(f"表型值无法转换为数值类型：{str(e)}\n请确保表型值列包含有效的数值。")

    logger.info(f"表型数据加载完成: {len(pheno_df)}个样本")
    gc.collect()
    return pheno_df

# ======================== 8.5. 表型类型判断和可视化 ========================
def determine_phenotype_type(pheno_df: pd.DataFrame, output_dir: Path) -> Literal["regression", "classification"]:
    """
    自动判断表型值类型（分类或回归）
    并绘制相应的可视化图表
    
    Returns:
        "regression" 或 "classification"
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import numpy as np
        # 设置字体
        try:
            from assoG2P.bin.font_utils import setup_matplotlib_font
            setup_matplotlib_font()
        except ImportError:
            pass
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        logger.warning("matplotlib未安装，跳过表型可视化")
    
    phenotype_values = pheno_df["phenotype"].values
    
    # 判断逻辑：
    # 1. 如果唯一值数量 <= 10，且都是整数，可能是分类任务
    # 2. 如果唯一值数量 > 10，或包含小数，可能是回归任务
    unique_values = np.unique(phenotype_values)
    unique_count = len(unique_values)
    
    # 检查是否都是整数
    is_all_integer = np.all(phenotype_values == np.round(phenotype_values))
    
    # 判断规则：
    # - 唯一值 <= 10 且都是整数 -> 分类
    # - 唯一值 > 10 或包含小数 -> 回归
    if unique_count <= 10 and is_all_integer:
        task_type = "classification"
        logger.info(f"表型类型: 分类（{unique_count}个类别）")
    else:
        task_type = "regression"
        logger.info(f"表型类型: 回归")
    
    # 绘制可视化图表
    if MATPLOTLIB_AVAILABLE:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if task_type == "classification":
            # 分类任务：绘制饼图
            value_counts = pd.Series(phenotype_values).value_counts().sort_index()
            
            plt.figure(figsize=(10, 6))
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(f"Phenotype Distribution (Classification)\nUnique Values: {unique_count}", fontsize=14)
            plt.axis('equal')
            
            plot_file = output_dir / "phenotype_distribution_pie.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # 回归任务：绘制密度分布直方图
            plt.figure(figsize=(10, 6))
            plt.hist(phenotype_values, bins=50, density=True, alpha=0.7, edgecolor='black')
            plt.xlabel("Phenotype Value", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.title(f"Phenotype Distribution (Regression)\nSamples: {len(phenotype_values)}, Unique Values: {unique_count}", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(phenotype_values)
            std_val = np.std(phenotype_values)
            plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1SD: {std_val:.2f}')
            plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
            plt.legend()
            
            plot_file = output_dir / "phenotype_distribution_histogram.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
    
    return task_type

# ======================== 9. SNP质量过滤 ========================
def filter_snps_by_quality(
    plink_prefix: str, 
    output_prefix: str,
    maf: float = 0.05,
    geno: float = 0.2,
    threads: int = 6
) -> str:
    """
    全局SNP质量过滤（MAF和缺失率）
    在分染色体处理前先过滤，可以显著减少后续处理的SNP数量
    :param plink_prefix: 输入PLINK文件前缀
    :param output_prefix: 输出PLINK文件前缀
    :param maf: 最小等位基因频率阈值
    :param geno: 最大缺失率阈值
    :param threads: PLINK线程数
    :return: 过滤后的PLINK文件前缀
    """
    output_prefix_abs = Path(output_prefix).absolute().as_posix()
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    
    filter_cmd = [
        PLINK_EXECUTABLE,
        "--bfile", plink_prefix_abs,
        "--maf", str(maf),
        "--geno", str(geno),
        "--make-bed",
        "--out", output_prefix_abs,
        "--allow-extra-chr",
        "--allow-no-sex",
        "--threads", str(max(1, min(threads, os.cpu_count() or 1, 8) if threads > 0 else 1))  # 全局过滤可以使用更多线程（最多8），但至少为1
    ]
    
    logger.info(f"SNP质量过滤（MAF≥{maf}, 缺失率≤{geno}）")
    
    try:
        result = subprocess.run(
            filter_cmd, 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "未知错误"
            logger.error(f"SNP质量过滤失败：{error_msg[:500]}")
            raise RuntimeError(f"SNP质量过滤失败: {error_msg[:500]}")
        
        # 统计过滤后的SNP数量
        bim_file = f"{output_prefix_abs}.bim"
        if Path(bim_file).exists():
            snp_count = sum(1 for _ in open(bim_file))
            logger.info(f"SNP质量过滤完成: {snp_count:,}个SNP")
        else:
            logger.warning("无法统计过滤后的SNP数量")
        
        # 清理临时文件
        if "temp_plink" in plink_prefix:
            delete_temp_files(plink_prefix, [".bed", ".bim", ".fam", ".log", ".nosex"])
        
        return output_prefix_abs
    except subprocess.TimeoutExpired:
        logger.error(f"SNP质量过滤超时（>2小时）")
        raise RuntimeError("SNP质量过滤超时")
    except Exception as e:
        logger.error(f"SNP质量过滤异常：{str(e)}")
        raise

# ======================== 10. 样本过滤 ========================
def filter_samples_by_phenotype(plink_prefix: str, pheno_df: pd.DataFrame, output_prefix: str) -> str:
    """按表型ID过滤PLINK样本"""
    output_prefix_abs = Path(output_prefix).absolute().as_posix()
    keep_file = f"{output_prefix_abs}_keep.txt"

    # 生成keep文件
    pheno_df[["sample", "sample"]].to_csv(keep_file, sep="\t", index=False, header=False)

    filter_cmd = [
        PLINK_EXECUTABLE,
        "--bfile", Path(plink_prefix).absolute().as_posix(),
        "--keep", keep_file,
        "--make-bed",
        "--out", output_prefix_abs,
        "--allow-extra-chr",
        "--allow-no-sex",
        "--threads", str(min(os.cpu_count(), 4))
    ]

    logger.info("过滤PLINK样本")
    try:
        subprocess.run(filter_cmd, capture_output=True, text=True, check=True)
        delete_temp_files(output_prefix_abs, ["_keep.txt"])
        if "temp_plink" in plink_prefix:
            delete_temp_files(plink_prefix, [".bed", ".bim", ".fam", ".log", ".nosex"])
        logger.info(f"样本过滤完成")
        return output_prefix_abs
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK样本过滤失败：{e.stderr}")
        raise RuntimeError(f"样本过滤失败: {e.stderr}")

# ======================== 10. 单染色体处理函数（供多进程调用） ========================
def process_single_chromosome(
    chr_num: str,
    plink_prefix_abs: str,
    pheno_file: str,  # 改为文件路径，避免序列化DataFrame
    plink_executable: str,
    plink_mem_limit: int,
    chunk_size: int = CHUNK_SIZE,
    snp_dtype: str = "Int8",
    plink_threads: int = 6,  # 新增参数：PLINK线程数
    force_conservative: bool = False  # 强制使用保守内存策略
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    单染色体处理函数（多进程核心）
    :return: (染色体编号, 处理后的DataFrame/None)
    """
    # 多进程环境下重新配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True
    )
    logger = logging.getLogger(__name__)
    
    try:
        
        # 在子进程中加载表型数据（避免序列化问题）
        try:
            pheno_df = pd.read_csv(
                pheno_file, 
                sep="\t", 
                index_col=0,
                dtype={"phenotype": float}
            )
            # 确保列名正确（如果文件只有一列，重命名为phenotype）
            if len(pheno_df.columns) == 1:
                pheno_df.columns = ["phenotype"]
            if "phenotype" not in pheno_df.columns:
                raise ValueError(f"表型文件缺少'phenotype'列，当前列：{pheno_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"{chr_num}号染色体：加载表型文件失败：{str(e)}")
            return (chr_num, None)
        
        chr_recode_prefix = f"{plink_prefix_abs}_chr{chr_num}"
        
        # PLINK命令：--recodeA + 指定染色体
        # 注意：MAF和缺失率过滤已在全局过滤步骤完成，此处不再重复过滤以加快速度
        # 使用传入的plink_threads参数（由主进程根据并行度计算，避免OOM）
        recode_cmd = [
            plink_executable,
            "--bfile", plink_prefix_abs,
            "--chr", chr_num,
            "--recodeA",
            "--out", chr_recode_prefix,
            "--allow-extra-chr",
            "--allow-no-sex",
            "--keep-allele-order",
            "--threads", str(plink_threads)  # 使用计算出的线程数
        ]
        
        # 内存优化：如果PLINK支持内存限制，添加--memory参数
        # PLINK 1.9支持--memory参数（单位：MB），但某些版本可能不支持
        if plink_mem_limit > 0:
            # 尝试添加内存限制（如果PLINK版本支持）
            # 注意：某些PLINK版本可能不支持此参数，如果失败会回退
            recode_cmd.extend(["--memory", str(plink_mem_limit)])
        
        # 在子进程中记录内存使用情况
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024**3)  # GB
            except Exception:
                pass
        
        # 执行PLINK命令，增强错误处理
        try:
            # 性能优化：增加超时时间，因为大数据集处理时间可能很长
            
            # 记录PLINK执行前的内存
            mem_before_plink = None
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process(os.getpid())
                    mem_before_plink = process.memory_info().rss / (1024**3)  # GB
                except Exception:
                    pass
            
            result = subprocess.run(
                recode_cmd, 
                capture_output=True, 
                text=True, 
                check=False,
                timeout=14400  # 4小时超时（从1小时增加到4小时）
            )
            
            # 记录PLINK执行后的内存
            if PSUTIL_AVAILABLE and mem_before_plink is not None:
                try:
                    process = psutil.Process(os.getpid())
                    mem_after_plink = process.memory_info().rss / (1024**3)  # GB
                except Exception:
                    pass
            
            # 内存优化：PLINK执行后立即清理，释放内存
            # 注意：PLINK进程已结束，但可能还有残留内存
            gc.collect()
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "未知错误"
                # 检查是否是内存相关错误
                if "memory" in error_msg.lower() or "oom" in error_msg.lower() or "killed" in error_msg.lower():
                    logger.error(f"{chr_num}号染色体：PLINK执行失败，疑似内存不足（OOM）")
                    logger.error(f"错误信息：{error_msg[:1000]}")
                else:
                    logger.error(f"{chr_num}号染色体：PLINK执行失败（返回码{result.returncode}）：{error_msg[:500]}")
                return (chr_num, None)
        except subprocess.TimeoutExpired:
            logger.error(f"{chr_num}号染色体：PLINK执行超时（>4小时），可能需要检查数据量或系统性能")
            return (chr_num, None)
        except Exception as e:
            logger.error(f"{chr_num}号染色体：PLINK执行异常：{str(e)}")
            return (chr_num, None)
        
        # 校验.raw文件
        chr_geno_file = f"{chr_recode_prefix}.raw"
        if not Path(chr_geno_file).exists():
            raise FileNotFoundError(f"{chr_num}号染色体.raw文件未生成")
        
        # 加载当前染色体的SNP映射（性能优化：在读取.raw文件前完成）
        chr_bim_file = f"{plink_prefix_abs}.bim"
        chr_snp_mapping = get_snp_chr_pos_mapping(chr_bim_file, chr_num=chr_num)
        
        # 分块读取.raw文件（内存优化：流式处理，边读边处理边合并）
        # 优化：根据可用内存动态调整chunk_size
        # 优化：使用流式处理避免累积所有chunks在内存中
        estimated_snp_count = len(chr_snp_mapping)
        
        # 内存优化：检查内存压力，决定是否使用保守策略
        available_gb = get_available_memory_gb()
        force_conservative = False
        if available_gb is not None:
            # 如果可用内存<8GB，使用非常保守的策略
            if available_gb < 8.0:
                force_conservative = True
                logger.warning(f"可用内存较低({available_gb:.2f}GB)，启用保守内存策略")
        
        optimized_chunk_size = calculate_optimal_chunk_size(
            chunk_size, 
            estimated_cols=estimated_snp_count,
            force_conservative=force_conservative
        )
        
        # 内存优化：使用流式处理，边读边合并，避免累积所有chunks
        # 优化：使用列表累积chunks，定期合并，比频繁concat更高效且内存友好
        chr_geno_chunks_list = []  # 累积处理后的chunks
        chr_geno_df = None  # 主DataFrame
        chunk_count = 0
        dtype_obj = pd.Int8Dtype() if snp_dtype == "Int8" else pd.Int8Dtype()
        # 内存优化：根据内存情况调整合并频率
        if force_conservative:
            max_chunks_before_merge = 3  # 内存紧张时，每3个chunk就合并
        else:
            max_chunks_before_merge = 10  # 正常情况，累积10个chunk后合并
        
        def process_chunk(chunk):
            """处理单个chunk：重命名、清理、类型转换"""
            # SNP列名重命名
            # PLINK的--recodeA生成的列名格式可能是：chr:pos_allele（如1:39206_T）
            # 需要转换为：chr_pos（如1_39206）
            new_columns = []
            for col in chunk.columns:
                if col in chr_snp_mapping:
                    # 如果列名在映射中，直接使用
                    new_columns.append(chr_snp_mapping[col])
                elif ':' in str(col) and '_' in str(col):
                    # 处理PLINK格式：chr:pos_allele -> chr_pos
                    # 例如：1:39206_T -> 1_39206
                    try:
                        parts = str(col).split(':')
                        if len(parts) == 2:
                            chr_part = parts[0]
                            pos_allele = parts[1]
                            # 提取位置部分（去掉等位基因后缀）
                            pos_part = pos_allele.split('_')[0]
                            new_name = f"{chr_part}_{pos_part}"
                            new_columns.append(new_name)
                        else:
                            new_columns.append(col)
                    except:
                        new_columns.append(col)
                else:
                    # 其他列名保持不变
                    new_columns.append(col)
            chunk.columns = new_columns
            
            # 清理列
            if "IID" in chunk.columns:
                chunk.rename(columns={"IID": "sample"}, inplace=True)
            drop_cols = ["FID", "PAT", "MAT", "SEX", "PHENOTYPE"]
            chunk = chunk.drop(columns=[col for col in drop_cols if col in chunk.columns])
            
            # 内存优化：在读取时就转换数据类型，使用更高效的方法
            snp_cols = [col for col in chunk.columns if col != "sample"]
            if snp_cols:
                # 优化：分批转换SNP列，根据内存情况调整批次大小
                # 内存紧张时使用更小的批次
                available_gb = get_available_memory_gb()
                if available_gb is not None and available_gb < 8.0:
                    batch_size = min(500, len(snp_cols))  # 内存紧张：小批次
                else:
                    batch_size = min(2000, len(snp_cols))  # 正常：较大批次
                
                for i in range(0, len(snp_cols), batch_size):
                    batch_cols = snp_cols[i:i+batch_size]
                    # 优化：逐列转换，使用pd.to_numeric直接转换，然后astype
                    # 这种方法比apply更高效，内存占用更少
                    for col in batch_cols:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype(dtype_obj)
                    
                    # 内存优化：每转换一批就检查内存，必要时GC
                    if available_gb is not None and available_gb < 4.0 and i % (batch_size * 2) == 0:
                        gc.collect()
            
            return chunk
        
        try:
            # 尝试使用C引擎（更快）
            for idx, chunk in enumerate(pd.read_csv(
                chr_geno_file,
                sep=r'\s+',  # 使用正则表达式分隔符
                dtype=str,
                engine='c',  # 使用C引擎，速度提升10-100倍
                chunksize=optimized_chunk_size
            )):
                chunk = process_chunk(chunk)
                chunk_count += 1
                
                # 内存优化：累积chunks，定期合并以减少concat次数
                chr_geno_chunks_list.append(chunk)
                
                # 每累积一定数量的chunks就合并一次，释放内存
                if len(chr_geno_chunks_list) >= max_chunks_before_merge:
                    if chr_geno_df is None:
                        chr_geno_df = pd.concat(chr_geno_chunks_list, ignore_index=True)
                    else:
                        # 合并累积的chunks到主DataFrame
                        chr_geno_df = pd.concat([chr_geno_df] + chr_geno_chunks_list, ignore_index=True)
                    # 清空列表并释放内存
                    chr_geno_chunks_list.clear()
                    gc.collect()  # 立即GC释放中间DataFrame
                    
                    # 内存优化：如果内存紧张，每合并一次就检查并强制GC
                    if force_conservative:
                        available_gb = get_available_memory_gb()
                        if available_gb is not None and available_gb < 4.0:
                            gc.collect()  # 再次强制GC
                            logger.warning(f"内存紧张({available_gb:.2f}GB)，已强制GC")
                
                # 注意：不再需要del chunk，因为chunk已添加到列表中
                
        except Exception as e:
            # 如果C引擎失败（例如分隔符问题），回退到Python引擎
            logger.warning(f"C引擎读取失败，回退到Python引擎：{str(e)}")
            for idx, chunk in enumerate(pd.read_csv(
                chr_geno_file,
                sep="\s+",
                dtype=str,
                engine='python',
                chunksize=optimized_chunk_size
            )):
                chunk = process_chunk(chunk)
                chunk_count += 1
                
                # 内存优化：累积chunks，定期合并
                chr_geno_chunks_list.append(chunk)
                
                if len(chr_geno_chunks_list) >= max_chunks_before_merge:
                    if chr_geno_df is None:
                        chr_geno_df = pd.concat(chr_geno_chunks_list, ignore_index=True)
                    else:
                        chr_geno_df = pd.concat([chr_geno_df] + chr_geno_chunks_list, ignore_index=True)
                    chr_geno_chunks_list.clear()
                    gc.collect()
        
        # 合并剩余的chunks
        if chr_geno_chunks_list:
            if chr_geno_df is None:
                chr_geno_df = pd.concat(chr_geno_chunks_list, ignore_index=True)
            else:
                chr_geno_df = pd.concat([chr_geno_df] + chr_geno_chunks_list, ignore_index=True)
            chr_geno_chunks_list.clear()
            gc.collect()
        
        if chr_geno_df is None or chr_geno_df.empty:
            logger.warning(f"{chr_num}号染色体：未读取到有效数据")
            return (chr_num, None)
        
        # 注意：数据类型转换已在process_chunk中完成，无需再次转换
        
        # 内存优化：立即释放不再使用的变量
        del chr_snp_mapping, chr_geno_chunks_list
        gc.collect()
        
        # 即时删除当前染色体临时文件（在处理数据前删除，释放磁盘和内存）
        delete_temp_files(chr_recode_prefix, [".raw", ".log", ".nosex", ".ped", ".map"])
        gc.collect()
        
        # 合并表型
        chr_geno_df.set_index("sample", inplace=True)
        chr_geno_df = chr_geno_df.fillna(-1)
        # pheno_df 已经以 sample 为索引（从文件加载时已设置 index_col=0）
        chr_geno_df = chr_geno_df.merge(
            pheno_df,
            left_index=True,
            right_index=True,
            how="inner"
        )
        
        # 最终内存清理
        del pheno_df
        gc.collect()
        
        # 计算SNP数量（排除sample列）
        # snp_count = len([col for col in chr_geno_df.columns if col != "sample"])
        # 按用户需求，分染色体处理的详细日志不再输出，这里仅返回结果
        return (chr_num, chr_geno_df)
    
    except Exception as e:
        logger.error(f"{chr_num}号染色体处理失败：{str(e)}", exc_info=True)
        return (chr_num, None)

# ======================== 11. 多进程分染色体生成训练数据 ========================
def plink_to_training_data(
    plink_prefix: str,
    pheno_df: pd.DataFrame,
    output_file: str,
    threads: int = 6,
    tmp_dir: Optional[Union[str, Path]] = None,
    cleanup_registry: Optional[List[str]] = None  # 记录当前运行生成的临时文件，便于异常时清理
) -> Tuple[pd.DataFrame, str]:
    """
    多进程并行处理染色体：
    1. 自动识别染色体列表；
    2. 按进程数并行处理（所有染色体同时处理）；
    3. 合并所有染色体结果。
    """
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    output_file_path = Path(output_file).absolute()
    output_file_abs = output_file_path.as_posix()

    # 统一临时目录：除最终输出外的中间文件全部写入 tmp_p 目录（preprocess专用）
    if tmp_dir is None:
        tmp_base = output_file_path.parent / "tmp_p"
    else:
        tmp_base = Path(tmp_dir).absolute()
    tmp_base.mkdir(parents=True, exist_ok=True)
    
    # 步骤1：自动识别染色体列表
    chr_list = auto_detect_chromosomes(plink_prefix_abs)
    if not chr_list:
        raise RuntimeError("未识别到任何染色体")
    
    # 步骤2：将pheno_df保存到临时文件（避免多进程序列化问题）
    # 放在统一的tmp目录下
    temp_pheno_file = str(tmp_base / "temp_pheno.tsv")
    try:
        pheno_df.set_index("sample").to_csv(temp_pheno_file, sep="\t")
        if cleanup_registry is not None:
            cleanup_registry.append(temp_pheno_file)
    except Exception as e:
        raise RuntimeError(f"保存表型临时文件失败：{str(e)}")
    
    # 步骤3：按“每个进程处理两条染色体”分配进程数
    chr_count = len(chr_list)
    base_processes = max(1, math.ceil(chr_count / 2))
    # 用户未指定（threads<=0）时，完全根据染色体数自动分配进程数；否则作为上限
    parallel_processes = base_processes if threads <= 0 else min(base_processes, threads)
    available_mem_gb = get_available_memory_gb() or 32  # 默认假设32GB
    
    # 计算每个进程的PLINK线程数，目标：总线程数 ≈ 染色体数，且不超过CPU核数
    available_cpus = os.cpu_count() or 1
    if threads > 0:
        # 手动设置时，总线程数不超过threads和CPU核数
        total_thread_cap = min(threads, available_cpus)
    else:
        # 自动模式：总线程数不超过“染色体数”和CPU核数
        total_thread_cap = min(chr_count, available_cpus)
    # 至少保证每个进程有1个线程
    if total_thread_cap < parallel_processes:
        total_thread_cap = parallel_processes
    plink_threads_per_process = max(1, total_thread_cap // parallel_processes)
    
    
    # 步骤4：处理染色体（串行或并行模式）
    chr_result_files = []  # 存储每个染色体的结果文件路径
    all_chr_dfs = []  # 如果不用延迟合并，直接存储DataFrame
    
    # 并行处理模式（默认）
    logger.info(f"并行处理模式: {parallel_processes}进程, {len(chr_list)}个染色体")

    # 并行模式进度条：总进度为所有染色体数，优化显示格式
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=len(chr_list),
            desc="处理染色体",
            unit="条",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=120,
            mininterval=0.5,
            maxinterval=2.0,
            smoothing=0.1
        )
    else:
        pbar = None

    try:
        with ProcessPoolExecutor(max_workers=parallel_processes) as executor:
            # 分批提交任务：每次最多提交 parallel_processes 个任务
            future_to_chr = {}
            completed_count = 0
            
            # 分批处理：将染色体列表分成批次
            for batch_start in range(0, len(chr_list), parallel_processes):
                batch_chrs = chr_list[batch_start:batch_start + parallel_processes]
                
                # 提交当前批次的任务（内存优化：添加启动延迟，避免同时启动导致OOM）
                batch_futures = {}
                for idx, chr_num in enumerate(batch_chrs):
                    # 内存优化：错开进程启动时间，避免同时启动导致峰值内存过高
                    if idx > 0:
                        # 每个进程延迟启动，给前一个进程时间初始化
                        delay_seconds = 2 * idx  # 递增延迟：0s, 2s, 4s, ...
                        time.sleep(delay_seconds)
                    
                    # 并行模式：根据内存情况决定是否使用保守策略
                    # 如果内存<16GB，使用保守策略
                    use_conservative = available_mem_gb is not None and available_mem_gb < 16
                    
                    future = executor.submit(
                        process_single_chromosome,
                        chr_num,
                        plink_prefix_abs,
                        temp_pheno_file,  # 传递文件路径而非DataFrame
                        PLINK_EXECUTABLE,
                        PLINK_MEM_LIMIT,
                        CHUNK_SIZE,
                        str(SNP_DTYPE),
                        plink_threads_per_process,  # 传递PLINK线程数
                        force_conservative=use_conservative  # 根据内存情况决定
                    )
                    batch_futures[future] = chr_num
                    future_to_chr[future] = chr_num
                
                # 等待当前批次完成
                try:
                    for future in as_completed(batch_futures):
                        chr_num = batch_futures[future]
                        # 更新进度条，显示正在处理的染色体
                        if pbar is not None:
                            pbar.set_postfix_str(f"处理Chr{chr_num}")
                        try:
                            # 性能优化：增加超时时间，因为大数据集处理时间可能很长
                            _, chr_df = future.result(timeout=28800)  # 8小时超时（从2小时增加到8小时）
                            if chr_df is not None and not chr_df.empty:
                                chr_result_file_base = f"{output_file_abs}_chr{chr_num}_temp"
                                chr_result_file = save_dataframe_optimized(
                                    chr_df,
                                    f"{chr_result_file_base}.pkl",
                                    use_parquet=USE_PARQUET
                                )
                                chr_result_files.append((chr_num, chr_result_file))
                            if cleanup_registry is not None:
                                cleanup_registry.append(chr_result_file)
                                del chr_df  # 立即释放内存
                            else:
                                logger.warning(f"{chr_num}号染色体无有效数据，跳过")

                            # 更新进度条（如果可用）
                            if pbar is not None:
                                pbar.set_postfix_str(f"Chr{chr_num}完成")
                                pbar.update(1)
                            completed_count += 1
                        except BrokenProcessPool as e:
                            # 进程池崩溃，可能是OOM或其他严重错误
                            logger.error(
                                f"{chr_num}号染色体进程崩溃："
                                f"可能是内存不足（OOM）或其他严重错误。"
                                f"建议：减少--threads参数或增加SLURM内存限制"
                            )
                            # 更新进度条，显示失败状态
                            if pbar is not None:
                                pbar.set_postfix_str(f"Chr{chr_num}崩溃")
                                pbar.update(1)
                            completed_count += 1
                            # 尝试获取子进程的实际异常（如果有）
                            try:
                                if hasattr(e, '_exception') and e._exception:
                                    logger.error(f"子进程异常详情：{str(e._exception)}")
                            except:
                                pass
                            # 进程池已崩溃，无法继续使用，需要退出循环
                            logger.error("进程池已崩溃，无法继续处理后续任务")
                            break
                        except Exception as e:
                            logger.error(f"{chr_num}号染色体任务异常：{str(e)}", exc_info=True)
                            # 更新进度条（即使失败也更新，保持进度准确）
                            if pbar is not None:
                                pbar.set_postfix_str(f"Chr{chr_num}异常")
                                pbar.update(1)
                            completed_count += 1
                except BrokenProcessPool as e:
                    # 进程池在提交任务时就崩溃了
                    logger.error(
                        f"进程池在提交任务时崩溃，可能是内存不足（OOM）。"
                        f"当前批次：{batch_chrs}，已处理：{completed_count}/{len(chr_list)}"
                    )
                    logger.error(f"建议：1) 减少--threads参数（当前：{threads}）")
                    logger.error(f"      2) 增加SLURM作业内存限制")
                    logger.error(f"      3) 检查数据文件大小和系统资源")
                    raise RuntimeError("进程池崩溃，无法继续处理") from e
                
                # 批次间强制GC，释放内存
                gc.collect()
    finally:
        # 关闭进度条
        if pbar is not None:
            pbar.close()
        # 清理临时文件（并行模式）
        if Path(temp_pheno_file).exists():
            os.remove(temp_pheno_file)
            logger.info(f"已删除临时表型文件：{temp_pheno_file}")
    
    # 步骤5：合并所有染色体结果
    if USE_DELAYED_MERGE:
        # 延迟合并策略：从磁盘加载并合并
        if not chr_result_files:
            raise RuntimeError("所有染色体均无有效数据，处理失败")
        
        logger.info(f"合并{len(chr_result_files)}个染色体的结果...")
        
        # 内存优化：分批加载和合并，使用列表累积减少concat次数
        final_train_df = None
        temp_files_to_cleanup = []
        merge_batch_size = 3  # 初始合并批次大小
        chr_dfs_batch = []  # 累积DataFrame列表，减少concat调用
        
        try:
            for idx, (chr_num, chr_file) in enumerate(chr_result_files):
                # 高级优化：智能加载，自动识别Parquet或pickle格式
                chr_df = load_dataframe_optimized(chr_file)
                
                # 内存优化：累积多个DataFrame后一次性合并，减少concat开销
                if final_train_df is None:
                    # 第一个DataFrame直接使用
                    final_train_df = chr_df
                else:
                    # 累积到批次列表
                    chr_dfs_batch.append(chr_df)
                
                # 立即释放原始DataFrame引用（列表中的引用仍然有效）
                del chr_df
                temp_files_to_cleanup.append(chr_file)
                
                # 检查内存压力并决定是否合并
                available_gb = get_available_memory_gb()
                should_merge = False
                
                if available_gb is not None:
                    if available_gb < 4.0:
                        # 内存紧张：立即合并
                        merge_batch_size = 1
                        should_merge = len(chr_dfs_batch) >= 1
                    elif available_gb < 8.0:
                        merge_batch_size = 1
                        should_merge = len(chr_dfs_batch) >= 1
                    elif available_gb < 16.0:
                        merge_batch_size = 2
                        should_merge = len(chr_dfs_batch) >= 2
                    else:
                        merge_batch_size = 3
                        should_merge = len(chr_dfs_batch) >= 3
                else:
                    # 无法检测内存，使用默认策略
                    should_merge = len(chr_dfs_batch) >= merge_batch_size
                
                # 执行批量合并
                if should_merge and chr_dfs_batch:
                    # 一次性合并所有累积的DataFrame，比逐个合并更高效
                    final_train_df = pd.concat([final_train_df] + chr_dfs_batch, axis=1, join="inner")
                    chr_dfs_batch.clear()  # 清空列表
                    gc.collect()  # 立即GC释放中间DataFrame
            
            # 合并剩余的DataFrame
            if chr_dfs_batch:
                final_train_df = pd.concat([final_train_df] + chr_dfs_batch, axis=1, join="inner")
                chr_dfs_batch.clear()
                gc.collect()
        finally:
            # 清理临时文件
            for temp_file in temp_files_to_cleanup:
                if Path(temp_file).exists():
                    os.remove(temp_file)
    else:
        # 直接合并策略：从内存合并
        if not all_chr_dfs:
            raise RuntimeError("所有染色体均无有效数据，处理失败")
        
        logger.info(f"合并{len(all_chr_dfs)}个染色体的结果...")
        final_train_df = pd.concat(all_chr_dfs, axis=1, join="inner")
        del all_chr_dfs
        gc.collect()
    
    # 保存最终训练数据（支持压缩，添加.txt后缀）
    # 如果输出文件没有.txt后缀，添加它
    if not output_file_abs.endswith('.txt') and not output_file_abs.endswith('.txt.gz'):
        if output_file_abs.endswith('.gz'):
            output_file_abs = output_file_abs.replace('.gz', '.txt.gz')
        else:
            output_file_abs = output_file_abs + '.txt'
    
    logger.info(f"保存训练数据")
    compression = "gzip" if output_file_abs.endswith(".gz") else None
    final_train_df.to_csv(
        output_file_abs,
        sep="\t",
        index=True,
        compression=compression
    )
    logger.info(f"训练数据保存完成: {final_train_df.shape[0]}样本, {final_train_df.shape[1]-1}特征")
    
    # 最终GC回收
    gc.collect()
    
    # 返回DataFrame和实际保存的文件路径
    return final_train_df, output_file_abs

# ======================== 12. 生成元数据 ========================
def generate_metadata(
    output_prefix: str,
    plink_prefix: str,
    train_file: str,
    valid_samples: List[str],
    genotype_format: str,
    threads: int,
    task_type: Optional[Literal["regression", "classification"]] = None,
    preprocess_tmp_dir: Optional[Union[str, Path]] = None
) -> None:
    """生成元数据文件"""
    chr_list = auto_detect_chromosomes(plink_prefix)
    
    metadata = {
        "preprocess_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "genotype_format": genotype_format,
        "phenotype_format": "无表头2列（样本ID+表型值）",
        "output_train_file": train_file,
        "valid_samples": valid_samples,
        "sample_count": len(valid_samples),
        "chromosome_list": chr_list,
        "chromosome_count": len(chr_list),
        "parallel_processes": min(threads, len(chr_list)),
        "plink_memory_limit_MB": "unlimited (no --memory passed to PLINK)",
        "snp_dtype": str(SNP_DTYPE),
        "snp_naming_rule": "染色体_物理位置（如'1_123456'）",
        "plink_executable": PLINK_EXECUTABLE,
        "plink_recode_param": "--recodeA",
        "gwas_genotype_prefix": str(Path(plink_prefix).absolute()),  # 保存过滤后的PLINK二进制文件前缀（绝对路径），供GWAS/LD使用
        "preprocess_tmp_dir": str(Path(preprocess_tmp_dir).absolute()) if preprocess_tmp_dir else None
    }
    
    # 添加表型值类别（如果提供）
    if task_type:
        metadata["task_type"] = task_type
    

    metadata_file = f"{output_prefix}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"元数据保存完成")

# ======================== 13. 主函数 ========================
def run_preprocess(
    genotype_file: str,
    phenotype_file: str,
    output_file: str,
    threads: int = 6,
    pheno_col: Optional[str] = None
) -> int:
    """预处理主函数"""
    # 当前运行生成的临时文件/前缀登记，仅在异常时清理
    cleanup_registry: List[str] = []
    temp_prefixes: List[str] = []

    def cleanup_current_run() -> None:
        """异常时，仅删除本次运行生成的临时文件/前缀"""
        # 删除按前缀生成的文件
        for prefix in temp_prefixes:
            delete_temp_files(prefix, [".bed", ".bim", ".fam", ".log", ".nosex", ".raw", ".map", ".ped"])
        # 删除精确记录的文件
        for fp in cleanup_registry:
            try:
                path = Path(fp)
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    try:
        # 参数检查
        if not Path(phenotype_file).exists():
            raise FileNotFoundError(f"表型文件不存在：{phenotype_file}")

        # 处理输出路径：如果指定的是目录，则在目录下创建默认文件名
        output_path = Path(output_file).absolute()
        if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
            # 如果指定的是目录或没有扩展名的路径，创建目录并设置默认文件名
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_name = "train_data.txt"
            output_file_path = output_dir / output_file_name
            output_prefix = output_file_name.replace('.txt', '').replace('.gz', '')
        else:
            # 如果指定的是文件路径，使用该文件的目录
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_path
            output_prefix = output_path.stem

        # 在输出目录下创建统一临时目录 tmp_p，用于存放所有中间文件（preprocess专用）
        tmp_dir = output_dir / "tmp_p"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 所有PLINK中间前缀统一放在 tmp 目录下
        temp_plink_prefix = str(tmp_dir / "temp_plink")
        temp_prefixes.append(temp_plink_prefix)

        # 步骤1：基因型格式转换
        genotype_format = detect_genotype_format(genotype_file)
        plink_prefix = genotype_to_plink(genotype_file, temp_plink_prefix, threads)
        
        # 步骤1.5：统一化染色体名称格式
        normalize_chromosome_names(plink_prefix)

        # 步骤2：加载表型数据
        pheno_df = load_phenotype(phenotype_file, pheno_col)

        # 步骤2.1：在绘制表型分布前，先与基因型样本进行匹配，只保留“既有表型又有基因型”的样本
        fam_file = f"{plink_prefix}.fam"
        if Path(fam_file).exists():
            try:
                fam_df = pd.read_csv(
                    fam_file,
                    sep=r"\s+",
                    header=None,
                    dtype=str,
                    engine="python"
                )
                # PLINK .fam 格式：FID IID PAT MAT SEX PHENO
                if fam_df.shape[1] >= 2:
                    geno_samples = set(fam_df.iloc[:, 1].astype(str))
                    original_pheno_n = len(pheno_df)
                    pheno_df = pheno_df[pheno_df["sample"].isin(geno_samples)].reset_index(drop=True)
                    if pheno_df.empty:
                        raise ValueError(
                            "表型样本与基因型样本无交集，请检查表型文件中的样本ID是否与PLINK文件一致。"
                        )
                    logger.info(
                        f"表型与基因型样本匹配完成：原始表型样本 {original_pheno_n} 个，"
                        f"与基因型匹配后剩余 {len(pheno_df)} 个样本"
                    )
                else:
                    logger.warning(
                        f"FAM文件格式异常（列数<2），跳过表型与基因型样本匹配：{fam_file}"
                    )
            except Exception as e:
                logger.warning(
                    f"读取FAM文件进行样本匹配时出错，跳过匹配步骤：{fam_file}，错误：{str(e)}"
                )
        else:
            logger.warning(
                f"未找到FAM文件，无法基于基因型样本过滤表型：{fam_file}。"
                f"将使用所有通过质量检查的表型样本绘制分布图。"
            )
        
        # 步骤2.5：判断表型类型并绘制可视化图表（基于已与基因型匹配后的表型样本）
        task_type = determine_phenotype_type(pheno_df, output_dir)

        # 步骤3：样本过滤（中间结果放在 tmp 目录）
        sample_filtered_prefix = str(tmp_dir / "sample_filtered_plink")
        temp_prefixes.append(sample_filtered_prefix)
        sample_filtered_plink = filter_samples_by_phenotype(plink_prefix, pheno_df, sample_filtered_prefix)

        # 步骤3.5：全局SNP质量过滤（MAF和缺失率）
        # 在分染色体处理前先过滤，可以显著减少后续处理的SNP数量，加快速度
        # 过滤后的PLINK前缀同样放在 tmp 目录
        filtered_plink_prefix = str(tmp_dir / "filtered_plink")
        temp_prefixes.append(filtered_plink_prefix)
        filtered_plink = filter_snps_by_quality(
            sample_filtered_plink, 
            filtered_plink_prefix,
            maf=0.05,
            geno=0.2,
            threads=threads
        )
        
        # 清理中间文件
        if "temp_plink" in sample_filtered_plink or "sample_filtered" in sample_filtered_plink:
            delete_temp_files(sample_filtered_plink, [".bed", ".bim", ".fam", ".log", ".nosex"])

        # 步骤4：多线程分染色体生成训练数据（所有中间结果写入 tmp）
        train_df, actual_train_file = plink_to_training_data(
            filtered_plink,
            pheno_df,
            str(output_file_path),
            threads,
            tmp_dir=tmp_dir,
            cleanup_registry=cleanup_registry
        )

        # 步骤5：生成元数据（使用实际保存的文件路径，保存在输出目录下）
        # output_prefix 已经是文件名（不含扩展名），直接使用它来生成元数据文件名
        metadata_file = output_dir / f"{output_prefix}_metadata.json"
        generate_metadata(
            output_prefix=str(output_dir / output_prefix),  # 使用原始output_prefix，不要用metadata_file.stem
            plink_prefix=filtered_plink,
            train_file=actual_train_file,
            valid_samples=train_df.index.tolist(),
            genotype_format=genotype_format,
            threads=threads,
            task_type=task_type,
            preprocess_tmp_dir=tmp_dir
        )

        # 最终清理
        gc.collect()

        logger.info(f"预处理完成: {len(train_df)}样本, {train_df.shape[1]-1}特征")
        return 0

    except Exception as e:
        logger.error(f"预处理失败：{str(e)}", exc_info=True)
        # 异常时删除本次运行生成的临时文件
        cleanup_current_run()
        return 1

# ======================== 14. 命令行入口 ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分染色体多进程并行预处理模块（兼容所有PLINK版本）")
    parser.add_argument("-g", "--genotype", required=True, help="基因型文件（VCF/PLINK二进制/文本前缀）")
    parser.add_argument("-p", "--phenotype", required=True, help="表型文件（无表头2列）")
    parser.add_argument("-o", "--output", required=True, help="输出目录或文件路径。如果指定目录，将在目录下创建train_data.txt；如果指定文件路径，将在文件所在目录下创建所有输出文件")
    parser.add_argument(
        "--threads", 
        type=int, 
        default=6, 
        help="并行进程上限。默认6表示自动根据染色体数和CPU核数分配（约每个进程处理2条染色体，总线程数≈染色体数）；"
             "如需手动限制并行度，可指定为正整数。如遇OOM可适当降低此值。"
    )
    # 串行模式已移除，始终使用并行执行（进程数由 --threads 控制）
    # --mem-limit 参数保留以兼容旧命令，但已不再生效（脚本不再向 PLINK 传递 --memory）
    parser.add_argument("--mem-limit", type=int, default=PLINK_MEM_LIMIT, help="（已废弃）PLINK内存限制（当前版本不生效）")
    parser.add_argument("--pheno-col", help="兼容参数，无实际作用")

    args = parser.parse_args()

    # 执行预处理
    sys.exit(run_preprocess(
        genotype_file=args.genotype,
        phenotype_file=args.phenotype,
        output_file=args.output,
        threads=args.threads,
        pheno_col=args.pheno_col
    ))