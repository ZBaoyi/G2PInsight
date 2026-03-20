#!/usr/bin/env python3
"""
GEMMA GWAS模块 - 严格匹配shell命令逻辑
固定输入格式：PLINK二进制文件（.bed/.bim/.fam）
"""

import subprocess
import logging
import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# ======================== 基础配置 =========================
# 初始化基础日志（仅控制台输出）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 软件路径：优先使用项目内集成的软件，其次回退到系统PATH中的可执行文件（standardized）
bin_dir = Path(__file__).parent
software_dir = bin_dir / "software"
_bundled_plink = software_dir / "plink"
_bundled_gemma = software_dir / "gemma-0.98.5-linux-static-AMD64"


def _get_plink_executable() -> str:
    """
    获取PLINK可执行路径（standardized）
    优先顺序：
      1) 项目内 software 目录下的集成版本
      2) 系统PATH中的 plink
    """
    if _bundled_plink.exists():
        return str(_bundled_plink)
    found = shutil.which("plink")
    if found:
        logger.info(f"Using system PLINK executable from PATH: {found}")
        return found
    raise FileNotFoundError(
        f"PLINK executable not found. "
        f"Checked bundled path: {_bundled_plink} and system PATH."
    )


def _get_gemma_executable() -> str:
    """
    获取GEMMA可执行路径（standardized）
    优先顺序：
      1) 项目内 software 目录下的集成版本
      2) 系统PATH中的 gemma
    """
    if _bundled_gemma.exists():
        return str(_bundled_gemma)
    # GEMMA 在PATH中的命令名可能为 gemma，这里做一次简单尝试
    found = shutil.which("gemma")
    if found:
        logger.info(f"Using system GEMMA executable from PATH: {found}")
        return found
    raise FileNotFoundError(
        f"GEMMA executable not found. "
        f"Checked bundled path: {_bundled_gemma} and system PATH."
    )

# 初始化可执行文件路径（在模块加载时执行，standardized）
plink_executable = _get_plink_executable()
gemma_executable = _get_gemma_executable()

# 固定输入格式要求
REQUIRED_INPUT_FILES = {
    "bed": ".bed",
    "bim": ".bim",
    "fam": ".fam"
}

# ======================== 参数合法性校验 =========================
def validate_input_files(input_prefix: str) -> None:
    """验证PLINK二进制文件完整性"""
    if not isinstance(input_prefix, str) or not input_prefix.strip():
        raise ValueError("input_prefix必须为非空字符串")
    
    missing_files = []
    for file_type, suffix in REQUIRED_INPUT_FILES.items():
        file_path = f"{input_prefix}{suffix}"
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(
            f"缺失PLINK二进制文件：{', '.join(missing_files)}"
        )
    logger.info(f"输入文件验证通过：{input_prefix}")

def validate_phenotype_file(phenotype_file: str) -> pd.DataFrame:
    """验证表型文件存在性，自动读取前两列并打印前5行"""
    if not isinstance(phenotype_file, str) or not phenotype_file.strip():
        raise ValueError("phenotype_file必须为非空字符串")
    if not Path(phenotype_file).exists():
        raise FileNotFoundError(f"表型文件不存在：{phenotype_file}")
    
    # 读取表型文件（仅取前两列）
    try:
        # usecols=[0,1] 强制读取前两列，skipinitialspace=True 忽略列前空格
        pheno_df = pd.read_csv(
            phenotype_file, 
            sep=r'\s+', 
            header=None, 
            usecols=[0, 1],  # 强制仅读取前两列
            skipinitialspace=True
        )
        # 重命名列
        pheno_df.columns = ['IID', 'PHENO']
        
        # 校验前两列是否为空
        if pheno_df['IID'].isnull().any() or pheno_df['PHENO'].isnull().any():
            logger.warning("表型文件前两列包含空值，请检查！")
        
        return pheno_df
    except Exception as e:
        raise RuntimeError(f"表型文件读取失败：{str(e)}")

# ======================== 核心执行函数 =========================
def run_shell_command(cmd: list, step_name: str) -> None:
    """执行shell命令（仅保留核心逻辑）"""
    cmd_str = " ".join(cmd)
    logger.info(f"[执行命令] {step_name}：{cmd_str}")
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    if result.stdout:
        # 标准输出通常信息量较大，仅在调试时完整查看
        pass
    
    # 处理标准错误输出：区分真正的错误和信息性消息，并过滤掉GEMMA的进度条
    stderr_content = ""
    if result.stderr:
        stderr_content = result.stderr[:1000]
        # GEMMA 在stderr中用类似“空格 + = + 百分比”的形式打印进度条，
        # 直接记录到日志会变成多行噪声，这里先按行过滤掉这类仅含进度条的行。
        lines = stderr_content.splitlines()
        filtered_lines = []
        for line in lines:
            # 去掉左右空白后，如果只剩下 0-9、% 和 =，视为进度条行，丢弃
            stripped = line.strip()
            if stripped and all(ch in "=%0123456789" for ch in stripped):
                continue
            filtered_lines.append(line)
        stderr_content = "\n".join(filtered_lines).strip()
    
    if stderr_content:
        # GEMMA 会将信息性消息（如 "**** INFO: Done."）输出到 stderr
        # 如果命令成功执行（returncode == 0），且 stderr 中只包含 INFO/Done 等关键词，则记录为 info 级别
        if result.returncode == 0:
            # 检查是否包含真正的错误关键词
            error_keywords = ['ERROR', 'FATAL', 'FAIL', 'Exception', 'Traceback']
            if any(keyword in stderr_content.upper() for keyword in error_keywords):
                logger.warning(f"标准错误（可能存在问题）：{stderr_content}")
            else:
                # 信息性消息，记录为 info 级别
                logger.info(f"标准错误（信息性消息）：{stderr_content}")
        else:
            # 命令失败，stderr 肯定是错误
            logger.error(f"标准错误：{stderr_content}")
    
    if result.returncode != 0:
        raise RuntimeError(f"{step_name}执行失败，返回码：{result.returncode}")
    logger.info(f"[完成] {step_name}")

def plink_quality_control(input_prefix: str, output_prefix: str) -> None:
    """PLINK质控（geno 0.2、maf 0.05）"""
    cmd = [
        plink_executable,
        "--bfile", input_prefix,
        "--maf", "0.05",
        "--geno", "0.2",
        "--allow-extra-chr",
        "--allow-no-sex",
        "--make-bed",
        "--out", output_prefix
    ]
    run_shell_command(cmd, "PLINK质控")

def merge_phenotype_to_fam(pheno_df: pd.DataFrame, fam_file: str) -> None:
    """表型匹配到FAM文件（使用预处理好的前两列表型数据）"""
    # 备份原始fam文件
    bak_fam = f"{fam_file}.bak"
    if not Path(bak_fam).exists():
        cmd = ["mv", fam_file, bak_fam]
        run_shell_command(cmd, "备份FAM文件")
    
    # 构建表型映射（IID转字符串避免类型不匹配）
    pheno_dict = dict(zip(
        pheno_df['IID'].astype(str).str.strip(), 
        pheno_df['PHENO'].astype(str).str.strip()
    ))
    
    # 读取并替换FAM第六列
    fam_df = pd.read_csv(
        bak_fam, 
        sep=r'\s+', 
        header=None, 
        names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'OLD_PHENO']
    )
    # IID转字符串并去空格，避免匹配失败
    fam_df['IID'] = fam_df['IID'].astype(str).str.strip()
    # 替换表型，缺失填-9
    fam_df['PHENO'] = fam_df['IID'].map(pheno_dict).fillna("-9")
    
    # 保存新FAM文件
    new_fam_df = fam_df[['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO']]
    new_fam_df.to_csv(fam_file, sep=' ', header=False, index=False)
    
    # 基础统计
    matched = (fam_df['PHENO'] != "-9").sum()
    total = len(fam_df)
    logger.info(f"表型匹配完成：{matched}/{total} 样本匹配成功（缺失表型填-9）")

def calculate_pca(input_prefix: str, output_prefix: str) -> None:
    """计算PCA（前5个主成分）"""
    cmd = [
        plink_executable,
        "--bfile", input_prefix,
        "--pca", "5",
        "--allow-extra-chr",
        "--allow-no-sex",
        "--out", output_prefix
    ]
    run_shell_command(cmd, "PCA计算")

def filter_valid_samples(fam_file: str, output_sample_file: str) -> None:
    """筛选有效表型样本（排除-9）"""
    fam_df = pd.read_csv(
        fam_file, 
        sep=r'\s+', 
        header=None, 
        names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO']
    )
    valid_df = fam_df[fam_df['PHENO'] != "-9"][['FID', 'IID']]
    valid_df.to_csv(output_sample_file, sep=' ', header=False, index=False)
    logger.info(f"有效样本保存到：{output_sample_file}，共{len(valid_df)}个样本")

def filter_plink_samples(input_prefix: str, sample_file: str, output_prefix: str) -> None:
    """过滤PLINK样本（仅保留有效样本）"""
    cmd = [
        plink_executable,
        "--bfile", input_prefix,
        "--keep", sample_file,
        "--allow-extra-chr",
        "--allow-no-sex",
        "--make-bed",
        "--out", output_prefix
    ]
    run_shell_command(cmd, "PLINK样本过滤")

def filter_pca_by_samples(pca_file: str, sample_file: str, output_pca_file: str) -> None:
    """过滤PCA结果（仅保留有效样本）"""
    # 读取有效样本
    valid_df = pd.read_csv(
        sample_file, 
        sep=r'\s+', 
        header=None, 
        names=['FID', 'IID']
    )
    valid_set = set(zip(
        valid_df['FID'].astype(str).str.strip(), 
        valid_df['IID'].astype(str).str.strip()
    ))
    
    # 过滤PCA文件
    pca_df = pd.read_csv(pca_file, sep=r'\s+', header=None)
    pca_df['key'] = list(zip(
        pca_df[0].astype(str).str.strip(), 
        pca_df[1].astype(str).str.strip()
    ))
    filtered_df = pca_df[pca_df['key'].isin(valid_set)].drop('key', axis=1)
    filtered_df.to_csv(output_pca_file, sep=' ', header=False, index=False)
    logger.info(f"过滤后PCA保存到：{output_pca_file}，共{len(filtered_df)}个样本")

def extract_pca_covariates(pca_file: str, output_cov_file: str) -> None:
    """提取PCA协变量（仅保留主成分数值）"""
    pca_df = pd.read_csv(pca_file, sep=r'\s+', header=None)
    cov_df = pca_df.iloc[:, 2:]  # 跳过前2列（FID/IID）
    cov_df.to_csv(output_cov_file, sep=' ', header=False, index=False)
    logger.info(f"PCA协变量保存到：{output_cov_file}，共{cov_df.shape[1]}个主成分")

def calculate_kinship_matrix(genotype_prefix: str, output_prefix: str) -> str:
    """计算亲缘关系矩阵（移除-p参数）"""
    base_prefix = Path(output_prefix).name
    kinship_prefix = f"{base_prefix}_kinship"
    cmd = [
        gemma_executable,
        "-bfile", genotype_prefix,
        "-gk", "1",
        "-o", kinship_prefix
    ]
    run_shell_command(cmd, "亲缘矩阵计算")
    
    # 返回kinship文件路径
    kinship_file = Path("output") / f"{kinship_prefix}.cXX.txt"
    if not kinship_file.exists():
        raise FileNotFoundError(f"亲缘矩阵文件未生成：{kinship_file}")
    return kinship_file.absolute().as_posix()

def run_gemma_gwas(genotype_prefix: str, kinship_file: str, cov_file: str, output_prefix: str) -> None:
    """运行GWAS分析（移除-p参数）"""
    # 使用与亲缘矩阵相同的策略：仅使用前缀的 basename，避免在 GEMMA 的 -o 参数中出现绝对路径
    # 注意：GEMMA 的 -o 参数是相对于当前工作目录的，所以使用相对路径
    # GEMMA 会在当前工作目录下创建 output/ 目录，并在其中生成结果文件
    base_prefix = Path(output_prefix).name
    gwas_prefix = f"{base_prefix}_gwas"
    
    # 记录当前工作目录，用于调试
    current_dir = os.getcwd()
    logger.info(f"GWAS执行目录（当前工作目录）：{current_dir}")
    logger.info(f"GWAS输出前缀（相对路径）：{gwas_prefix}（将在 {current_dir}/output/ 下生成结果文件）")
    
    cmd = [
        gemma_executable,
        "-bfile", genotype_prefix,
        "-k", kinship_file,
        "-c", cov_file,
        "-lmm", "1",
        "-o", gwas_prefix
    ]
    run_shell_command(cmd, "GWAS关联分析")
    
    # 验证结果文件是否生成
    expected_result_file = Path(current_dir) / "output" / f"{gwas_prefix}.assoc.txt"
    if expected_result_file.exists():
        logger.info(f"GWAS结果文件已生成：{expected_result_file}")
    else:
        logger.warning(f"GWAS结果文件未找到（可能路径不同）：{expected_result_file}")

def run_complete_gwas_pipeline(
    input_plink_prefix: str,
    phenotype_file: str,
    output_prefix: str,
    perform_plink_qc: bool = True,
) -> None:
    """完整GWAS流程
    
    Args:
        input_plink_prefix: 输入PLINK前缀
        phenotype_file: 表型文件路径（两列：IID PHENO）
        output_prefix: 输出前缀（可包含路径）
        perform_plink_qc: 是否在GWAS前执行PLINK质控（--maf 0.05, --geno 0.2）
    """
    # 输入校验
    validate_input_files(input_plink_prefix)
    # 读取表型文件（自动取前两列+打印前5行）
    pheno_df = validate_phenotype_file(phenotype_file)
    
    # 1. PLINK质控（可选）
    if perform_plink_qc:
        clean_geno_prefix = f"{output_prefix}_clean_geno"
        plink_quality_control(input_plink_prefix, clean_geno_prefix)
    else:
        logger.info("跳过GWAS阶段的PLINK质控（根据上游参数设置）")
        clean_geno_prefix = input_plink_prefix
    
    # 2. 表型匹配到FAM（使用预处理的表型数据）
    clean_fam_file = f"{clean_geno_prefix}.fam"
    merge_phenotype_to_fam(pheno_df, clean_fam_file)
    
    # 3. 计算PCA
    pca_prefix = f"{output_prefix}_geno_pca"
    calculate_pca(clean_geno_prefix, pca_prefix)
    
    # 4. 筛选有效样本
    valid_sample_file = f"{output_prefix}_valid_samples.txt"
    filter_valid_samples(clean_fam_file, valid_sample_file)
    
    # 5. 过滤PLINK样本
    filtered_geno_prefix = f"{output_prefix}_clean_geno_filtered"
    filter_plink_samples(clean_geno_prefix, valid_sample_file, filtered_geno_prefix)
    
    # 6. 过滤PCA（从过滤后样本提取协变量）
    pca_eigenvec_file = f"{pca_prefix}.eigenvec"
    filtered_pca_file = f"{output_prefix}_geno_pca_filtered.eigenvec"
    filter_pca_by_samples(pca_eigenvec_file, valid_sample_file, filtered_pca_file)
    
    # 7. 提取PCA协变量
    cov_file = f"{output_prefix}_covariates_filtered.txt"
    extract_pca_covariates(filtered_pca_file, cov_file)
    
    # 8. 计算亲缘矩阵（无-p参数）
    kinship_file = calculate_kinship_matrix(filtered_geno_prefix, output_prefix)
    
    # 9. 运行GWAS（无-p参数）
    run_gemma_gwas(filtered_geno_prefix, kinship_file, cov_file, output_prefix)
    
    logger.info("GWAS流程执行完成！")

if __name__ == "__main__":
    # 基础调用逻辑
    if len(sys.argv) != 4:
        print("使用方式：python gwas_module.py <输入PLINK前缀> <表型文件> <输出前缀>")
        sys.exit(1)
    
    try:
        input_plink_prefix = sys.argv[1]
        phenotype_file = sys.argv[2]
        output_prefix = sys.argv[3]
        run_complete_gwas_pipeline(input_plink_prefix, phenotype_file, output_prefix)
    except Exception as e:
        logger.error(f"流程执行失败：{str(e)}")
        sys.exit(1)