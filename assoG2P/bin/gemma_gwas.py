#!/usr/bin/env python3
"""
GEMMA GWAS模块 - 严格匹配shell命令逻辑
固定输入格式：PLINK二进制文件（.bed/.bim/.fam）
支持命令行直接运行，参数解析+完整错误处理
"""

import subprocess
import logging
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# ======================== 固定配置（完全匹配shell环境）========================
logger = logging.getLogger(__name__)

# 1. 固定软件路径（不再检测系统路径）
bin_dir = Path(__file__).parent
software_dir = bin_dir / "software"
plink_executable = str(software_dir / "plink")
gemma_executable = str(software_dir / "gemma-0.98.5-linux-static-AMD64")

# 2. 固定输入格式要求（仅支持PLINK二进制格式）
REQUIRED_INPUT_FILES = {
    "bed": ".bed",  # 基因型二进制文件
    "bim": ".bim",  # 标记信息文件
    "fam": ".fam"   # 样本信息文件
}

# ======================== 核心函数（严格对应shell命令）========================
def validate_input_files(input_prefix: str) -> None:
    """
    验证输入文件格式（固定为PLINK二进制格式）
    """
    missing_files = []
    for file_type, suffix in REQUIRED_INPUT_FILES.items():
        file_path = f"{input_prefix}{suffix}"
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(
            f"输入文件格式错误（仅支持PLINK二进制格式）！缺失文件：{', '.join(missing_files)}\n"
            f"要求：输入前缀{input_prefix}必须对应以下文件：\n"
            f"  - {input_prefix}.bed (基因型二进制文件)\n"
            f"  - {input_prefix}.bim (SNP标记信息文件)\n"
            f"  - {input_prefix}.fam (样本表型信息文件)"
        )
    logger.info(f"输入文件验证通过：{input_prefix}.*")

def run_shell_command(cmd: list, step_name: str) -> None:
    """
    执行shell命令（严格复刻命令执行逻辑）
    """
    cmd_str = " ".join(cmd)
    logger.info(f"\n【执行命令】{step_name}:\n{cmd_str}")
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    # 输出命令执行结果
    if result.stdout:
        logger.info(f"标准输出:\n{result.stdout[:1000]}")  # 只显示前1000字符避免刷屏
    if result.stderr:
        logger.warning(f"标准错误:\n{result.stderr[:1000]}")
    
    # 检查执行状态
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} 执行失败（返回码：{result.returncode}）")
    logger.info(f"【完成】{step_name}")

def plink_quality_control(input_prefix: str, output_prefix: str) -> None:
    """
    PLINK质控 - 严格对应shell命令：plink --bfile input --maf 0.05 --geno 0.2 --make-bed --out output
    """
    cmd = [
        plink_executable,
        "--bfile", input_prefix,
        "--maf", "0.05",       # 固定参数，匹配shell
        "--geno", "0.2",       # 固定参数，匹配shell
        "--make-bed",
        "--out", output_prefix
    ]
    run_shell_command(cmd, "PLINK质控（MAF/GENO过滤）")

def merge_phenotype_to_fam(phenotype_file: str, fam_file: str) -> None:
    """
    表型匹配到FAM文件 - 严格对应shell的awk命令逻辑
    
    支持两种常见表型文件格式：
      1) 两列：IID PHENO
      2) 三列：FID IID PHENO  （推荐，用于直接传给GEMMA的 -p）
    """
    # 1. 备份原始fam文件（对应shell: mv $clean_fam $clean_fam.bak）
    bak_fam = f"{fam_file}.bak"
    if not Path(bak_fam).exists():
        cmd = ["mv", fam_file, bak_fam]
        run_shell_command(cmd, f"备份FAM文件到 {bak_fam}")
    
    # 2. 读取表型文件构建映射（对应shell: awk 'NR==FNR{{pheno[$1]=$2;next}}{{print $0, pheno[$2]}}'）
    logger.info(f"读取表型文件：{phenotype_file}")
    pheno_df_raw = pd.read_csv(phenotype_file, sep='\s+', header=None)
    if pheno_df_raw.shape[1] >= 3:
        # 三列表型：FID IID PHENO（兼容GEMMA标准三列格式）
        pheno_df = pheno_df_raw.iloc[:, :3]
        pheno_df.columns = ['FID', 'IID', 'PHENO']
    elif pheno_df_raw.shape[1] == 2:
        # 两列表型：IID PHENO（旧版格式）
        pheno_df = pheno_df_raw.copy()
        pheno_df.columns = ['IID', 'PHENO']
    else:
        raise ValueError(
            f"表型文件格式不正确（需要2列 IID PHENO 或 3列 FID IID PHENO）：{phenotype_file}，"
            f"当前列数={pheno_df_raw.shape[1]}"
        )
    pheno_dict = dict(zip(pheno_df['IID'].astype(str), pheno_df['PHENO'].astype(str)))
    
    # 3. 重新生成fam文件（保留前5列，第6列替换为匹配的表型，缺失填-9）
    logger.info(f"匹配表型到FAM文件：{fam_file}")
    fam_df = pd.read_csv(bak_fam, sep='\s+', header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'OLD_PHENO'])
    fam_df['PHENO'] = fam_df['IID'].astype(str).map(pheno_dict).fillna("-9")
    
    # 保留前5列 + 新表型列（对应shell的print $0, pheno[$2]）
    new_fam_df = fam_df[['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO']]
    new_fam_df.to_csv(fam_file, sep=' ', header=False, index=False)
    
    # 统计匹配结果
    matched = (fam_df['PHENO'] != "-9").sum()
    total = len(fam_df)
    logger.info(f"表型匹配完成：{matched}/{total} 样本匹配成功，缺失表型填充-9")

def calculate_pca(input_prefix: str, output_prefix: str) -> None:
    """
    PCA计算 - 严格对应shell命令：plink --bfile input --pca 5 --out output
    """
    cmd = [
        plink_executable,
        "--bfile", input_prefix,
        "--pca", "5",          # 固定参数，匹配shell
        "--out", output_prefix
    ]
    run_shell_command(cmd, "PLINK PCA计算（前5主成分）")

def filter_valid_samples(fam_file: str, output_sample_file: str) -> None:
    """
    筛选有效表型样本 - 严格对应shell命令：awk '$6 != "-9" {print $1, $2}' $clean_fam > $valid_samples
    """
    logger.info(f"筛选有效表型样本（排除-9）：{fam_file}")
    fam_df = pd.read_csv(fam_file, sep='\s+', header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
    valid_df = fam_df[fam_df['PHENO'] != "-9"][['FID', 'IID']]
    valid_df.to_csv(output_sample_file, sep=' ', header=False, index=False)
    logger.info(f"有效样本保存到：{output_sample_file}（共{len(valid_df)}个样本）")

def filter_plink_samples(input_prefix: str, sample_file: str, output_prefix: str) -> None:
    """
    过滤PLINK样本 - 严格对应shell命令：plink --bfile input --keep sample_file --make-bed --out output
    """
    cmd = [
        plink_executable,
        "--bfile", input_prefix,
        "--keep", sample_file,
        "--make-bed",
        "--out", output_prefix
    ]
    run_shell_command(cmd, "PLINK样本过滤（仅保留有效表型样本）")

def filter_pca_by_samples(pca_file: str, sample_file: str, output_pca_file: str) -> None:
    """
    过滤PCA结果 - 严格对应shell命令：awk 'NR==FNR{valid[$1,$2]=1;next} valid[$1,$2]' $valid_samples $pca_eigenvec > $filtered_pca
    """
    logger.info(f"过滤PCA文件：{pca_file}")
    
    # 读取有效样本
    valid_df = pd.read_csv(sample_file, sep='\s+', header=None, names=['FID', 'IID'])
    valid_set = set(zip(valid_df['FID'].astype(str), valid_df['IID'].astype(str)))
    
    # 读取并过滤PCA文件
    pca_df = pd.read_csv(pca_file, sep='\s+', header=None)
    pca_df['key'] = list(zip(pca_df[0].astype(str), pca_df[1].astype(str)))
    filtered_df = pca_df[pca_df['key'].isin(valid_set)].drop('key', axis=1)
    
    # 保存过滤结果
    filtered_df.to_csv(output_pca_file, sep=' ', header=False, index=False)
    logger.info(f"过滤后PCA保存到：{output_pca_file}（共{len(filtered_df)}个样本）")

def extract_pca_covariates(pca_file: str, output_cov_file: str) -> None:
    """
    提取PCA协变量 - 严格对应shell命令：cut -d ' ' -f 3- $filtered_pca > $cov_file
    """
    logger.info(f"提取PCA协变量（跳过前2列）：{pca_file}")
    pca_df = pd.read_csv(pca_file, sep='\s+', header=None)
    cov_df = pca_df.iloc[:, 2:]  # 跳过FID/IID列（前2列）
    cov_df.to_csv(output_cov_file, sep=' ', header=False, index=False)
    logger.info(f"协变量文件保存到：{output_cov_file}（共{cov_df.shape[1]}个主成分）")

def calculate_kinship_matrix(genotype_prefix: str, phenotype_file: str, output_prefix: str) -> str:
    """
    计算Kinship矩阵 - 严格对应shell命令：gemma -bfile input -p phenotype -gk 1 -o output
    """
    # 注意：GEMMA会将 -o 参数当作“文件名”，并固定写到当前工作目录下的 output/ 目录，
    # 因此这里必须只传递“basename”，不能包含路径，否则会出现
    #   error writing file: ./output//path/to/xxx_kinship.cXX.txt
    # 这样的报错。
    base_prefix = Path(output_prefix).name
    kinship_prefix = f"{base_prefix}_kinship"
    cmd = [
        gemma_executable,
        "-bfile", genotype_prefix,
        "-p", phenotype_file,
        "-gk", "1",            # 固定参数，匹配shell
        "-o", kinship_prefix
    ]
    run_shell_command(cmd, "GEMMA Kinship矩阵计算")
    
    # 返回kinship文件路径（GEMMA默认输出到当前工作目录下的output目录）
    # 使用绝对路径，确保无论工作目录如何切换都能找到文件
    kinship_file = Path("output") / f"{kinship_prefix}.cXX.txt"
    if not kinship_file.exists():
        raise FileNotFoundError(f"Kinship矩阵文件未生成：{kinship_file.absolute()}")
    # 返回绝对路径，确保调用方可以正确使用
    return kinship_file.absolute().as_posix()

def run_gemma_gwas(genotype_prefix: str, phenotype_file: str, kinship_file: str, 
                   cov_file: str, output_prefix: str) -> None:
    """
    运行GWAS分析 - 严格对应shell命令：gemma -bfile input -p phenotype -k kinship -c cov -lmm 1 -o output
    """
    # 同 kinship，一样只使用 basename 作为 -o 前缀，避免在 output/ 目录下再嵌套路径
    base_prefix = Path(output_prefix).name
    gwas_prefix = f"{base_prefix}_gwas"
    cmd = [
        gemma_executable,
        "-bfile", genotype_prefix,
        "-p", phenotype_file,
        "-k", kinship_file,
        "-c", cov_file,
        "-lmm", "1",           # 固定参数，匹配shell
        "-o", gwas_prefix
    ]
    run_shell_command(cmd, "GEMMA GWAS分析（LMM模型）")

def run_complete_gwas_pipeline(
    input_plink_prefix: str,  # 固定输入：PLINK二进制文件前缀
    phenotype_file: str,      # 固定输入：表型文件（两列：IID PHENO）
    output_prefix: str        # 输出前缀
) -> None:
    """
    完整GWAS流程 - 严格按shell命令顺序执行，固定输入格式要求
    """
    # ======================== 输入验证 ========================
    logger.info("\n【步骤0】验证输入文件格式")
    validate_input_files(input_plink_prefix)
    if not Path(phenotype_file).exists():
        raise FileNotFoundError(f"表型文件不存在：{phenotype_file}")
    logger.info(f"表型文件验证通过：{phenotype_file}")
    
    # ======================== 流程执行（严格对应shell命令顺序）========================
    # 步骤1：PLINK质控
    clean_geno_prefix = f"{output_prefix}_clean_geno"
    plink_quality_control(input_plink_prefix, clean_geno_prefix)
    
    # 步骤2：匹配表型到FAM文件
    clean_fam_file = f"{clean_geno_prefix}.fam"
    merge_phenotype_to_fam(phenotype_file, clean_fam_file)
    
    # 步骤3：计算PCA
    pca_prefix = f"{output_prefix}_geno_pca"
    calculate_pca(clean_geno_prefix, pca_prefix)
    
    # 步骤4：筛选有效表型样本
    valid_sample_file = f"{output_prefix}_valid_samples.txt"
    filter_valid_samples(clean_fam_file, valid_sample_file)
    
    # 步骤5：过滤PLINK样本
    filtered_geno_prefix = f"{output_prefix}_clean_geno_filtered"
    filter_plink_samples(clean_geno_prefix, valid_sample_file, filtered_geno_prefix)
    
    # 步骤6：过滤PCA结果
    pca_eigenvec_file = f"{pca_prefix}.eigenvec"
    filtered_pca_file = f"{output_prefix}_geno_pca_filtered.eigenvec"
    filter_pca_by_samples(pca_eigenvec_file, valid_sample_file, filtered_pca_file)
    
    # 步骤7：提取PCA协变量
    cov_file = f"{output_prefix}_covariates_filtered.txt"
    extract_pca_covariates(filtered_pca_file, cov_file)
    
    # 步骤8：计算Kinship矩阵
    kinship_file = calculate_kinship_matrix(filtered_geno_prefix, phenotype_file, output_prefix)
    
    # 步骤9：运行GWAS分析
    run_gemma_gwas(filtered_geno_prefix, phenotype_file, kinship_file, cov_file, output_prefix)
    
    # ======================== 完成 ========================
    logger.info("\n" + "="*80)
    logger.info("GWAS流程全部完成！")
    logger.info(f"核心输出文件：")
    logger.info(f"  - GWAS结果：output/{output_prefix}_gwas.assoc.txt")
    logger.info(f"  - Kinship矩阵：{kinship_file}")
    logger.info(f"  - 协变量文件：{cov_file}")
    logger.info("="*80)

if __name__ == "__main__":
    # 防止直接运行该模块，提示需通过model train调用
    logger.error("该模块不可独立运行，请通过model train模块调用！")
    sys.exit(1)