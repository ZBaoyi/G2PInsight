#!/usr/bin/env python3
import subprocess
import logging
import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
bin_dir = Path(__file__).parent
software_dir = bin_dir / 'software'
_bundled_plink = software_dir / 'plink'
_bundled_gemma = software_dir / 'gemma-0.98.5-linux-static-AMD64'

def _get_plink_executable() -> str:
    if _bundled_plink.exists():
        return str(_bundled_plink)
    found = shutil.which('plink')
    if found:
        logger.debug(f'Using system PLINK executable from PATH: {found}')
        return found
    raise FileNotFoundError(f'PLINK executable not found. Checked bundled path: {_bundled_plink} and system PATH.')

def _get_gemma_executable() -> str:
    if _bundled_gemma.exists():
        return str(_bundled_gemma)
    found = shutil.which('gemma')
    if found:
        logger.debug(f'Using system GEMMA executable from PATH: {found}')
        return found
    raise FileNotFoundError(f'GEMMA executable not found. Checked bundled path: {_bundled_gemma} and system PATH.')
plink_executable = _get_plink_executable()
gemma_executable = _get_gemma_executable()
REQUIRED_INPUT_FILES = {'bed': '.bed', 'bim': '.bim', 'fam': '.fam'}

def validate_input_files(input_prefix: str) -> None:
    if not isinstance(input_prefix, str) or not input_prefix.strip():
        raise ValueError('input_prefix must be a non-empty string')
    missing_files = []
    for file_type, suffix in REQUIRED_INPUT_FILES.items():
        file_path = f'{input_prefix}{suffix}'
        if not Path(file_path).exists():
            missing_files.append(file_path)
    if missing_files:
        raise FileNotFoundError(f"Missing PLINK binary files: {', '.join(missing_files)}")
    logger.debug(f'Input file validation passed: {input_prefix}')

def validate_phenotype_file(phenotype_file: str) -> pd.DataFrame:
    if not isinstance(phenotype_file, str) or not phenotype_file.strip():
        raise ValueError('phenotype_file must be a non-empty string')
    if not Path(phenotype_file).exists():
        raise FileNotFoundError(f'Phenotype file not found: {phenotype_file}')
    try:
        pheno_df = pd.read_csv(phenotype_file, sep='\\s+', header=None, usecols=[0, 1], skipinitialspace=True)
        pheno_df.columns = ['IID', 'PHENO']
        if pheno_df['IID'].isnull().any() or pheno_df['PHENO'].isnull().any():
            logger.debug('The first two columns in the phenotype file contain missing values')
        return pheno_df
    except Exception as e:
        raise RuntimeError(f'Failed to read phenotype file: {str(e)}')

def run_shell_command(cmd: list, step_name: str) -> None:
    cmd_str = ' '.join(cmd)
    logger.debug(f'[RUN] {step_name}: {cmd_str}')
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    stderr_content = ''
    if result.stderr:
        stderr_content = result.stderr[:1000]
        lines = stderr_content.splitlines()
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and all((ch in '=%0123456789' for ch in stripped)):
                continue
            filtered_lines.append(line)
        stderr_content = '\n'.join(filtered_lines).strip()
    if stderr_content:
        if result.returncode == 0:
            error_keywords = ['ERROR', 'FATAL', 'FAIL', 'Exception', 'Traceback']
            if any((keyword in stderr_content.upper() for keyword in error_keywords)):
                logger.debug(f'stderr (potential issue): {stderr_content}')
            else:
                logger.debug(f'stderr (informational): {stderr_content}')
        else:
            logger.error(f'stderr: {stderr_content}')
    if result.returncode != 0:
        raise RuntimeError(f'{step_name} failed with return code: {result.returncode}')
    logger.debug(f'[DONE] {step_name}')

def plink_quality_control(input_prefix: str, output_prefix: str) -> None:
    cmd = [plink_executable, '--bfile', input_prefix, '--maf', '0.05', '--geno', '0.2', '--allow-extra-chr', '--allow-no-sex', '--make-bed', '--out', output_prefix]
    run_shell_command(cmd, 'PLINK quality control')

def merge_phenotype_to_fam(pheno_df: pd.DataFrame, fam_file: str) -> None:
    bak_fam = f'{fam_file}.bak'
    if not Path(bak_fam).exists():
        cmd = ['mv', fam_file, bak_fam]
        run_shell_command(cmd, 'Backup FAM file')
    pheno_dict = dict(zip(pheno_df['IID'].astype(str).str.strip(), pheno_df['PHENO'].astype(str).str.strip()))
    fam_df = pd.read_csv(bak_fam, sep='\\s+', header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'OLD_PHENO'])
    fam_df['IID'] = fam_df['IID'].astype(str).str.strip()
    fam_df['PHENO'] = fam_df['IID'].map(pheno_dict).fillna('-9')
    new_fam_df = fam_df[['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO']]
    new_fam_df.to_csv(fam_file, sep=' ', header=False, index=False)
    matched = (fam_df['PHENO'] != '-9').sum()
    total = len(fam_df)
    logger.debug(f'Phenotype matching completed: {matched}/{total} samples matched (-9 used for missing phenotypes)')

def calculate_pca(input_prefix: str, output_prefix: str) -> None:
    cmd = [plink_executable, '--bfile', input_prefix, '--pca', '5', '--allow-extra-chr', '--allow-no-sex', '--out', output_prefix]
    run_shell_command(cmd, 'PCA computation')

def filter_valid_samples(pheno_df: pd.DataFrame, fam_file: str, output_sample_file: str) -> None:
    fam_df = pd.read_csv(fam_file, sep='\\s+', header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
    valid_iids = set(pheno_df['IID'].astype(str).str.strip())
    fam_df['IID'] = fam_df['IID'].astype(str).str.strip()
    valid_df = fam_df[fam_df['IID'].isin(valid_iids)][['FID', 'IID']]
    valid_df.to_csv(output_sample_file, sep=' ', header=False, index=False)
    logger.debug(f'Valid samples saved to: {output_sample_file} (n={len(valid_df)})')

def filter_plink_samples(input_prefix: str, sample_file: str, output_prefix: str) -> None:
    cmd = [plink_executable, '--bfile', input_prefix, '--keep', sample_file, '--allow-extra-chr', '--allow-no-sex', '--make-bed', '--out', output_prefix]
    run_shell_command(cmd, 'PLINK sample filtering')

def filter_pca_by_samples(pca_file: str, sample_file: str, output_pca_file: str) -> None:
    valid_df = pd.read_csv(sample_file, sep='\\s+', header=None, names=['FID', 'IID'])
    valid_set = set(zip(valid_df['FID'].astype(str).str.strip(), valid_df['IID'].astype(str).str.strip()))
    pca_df = pd.read_csv(pca_file, sep='\\s+', header=None)
    pca_df['key'] = list(zip(pca_df[0].astype(str).str.strip(), pca_df[1].astype(str).str.strip()))
    filtered_df = pca_df[pca_df['key'].isin(valid_set)].drop('key', axis=1)
    filtered_df.to_csv(output_pca_file, sep=' ', header=False, index=False)
    logger.debug(f'Filtered PCA saved to: {output_pca_file} (n={len(filtered_df)} samples)')

def extract_pca_covariates(pca_file: str, output_cov_file: str) -> None:
    pca_df = pd.read_csv(pca_file, sep='\\s+', header=None)
    cov_df = pca_df.iloc[:, 2:]
    cov_df.to_csv(output_cov_file, sep=' ', header=False, index=False)
    logger.debug(f'PCA covariates saved to: {output_cov_file} (PCs={cov_df.shape[1]})')

def calculate_kinship_matrix(genotype_prefix: str, output_prefix: str) -> str:
    base_prefix = Path(output_prefix).name
    kinship_prefix = f'{base_prefix}_kinship'
    cmd = [gemma_executable, '-bfile', genotype_prefix, '-gk', '1', '-o', kinship_prefix]
    run_shell_command(cmd, 'Kinship matrix computation')
    kinship_file = Path('output') / f'{kinship_prefix}.cXX.txt'
    if not kinship_file.exists():
        raise FileNotFoundError(f'Kinship matrix file was not generated: {kinship_file}')
    return kinship_file.absolute().as_posix()

def run_gemma_gwas(genotype_prefix: str, kinship_file: str, cov_file: str, output_prefix: str) -> None:
    base_prefix = Path(output_prefix).name
    gwas_prefix = f'{base_prefix}_gwas'
    current_dir = os.getcwd()
    logger.debug(f'GWAS working directory: {current_dir}')
    logger.debug(f'GWAS output prefix: {gwas_prefix} (results expected under {current_dir}/output/)')
    cmd = [gemma_executable, '-bfile', genotype_prefix, '-k', kinship_file, '-c', cov_file, '-lmm', '1', '-o', gwas_prefix]
    run_shell_command(cmd, 'GWAS association analysis')
    expected_result_file = Path(current_dir) / 'output' / f'{gwas_prefix}.assoc.txt'
    if expected_result_file.exists():
        logger.debug(f'GWAS result file generated: {expected_result_file}')
    else:
        logger.debug(f'GWAS result file not found at expected path: {expected_result_file}')

def run_complete_gwas_pipeline(input_plink_prefix: str, phenotype_file: str, output_prefix: str, perform_plink_qc: bool=True) -> None:
    validate_input_files(input_plink_prefix)
    pheno_df = validate_phenotype_file(phenotype_file)
    if perform_plink_qc:
        clean_geno_prefix = f'{output_prefix}_clean_geno'
        plink_quality_control(input_plink_prefix, clean_geno_prefix)
    else:
        logger.debug('Skipping PLINK QC in GWAS step based on upstream configuration')
        clean_geno_prefix = input_plink_prefix
    pca_prefix = f'{output_prefix}_geno_pca'
    calculate_pca(clean_geno_prefix, pca_prefix)
    valid_sample_file = f'{output_prefix}_valid_samples.txt'
    filter_valid_samples(pheno_df, f'{clean_geno_prefix}.fam', valid_sample_file)
    filtered_geno_prefix = f'{output_prefix}_clean_geno_filtered'
    filter_plink_samples(clean_geno_prefix, valid_sample_file, filtered_geno_prefix)
    filtered_fam_file = f'{filtered_geno_prefix}.fam'
    merge_phenotype_to_fam(pheno_df, filtered_fam_file)
    pca_eigenvec_file = f'{pca_prefix}.eigenvec'
    filtered_pca_file = f'{output_prefix}_geno_pca_filtered.eigenvec'
    filter_pca_by_samples(pca_eigenvec_file, valid_sample_file, filtered_pca_file)
    cov_file = f'{output_prefix}_covariates_filtered.txt'
    extract_pca_covariates(filtered_pca_file, cov_file)
    kinship_file = calculate_kinship_matrix(filtered_geno_prefix, output_prefix)
    run_gemma_gwas(filtered_geno_prefix, kinship_file, cov_file, output_prefix)
    logger.debug('GWAS pipeline completed')
