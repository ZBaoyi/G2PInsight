#!/usr/bin/env python3
import subprocess
import logging
import gzip
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict
logger = logging.getLogger(__name__)
def get_bundled_plink_path() -> str:
    bin_dir = Path(__file__).parent
    software_dir = bin_dir / 'software'
    plink_path = software_dir / 'plink'
    return str(plink_path)
def detect_input_type(input_path: str) -> Tuple[str, Dict]:
    input_path_abs = str(Path(input_path).absolute())
    path = Path(input_path_abs)
    if path.exists() and path.suffix in ['.vcf', '.vcf.gz']:
        return ('vcf', {'vcf_file': str(path)})
    bed_file = Path(f'{input_path_abs}.bed')
    bim_file = Path(f'{input_path_abs}.bim')
    fam_file = Path(f'{input_path_abs}.fam')
    if bed_file.exists() and bim_file.exists() and fam_file.exists():
        return ('bed', {'bed_prefix': input_path_abs, 'bed_file': str(bed_file), 'bim_file': str(bim_file), 'fam_file': str(fam_file)})
    ped_file = Path(f'{input_path_abs}.ped')
    map_file = Path(f'{input_path_abs}.map')
    if ped_file.exists() and map_file.exists():
        return ('ped', {'ped_prefix': input_path_abs, 'ped_file': str(ped_file), 'map_file': str(map_file)})
    if str(input_path_abs).endswith('.ped'):
        map_candidate = str(input_path_abs).replace('.ped', '.map')
        if Path(map_candidate).exists():
            return ('ped', {'ped_prefix': input_path_abs[:-4], 'ped_file': input_path_abs, 'map_file': map_candidate})
    if str(input_path_abs).endswith('.vcf') or str(input_path_abs).endswith('.vcf.gz'):
        if Path(input_path_abs).exists():
            return ('vcf', {'vcf_file': input_path_abs})
    return ('unknown', {})
def build_plink_input_args(input_type: str, input_info: Dict) -> List[str]:
    if input_type == 'vcf':
        vcf_file = input_info['vcf_file']
        return ['--vcf', vcf_file]
    elif input_type == 'bed':
        return ['--bfile', input_info['bed_prefix']]
    elif input_type == 'ped':
        return ['--file', input_info['ped_prefix']]
    else:
        raise ValueError(f'Unsupported input format: {input_type}')
def validate_input_files(input_type: str, input_info: Dict) -> bool:
    try:
        if input_type == 'vcf':
            vcf_file = Path(input_info['vcf_file'])
            if not vcf_file.exists():
                logger.error(f'VCF file does not exist: {vcf_file}')
                return False
            try:
                if vcf_file.suffix == '.gz':
                    with gzip.open(vcf_file, 'rt') as f:
                        header = f.readline()
                else:
                    with open(vcf_file, 'r') as f:
                        header = f.readline()
                if not header.startswith('#'):
                    logger.error(f'Invalid VCF file format: {vcf_file}')
                    return False
            except Exception as e:
                logger.error(f'Cannot read VCF file: {e}')
                return False
            return True
        elif input_type == 'bed':
            bed_prefix = input_info['bed_prefix']
            for ext in ['.bed', '.bim', '.fam']:
                file_path = Path(f'{bed_prefix}{ext}')
                if not file_path.exists():
                    logger.error(f'PLINK binary file missing: {file_path}')
                    return False
            return True
        elif input_type == 'ped':
            ped_prefix = input_info['ped_prefix']
            for ext in ['.ped', '.map']:
                file_path = Path(f'{ped_prefix}{ext}')
                if not file_path.exists():
                    logger.error(f'PLINK text file missing: {file_path}')
                    return False
            return True
        else:
            return False
    except Exception as e:
        logger.error(f'File validation failed: {e}')
        return False
def run_plink_command(plink_path: str, args: list, step_name: str) -> bool:
    cmd = [plink_path] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f'{step_name} failed: {e.stderr}')
        return False
    except Exception as e:
        logger.error(f'{step_name} failed: {e}')
        return False
def cleanup_intermediate_files(output_prefix: str, input_type: str) -> None:
    patterns_to_delete = [f'{output_prefix}.prune.in', f'{output_prefix}.prune.out', f'{output_prefix}.log', f'{output_prefix}_filtered.log', f'{output_prefix}_filtered.nosex']
    if input_type == 'vcf':
        patterns_to_delete.extend([f'{output_prefix}_filtered.bed', f'{output_prefix}_filtered.bim', f'{output_prefix}_filtered.fam'])
    if input_type == 'ped':
        patterns_to_delete.extend([f'{output_prefix}_filtered.bed', f'{output_prefix}_filtered.bim', f'{output_prefix}_filtered.fam'])
    for pattern in patterns_to_delete:
        file_path = Path(pattern)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
def check_prune_file_exists(output_prefix: str) -> bool:
    output_prefix_abs = str(Path(output_prefix).absolute())
    prune_file = Path(f'{output_prefix_abs}.prune.in')
    if not prune_file.exists():
        logger.error(f'LD filtering did not generate prune.in file: {prune_file}')
        return False
    try:
        with open(prune_file, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            logger.error(f'prune.in file is empty: {prune_file}')
            return False
        return True
    except Exception as e:
        logger.error(f'Cannot read prune.in file: {e}')
        return False
def run_ld_filtering(input_path: str, output_prefix: str, ld_window_kb: int=50, ld_window: int=5, ld_window_r2: float=0.2, keep_intermediate: bool=False, threads: int=8, keep_samples_file: Optional[str]=None, extract_snps_file: Optional[str]=None) -> int:
    try:
        input_type, input_info = detect_input_type(input_path)
        if input_type == 'unknown':
            logger.error(f'Cannot identify input file format: {input_path}')
            logger.error('Supported formats:')
            logger.error('  1. VCF format: .vcf or .vcf.gz')
            logger.error('  2. PLINK binary format: .bed/.bim/.fam')
            logger.error('  3. PLINK text format: .ped/.map')
            logger.error('')
            logger.error('Common issues:')
            logger.error('  1. Check if file path is correct')
            logger.error('  2. Check if file extension follows the specification')
            logger.error('  3. For PLINK format, ensure all required files exist')
            return 1
        if not validate_input_files(input_type, input_info):
            return 1
        input_args = build_plink_input_args(input_type, input_info)
        output_prefix_abs = str(Path(output_prefix).absolute())
        prune_args = input_args + ['--indep-pairwise', str(ld_window_kb), str(ld_window), str(ld_window_r2), '--allow-extra-chr', '--allow-no-sex', '--threads', str(threads), '--out', output_prefix_abs]
        if keep_samples_file:
            prune_args.extend(['--keep', keep_samples_file])
        if extract_snps_file:
            prune_args.extend(['--extract', extract_snps_file])
        bundled_plink = Path(get_bundled_plink_path())
        if bundled_plink.exists():
            plink_executable = str(bundled_plink)
        else:
            plink_executable = shutil.which('plink') or ''
        if not Path(plink_executable).exists():
            logger.error(f'PLINK executable not found, please check PATH or software directory: {plink_executable}')
            return 1
        if not run_plink_command(plink_executable, prune_args, 'LD filtering'):
            return 1
        if not check_prune_file_exists(output_prefix):
            return 1
        # Prune may use --keep (e.g. train-only LD), but the final bed must keep
        # all input samples so held-out test IDs remain in the training matrix.
        extract_args = input_args + ['--extract', f'{output_prefix_abs}.prune.in', '--make-bed', '--allow-extra-chr', '--allow-no-sex', '--threads', str(threads), '--out', output_prefix_abs]
        if not run_plink_command(plink_executable, extract_args, 'Extracting SNPs to generate bed/bim/fam'):
            return 1
        bed_file = Path(f'{output_prefix_abs}.bed')
        bim_file = Path(f'{output_prefix_abs}.bim')
        fam_file = Path(f'{output_prefix_abs}.fam')
        generated_files = [bed_file, bim_file, fam_file]
        if all((p.exists() for p in generated_files)):
            if not keep_intermediate:
                cleanup_intermediate_files(output_prefix, input_type)
                prune_in_file = Path(f'{output_prefix_abs}.prune.in')
                if prune_in_file.exists():
                    try:
                        prune_in_file.unlink()
                    except Exception:
                        pass
            return 0
        logger.error('PLINK binary files not generated, please check PLINK output logs.')
        return 1
    except Exception as e:
        logger.error(f'LD filtering failed: {e}', exc_info=True)
        return 1
