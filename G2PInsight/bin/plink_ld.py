#!/usr/bin/env python3
"""
PLINK LD过滤模块 - 支持多种输入格式的LD过滤并输出VCF格式
支持的输入格式：
1. VCF格式 (.vcf, .vcf.gz)
2. PLINK二进制格式 (.bed/.bim/.fam)
3. PLINK文本格式 (.ped/.map)
"""

import subprocess
import logging
import sys
import os
import gzip
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)

def get_bundled_plink_path() -> str:
    """获取集成的PLINK软件路径"""
    bin_dir = Path(__file__).parent
    software_dir = bin_dir / "software"
    
    plink_path = software_dir / "plink"
    
    return str(plink_path)

def detect_input_type(input_path: str) -> Tuple[str, Dict]:
    """
    自动检测输入文件类型
    
    Args:
        input_path: 输入文件路径（可以是前缀或完整路径）
    
    Returns:
        (文件类型, 参数字典)
        文件类型: "vcf", "ped", "bed", "unknown"
    """
    # 确保使用绝对路径进行检测
    input_path_abs = str(Path(input_path).absolute())
    path = Path(input_path_abs)
    
    # 情况1: VCF文件
    if path.exists() and path.suffix in ['.vcf', '.vcf.gz']:
            return "vcf", {"vcf_file": str(path)}
    
    # 情况2: 检查PLINK二进制格式
    bed_file = Path(f"{input_path_abs}.bed")
    bim_file = Path(f"{input_path_abs}.bim")
    fam_file = Path(f"{input_path_abs}.fam")
    
    if bed_file.exists() and bim_file.exists() and fam_file.exists():
        return "bed", {
            "bed_prefix": input_path_abs,
            "bed_file": str(bed_file),
            "bim_file": str(bim_file),
            "fam_file": str(fam_file)
        }
    
    # 情况3: 检查PLINK文本格式
    ped_file = Path(f"{input_path_abs}.ped")
    map_file = Path(f"{input_path_abs}.map")
    
    if ped_file.exists() and map_file.exists():
        return "ped", {
            "ped_prefix": input_path_abs,
            "ped_file": str(ped_file),
            "map_file": str(map_file)
        }
    
    # 情况4: 直接提供文件路径的情况
    if str(input_path_abs).endswith('.ped'):
        map_candidate = str(input_path_abs).replace('.ped', '.map')
        if Path(map_candidate).exists():
            return "ped", {
                "ped_prefix": input_path_abs[:-4],
                "ped_file": input_path_abs,
                "map_file": map_candidate
            }
    
    # 情况5: 直接提供VCF文件路径但后缀检查失败的情况
    if str(input_path_abs).endswith('.vcf') or str(input_path_abs).endswith('.vcf.gz'):
        if Path(input_path_abs).exists():
            return "vcf", {"vcf_file": input_path_abs}
    
    return "unknown", {}

def build_plink_input_args(input_type: str, input_info: Dict) -> List[str]:
    """
    根据输入类型构建PLINK输入参数
    
    Args:
        input_type: 输入文件类型
        input_info: 检测到的文件信息
    
    Returns:
        PLINK命令行参数列表
    """
    if input_type == "vcf":
        vcf_file = input_info["vcf_file"]
        return ["--vcf", vcf_file]
    
    elif input_type == "bed":
        return ["--bfile", input_info["bed_prefix"]]
    
    elif input_type == "ped":
        return ["--file", input_info["ped_prefix"]]
    
    else:
        raise ValueError(f"Unsupported input format: {input_type}")

def validate_input_files(input_type: str, input_info: Dict) -> bool:
    """
    验证输入文件的完整性和可用性
    
    Args:
        input_type: 输入文件类型
        input_info: 输入文件信息
    
    Returns:
        True如果文件有效，False否则
    """
    try:
        if input_type == "vcf":
            vcf_file = Path(input_info["vcf_file"])
            if not vcf_file.exists():
                logger.error(f"VCF file does not exist: {vcf_file}")
                return False
            
            # 检查VCF文件格式
            try:
                if vcf_file.suffix == '.gz':
                    with gzip.open(vcf_file, 'rt') as f:
                        header = f.readline()
                else:
                    with open(vcf_file, 'r') as f:
                        header = f.readline()
                
                if not header.startswith('#'):
                    logger.error(f"Invalid VCF file format: {vcf_file}")
                    return False
            except Exception as e:
                logger.error(f"Cannot read VCF file: {e}")
                return False
            
            return True
        
        elif input_type == "bed":
            # 检查所有必需文件
            for ext in ['.bed', '.bim', '.fam']:
                file_path = Path(f"{input_info['bed_prefix']}{ext}")
                if not file_path.exists():
                    logger.error(f"PLINK binary file missing: {file_path}")
                    return False
            
            # 检查文件大小
            bed_file = Path(f"{input_info['bed_prefix']}.bed")
            if bed_file.stat().st_size < 100:
                logger.warning(f"BED file may be too small: {bed_file}")
            
            return True
        
        elif input_type == "ped":
            # 检查所有必需文件
            for ext in ['.ped', '.map']:
                file_path = Path(f"{input_info['ped_prefix']}{ext}")
                if not file_path.exists():
                    logger.error(f"PLINK text file missing: {file_path}")
                    return False
            
            # 检查PED文件是否有内容
            ped_file = Path(f"{input_info['ped_prefix']}.ped")
            if ped_file.stat().st_size < 100:
                logger.warning(f"PED file may be too small: {ped_file}")
            
            return True
        
        else:
            return False
            
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        return False

def run_plink_command(plink_path: str, args: list, step_name: str) -> bool:
    """运行PLINK命令"""
    cmd = [plink_path] + args
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"{step_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{step_name} failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"{step_name} failed: {e}")
        return False

def cleanup_intermediate_files(output_prefix: str, input_type: str) -> None:
    """
    清理所有中间文件（包括prune.in，因为所有临时文件都不保留）
    
    Args:
        output_prefix: 输出文件前缀
        input_type: 输入文件类型
    """
    # 基础中间文件（包括prune.in，所有临时文件都不保留）
    patterns_to_delete = [
        f"{output_prefix}.prune.in",  # 删除prune.in文件
        f"{output_prefix}.prune.out", 
        f"{output_prefix}.log",
        f"{output_prefix}_filtered.log",
        f"{output_prefix}_filtered.nosex"
    ]
    
    # 对于VCF输入，PLINK可能会生成额外的二进制文件
    if input_type == "vcf":
        patterns_to_delete.extend([
            f"{output_prefix}_filtered.bed",
            f"{output_prefix}_filtered.bim",
            f"{output_prefix}_filtered.fam"
        ])
    
    # 对于文本PLINK输入，可能会生成二进制中间文件
    if input_type == "ped":
        patterns_to_delete.extend([
            f"{output_prefix}_filtered.bed",
            f"{output_prefix}_filtered.bim",
            f"{output_prefix}_filtered.fam"
        ])
    
    deleted_files = []
    for pattern in patterns_to_delete:
        file_path = Path(pattern)
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_files.append(pattern)
            except Exception as e:
                logger.warning(f"Cannot delete file {pattern}: {e}")
    
    if deleted_files:
        logger.info(f"Cleaned up intermediate files: {', '.join(deleted_files)}")

def check_prune_file_exists(output_prefix: str) -> bool:
    """
    检查LD过滤是否生成了有效的prune.in文件
    Args:
        output_prefix: 输出文件前缀（可以是相对或绝对路径）
    Returns:
        True如果prune.in文件存在且有内容
    """
    # 确保使用绝对路径
    output_prefix_abs = str(Path(output_prefix).absolute())
    prune_file = Path(f"{output_prefix_abs}.prune.in")
    
    if not prune_file.exists():
        logger.error(f"LD filtering did not generate prune.in file: {prune_file}")
        return False
    
    # 检查文件是否有内容
    try:
        with open(prune_file, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            logger.error(f"prune.in file is empty: {prune_file}")
            return False
        logger.info(f"LD filtering retained {len(lines):,} SNPs")
        return True
    except Exception as e:
        logger.error(f"Cannot read prune.in file: {e}")
        return False

def run_ld_filtering(
    input_path: str,
    output_prefix: str,
    ld_window_kb: int = 50,
    ld_window: int = 5,
    ld_window_r2: float = 0.2,
    keep_intermediate: bool = False,
    threads: int = 8,
    keep_samples_file: Optional[str] = None,
    extract_snps_file: Optional[str] = None
) -> int:
    """
    使用PLINK进行LD过滤并输出VCF格式，支持多种输入格式
    Args:
        input_path: 输入文件路径或前缀
        output_prefix: 输出文件前缀
        ld_window_kb: LD窗口大小（KB）
        ld_window: LD窗口大小（变体数）
        ld_window_r2: LD r²阈值
        keep_intermediate: 是否保留中间文件（用于调试）
        threads: 使用的线程数
        keep_samples_file: 样本列表文件（可选，仅对指定样本进行LD过滤）
        extract_snps_file: SNP列表文件（可选，仅对指定SNP进行LD过滤，用于模式4：先GWAS后LD）
    Returns:
        0表示成功，1表示失败
    """
    try:
        # 1. 检测输入文件类型
        logger.info(f"Detecting input file type: {input_path}")
        input_type, input_info = detect_input_type(input_path)
        
        if input_type == "unknown":
            logger.error(f"Cannot identify input file format: {input_path}")
            logger.error("Supported formats:")
            logger.error("  1. VCF format: .vcf or .vcf.gz")
            logger.error("  2. PLINK binary format: .bed/.bim/.fam")
            logger.error("  3. PLINK text format: .ped/.map")
            logger.error("")
            logger.error("Common issues:")
            logger.error("  1. Check if file path is correct")
            logger.error("  2. Check if file extension follows the specification")
            logger.error("  3. For PLINK format, ensure all required files exist")
            return 1
        
        logger.info(f"Detected input format: {input_type}")
        logger.info(f"Input file information: {input_info}")
        
        # 2. 验证输入文件
        logger.info("Validating input files...")
        if not validate_input_files(input_type, input_info):
            return 1
        
        # 4. 构建输入参数
        input_args = build_plink_input_args(input_type, input_info)
        
        # 5. 步骤1: LD pruning
        logger.info("Starting LD filtering...")
        logger.info(f"LD parameters: window size={ld_window_kb}KB, variant count={ld_window}, r² threshold={ld_window_r2}")
        
        # 确保输出前缀是绝对路径
        output_prefix_abs = str(Path(output_prefix).absolute())
        
        prune_args = input_args + [
            "--indep-pairwise", 
            str(ld_window_kb), 
            str(ld_window), 
            str(ld_window_r2),
            "--allow-extra-chr",
            "--allow-no-sex",
            "--threads", str(threads),
            "--out", output_prefix_abs
        ]
        # 如果提供了样本子集文件，只对指定样本进行LD过滤
        if keep_samples_file:
            prune_args.extend(["--keep", keep_samples_file])
        # 如果提供了SNP列表文件，只对指定SNP进行LD过滤
        if extract_snps_file:
            prune_args.extend(["--extract", extract_snps_file])
            logger.info(f"Performing LD filtering only on specified SNP list: {extract_snps_file}")
        
        # 确定PLINK可执行路径：优先项目内集成软件，其次系统PATH（standardized）
        bundled_plink = Path(get_bundled_plink_path())
        if bundled_plink.exists():
            plink_executable = str(bundled_plink)
        else:
            plink_executable = shutil.which("plink") or ""
        if not Path(plink_executable).exists():
            logger.error(f"PLINK executable not found, please check PATH or software directory: {plink_executable}")
            return 1
        
        if not run_plink_command(plink_executable, prune_args, "LD filtering"):
            return 1
        
        # 6. 检查prune.in文件
        if not check_prune_file_exists(output_prefix):
            return 1
        
        # 7. 步骤2: 提取过滤后的SNP并输出PLINK二进制格式（供GWAS使用）
        logger.info("Extracting filtered SNPs and generating PLINK binary files...")
        extract_args = input_args + [
            "--extract", f"{output_prefix_abs}.prune.in",
            "--make-bed",
            "--allow-extra-chr",
            "--allow-no-sex",
            "--threads", str(threads),
            "--out", output_prefix_abs
        ]
        # 保持与LD过滤阶段一致的样本子集
        if keep_samples_file:
            extract_args.extend(["--keep", keep_samples_file])
        # 注意：extract阶段不需要再指定extract_snps_file，因为prune.in已经包含了过滤后的SNP
        
        if not run_plink_command(plink_executable, extract_args, "Extracting SNPs to generate bed/bim/fam"):
            return 1
        
        # 8. 验证PLINK二进制文件是否生成
        bed_file = Path(f"{output_prefix_abs}.bed")
        bim_file = Path(f"{output_prefix_abs}.bim")
        fam_file = Path(f"{output_prefix_abs}.fam")
        generated_files = [bed_file, bim_file, fam_file]
        if all(p.exists() for p in generated_files):
            logger.info(
                "LD filtering completed! Generated PLINK binary files: %s, %s, %s",
                bed_file, bim_file, fam_file
            )
            logger.info(f"BED file size: {bed_file.stat().st_size / (1024*1024):.2f} MB")
            
            # 9. 清理所有中间文件（包括prune.in，因为所有临时文件都不保留）
            if not keep_intermediate:
                cleanup_intermediate_files(output_prefix, input_type)
                # 删除 prune.in 文件（所有临时文件都不保留）
                prune_in_file = Path(f"{output_prefix_abs}.prune.in")
                if prune_in_file.exists():
                    try:
                        prune_in_file.unlink()
                    except Exception as e:
                        pass
            else:
                logger.info("Keeping all intermediate files (debug mode)")
            
            return 0
        else:
            logger.error("PLINK binary files not generated, please check PLINK output logs.") 
            return 1
        
    except Exception as e:
        logger.error(f"LD filtering failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        