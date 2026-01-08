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
    path = Path(input_path)
    
    # 情况1: VCF文件
    if path.exists() and path.suffix in ['.vcf', '.vcf.gz']:
            return "vcf", {"vcf_file": str(path)}
    
    # 情况2: 检查PLINK二进制格式
    bed_file = Path(f"{input_path}.bed")
    bim_file = Path(f"{input_path}.bim")
    fam_file = Path(f"{input_path}.fam")
    
    if bed_file.exists() and bim_file.exists() and fam_file.exists():
        return "bed", {
            "bed_prefix": input_path,
            "bed_file": str(bed_file),
            "bim_file": str(bim_file),
            "fam_file": str(fam_file)
        }
    
    # 情况3: 检查PLINK文本格式
    ped_file = Path(f"{input_path}.ped")
    map_file = Path(f"{input_path}.map")
    
    if ped_file.exists() and map_file.exists():
        return "ped", {
            "ped_prefix": input_path,
            "ped_file": str(ped_file),
            "map_file": str(map_file)
        }
    
    # 情况4: 直接提供文件路径的情况
    if str(input_path).endswith('.ped'):
        map_candidate = str(input_path).replace('.ped', '.map')
        if Path(map_candidate).exists():
            return "ped", {
                "ped_prefix": input_path[:-4],
                "ped_file": input_path,
                "map_file": map_candidate
            }
    
    # 情况5: 直接提供VCF文件路径但后缀检查失败的情况
    if str(input_path).endswith('.vcf') or str(input_path).endswith('.vcf.gz'):
        if Path(input_path).exists():
            return "vcf", {"vcf_file": input_path}
    
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
        raise ValueError(f"不支持的输入格式: {input_type}")

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
                logger.error(f"VCF文件不存在: {vcf_file}")
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
                    logger.error(f"VCF文件格式不正确: {vcf_file}")
                    return False
            except Exception as e:
                logger.error(f"无法读取VCF文件: {e}")
                return False
            
            return True
        
        elif input_type == "bed":
            # 检查所有必需文件
            for ext in ['.bed', '.bim', '.fam']:
                file_path = Path(f"{input_info['bed_prefix']}{ext}")
                if not file_path.exists():
                    logger.error(f"PLINK二进制文件缺失: {file_path}")
                    return False
            
            # 检查文件大小
            bed_file = Path(f"{input_info['bed_prefix']}.bed")
            if bed_file.stat().st_size < 100:
                logger.warning(f"BED文件可能过小: {bed_file}")
            
            return True
        
        elif input_type == "ped":
            # 检查所有必需文件
            for ext in ['.ped', '.map']:
                file_path = Path(f"{input_info['ped_prefix']}{ext}")
                if not file_path.exists():
                    logger.error(f"PLINK文本文件缺失: {file_path}")
                    return False
            
            # 检查PED文件是否有内容
            ped_file = Path(f"{input_info['ped_prefix']}.ped")
            if ped_file.stat().st_size < 100:
                logger.warning(f"PED文件可能过小: {ped_file}")
            
            return True
        
        else:
            return False
            
    except Exception as e:
        logger.error(f"文件验证失败: {e}")
        return False

def run_plink_command(plink_path: str, args: list, step_name: str) -> bool:
    """运行PLINK命令"""
    cmd = [plink_path] + args
    logger.info(f"执行: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"{step_name} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{step_name} 失败: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"{step_name} 失败: {e}")
        return False

def cleanup_intermediate_files(output_prefix: str, input_type: str, input_info: Dict) -> None:
    """
    清理所有中间文件，只保留最终的VCF文件
    
    Args:
        output_prefix: 输出文件前缀
        input_type: 输入文件类型
        input_info: 输入文件信息
    """
    # 基础中间文件
    # 注意：不删除 prune.in 文件，因为 modeltraining 模块需要它来读取 LD 过滤后的 SNP 列表
    patterns_to_delete = [
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
                logger.warning(f"无法删除文件 {pattern}: {e}")
    
    if deleted_files:
        logger.info(f"已清理中间文件: {', '.join(deleted_files)}")

def check_prune_file_exists(output_prefix: str) -> bool:
    """
    检查LD过滤是否生成了有效的prune.in文件
    Args:
        output_prefix: 输出文件前缀
    Returns:
        True如果prune.in文件存在且有内容
    """
    prune_file = Path(f"{output_prefix}.prune.in")
    
    if not prune_file.exists():
        logger.error(f"LD过滤未生成prune.in文件: {prune_file}")
        return False
    
    # 检查文件是否有内容
    try:
        with open(prune_file, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            logger.error(f"prune.in文件为空: {prune_file}")
            return False
        logger.info(f"LD过滤保留了 {len(lines)} 个SNP")
        return True
    except Exception as e:
        logger.error(f"无法读取prune.in文件: {e}")
        return False

def run_ld_filtering(
    input_path: str,
    output_prefix: str,
    ld_window_kb: int = 50,
    ld_window: int = 5,
    ld_window_r2: float = 0.2,
    plink_path: Optional[str] = None,
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
        plink_path: PLINK可执行文件路径
        keep_intermediate: 是否保留中间文件（用于调试）
        threads: 使用的线程数
        keep_samples_file: 样本列表文件（可选，仅对指定样本进行LD过滤）
        extract_snps_file: SNP列表文件（可选，仅对指定SNP进行LD过滤，用于模式4：先GWAS后LD）
    Returns:
        0表示成功，1表示失败
    """
    try:
        # 1. 检测输入文件类型
        logger.info(f"检测输入文件类型: {input_path}")
        input_type, input_info = detect_input_type(input_path)
        
        if input_type == "unknown":
            logger.error(f"无法识别输入文件格式: {input_path}")
            logger.error("支持的格式：")
            logger.error("  1. VCF格式: .vcf 或 .vcf.gz")
            logger.error("  2. PLINK二进制格式: .bed/.bim/.fam")
            logger.error("  3. PLINK文本格式: .ped/.map")
            logger.error("")
            logger.error("常见问题：")
            logger.error("  1. 检查文件路径是否正确")
            logger.error("  2. 检查文件扩展名是否符合规范")
            logger.error("  3. 对于PLINK格式，确保所有必需文件都存在")
            return 1
        
        logger.info(f"检测到输入格式: {input_type}")
        logger.info(f"输入文件信息: {input_info}")
        
        # 2. 验证输入文件
        logger.info("验证输入文件...")
        if not validate_input_files(input_type, input_info):
            return 1
        
        # 4. 构建输入参数
        input_args = build_plink_input_args(input_type, input_info)
        
        # 5. 步骤1: LD pruning
        logger.info("开始LD过滤...")
        logger.info(f"LD参数: 窗口大小={ld_window_kb}KB, 变体数={ld_window}, r²阈值={ld_window_r2}")
        
        prune_args = input_args + [
            "--indep-pairwise", 
            str(ld_window_kb), 
            str(ld_window), 
            str(ld_window_r2),
            "--threads", str(threads),
            "--out", output_prefix
        ]
        # 如果提供了样本子集文件，只对指定样本进行LD过滤
        if keep_samples_file:
            prune_args.extend(["--keep", keep_samples_file])
        # 如果提供了SNP列表文件，只对指定SNP进行LD过滤
        if extract_snps_file:
            prune_args.extend(["--extract", extract_snps_file])
            logger.info(f"仅对指定SNP列表进行LD过滤: {extract_snps_file}")
        
        # 确定PLINK可执行路径：优先系统PATH，其次内置软件目录
        plink_executable = shutil.which("plink") or get_bundled_plink_path()
        if not Path(plink_executable).exists():
            logger.error(f"未找到PLINK可执行文件，请检查PATH或软件目录: {plink_executable}")
            return 1
        
        if not run_plink_command(plink_executable, prune_args, "LD过滤"):
            return 1
        
        # 6. 检查prune.in文件
        if not check_prune_file_exists(output_prefix):
            return 1
        
        # 7. 步骤2: 提取过滤后的SNP并输出PLINK二进制格式（供GWAS使用）
        logger.info("提取过滤后的SNP并生成PLINK二进制文件...")
        extract_args = input_args + [
            "--extract", f"{output_prefix}.prune.in",
            "--make-bed",
            "--threads", str(threads),
            "--out", output_prefix
        ]
        # 保持与LD过滤阶段一致的样本子集
        if keep_samples_file:
            extract_args.extend(["--keep", keep_samples_file])
        # 注意：extract阶段不需要再指定extract_snps_file，因为prune.in已经包含了过滤后的SNP
        
        if not run_plink_command(plink_executable, extract_args, "提取SNP生成bed/bim/fam"):
            return 1
        
        # 8. 验证PLINK二进制文件是否生成
        bed_file = Path(f"{output_prefix}.bed")
        bim_file = Path(f"{output_prefix}.bim")
        fam_file = Path(f"{output_prefix}.fam")
        generated_files = [bed_file, bim_file, fam_file]
        if all(p.exists() for p in generated_files):
            logger.info(
                "LD过滤完成! 生成PLINK二进制文件: %s, %s, %s",
                bed_file, bim_file, fam_file
            )
            logger.info(f"BED文件大小: {bed_file.stat().st_size / (1024*1024):.2f} MB")
            
            # 9. 清理中间文件（保留prune.in供上层读取）
            if not keep_intermediate:
                cleanup_intermediate_files(output_prefix, input_type, input_info)
            else:
                logger.info("保留所有中间文件（调试模式）")
            
            return 0
        else:
            logger.error("未生成PLINK二进制文件，请检查PLINK输出日志。") 
            return 1
        
    except Exception as e:
        logger.error(f"LD过滤失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
if __name__ == "__main__":
    logger.error("该模块不可独立运行，请通过model train模块调用！")
    sys.exit(1)