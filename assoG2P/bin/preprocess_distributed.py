#!/usr/bin/env python3
"""
分布式预处理模块 - 基于Dask解决内存溢出问题
可以作为preprocess.py的扩展模块使用
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入Dask相关库
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, as_completed
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask未安装，分布式功能不可用。安装：pip install dask[complete] distributed")

try:
    import pyarrow
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("PyArrow未安装，Parquet格式不可用。安装：pip install pyarrow")


class DistributedPreprocessor:
    """分布式预处理器 - 解决内存溢出问题"""
    
    def __init__(
        self,
        scheduler_url: Optional[str] = None,
        n_workers: int = 4,
        threads_per_worker: int = 1,
        memory_limit: str = "8GB",
        storage_backend: str = "local"
    ):
        """
        初始化分布式预处理器
        
        :param scheduler_url: Dask调度器地址（None表示本地集群）
        :param n_workers: 工作节点数
        :param threads_per_worker: 每个工作节点的线程数
        :param memory_limit: 每个工作节点的内存限制
        :param storage_backend: 存储后端（local/hdfs/s3）
        """
        if not DASK_AVAILABLE:
            raise ImportError("Dask未安装，无法使用分布式功能")
        
        self.storage_backend = storage_backend
        
        # 初始化Dask客户端
        if scheduler_url:
            self.client = Client(scheduler_url)
            logger.info(f"连接到Dask集群: {scheduler_url}")
        else:
            # 创建本地集群
            self.client = Client(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit
            )
            logger.info(f"创建本地Dask集群")
        
        logger.info(f"   工作节点数: {n_workers}")
        logger.info(f"   每个节点内存限制: {memory_limit}")
        logger.info(f"   集群信息: {self.client}")
    
    def process_chromosome_distributed(
        self,
        chr_num: str,
        raw_file: str,
        pheno_file: str,
        snp_mapping: Dict[str, str],
        output_dir: str,
        chunk_size_mb: int = 100
    ) -> str:
        """
        分布式处理单个染色体
        
        :param chr_num: 染色体编号
        :param raw_file: PLINK .raw文件路径
        :param pheno_file: 表型文件路径
        :param snp_mapping: SNP名称映射字典
        :param output_dir: 输出目录
        :param chunk_size_mb: 分块大小（MB）
        :return: 输出文件路径
        """
        output_file = f"{output_dir}/chr{chr_num}_processed.parquet"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 开始分布式处理染色体{chr_num}...")
        
        # 使用Dask DataFrame分块读取.raw文件
        # blocksize以字节为单位
        blocksize = chunk_size_mb * 1024 * 1024
        
        try:
            ddf = dd.read_csv(
                raw_file,
                sep=r'\s+',
                dtype=str,
                blocksize=blocksize,
                engine='python',
                assume_missing=True
            )
            
            logger.info(f"   读取完成，分区数: {ddf.npartitions}")
            
            # 分布式处理：重命名SNP列
            if snp_mapping:
                # 重命名列（分布式执行）
                new_columns = [snp_mapping.get(col, col) for col in ddf.columns]
                ddf.columns = new_columns
            
            # 清理列
            drop_cols = ["FID", "PAT", "MAT", "SEX", "PHENOTYPE"]
            for col in drop_cols:
                if col in ddf.columns:
                    ddf = ddf.drop(columns=[col])
            
            # 重命名IID为sample
            if "IID" in ddf.columns:
                ddf = ddf.rename(columns={"IID": "sample"})
            
            # 设置sample为索引
            ddf = ddf.set_index("sample")
            
            # 加载表型数据（小文件，单机加载）
            pheno_df = pd.read_csv(
                pheno_file,
                sep="\t",
                index_col=0,
                dtype={"phenotype": float}
            )
            
            # 转换为Dask DataFrame
            pheno_ddf = dd.from_pandas(pheno_df, npartitions=1)
            
            # 分布式合并表型
            ddf = ddf.merge(
                pheno_ddf,
                left_index=True,
                right_index=True,
                how="inner"
            )
            
            # 填充缺失值
            ddf = ddf.fillna(-1)
            
            # 转换SNP列为int8（分布式执行）
            snp_cols = [col for col in ddf.columns if col != "phenotype"]
            if snp_cols:
                # 分批转换，避免内存峰值
                for i in range(0, len(snp_cols), 1000):
                    batch_cols = snp_cols[i:i+1000]
                    for col in batch_cols:
                        ddf[col] = dd.to_numeric(ddf[col], errors='coerce').astype('int8')
            
            # 保存为Parquet格式（列式存储，支持分布式）
            if PARQUET_AVAILABLE:
                ddf.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy',
                    write_index=True
                )
            else:
                # 回退到CSV（分块保存）
                ddf.to_csv(
                    output_file.replace('.parquet', '.csv'),
                    sep='\t',
                    single_file=False,
                    index=True
                )
            
            logger.info(f"染色体{chr_num}处理完成: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"染色体{chr_num}处理失败: {str(e)}", exc_info=True)
            raise
    
    def merge_chromosomes_distributed(
        self,
        chr_files: List[str],
        output_file: str,
        merge_batch_size: int = 3
    ) -> str:
        """
        分布式合并所有染色体结果
        
        :param chr_files: 染色体文件列表（Parquet格式）
        :param output_file: 输出文件路径
        :param merge_batch_size: 合并批次大小
        :return: 输出文件路径
        """
        logger.info(f"开始分布式合并{len(chr_files)}个染色体...")
        
        if not chr_files:
            raise ValueError("没有染色体文件需要合并")
        
        # 使用Dask DataFrame读取所有染色体文件
        ddfs = []
        for chr_file in chr_files:
            if Path(chr_file).exists():
                if PARQUET_AVAILABLE and chr_file.endswith('.parquet'):
                    ddf = dd.read_parquet(chr_file, engine='pyarrow')
                else:
                    # CSV格式
                    ddf = dd.read_csv(chr_file, sep='\t', index_col=0)
                ddfs.append(ddf)
                logger.info(f"   已加载: {Path(chr_file).name}")
        
        if not ddfs:
            raise ValueError("没有有效的染色体文件")
        
        # 分批合并，避免内存峰值
        logger.info(f"   分批合并，批次大小: {merge_batch_size}")
        
        final_ddf = ddfs[0]
        for i in range(1, len(ddfs), merge_batch_size):
            batch_ddfs = ddfs[i:i+merge_batch_size]
            batch_merged = dd.concat([final_ddf] + batch_ddfs, axis=1, join='inner')
            
            # 释放中间结果
            del final_ddf
            final_ddf = batch_merged
            
            logger.info(f"   已合并 {min(i+merge_batch_size, len(ddfs))}/{len(ddfs)} 个染色体")
        
        # 估算最终文件大小
        estimated_size_gb = self._estimate_size(final_ddf)
        logger.info(f"   估算最终文件大小: {estimated_size_gb:.2f}GB")
        
        # 保存最终结果
        if estimated_size_gb > 50:  # > 50GB，分块保存
            logger.info("   文件较大，使用分块保存...")
            output_dir = Path(output_file).parent
            output_base = Path(output_file).stem
            final_ddf.to_csv(
                output_file,
                sep='\t',
                single_file=False,  # 保存为多个文件
                index=True
            )
        else:
            # 单文件保存
            if PARQUET_AVAILABLE and output_file.endswith('.parquet'):
                final_ddf.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy',
                    write_index=True
                )
            else:
                final_ddf.to_csv(
                    output_file,
                    sep='\t',
                    single_file=True,
                    index=True
                )
        
        logger.info(f"合并完成: {output_file}")
        return output_file
    
    def _estimate_size(self, ddf: dd.DataFrame) -> float:
        """估算DataFrame大小（GB）"""
        try:
            # 计算内存使用量
            memory_usage = ddf.memory_usage(deep=True).sum().compute()
            return memory_usage / (1024**3)
        except:
            # 如果计算失败，返回估算值
            return ddf.npartitions * 1.0  # 粗略估算：每个分区1GB
    
    def close(self):
        """关闭Dask客户端"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("Dask客户端已关闭")


def run_distributed_preprocess(
    genotype_file: str,
    phenotype_file: str,
    output_file: str,
    scheduler_url: Optional[str] = None,
    n_workers: int = 4,
    memory_limit: str = "8GB",
    chunk_size_mb: int = 100
) -> int:
    """
    分布式预处理主函数
    
    这是preprocess.py的分布式版本，可以处理超出单机内存的数据集
    
    :param genotype_file: 基因型文件
    :param phenotype_file: 表型文件
    :param output_file: 输出文件
    :param scheduler_url: Dask调度器地址（None表示本地集群）
    :param n_workers: 工作节点数
    :param memory_limit: 每个节点的内存限制
    :param chunk_size_mb: 分块大小（MB）
    :return: 退出码
    """
    if not DASK_AVAILABLE:
        logger.error("Dask未安装，无法使用分布式功能")
        logger.error("   安装命令: pip install dask[complete] distributed")
        return 1
    
    try:
        # 导入原有preprocess模块的函数
        from assoG2P.bin.preprocess import (
            detect_genotype_format,
            genotype_to_plink,
            load_phenotype,
            filter_samples_by_phenotype,
            filter_snps_by_quality,
            auto_detect_chromosomes,
            get_snp_chr_pos_mapping,
            delete_temp_files
        )
        
        # 初始化分布式预处理器
        preprocessor = DistributedPreprocessor(
            scheduler_url=scheduler_url,
            n_workers=n_workers,
            memory_limit=memory_limit
        )
        
        output_prefix = Path(output_file).absolute().stem
        output_dir = Path(output_file).parent
        temp_plink_prefix = f"{output_prefix}_temp_plink"
        
        # 步骤1：基因型格式转换（单机执行，数据量小）
        logger.info("步骤1: 基因型格式转换...")
        genotype_format = detect_genotype_format(genotype_file)
        plink_prefix = genotype_to_plink(genotype_file, temp_plink_prefix, threads=1)
        
        # 步骤2：加载表型数据
        logger.info("步骤2: 加载表型数据...")
        pheno_df = load_phenotype(phenotype_file, None)
        
        # 保存表型到临时文件（供分布式处理使用）
        temp_pheno_file = f"{output_prefix}_temp_pheno.tsv"
        pheno_df.set_index("sample").to_csv(temp_pheno_file, sep="\t")
        
        # 步骤3：样本过滤
        logger.info("步骤3: 样本过滤...")
        sample_filtered_prefix = f"{output_prefix}_sample_filtered_plink"
        sample_filtered_plink = filter_samples_by_phenotype(
            plink_prefix, pheno_df, sample_filtered_prefix
        )
        
        # 步骤4：SNP质量过滤
        logger.info("步骤4: SNP质量过滤...")
        filtered_plink_prefix = f"{output_prefix}_filtered_plink"
        filtered_plink = filter_snps_by_quality(
            sample_filtered_plink,
            filtered_plink_prefix,
            maf=0.05,
            geno=0.2,
            threads=1
        )
        
        # 清理中间文件
        delete_temp_files(sample_filtered_plink, [".bed", ".bim", ".fam", ".log", ".nosex"])
        
        # 步骤5：识别染色体
        logger.info("步骤5: 识别染色体...")
        chr_list = auto_detect_chromosomes(filtered_plink)
        logger.info(f"   识别到{len(chr_list)}个染色体: {chr_list}")
        
        # 步骤6：分布式处理每个染色体
        logger.info("步骤6: 分布式处理染色体...")
        chr_result_files = []
        chr_output_dir = f"{output_dir}/chr_results"
        Path(chr_output_dir).mkdir(parents=True, exist_ok=True)
        
        for chr_num in chr_list:
            try:
                # 生成PLINK .raw文件（单机执行）
                from assoG2P.bin.preprocess import PLINK_EXECUTABLE
                chr_recode_prefix = f"{filtered_plink}_chr{chr_num}"
                
                import subprocess
                recode_cmd = [
                    PLINK_EXECUTABLE,
                    "--bfile", filtered_plink,
                    "--chr", chr_num,
                    "--recodeA",
                    "--out", chr_recode_prefix,
                    "--allow-extra-chr",
                    "--allow-no-sex",
                    "--threads", "1"
                ]
                
                logger.info(f"   生成{chr_num}号染色体.raw文件...")
                subprocess.run(recode_cmd, capture_output=True, text=True, check=True)
                
                # 获取SNP映射
                chr_bim_file = f"{filtered_plink}.bim"
                snp_mapping = get_snp_chr_pos_mapping(chr_bim_file, chr_num=chr_num)
                
                # 分布式处理染色体
                chr_raw_file = f"{chr_recode_prefix}.raw"
                chr_file = preprocessor.process_chromosome_distributed(
                    chr_num=chr_num,
                    raw_file=chr_raw_file,
                    pheno_file=temp_pheno_file,
                    snp_mapping=snp_mapping,
                    output_dir=chr_output_dir,
                    chunk_size_mb=chunk_size_mb
                )
                chr_result_files.append(chr_file)
                
                # 删除临时.raw文件
                delete_temp_files(chr_recode_prefix, [".raw", ".log", ".nosex"])
                
            except Exception as e:
                logger.error(f"染色体{chr_num}处理失败: {str(e)}")
                continue
        
        # 步骤7：分布式合并
        logger.info("步骤7: 分布式合并染色体结果...")
        preprocessor.merge_chromosomes_distributed(chr_result_files, output_file)
        
        # 步骤8：生成元数据（使用原有函数）
        from assoG2P.bin.preprocess import generate_metadata
        generate_metadata(
            output_prefix=output_prefix,
            plink_prefix=filtered_plink,
            train_file=output_file,
            valid_samples=pheno_df["sample"].tolist(),
            genotype_format=genotype_format,
            threads=n_workers
        )
        
        # 清理
        preprocessor.close()
        if Path(temp_pheno_file).exists():
            os.remove(temp_pheno_file)
        
        logger.info("分布式预处理完成！")
        return 0
        
    except Exception as e:
        logger.error(f"分布式预处理失败: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分布式预处理模块（基于Dask）")
    parser.add_argument("-g", "--genotype", required=True, help="基因型文件")
    parser.add_argument("-p", "--phenotype", required=True, help="表型文件")
    parser.add_argument("-o", "--output", required=True, help="输出文件")
    parser.add_argument("--scheduler-url", help="Dask调度器地址（默认：本地集群）")
    parser.add_argument("--n-workers", type=int, default=4, help="工作节点数（默认：4）")
    parser.add_argument("--memory-limit", default="8GB", help="每个节点内存限制（默认：8GB）")
    parser.add_argument("--chunk-size-mb", type=int, default=100, help="分块大小MB（默认：100）")
    
    args = parser.parse_args()
    
    sys.exit(run_distributed_preprocess(
        genotype_file=args.genotype,
        phenotype_file=args.phenotype,
        output_file=args.output,
        scheduler_url=args.scheduler_url,
        n_workers=args.n_workers,
        memory_limit=args.memory_limit,
        chunk_size_mb=args.chunk_size_mb
    ))

