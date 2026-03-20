#!/usr/bin/env python3
"""
preprocess模块 - 第一层优化版本
核心优化:
1. 加强向量化计算 - 全面向量化操作，消除显式循环
2. 智能缓存系统 - 带LRU淘汰和文件修改时间检测的缓存
3. PLINK参数调优 - 根据数据规模自动优化PLINK参数
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
import numpy as np
import math
import mmap
import functools
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Optional, Dict, List, Literal, Tuple, Union, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from collections import OrderedDict

# GWAS / LD 相关模块（供预处理阶段调用）
from assoG2P.bin import gemma_gwas, plink_ld

warnings.filterwarnings('ignore')

# ======================== 1. 基础配置和导入 ========================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

# 尝试导入psutil用于内存监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed, memory monitoring functionality limited")

# 尝试导入datatable用于快速读取
try:
    import datatable as dt
    DATATABLE_AVAILABLE = True
except ImportError:
    DATATABLE_AVAILABLE = False

# SNP数据类型常量
SNP_DTYPE = np.int8

# SNP映射缓存（用于get_snp_chr_pos_mapping_cached函数）
_snp_mapping_cache: Dict[str, dict] = {}

# ======================== 2. 向量化计算优化 ========================

class VectorizedProcessor:
    """
    向量化计算优化器
    消除显式循环，使用numpy/pandas向量化操作
    """
    
    @staticmethod
    def batch_rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        批量重命名列（向量化版本）
        消除显式循环，使用pandas的rename方法
        """
        if not mapping:
            return df
        
        # 只重命名存在的列
        existing_cols = [col for col in mapping.keys() if col in df.columns]
        if not existing_cols:
            return df
        
        rename_dict = {col: mapping[col] for col in existing_cols}
        return df.rename(columns=rename_dict)
    
    @staticmethod
    def _convert_chunk(chunk_data: np.ndarray, is_numeric: bool, target_dtype: type) -> np.ndarray:
        """
        转换单个数据块（用于并行处理）
        """
        if is_numeric:
            # 已经是数字类型，直接转换
            return chunk_data.astype(target_dtype)
        else:
            # 需要字符串到数字的转换
            try:
                # 先尝试直接转换（可能已经是数字字符串）
                return chunk_data.astype(float).astype(target_dtype)
            except (ValueError, TypeError):
                # 需要pd.to_numeric处理混合类型
                chunk_converted = pd.to_numeric(chunk_data, errors='coerce').values
                return np.nan_to_num(chunk_converted, nan=-1).astype(target_dtype)
    
    @staticmethod
    def vectorized_type_conversion(df: pd.DataFrame, exclude_cols: List[str] = None, 
                                   target_dtype: type = np.int8, max_workers: int = None) -> pd.DataFrame:
        """
        向量化类型转换
        将除指定列外的所有列转换为目标类型
        
        Args:
            df: 要转换的DataFrame
            exclude_cols: 排除的列名列表
            target_dtype: 目标数据类型
            max_workers: 并行处理的最大线程数（None表示自动，0表示不使用并行）
        """
        if df.empty:
            return df
        
        if exclude_cols is None:
            exclude_cols = ["sample"]
        
        # 确定需要转换的列
        cols_to_convert = [col for col in df.columns if col not in exclude_cols]
        if not cols_to_convert:
            return df
        
        # 获取需要转换的列的数据
        data_to_convert = df[cols_to_convert]
        
        # 向量化转换：使用numpy批量操作，避免逐列处理
        total_cols = len(cols_to_convert)
        arr = data_to_convert.values
        
        # 检查数据类型，如果已经是数字类型，可以直接转换
        arr_dtype = arr.dtype
        is_numeric = np.issubdtype(arr_dtype, np.number)
        
        # 对于所有情况，优先使用批量向量化操作
        try:
            total_elements = arr.size
            if total_elements > 50_000_000:  # 超过5000万元素，分块处理（串行，避免多进程pickling问题）
                chunk_size = 10_000_000  # 每块1000万元素
                total_chunks = (total_elements + chunk_size - 1) // chunk_size
                
                # 准备所有块的数据
                chunk_ranges = []
                for chunk_start in range(0, total_elements, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_elements)
                    chunk_ranges.append((chunk_start, chunk_end))
                
                # 串行处理所有chunk，避免多进程带来的pickling限制
                arr_converted_list = []
                for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                    if chunk_idx % 10 == 0:
                        pass
                    chunk_flat = arr.ravel()[chunk_start:chunk_end]
                    chunk_converted = VectorizedProcessor._convert_chunk(chunk_flat, is_numeric, target_dtype)
                    arr_converted_list.append(chunk_converted)
                
                arr_converted = np.concatenate(arr_converted_list).reshape(arr.shape)
            else:
                # 中小型数组：直接批量转换
                if is_numeric:
                    # 已经是数字类型，直接转换（最快）
                    arr_converted = arr.astype(target_dtype)
                else:
                    # 需要类型转换
                    try:
                        # 尝试直接转换（可能已经是数字字符串）
                        arr_converted = arr.astype(float).astype(target_dtype)
                    except (ValueError, TypeError):
                        # 需要pd.to_numeric处理混合类型
                        arr_flat = pd.to_numeric(arr.ravel(), errors='coerce').values
                        arr_converted = arr_flat.reshape(arr.shape)
                        arr_converted = np.nan_to_num(arr_converted, nan=-1).astype(target_dtype)
            
            # 批量赋值回DataFrame
            df[cols_to_convert] = arr_converted
            
        except Exception as e:
            logger.warning(f"Bulk vectorized conversion failed, falling back to column-wise conversion: {e}")
            # 回退到逐列转换（但使用向量化操作）
            # 对于超大量列，至少分批处理以减少内存压力
            if total_cols > 1000:
                batch_size = 1000
                for i in range(0, total_cols, batch_size):
                    batch_cols = cols_to_convert[i:i+batch_size]
                    if (i // batch_size) % 10 == 0:
                        pass
                    for col in batch_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(target_dtype)
            else:
                for col in cols_to_convert:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(target_dtype)
        
        return df
    
    @staticmethod
    def parse_plink_column_name(col_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析PLINK列名格式：chr:pos_allele -> (chr, pos)
        使用正则表达式向量化解析
        """
        # 预编译正则表达式，提高效率
        pattern = re.compile(r'^([^:]+):([^_]+)_[^_]+$')
        match = pattern.match(str(col_name))
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    @staticmethod
    def batch_parse_plink_columns(col_names: List[str]) -> Dict[str, str]:
        """
        批量解析PLINK列名（向量化版本）
        返回映射字典：原始列名 -> 新列名(chr_pos)
        """
        mapping = {}
        pattern = re.compile(r'^([^:]+):([^_]+)_[^_]+$')
        
        for col in col_names:
            match = pattern.match(str(col))
            if match:
                chr_part = match.group(1)
                pos_part = match.group(2)
                mapping[col] = f"{chr_part}_{pos_part}"
        
        return mapping
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame, categorical_threshold: float = 0.5) -> pd.DataFrame:
        """
        优化DataFrame内存使用（向量化版本）
        - 将高重复率列转换为category类型
        - 将整数列转换为最小类型
        """
        if df.empty:
            return df
        
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_data = optimized_df[col]
            dtype = col_data.dtype
            
            # 跳过已经是category或object的列
            if dtype.name == 'category' or dtype == object:
                continue
            
            # 检查是否可以转换为category
            if len(col_data) > 0:
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < categorical_threshold:
                    optimized_df[col] = col_data.astype('category')
                    continue
            
            # 整数类型优化
            if pd.api.types.is_integer_dtype(dtype):
                col_min = col_data.min()
                col_max = col_data.max()
                
                # 选择最小整数类型
                if col_min >= 0:
                    if col_max <= 255:
                        optimized_df[col] = col_data.astype(np.uint8)
                    elif col_max <= 65535:
                        optimized_df[col] = col_data.astype(np.uint16)
                    elif col_max <= 4294967295:
                        optimized_df[col] = col_data.astype(np.uint32)
                else:
                    if col_min >= -128 and col_max <= 127:
                        optimized_df[col] = col_data.astype(np.int8)
                    elif col_min >= -32768 and col_max <= 32767:
                        optimized_df[col] = col_data.astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        optimized_df[col] = col_data.astype(np.int32)
        
        return optimized_df

# ======================== 3. 智能缓存系统 ========================

class SmartCache:
    """
    智能缓存管理器
    支持LRU淘汰、文件修改时间检测、内存限制
    """
    
    def __init__(self, max_size_mb: int = 512, max_items: int = 1000):
        """
        初始化缓存
        :param max_size_mb: 最大缓存大小（MB）
        :param max_items: 最大缓存项数
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_items = max_items
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
        # 缓存统计
        self.stats = {
            'total_requests': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_mb': 0
        }
    
    def _get_item_size(self, item: Any) -> int:
        """估算对象大小"""
        try:
            if isinstance(item, (pd.DataFrame, pd.Series)):
                return item.memory_usage(deep=True).sum()
            elif isinstance(item, np.ndarray):
                return item.nbytes
            elif isinstance(item, dict):
                # 简单估算字典大小
                return sum(len(str(k)) + len(str(v)) for k, v in item.items()) * 2
            else:
                return sys.getsizeof(item)
        except:
            # 如果无法计算大小，返回保守估计
            return 1024 * 1024  # 1MB
    
    def _evict_if_needed(self, new_item_size: int):
        """如果需要，淘汰最久未使用的项"""
        with self.lock:
            while (self.current_size_bytes + new_item_size > self.max_size_bytes or 
                   len(self.cache) >= self.max_items) and self.cache:
                # LRU淘汰：移除第一个（最久未使用）的项
                key, item = self.cache.popitem(last=False)
                self.current_size_bytes -= item['size']
                self.stats['evictions'] += 1
    
    def get(self, key: str, check_file_mtime: bool = False, file_path: Optional[str] = None) -> Optional[Any]:
        """
        从缓存获取项
        :param key: 缓存键
        :param check_file_mtime: 是否检查文件修改时间
        :param file_path: 关联的文件路径（用于检查修改时间）
        :return: 缓存项或None
        """
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            cache_item = self.cache[key]
            
            # 检查文件修改时间（如果需要）
            if check_file_mtime and file_path:
                try:
                    file_mtime = os.path.getmtime(file_path)
                    cached_mtime = cache_item.get('file_mtime', 0)
                    if file_mtime > cached_mtime:
                        # 文件已修改，删除缓存
                        self._delete(key)
                        self.stats['misses'] += 1
                        return None
                except (OSError, FileNotFoundError) as e:
                    self._delete(key)
                    self.stats['misses'] += 1
                    return None
            
            # 更新访问时间和顺序（移动到末尾）
            self.cache.move_to_end(key)
            cache_item['last_access'] = time.time()
            cache_item['access_count'] = cache_item.get('access_count', 0) + 1
            
            self.stats['hits'] += 1
            return cache_item['data']
    
    def set(self, key: str, data: Any, file_path: Optional[str] = None, 
            ttl_seconds: Optional[int] = None):
        """
        设置缓存项
        :param key: 缓存键
        :param data: 要缓存的数据
        :param file_path: 关联的文件路径（用于记录修改时间）
        :param ttl_seconds: 生存时间（秒）
        """
        with self.lock:
            # 计算数据大小
            data_size = self._get_item_size(data)
            
            # 如果需要，淘汰旧项
            self._evict_if_needed(data_size)
            
            # 获取文件修改时间（如果提供）
            file_mtime = 0
            if file_path:
                try:
                    file_mtime = os.path.getmtime(file_path)
                except (OSError, FileNotFoundError):
                    pass
            
            # 创建缓存项
            cache_item = {
                'data': data,
                'size': data_size,
                'created': time.time(),
                'last_access': time.time(),
                'access_count': 1,
                'file_path': file_path,
                'file_mtime': file_mtime,
                'ttl': ttl_seconds
            }
            
            # 添加到缓存
            self.cache[key] = cache_item
            self.current_size_bytes += data_size
            
            # 更新统计
            self.stats['size_mb'] = self.current_size_bytes / (1024 * 1024)
    
    def _delete(self, key: str):
        """删除缓存项"""
        if key in self.cache:
            item = self.cache.pop(key)
            self.current_size_bytes -= item['size']
    
    def delete(self, key: str):
        """删除缓存项（线程安全）"""
        with self.lock:
            self._delete(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.stats['size_mb'] = 0
    
    def cleanup_expired(self):
        """清理过期项"""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, item in self.cache.items():
                # 检查TTL过期
                if item.get('ttl'):
                    if current_time - item['created'] > item['ttl']:
                        expired_keys.append(key)
                        continue
                
                # 检查文件修改（如果启用了文件关联）
                file_path = item.get('file_path')
                if file_path:
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        if file_mtime > item.get('file_mtime', 0):
                            expired_keys.append(key)
                    except (OSError, FileNotFoundError):
                        # 文件不存在，也视为过期
                        expired_keys.append(key)
            
            # 删除过期项
            for key in expired_keys:
                self._delete(key)
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self.lock:
            stats = self.stats.copy()
            stats['current_items'] = len(self.cache)
            stats['hit_ratio'] = (stats['hits'] / stats['total_requests']) if stats['total_requests'] > 0 else 0
            return stats
    
    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()

# 全局缓存实例
GLOBAL_CACHE = SmartCache(max_size_mb=1024*4)  

# 缓存装饰器
def cached(key_func: Optional[Callable] = None, ttl_seconds: Optional[int] = None):
    """
    缓存装饰器
    :param key_func: 生成缓存键的函数，默认为参数哈希
    :param ttl_seconds: 缓存生存时间（秒）
    """
    def _make_hashable(obj: Any) -> Any:
        """将对象转换为可哈希的形式"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return tuple(_make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            # 对字典按键排序，确保顺序一致
            return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, pd.DataFrame):
            # 对于DataFrame，使用形状、列名和数据的哈希
            try:
                # 使用更稳定的哈希：形状 + 列名 + 数据哈希
                data_hash = hashlib.md5(
                    obj.values.tobytes() if hasattr(obj.values, 'tobytes') 
                    else str(obj.values).encode()
                ).hexdigest()[:16]
                return f"df_{obj.shape}_{tuple(obj.columns)}_{data_hash}"
            except:
                return f"df_{id(obj)}"
        elif isinstance(obj, np.ndarray):
            # 对于numpy数组，使用形状和数据的哈希
            try:
                data_hash = hashlib.md5(obj.tobytes()).hexdigest()[:16]
                return f"ndarray_{obj.shape}_{obj.dtype}_{data_hash}"
            except:
                return f"ndarray_{id(obj)}"
        elif isinstance(obj, Path):
            # 对于Path对象，转换为字符串并包含文件修改时间
            try:
                path_str = str(obj)
                if obj.exists():
                    mtime = os.path.getmtime(path_str)
                    return f"path_{path_str}_{mtime}"
                return f"path_{path_str}"
            except:
                return f"path_{str(obj)}"
        else:
            # 对于其他对象，尝试使用字符串表示
            try:
                return str(obj)
            except:
                return f"obj_{id(obj)}"
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认：使用函数名和参数哈希
                # 处理args：转换为可哈希形式
                hashable_args = tuple(_make_hashable(arg) for arg in args)
                # 处理kwargs：按键排序，确保顺序一致
                hashable_kwargs = tuple(sorted(
                    (k, _make_hashable(v)) for k, v in kwargs.items()
                ))
                # 组合所有部分
                key_parts = (func.__name__, hashable_args, hashable_kwargs)
                # 生成稳定的MD5哈希
                cache_key = hashlib.md5(
                    pickle.dumps(key_parts, protocol=pickle.HIGHEST_PROTOCOL)
                ).hexdigest()
            
            # 尝试从缓存获取
            cached_result = GLOBAL_CACHE.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 缓存未命中，执行函数
            result = func(*args, **kwargs)
            
            # 存储到缓存
            GLOBAL_CACHE.set(cache_key, result, ttl_seconds=ttl_seconds)
            
            return result
        return wrapper
    return decorator

# ======================== 4. PLINK参数调优 ========================

class PLINKOptimizer:
    """
    PLINK参数优化器
    根据数据规模自动优化PLINK参数
    """
    
    @staticmethod
    def optimize_plink_parameters(available_memory_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        优化PLINK参数
        :return: 优化的PLINK参数字典
        """
        # 获取可用内存
        if available_memory_mb is None and PSUTIL_AVAILABLE:
            try:
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            except:
                available_memory_mb = 4096  # 默认4GB
        
        # 基础参数
        params = {
            'memory_mb': 4096,  # 默认4GB
            'window_size': 50,  # 默认窗口大小
            'window_kb': 1000,  # 默认窗口KB
        }
        
        # 根据CPU核心数设置每个PLINK进程使用的CPU线程数（固定为4，但不超过实际CPU数）
        cpu_count = os.cpu_count() or 1
        params['threads'] = int(min(4, cpu_count))
        
        # 根据可用内存调整内存限制
        if available_memory_mb:
            # 保守策略：使用可用内存的70%
            safe_memory = available_memory_mb * 0.7
            params['memory_mb'] = int(min(safe_memory, 32768))  # 最大32GB
            # 确保至少1GB
            params['memory_mb'] = max(1024, params['memory_mb'])
        
        # 添加性能优化标志
        params['performance_flags'] = [
            '--allow-extra-chr',
            '--allow-no-sex',
            '--keep-allele-order',
            '--nonfounders'
        ]
        
        
        return params
    
    @staticmethod
    def build_plink_command(base_cmd: List[str], 
                           optimized_params: Dict[str, Any],
                           additional_args: Optional[List[str]] = None) -> List[str]:
        """
        构建优化的PLINK命令
        """
        cmd = base_cmd.copy()
        
        # 添加内存限制（如果PLINK版本支持）
        if optimized_params.get('memory_mb'):
            cmd.extend(['--memory', str(int(optimized_params['memory_mb']))])
        
        # 添加线程数设置（每个PLINK进程使用固定数量的CPU线程）
        if optimized_params.get('threads'):
            cmd.extend(['--threads', str(int(optimized_params['threads']))])
        
        # 添加性能标志
        if optimized_params.get('performance_flags'):
            cmd.extend(optimized_params['performance_flags'])
        
        # 添加额外参数
        if additional_args:
            cmd.extend(additional_args)
        
        return cmd

# ======================== 5. 优化的.raw文件处理（集成第一层优化） ========================

def _make_cache_key_for_bim(bim_file: str, chr_num: Optional[str] = None) -> str:
    """为bim文件生成稳定的缓存键"""
    # 规范化路径为绝对路径
    bim_path = Path(bim_file).absolute()
    # 包含文件修改时间，确保文件变化时缓存失效
    try:
        mtime = os.path.getmtime(str(bim_path))
        key_parts = (str(bim_path), str(chr_num), mtime)
    except:
        key_parts = (str(bim_path), str(chr_num))
    return hashlib.md5(
        pickle.dumps(key_parts, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()

@cached(key_func=_make_cache_key_for_bim, ttl_seconds=3600)  # 缓存1小时
def get_snp_chr_pos_mapping_optimized(bim_file: str, chr_num: Optional[str] = None) -> dict:
    """
    优化的SNP映射获取（带缓存）
    使用向量化操作提高效率
    """
    # 规范化路径
    bim_file = str(Path(bim_file).absolute())
    if not Path(bim_file).exists():
        raise FileNotFoundError(f".bim文件不存在:{bim_file}")
    
    # 使用向量化读取（如果文件不大）
    file_size = Path(bim_file).stat().st_size
    if file_size < 100 * 1024 * 1024:  # 小于100MB，直接读取
        bim_df = pd.read_csv(
            bim_file,
            sep=r"\s+",
            header=None,
            names=["chr", "snp_id", "genetic_dist", "physical_pos", "a1", "a2"],
            dtype={"chr": str, "snp_id": str, "physical_pos": str, "a1": str, "a2": str}
        )
        
        if chr_num is not None:
            bim_df = bim_df[bim_df["chr"] == chr_num]
        
        # 向量化生成新名称
        bim_df["new_snp_name"] = bim_df["chr"] + "_" + bim_df["physical_pos"].astype(str)
        
        # 构建映射
        snp_mapping = dict(zip(bim_df["snp_id"], bim_df["new_snp_name"]))
        
        # 添加PLINK列名格式映射
        for _, row in bim_df.iterrows():
            chr_val = str(row["chr"])
            pos_val = str(row["physical_pos"])
            a1_val = str(row["a1"])
            a2_val = str(row["a2"])
            snp_id = str(row["snp_id"])
            new_name = f"{chr_val}_{pos_val}"
            
            # PLINK可能使用的格式
            plink_formats = [
                f"{chr_val}:{pos_val}_{a1_val}",
                f"{chr_val}:{pos_val}_{a2_val}",
                f"{chr_val}:{pos_val}"
            ]
            
            # 添加rs_allele格式的映射（如rs123_A, rs123_G等）
            # 支持所有可能的等位基因格式
            rs_allele_formats = [
                f"{snp_id}_{a1_val}",
                f"{snp_id}_{a2_val}",
                f"{snp_id}_{a1_val}/{a2_val}",  # 如rs123_A/G
                f"{snp_id}_{a2_val}/{a1_val}",  # 如rs123_G/A
            ]
            
            for fmt in plink_formats + rs_allele_formats:
                snp_mapping[fmt] = new_name
        
        return snp_mapping
    else:
        # 大文件：分块读取
        return get_snp_chr_pos_mapping(bim_file, chr_num)

def get_snp_chr_pos_mapping(bim_file: str, chr_num: Optional[str] = None) -> dict:
    """
    基础SNP映射获取函数（用于大文件分块读取）
    从.bim文件读取SNP映射：snp_id -> chr_pos
    """
    if not Path(bim_file).exists():
        raise FileNotFoundError(f".bim文件不存在:{bim_file}")
    
    snp_mapping = {}
    
    # 分块读取大文件
    chunk_size = 100000
    for chunk in pd.read_csv(
        bim_file,
        sep=r"\s+",
        header=None,
        names=["chr", "snp_id", "genetic_dist", "physical_pos", "a1", "a2"],
        dtype={"chr": str, "snp_id": str, "physical_pos": str, "a1": str, "a2": str},
        chunksize=chunk_size,
        engine='python'
    ):
        # 如果指定了染色体，过滤
        if chr_num is not None:
            chunk = chunk[chunk["chr"] == chr_num]
        
        # 生成新名称
        chunk["new_snp_name"] = chunk["chr"] + "_" + chunk["physical_pos"].astype(str)
        
        # 构建映射
        for _, row in chunk.iterrows():
            snp_id = str(row["snp_id"])
            new_name = str(row["new_snp_name"])
            snp_mapping[snp_id] = new_name
            
            # 添加PLINK列名格式映射
            chr_val = str(row["chr"])
            pos_val = str(row["physical_pos"])
            a1_val = str(row["a1"])
            a2_val = str(row["a2"])
            
            plink_formats = [
                f"{chr_val}:{pos_val}_{a1_val}",
                f"{chr_val}:{pos_val}_{a2_val}",
                f"{chr_val}:{pos_val}"
            ]
            
            # 添加rs_allele格式的映射（如rs123_A, rs123_G等）
            # 支持所有可能的等位基因格式
            rs_allele_formats = [
                f"{snp_id}_{a1_val}",
                f"{snp_id}_{a2_val}",
                f"{snp_id}_{a1_val}/{a2_val}",  # 如rs123_A/G
                f"{snp_id}_{a2_val}/{a1_val}",  # 如rs123_G/A
            ]
            
            for fmt in plink_formats + rs_allele_formats:
                snp_mapping[fmt] = new_name
    
    return snp_mapping


def standardize_snp_column_names(
    df: pd.DataFrame,
    bim_file: str,
    pheno_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    统一训练矩阵中的 SNP 列名为 chr_position 格式（例如: chr1_12345）
    
    设计目的：
    - 有些情况下 .raw 头信息的列名格式可能与预期不完全一致，导致前面在解析阶段的重命名未完全生效；
    - 这里在「所有染色体合并完成后、写出 train_data.txt 之前」做一道兜底重命名。
    
    规则：
    - 只对「非表型列」尝试重命名；
    - 优先使用 .bim 生成的 snp_id/PLINK 列名 -> chr_pos 映射；
    - 若映射中无对应键，则尝试从 'chr:pos_allele' 形式解析为 'chr_pos'；
    - 未能识别的列名保持不变。
    """
    if df.empty:
        return df
    
    bim_file = str(Path(bim_file).absolute())
    if not Path(bim_file).exists():
        logger.warning(f"standardize_snp_column_names: .bim file not found, skip renaming: {bim_file}")
        return df
    
    # 确定表型列，避免误改
    pheno_cols: set = set(pheno_columns or [])
    # 常见表型列名兜底
    pheno_cols.update({"phenotype"})
    
    # 构建 SNP 映射（不按染色体拆分）
    try:
        snp_mapping = get_snp_chr_pos_mapping_optimized(bim_file, chr_num=None)
    except Exception as e:
        logger.warning(f"standardize_snp_column_names: failed to build SNP mapping from .bim: {e}")
        return df
    
    rename_dict: Dict[str, str] = {}
    
    # 正则：解析 'chr:pos_allele' 形式
    pattern_plink = re.compile(r'^([^:]+):([^_]+)_[^_]+$')
    
    for col in df.columns:
        # 跳过明显的表型列
        if col in pheno_cols or any(str(col).lower().startswith(str(p).lower()) for p in pheno_cols):
            continue
        
        col_str = str(col)
        new_name = None
        
        # 1) 优先使用 .bim 提供的映射
        if col_str in snp_mapping:
            new_name = snp_mapping[col_str]
        else:
            # 2) 尝试从 'chr:pos_allele' 解析
            m = pattern_plink.match(col_str)
            if m:
                chr_part, pos_part = m.group(1), m.group(2)
                new_name = f"{chr_part}_{pos_part}"
        
        if new_name and new_name != col_str:
            rename_dict[col] = new_name
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    else:
        pass
    
    return df

def get_plink_path() -> str:
    """自动定位PLINK1.9可执行文件"""
    current_file = Path(__file__).absolute()
    software_dir = current_file.parent / "software"
    plink_candidates = [software_dir / "plink"]
    return str(plink_candidates[0])

PLINK_EXECUTABLE = get_plink_path()

def _process_chunk_parallel(chunk_data: Tuple[int, pd.DataFrame, Dict[str, str], Optional[int], List[str], type]) -> Tuple[int, pd.DataFrame]:
    """
    并行处理单个chunk的辅助函数（用于多进程）
    在每个进程内部可以使用多线程进一步并行化
    
    Args:
        chunk_data: (chunk_idx, chunk_df, batch_mapping, sample_col_idx, original_columns, dtype_obj)
    
    Returns:
        (chunk_idx, processed_chunk_df)
    """
    chunk_idx, chunk_df, batch_mapping, sample_col_idx, original_columns, dtype_obj = chunk_data
    
    try:
        # 1. 批量重命名
        chunk_df = VectorizedProcessor.batch_rename_columns(chunk_df, batch_mapping)
        
        # 2. 重命名样本列
        if sample_col_idx is not None:
            sample_col_name = original_columns[sample_col_idx]
            if sample_col_name in chunk_df.columns:
                chunk_df = chunk_df.rename(columns={sample_col_name: "sample"})
        
        # 3. 向量化类型转换（在子进程内串行处理，避免嵌套并行带来的复杂性）
        chunk_df = VectorizedProcessor.vectorized_type_conversion(
            chunk_df, ["sample"], dtype_obj, max_workers=0
        )
        
        # 4. 设置索引
        if "sample" in chunk_df.columns:
            chunk_df = chunk_df.set_index("sample")
        
        return (chunk_idx, chunk_df)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"处理chunk {chunk_idx}失败: {e}")
        raise


def parse_raw_file_optimized_v2(
    raw_file: str,
    chr_snp_mapping: Dict[str, str],
    dtype_obj: type = np.int8,
    chunk_rows: int = 10000,
    use_cache: bool = True,
    max_workers: Optional[int] = None,
    use_parallel: bool = False
) -> pd.DataFrame:
    """
    第二版优化的.raw文件解析
    集成向量化计算、缓存优化、分块累计和多核并行处理
    
    Args:
        raw_file: raw文件路径
        chr_snp_mapping: SNP映射字典
        dtype_obj: 目标数据类型
        chunk_rows: 每个chunk的行数
        use_cache: 是否使用缓存
        max_workers: 最大并行进程数（None表示自动）
        use_parallel: 是否使用多进程并行处理
    """
    # 规范化路径为绝对路径
    raw_file_abs = str(Path(raw_file).absolute())
    
    # 生成缓存键（使用稳定的哈希方法）
    # 对字典进行排序后哈希，确保顺序一致
    try:
        sorted_items = sorted(chr_snp_mapping.items())
        mapping_hash = hashlib.md5(
            pickle.dumps(sorted_items, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest()[:16]
    except Exception:
        # 如果无法序列化，使用字符串表示
        mapping_hash = hashlib.md5(
            str(sorted(chr_snp_mapping.items())).encode()
        ).hexdigest()[:16]
    
    cache_key = f"raw_parse_{raw_file_abs}_{mapping_hash}"
    
    # 尝试从缓存获取
    if use_cache:
        cached_result = GLOBAL_CACHE.get(cache_key, check_file_mtime=True, file_path=raw_file_abs)
        if cached_result is not None:
            return cached_result
    
    
    # 步骤1: 使用内存映射打开文件
    with open(raw_file, 'rb') as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        try:
            # 步骤2: 解析表头
            header_end = mmapped_file.find(b'\n')
            if header_end == -1:
                raise ValueError("Cannot find header line")
            
            header_line = mmapped_file[:header_end].decode('utf-8').strip()
            original_columns = header_line.split()
            
            # 向量化解析列结构
            sample_col_idx = None
            snp_col_indices = []
            snp_col_name_mapping = {}
            
            for i, col in enumerate(original_columns):
                if col == "IID":
                    sample_col_idx = i
                elif col not in ["FID", "PAT", "MAT", "SEX", "PHENOTYPE"]:
                    snp_col_indices.append(i)
                    # 使用向量化解析函数
                    chr_part, pos_part = VectorizedProcessor.parse_plink_column_name(col)
                    if chr_part and pos_part:
                        snp_col_name_mapping[i] = f"{chr_part}_{pos_part}"
                    else:
                        snp_col_name_mapping[i] = col
            
            
            # 步骤3: 构建批量映射（向量化）
            batch_mapping = VectorizedProcessor.batch_parse_plink_columns(original_columns)
            # 添加传入的映射（优先级更高）
            batch_mapping.update(chr_snp_mapping)
            
            # 步骤4: 确定要读取的列
            usecols_list = None
            if sample_col_idx is not None:
                usecols_list = [sample_col_idx] + snp_col_indices
            
            # 步骤5: 读取数据（分块累计 + 多核并行）
            data_start = header_end + 1
            import io
            mmap_file_obj = io.BytesIO(mmapped_file[data_start:])
            
            # 确定并行策略
            if max_workers is None:
                # 固定使用 6 个进程并行处理 .raw 文件
                max_workers = 6
            
            # 动态调整chunk大小以提高利用率
            # 目标：每个chunk处理时间约1-2秒，保持CPU忙碌
            if chunk_rows < 5000:
                chunk_rows = 5000  # 最小chunk大小
            elif chunk_rows > 50000:
                chunk_rows = 50000  # 最大chunk大小，避免内存溢出
            
            
            # 第一阶段：分块读取所有chunk（I/O密集型，串行）
            raw_chunks = []
            chunk_count = 0
            
            for chunk in pd.read_csv(
                mmap_file_obj,
                sep=r'\s+',
                header=None,
                names=original_columns,
                dtype=str,
                engine='c',
                chunksize=chunk_rows,
                usecols=usecols_list
            ):
                raw_chunks.append((chunk_count, chunk))
                chunk_count += 1
                
                # 每读取10个chunk输出一次进度（调试级别）
                if chunk_count % 10 == 0:
                    pass
            
            
            if not raw_chunks:
                raise ValueError("No data read from file")
            
            # 第二阶段：并行处理所有chunk（CPU密集型，并行）
            processed_chunks = []
            
            if use_parallel and max_workers > 1 and chunk_count > 1:
                # 使用多进程并行处理（每个子进程内部使用串行处理，避免嵌套并行带来的问题）
                chunk_data_list = [
                    (idx, chunk, batch_mapping, sample_col_idx, original_columns, dtype_obj)
                    for idx, chunk in raw_chunks
                ]
                
                
                # 使用ProcessPoolExecutor进行真正的多核并行
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_process_chunk_parallel, chunk_data): chunk_data[0] 
                              for chunk_data in chunk_data_list}
                    
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            chunk_idx, processed_chunk = future.result()
                            processed_chunks.append((chunk_idx, processed_chunk))
                            completed += 1
                            
                            # 每完成10%输出一次进度（调试级别）
                            if completed % max(1, chunk_count // 10) == 0:
                                pass
                        except Exception as e:
                            chunk_idx = futures[future]
                            logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                            raise
                
                # 按chunk_idx排序，确保顺序正确
                processed_chunks.sort(key=lambda x: x[0])
                chunks = [chunk for _, chunk in processed_chunks]
                
            else:
                # 串行处理（兼容模式或单chunk）
                chunks = []
                for chunk_idx, chunk in raw_chunks:
                    chunk_data = (chunk_idx, chunk, batch_mapping, sample_col_idx, original_columns, dtype_obj)
                    _, processed_chunk = _process_chunk_parallel(chunk_data)
                    chunks.append(processed_chunk)
                    
                    if (chunk_idx + 1) % 10 == 0:
                        pass
            
            
            # 第三阶段：累计合并所有chunk
            if len(chunks) == 1:
                result_df = chunks[0]
            else:
                # 分批合并，避免一次性concat导致内存峰值
                batch_size = max(10, max_workers * 2) if use_parallel else 10
                
                merged_batches = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    if len(batch) == 1:
                        merged_batches.append(batch[0])
                    else:
                        merged_batch = pd.concat(batch, axis=0)
                        merged_batches.append(merged_batch)
                        # 释放中间chunk内存
                        del batch
                        gc.collect()
                
                # 最终合并所有批次
                if len(merged_batches) == 1:
                    result_df = merged_batches[0]
                else:
                    result_df = pd.concat(merged_batches, axis=0)
                    del merged_batches
                    gc.collect()
            
            # 优化内存
            result_df = VectorizedProcessor.optimize_dataframe_memory(result_df)
            
            # 缓存结果
            if use_cache:
                GLOBAL_CACHE.set(cache_key, result_df, file_path=raw_file_abs, ttl_seconds=1800)
            
            logger.info(f".raw file parsing completed: shape={result_df.shape[0]:,} samples × {result_df.shape[1]:,} features")
            
            return result_df
            
        finally:
            mmapped_file.close()

# ======================== 6. 优化的染色体处理函数 ========================

def process_single_chromosome_optimized(
    chr_num: str,
    plink_prefix_abs: str,
    pheno_df: pd.DataFrame,
    plink_executable: str,
    use_cache: bool = True,
    skip_recode: bool = False,
    chr_recode_prefix: Optional[str] = None,
    cleanup_recode_files: bool = True,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    优化的单染色体处理函数
    集成第一层优化
    """
    try:
        # 生成缓存键（使用稳定的哈希方法）
        # 对pheno_df使用MD5哈希，确保稳定性
        try:
            # 对DataFrame进行排序后哈希，确保顺序一致
            pheno_sorted = pheno_df.sort_index()
            pheno_hash = hashlib.md5(
                pheno_sorted.values.tobytes() if hasattr(pheno_sorted.values, 'tobytes')
                else pickle.dumps(pheno_sorted.values, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest()[:16]
        except Exception:
            # 如果无法哈希DataFrame，使用形状和列名
            pheno_hash = hashlib.md5(
                f"{pheno_df.shape}_{tuple(pheno_df.columns)}".encode()
            ).hexdigest()[:16]
        
        # 规范化路径
        plink_prefix_abs = str(Path(plink_prefix_abs).absolute())
        cache_key = f"chr_{chr_num}_{plink_prefix_abs}_{pheno_hash}"
        
        # 尝试从缓存获取
        if use_cache:
            cached_result = GLOBAL_CACHE.get(cache_key)
            if cached_result is not None:
                # 兼容旧缓存：历史版本可能把 phenotype/index 等列 merge 进来了
                try:
                    df_cached = cached_result
                    # 只保留基因型列（删除 phenotype 及其重复列、以及 index/index.N 伪列）
                    drop_cols = []
                    for c in df_cached.columns:
                        c_str = str(c).lower()
                        if 'phenotype' in c_str:
                            drop_cols.append(c)
                        # pandas read_csv 对重复列会生成 index.1/index.2...，这里统一过滤
                        if c_str == 'index' or c_str.startswith('index.'):
                            drop_cols.append(c)
                    if drop_cols:
                        df_cached = df_cached.drop(columns=list(dict.fromkeys(drop_cols)), errors='ignore')

                    # 仍然做一次样本交集过滤，确保与表型样本一致
                    if 'sample' in pheno_df.columns and pheno_df.index.name != 'sample':
                        pheno_df = pheno_df.set_index('sample')
                    elif pheno_df.index.name is None and 'sample' not in pheno_df.columns:
                        pheno_df = pheno_df.copy()
                        pheno_df.index.name = 'sample'
                    df_cached = df_cached.loc[df_cached.index.intersection(pheno_df.index)]
                    return (chr_num, df_cached)
                except Exception:
                    return (chr_num, cached_result)
        

        # 统一输出前缀（允许外部先并行生成.raw，再进入解析）
        if chr_recode_prefix is None:
            chr_recode_prefix = f"{plink_prefix_abs}_chr{chr_num}"

        if not skip_recode:
            # 步骤1: 优化PLINK参数
            optimized_params = PLINKOptimizer.optimize_plink_parameters()

            # 步骤3: 构建PLINK命令
            base_cmd = [
                plink_executable,
                "--bfile", plink_prefix_abs,
                "--chr", chr_num,
                "--recodeA",
                "--out", chr_recode_prefix
            ]

            plink_cmd = PLINKOptimizer.build_plink_command(base_cmd, optimized_params)

            # 步骤4: 执行PLINK
            try:
                result = subprocess.run(
                    plink_cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    logger.error(f"Chromosome {chr_num}: PLINK execution failed: {error_msg[:500]}")
                    return (chr_num, None)
            except Exception as e:
                logger.error(f"Chromosome {chr_num}: PLINK execution exception: {str(e)}")
                return (chr_num, None)
        
        # 步骤5: 解析.raw文件
        chr_geno_file = f"{chr_recode_prefix}.raw"
        if not Path(chr_geno_file).exists():
            logger.error(f"染色体 {chr_num}: .raw文件未生成")
            return (chr_num, None)
        
        # 获取SNP映射（带缓存）
        chr_bim_file = f"{plink_prefix_abs}.bim"
        chr_snp_mapping = get_snp_chr_pos_mapping_optimized(chr_bim_file, chr_num=chr_num)
        
        # 使用优化的解析函数
        chr_geno_df = parse_raw_file_optimized_v2(
            chr_geno_file,
            chr_snp_mapping,
            dtype_obj=np.int8,
            use_cache=False  # 不缓存中间结果
        )
        
        if chr_geno_df is None or chr_geno_df.empty:
            logger.warning(f"Chromosome {chr_num}: No valid data read")
            return (chr_num, None)
        
        # 步骤6: 合并表型
        # 重要：这里不要把表型列 merge 进每条染色体的基因型矩阵里，
        # 否则在多染色体按列 concat 时会产生大量重复的 phenotype/index 列，
        # 后续 read_csv 会自动“列名去重”生成 index.1/index.2/... 之类伪特征。
        # 我们只在这里做样本交集过滤，表型列在全染色体合并完成后再追加一次。
        #
        # 确保 pheno_df 以 sample 为索引，用于过滤样本
        if 'sample' in pheno_df.columns and pheno_df.index.name != 'sample':
            pheno_df = pheno_df.set_index('sample')
        elif pheno_df.index.name is None and 'sample' not in pheno_df.columns:
            # 兜底：如果 sample 存在于索引但索引未命名，则将其视为 sample
            pheno_df = pheno_df.copy()
            pheno_df.index.name = 'sample'

        common_index = chr_geno_df.index.intersection(pheno_df.index)
        chr_geno_df = chr_geno_df.loc[common_index]
        
        # 步骤7: 清理临时文件
        if cleanup_recode_files:
            delete_temp_files(chr_recode_prefix, [".raw", ".log", ".nosex"])
        
        # 步骤8: 缓存结果
        if use_cache:
            GLOBAL_CACHE.set(cache_key, chr_geno_df, ttl_seconds=7200)  # 缓存2小时
        
        
        return (chr_num, chr_geno_df)
        
    except Exception as e:
        logger.error(f"Chromosome {chr_num} processing failed: {str(e)}", exc_info=True)
        return (chr_num, None)


def _plink_recodeA_single_chr_worker(
    chr_num: str,
    plink_prefix_abs: str,
    plink_executable: str,
    optimized_params: Dict[str, Any],
) -> Tuple[str, int, str, str]:
    """
    子进程任务：对单条染色体执行 PLINK --recodeA，生成 {plink_prefix_abs}_chr{chr}.raw
    返回：(chr_num, returncode, chr_recode_prefix, stderr_or_stdout_snippet)
    """
    chr_recode_prefix = f"{plink_prefix_abs}_chr{chr_num}"
    base_cmd = [
        plink_executable,
        "--bfile", plink_prefix_abs,
        "--chr", chr_num,
        "--recodeA",
        "--out", chr_recode_prefix
    ]
    plink_cmd = PLINKOptimizer.build_plink_command(base_cmd, optimized_params)

    try:
        result = subprocess.run(
            plink_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        msg = (result.stderr or result.stdout or "")[:800]
        return chr_num, int(result.returncode), chr_recode_prefix, msg
    except Exception as e:
        return chr_num, 1, chr_recode_prefix, str(e)[:800]

# ======================== 7. 优化的主处理函数 ========================

def plink_to_training_data_optimized(
    plink_prefix: str,
    pheno_df: pd.DataFrame,
    output_file: str,
    tmp_dir: Optional[Union[str, Path]] = None,
    cleanup_registry: Optional[List[str]] = None,
    use_cache: bool = True,
    parallel_chr_recode: bool = False,
    chr_recode_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    优化的训练数据生成函数
    集成第一层优化
    """
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    output_file_path = Path(output_file).absolute()
    output_file_abs = output_file_path.as_posix()
    
    # 清理缓存中的过期项
    if use_cache:
        GLOBAL_CACHE.cleanup_expired()
    
    # 步骤1: 统一临时目录
    if tmp_dir is None:
        tmp_base = output_file_path.parent / "tmp_p"
    else:
        tmp_base = Path(tmp_dir).absolute()
    tmp_base.mkdir(parents=True, exist_ok=True)
    
    # 步骤2: 自动识别染色体列表
    chr_list = auto_detect_chromosomes(plink_prefix_abs)
    if not chr_list:
        raise RuntimeError("No chromosomes detected")
    
    logger.info(f"Detected {len(chr_list)} chromosome(s)")
    
    # 步骤3: 染色体级并行生成 .raw（PLINK --recodeA）
    # 说明：此处仅并行“生成raw文件”，解析与合并仍沿用原逻辑，避免内存压力过大
    chr_result_files = []
    
    # 将pheno_df保存为临时文件（避免在函数间传递）
    temp_pheno_file = str(tmp_base / "temp_pheno.feather")
    try:
        # 不要引入额外的“index”列：reset_index(drop=True) 或直接保存列即可
        pheno_for_save = pheno_df.copy()
        if 'sample' not in pheno_for_save.columns:
            # 如果 sample 在索引里（命名或未命名），转成列并确保列名为 sample
            pheno_for_save = pheno_for_save.reset_index()
            if 'sample' not in pheno_for_save.columns and 'index' in pheno_for_save.columns:
                pheno_for_save = pheno_for_save.rename(columns={'index': 'sample'})
        else:
            # 丢弃 RangeIndex，避免生成“index”列
            pheno_for_save = pheno_for_save.reset_index(drop=True)

        pheno_for_save.to_feather(temp_pheno_file)
        pheno_df_loaded = pd.read_feather(temp_pheno_file)
        if 'sample' not in pheno_df_loaded.columns:
            raise ValueError("Phenotype data missing required 'sample' column after serialization")
        pheno_df_loaded = pheno_df_loaded.set_index('sample')
    except Exception as e:
        logger.warning(f"Cannot use feather format, using pickle: {e}")
        temp_pheno_file = str(tmp_base / "temp_pheno.pkl")
        pheno_df.to_pickle(temp_pheno_file)
        pheno_df_loaded = pd.read_pickle(temp_pheno_file)
    
    try:
        # 3.1 先并行生成所有染色体的 .raw 文件
        chr_to_recode_prefix: Dict[str, str] = {}
        if parallel_chr_recode and len(chr_list) > 1:
            recode_workers = chr_recode_workers
            if recode_workers is None:
                cpu = os.cpu_count() or 2
                # 默认不超过4个并发PLINK进程，避免I/O和内存压力过大
                recode_workers = min(4, len(chr_list), max(1, cpu // 2))

            # 优化参数：并行跑多个PLINK时，限制每个PLINK的线程数，避免过度占用
            base_params = PLINKOptimizer.optimize_plink_parameters()
            per_plink_threads = max(1, int((os.cpu_count() or 1) // max(1, recode_workers)))
            per_plink_mem = base_params.get("memory_mb")
            if isinstance(per_plink_mem, (int, float)) and per_plink_mem > 0:
                per_plink_mem = max(1024, int(per_plink_mem // max(1, recode_workers)))
            optimized_params = dict(base_params)
            optimized_params["threads"] = per_plink_threads
            if per_plink_mem:
                optimized_params["memory_mb"] = per_plink_mem

            logger.info(
                f"Parallel PLINK recodeA (generate .raw): chromosomes={len(chr_list)}, "
                f"processes={recode_workers}, threads_per_plink={per_plink_threads}, "
                f"memory_mb_per_plink={optimized_params.get('memory_mb')}"
            )

            with ProcessPoolExecutor(max_workers=recode_workers) as executor:
                futures = {
                    executor.submit(
                        _plink_recodeA_single_chr_worker,
                        chr_num,
                        plink_prefix_abs,
                        PLINK_EXECUTABLE,
                        optimized_params,
                    ): chr_num
                    for chr_num in chr_list
                }

                completed = 0
                for future in as_completed(futures):
                    chr_num = futures[future]
                    try:
                        chr_num_ret, rc, recode_prefix, msg = future.result()
                    except Exception as e:
                        logger.error(f"Chromosome {chr_num}: recodeA worker crashed: {e}")
                        continue

                    completed += 1
                    if rc != 0:
                        logger.error(f"Chromosome {chr_num_ret}: PLINK recodeA failed: {msg}")
                        continue

                    raw_fp = f"{recode_prefix}.raw"
                    if not Path(raw_fp).exists():
                        logger.error(f"Chromosome {chr_num_ret}: .raw not found after recodeA: {raw_fp}")
                        continue

                    chr_to_recode_prefix[chr_num_ret] = recode_prefix

                    if completed % max(1, len(chr_list) // 10) == 0:
                        logger.info(f"PLINK recodeA progress: {completed}/{len(chr_list)}")

            ok = len(chr_to_recode_prefix)
            if ok == 0:
                raise RuntimeError("All chromosome recodeA tasks failed; no .raw generated")
            logger.info(f"PLINK recodeA completed: {ok}/{len(chr_list)} chromosome .raw generated")
        else:
            # 串行生成（旧逻辑）
            logger.info("Serial PLINK recodeA (generate .raw) mode")
            for chr_num in chr_list:
                chr_recode_prefix = f"{plink_prefix_abs}_chr{chr_num}"
                base_params = PLINKOptimizer.optimize_plink_parameters()
                base_params = dict(base_params)
                base_params["threads"] = 1  # 串行时也尽量不占用过多线程
                chr_num_ret, rc, recode_prefix, msg = _plink_recodeA_single_chr_worker(
                    chr_num,
                    plink_prefix_abs,
                    PLINK_EXECUTABLE,
                    base_params,
                )
                if rc != 0:
                    logger.error(f"Chromosome {chr_num_ret}: PLINK recodeA failed: {msg}")
                    continue
                if Path(f"{recode_prefix}.raw").exists():
                    chr_to_recode_prefix[chr_num_ret] = recode_prefix

        # 3.2 再按原流程解析每条染色体（这里保持串行，避免内存峰值）
        for chr_idx, chr_num in enumerate(chr_list, 1):
            if chr_num not in chr_to_recode_prefix:
                logger.warning(f"Chromosome {chr_num}: .raw not generated, skipping parsing")
                continue

            logger.info(f"Parsing chromosome {chr_idx}/{len(chr_list)} from .raw: {chr_num}")

            _, chr_df = process_single_chromosome_optimized(
                chr_num,
                plink_prefix_abs,
                pheno_df_loaded,
                PLINK_EXECUTABLE,
                use_cache=use_cache,
                skip_recode=True,
                chr_recode_prefix=chr_to_recode_prefix[chr_num],
                cleanup_recode_files=True,
            )
            
            if chr_df is not None and not chr_df.empty:
                # 保存染色体结果到临时文件
                chr_temp_file = str(tmp_base / f"chr_{chr_num}_temp.feather")
                try:
                    chr_df.reset_index().to_feather(chr_temp_file)
                except Exception as e:
                    logger.warning(f"Cannot use feather format for chromosome {chr_num}, using pickle: {e}")
                    chr_temp_file = str(tmp_base / f"chr_{chr_num}_temp.pkl")
                    chr_df.to_pickle(chr_temp_file)
                chr_result_files.append((chr_num, chr_temp_file))
                
                if cleanup_registry is not None:
                    cleanup_registry.append(chr_temp_file)
                
                # 释放内存
                del chr_df
                gc.collect()
                
                logger.info(f"Chromosome {chr_num} processing completed, saved to temporary file")
            else:
                logger.warning(f"Chromosome {chr_num} has no valid data, skipping")
        
        # 步骤5: 合并所有染色体结果
        if not chr_result_files:
            raise RuntimeError("All chromosomes have no valid data")
        
        logger.info(f"Starting merge of {len(chr_result_files)} chromosome results")
        
        # 使用批次合并优化内存
        batch_size = min(5, len(chr_result_files))
        all_batches = []
        
        for i in range(0, len(chr_result_files), batch_size):
            batch = chr_result_files[i:i+batch_size]
            batch_dfs = []
            
            # 加载批次内的染色体数据
            for chr_num, chr_file in batch:
                try:
                    if chr_file.endswith('.feather'):
                        chr_df = pd.read_feather(chr_file).set_index('sample')
                    else:
                        chr_df = pd.read_pickle(chr_file)
                    # 兼容旧流程产生的重复列：去掉 phenotype 与 index/index.N
                    if chr_df is not None and not chr_df.empty:
                        cols_lower = [str(c).lower() for c in chr_df.columns]
                        drop_cols = [
                            c for c, cl in zip(chr_df.columns, cols_lower)
                            if ('phenotype' in cl) or (cl == 'index') or cl.startswith('index.')
                        ]
                        if drop_cols:
                            chr_df = chr_df.drop(columns=drop_cols, errors='ignore')
                    batch_dfs.append(chr_df)
                except Exception as e:
                    logger.warning(f"Failed to load chromosome {chr_num} data: {e}")
            
            if batch_dfs:
                # 合并批次内的染色体
                if len(batch_dfs) == 1:
                    batch_merged = batch_dfs[0]
                else:
                    # 确保所有DataFrame有相同的索引
                    common_index = batch_dfs[0].index
                    for df in batch_dfs[1:]:
                        common_index = common_index.intersection(df.index)
                    
                    # 只保留共有样本
                    batch_dfs = [df.loc[common_index] for df in batch_dfs]
                    batch_merged = pd.concat(batch_dfs, axis=1, join='inner')
                
                all_batches.append(batch_merged)
                
                # 清理批次内存
                del batch_dfs, batch_merged
                gc.collect()
        
        # 合并所有批次（此时还没有 phenotype 列，避免重复列污染）
        if len(all_batches) == 1:
            final_train_df = all_batches[0]
        else:
            # 确保所有批次有相同的索引
            common_index = all_batches[0].index
            for batch in all_batches[1:]:
                common_index = common_index.intersection(batch.index)
            
            # 只保留共有样本
            all_batches = [batch.loc[common_index] for batch in all_batches]
            final_train_df = pd.concat(all_batches, axis=1, join='inner')

        # 步骤5.2: 合并完成后再追加一次 phenotype（避免在每条染色体重复）
        if 'phenotype' in pheno_df_loaded.columns:
            # 保证样本一致（inner join）
            final_train_df = final_train_df.merge(
                pheno_df_loaded[['phenotype']],
                left_index=True,
                right_index=True,
                how='inner'
            )
        else:
            raise ValueError("Phenotype column 'phenotype' not found in phenotype data")
        
        # 步骤5.5: 兜底统一 SNP 列名为 chr_position 格式
        try:
            # .bim 文件与当前用于生成训练数据的 PLINK 前缀对应
            # 在本函数内，该前缀为 plink_prefix_abs（run_preprocess 中传入的是 final_plink_prefix）
            bim_file = f"{plink_prefix_abs}.bim"
            # 使用在本函数开头加载的 pheno_df_loaded 的列名作为表型列集合
            pheno_columns = list(pheno_df_loaded.columns) if 'pheno_df_loaded' in locals() else None
            final_train_df = standardize_snp_column_names(
                final_train_df,
                bim_file=bim_file,
                pheno_columns=pheno_columns,
            )
        except Exception as e:
            logger.warning(f"Failed to standardize SNP column names to chr_pos format (ignored): {e}")
        
        # 步骤6: 保存最终训练数据
        # 确保输出文件有.txt后缀
        if not output_file_abs.endswith('.txt') and not output_file_abs.endswith('.txt.gz'):
            if output_file_abs.endswith('.gz'):
                output_file_abs = output_file_abs.replace('.gz', '.txt.gz')
            else:
                output_file_abs = output_file_abs + '.txt'
        
        # 使用gzip压缩（如果文件较大）
        use_gzip = len(final_train_df) * len(final_train_df.columns) > 1000000
        compression = "gzip" if use_gzip else None
        
        if use_gzip and not output_file_abs.endswith('.gz'):
            output_file_abs = output_file_abs + '.gz'
        
        final_train_df.to_csv(
            output_file_abs,
            sep="\t",
            index=True,
            compression=compression
        )
        
        # ======================== 关键日志：最终用于模型训练的样本数和特征SNP数 ========================
        # 说明：
        # - 行数(final_train_df.shape[0]) = 有效样本数
        # - 列数(final_train_df.shape[1]) 中包含 phenotype 表型列，因此减 1 得到特征SNP数量
        logger.info(
            f"Final training dataset (after GWAS/LD/combined strategy if applied): "
            f"{final_train_df.shape[0]:,} samples, {final_train_df.shape[1]-1:,} feature SNPs"
        )
        
        # 打印缓存统计
        if use_cache:
            GLOBAL_CACHE.print_stats()
        
        return final_train_df, output_file_abs
        
    finally:
        # 清理临时文件
        try:
            os.remove(temp_pheno_file)
        except:
            pass
        
        # 清理染色体临时文件
        for _, chr_file in chr_result_files:
            try:
                os.remove(chr_file)
            except:
                pass

# ======================== 8. 优化的主函数 ========================

def run_preprocess(
    genotype_file: str,
    phenotype_file: str,
    output_file: str,
    pheno_col: Optional[str] = None,
    filter_snps: bool = True,
    use_cache: bool = True,
    feature_selection_mode: int = 1,
    gwas_pvalue: float = 0.01,
    ld_config: str = "50,5,0.2",
    keep_temp_files: bool = False,
) -> int:
    """
    优化的预处理主函数
    集成第一层优化
    
    LD参数三合一配置 (standardized):
      - 参数名: ld_config
      - 参数格式: \"<window_kb>,<window_variants>,<r2_threshold>\"
      - 示例: \"50,5,0.2\" 表示 LD窗口50KB, 窗口内5个变体, r²阈值0.2
    """
    # 当前运行生成的临时文件登记
    cleanup_registry: List[str] = []
    temp_prefixes: List[str] = []
    
    def cleanup_current_run():
        """清理当前运行生成的临时文件（仅在 keep_temp_files=False 时生效）"""
        if keep_temp_files:
            # 用户显式要求保留临时文件，用于调试或复现
            return
        
        # 1) 按前缀批量删除典型 PLINK 中间文件
        for prefix in temp_prefixes:
            delete_temp_files(prefix, [".bed", ".bim", ".fam", ".log", ".nosex", ".raw", ".map", ".ped"])
        
        # 2) 删除登记在册的其它临时文件
        for fp in cleanup_registry:
            try:
                Path(fp).unlink()
            except Exception:
                pass
        
        # 3) 清理GWAS产生的临时文件和目录（如果存在）
        try:
            import shutil
            if 'stage_output_dir' in locals() or 'stage_output_dir' in globals():
                stage_output_dir_path = Path(stage_output_dir)
                # 删除GWAS产生的 output 目录
                gwas_output_dir = stage_output_dir_path / "output"
                if gwas_output_dir.exists():
                    shutil.rmtree(gwas_output_dir, ignore_errors=True)
                # 删除GWAS输出前缀相关的所有文件
                gwas_prefix_pattern = stage_output_dir_path / "preprocess_gwas*"
                for gwas_file in stage_output_dir_path.glob("preprocess_gwas*"):
                    try:
                        if gwas_file.is_file():
                            gwas_file.unlink()
                    except Exception:
                        pass
        except Exception as e:
            pass
        
        # 4) 删除整个临时目录 tmp（包括目录本身及其所有文件和子目录，包括LD产生的文件）
        try:
            import shutil
            if 'base_output_dir' in locals() or 'base_output_dir' in globals():
                # 删除整个 tmp 目录（output_dir/tmp），而不仅仅是 tmp/preprocess 子目录
                tmp_root_dir = base_output_dir / "tmp"
                if tmp_root_dir.exists():
                    shutil.rmtree(tmp_root_dir, ignore_errors=True)
            elif 'tmp_dir' in locals() or 'tmp_dir' in globals():
                # 如果 base_output_dir 不可用，则删除 tmp_dir 的父目录（即整个 tmp 目录）
                tmp_dir_path = Path(tmp_dir)
                tmp_root_dir = tmp_dir_path.parent  # 获取 tmp 目录（即 output_dir/tmp）
                if tmp_root_dir.exists() and tmp_root_dir.name == "tmp":
                    shutil.rmtree(tmp_root_dir, ignore_errors=True)
        except Exception as e:
            # 删除失败不影响主流程
            pass
    
    try:
        # ======================== 参数检查与标准化 ========================
        if not Path(phenotype_file).exists():
            raise FileNotFoundError(f"表型文件不存在:{phenotype_file}")
        
        # 解析LD三合一配置参数 (standardized)
        try:
            ld_parts = [p.strip() for p in str(ld_config).split(",") if p.strip() != ""]
            if len(ld_parts) != 3:
                raise ValueError
            ld_window_kb = int(ld_parts[0])
            ld_window = int(ld_parts[1])
            ld_window_r2 = float(ld_parts[2])
        except Exception:
            raise ValueError(
                f"无效的LD三合一配置 ld_config={ld_config!r}; "
                f"正确格式为 \"<window_kb>,<window_variants>,<r2_threshold>\", 例如 \"50,5,0.2\""
            )
        
        # ======================== 输出目录结构规范化（standardized） ========================
        # 新的目录结构：
        # - 如果 -o test，则在 test 目录下创建 preprocess 子目录
        # - 需要保存的文件放在 test/preprocess/ 中
        # - 临时文件放在 test/tmp/ 中（与 preprocess 同级）
        # - 阶段运行结束或异常结束时删除 test/tmp/ 目录
        output_path = Path(output_file).absolute()
        
        # 确定基础输出目录（-o 指定的目录）
        if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
            base_output_dir = output_path
        else:
            base_output_dir = output_path.parent
        
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 在基础输出目录下创建 preprocess 子目录（用于保存需要保留的文件）
        stage_output_dir = base_output_dir / "preprocess"
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 在基础输出目录下创建 tmp/preprocess 子目录（用于临时文件和中间文件）
        # 每个阶段在 tmp 下创建自己的子目录，避免不同阶段的临时文件互相干扰
        tmp_dir = base_output_dir / "tmp" / "preprocess"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定输出文件路径和前缀
        if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
            output_prefix = base_output_dir.name
        else:
            output_prefix = output_path.stem
        
        output_file_name = f"{output_prefix}_train_data.txt"
        output_file_path = stage_output_dir / output_file_name
        
        # PLINK中间前缀
        temp_plink_prefix = str(tmp_dir / "temp_plink")
        temp_prefixes.append(temp_plink_prefix)
        
        # 步骤1: 基因型格式转换（使用原始函数，这部分不是优化重点）
        genotype_format = detect_genotype_format(genotype_file)
        plink_prefix = genotype_to_plink(genotype_file, temp_plink_prefix)

        # 统一化染色体名称
        normalize_chromosome_names(plink_prefix) 
        
        # 步骤2: 加载表型数据
        pheno_df = load_phenotype(phenotype_file, pheno_col)
        
        # 判断表型类型
        task_type = determine_phenotype_type(pheno_df, stage_output_dir)
        pheno_df = convert_phenotype_dtype(pheno_df, task_type)
        
        # 回归任务剔除0值
        if task_type == "regression":
            original_count = len(pheno_df)
            pheno_df = pheno_df[pheno_df["phenotype"] != 0].reset_index(drop=True)
            removed_count = original_count - len(pheno_df)
            if removed_count > 0:
                logger.info(f"Regression task: removed {removed_count} samples with phenotype value = 0")
        
        # 样本匹配
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
                if fam_df.shape[1] >= 2:
                    geno_samples = set(fam_df.iloc[:, 1].astype(str))
                    original_pheno_n = len(pheno_df)
                    pheno_df = pheno_df[pheno_df["sample"].isin(geno_samples)].reset_index(drop=True)
                    if pheno_df.empty:
                        raise ValueError("No intersection between phenotype samples and genotype samples")
                    logger.info(f"Sample matching completed: {len(pheno_df):,} samples remaining after matching")
            except Exception as e:
                logger.warning(f"Sample matching failed: {e}")
        
        # 回归任务绘制分布图
        if task_type == "regression":
            plot_regression_phenotype_distribution(pheno_df, stage_output_dir)
        
        # 步骤3: 样本过滤
        sample_filtered_prefix = str(tmp_dir / "sample_filtered_plink")
        temp_prefixes.append(sample_filtered_prefix)
        sample_filtered_plink = filter_samples_by_phenotype(plink_prefix, pheno_df, sample_filtered_prefix)
        
        # 步骤4: SNP质量过滤（可选）
        logger.info(f"SNP过滤参数: filter_snps={filter_snps}")
        if filter_snps:
            filtered_plink_prefix = str(tmp_dir / "filtered_plink")
            temp_prefixes.append(filtered_plink_prefix)
            final_plink_prefix = filter_snps_by_quality(
                sample_filtered_plink,
                filtered_plink_prefix,
                maf=0.05,
                geno=0.2,
            )
            gwas_genotype_prefix = final_plink_prefix
            logger.info("SNP quality filtering completed")
        else:
            final_plink_prefix = sample_filtered_plink
            gwas_genotype_prefix = sample_filtered_plink
            logger.info("SNP filtering skipped")

        # 步骤4.5: 预处理阶段的GWAS/LD特征筛选（可选, LD参数基于三合一配置）
        if feature_selection_mode not in [1, 2, 3, 4]:
            raise ValueError(f"feature_selection_mode must be 1, 2, 3, or 4, got: {feature_selection_mode}")

        if feature_selection_mode != 1:
            logger.info(
                f"Running feature selection in preprocess (mode={feature_selection_mode}): "
                f"1=no selection, 2=GWAS, 3=LD, 4=GWAS+LD"
            )

        # 仅当需要GWAS或LD时才执行后续步骤
        if feature_selection_mode in [2, 3, 4]:
            # GWAS / LD 均基于当前过滤后的PLINK前缀
            # GWAS: 先运行GWAS获取显著SNP
            significant_snps: Optional[List[str]] = None

            # 为GWAS使用的表型：使用当前已清洗后的 pheno_df（与sample_filtered_plink一致）
            if feature_selection_mode in [2, 4]:
                try:
                    logger.info(f"Preprocess GWAS: p-value threshold={gwas_pvalue}")
                    significant_snps = run_gwas_preprocess(
                        plink_prefix=final_plink_prefix,
                        pheno_df=pheno_df,
                        output_dir=stage_output_dir,
                        tmp_dir=tmp_dir,
                        pvalue_threshold=gwas_pvalue,
                        # 关键：当用户指定 --no-filter-snps 时，同时关闭GWAS阶段的PLINK质控
                        perform_plink_qc=filter_snps,
                        keep_temp_files=keep_temp_files,
                    )
                    if not significant_snps:
                        logger.warning("No significant SNPs found by GWAS; feature selection will be skipped.")
                        feature_selection_mode = 1
                except Exception as e:
                    logger.error(f"GWAS in preprocess failed: {e}")
                    feature_selection_mode = 1

            # LD 过滤
            if feature_selection_mode in [3, 4]:
                # LD 仅在 GWAS 成功（模式4）或纯LD模式（模式3）时执行
                if feature_selection_mode == 4 and not significant_snps:
                    logger.warning("GWAS failed or no significant SNPs; skip LD filtering.")
                    feature_selection_mode = 1
                else:
                    logger.info(
                        f"[LD_CONFIG] Preprocess LD filtering with three-in-one config: "
                        f"window_kb={ld_window_kb}, window_variants={ld_window}, r²_threshold={ld_window_r2}"
                    )
                    # 确保使用绝对路径
                    ld_output_prefix = str(Path(tmp_dir / "ld_filtered_plink").absolute())
                    # 如果是模式4，仅对GWAS显著SNP执行LD过滤
                    extract_snps_file = None
                    if feature_selection_mode == 4 and significant_snps:
                        extract_snps_file = str(Path(tmp_dir / "gwas_significant_snps_for_ld.txt").absolute())
                        with open(extract_snps_file, "w") as f:
                            for snp in significant_snps:
                                f.write(f"{snp}\n")
                        logger.info(f"LD filtering will be performed on {len(significant_snps):,} GWAS-significant SNPs")

                    try:
                        # 确保输入路径也是绝对路径
                        final_plink_prefix_abs = str(Path(final_plink_prefix).absolute())
                        logger.info(f"LD filtering input: {final_plink_prefix_abs}")
                        logger.info(f"LD filtering output: {ld_output_prefix}")
                        ld_ret = plink_ld.run_ld_filtering(
                            input_path=final_plink_prefix_abs,
                            output_prefix=ld_output_prefix,
                            ld_window_kb=ld_window_kb,
                            ld_window=ld_window,
                            ld_window_r2=ld_window_r2,
                            keep_intermediate=keep_temp_files,
                            keep_samples_file=None,
                            extract_snps_file=extract_snps_file
                        )
                        if ld_ret != 0:
                            logger.error("LD filtering in preprocess failed, fallback to no feature selection.")
                            feature_selection_mode = 1
                        else:
                            final_plink_prefix = ld_output_prefix
                            gwas_genotype_prefix = ld_output_prefix
                            logger.info("LD filtering in preprocess completed.")
                    finally:
                        if extract_snps_file:
                            try:
                                Path(extract_snps_file).unlink()
                            except Exception:
                                pass
            elif feature_selection_mode == 2 and significant_snps:
                # 仅GWAS：直接按显著SNP抽取PLINK
                logger.info("Preprocess GWAS-only feature selection: extracting significant SNPs.")
                gwas_only_prefix = str(tmp_dir / "gwas_filtered_plink")
                try:
                    final_plink_prefix = extract_snps_with_plink(
                        plink_prefix=final_plink_prefix,
                        snp_list=significant_snps,
                        output_prefix=gwas_only_prefix,
                    )
                    gwas_genotype_prefix = final_plink_prefix
                except Exception as e:
                    logger.error(f"GWAS-only extraction failed, fallback to no feature selection: {e}")
                    feature_selection_mode = 1

        # 步骤5: 使用优化的函数生成训练数据
        train_df, actual_train_file = plink_to_training_data_optimized(
            final_plink_prefix,
            pheno_df,
            str(output_file_path),
            tmp_dir=tmp_dir,
            cleanup_registry=cleanup_registry,
            use_cache=use_cache
        )
        
        # 步骤6: 生成元数据
        generate_metadata(
            output_prefix=str(stage_output_dir / output_prefix),
            plink_prefix=gwas_genotype_prefix,
            train_file=actual_train_file,
            valid_samples=train_df.index.tolist(),
            genotype_format=genotype_format,
            task_type=task_type,
            preprocess_tmp_dir=tmp_dir,
            filter_snps=filter_snps,
            feature_selection_mode=feature_selection_mode,
            gwas_pvalue=gwas_pvalue,
            ld_window_kb=ld_window_kb,
            ld_window=ld_window,
            ld_window_r2=ld_window_r2
        )
        
        # ======================== 关键日志：预处理结束时的样本数和特征SNP数 ========================
        # 无论是否执行 GWAS / LD / GWAS+LD（由 feature_selection_mode 决定），
        # 此处的 train_df 均为最终用于 model training 的矩阵：
        # - 行数 = 样本数
        # - 列数 - 1 = 特征SNP数量（排除 phenotype 表型列）
        logger.info(
            f"Preprocess completed: {len(train_df):,} samples, {train_df.shape[1]-1:,} feature SNPs "
            f"(feature_selection_mode={feature_selection_mode})"
        )
        
        # 最终内存清理
        gc.collect()
        
        # 预处理成功结束后，根据 keep_temp_files 选项清理临时文件
        cleanup_current_run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        cleanup_current_run()
        return 1

def delete_temp_files(prefix: str, exts: list) -> None:
    """批量删除临时文件"""
    for ext in exts:
        temp_file = f"{prefix}{ext}"
        if Path(temp_file).exists():
            try:
                os.remove(temp_file)
            except:
                pass

def detect_genotype_format(genotype_path: str) -> Literal["vcf", "plink_binary", "plink_text"]:
    """自动识别基因型文件格式"""
    genotype_path = Path(genotype_path).absolute().as_posix()

    if genotype_path.endswith((".vcf", ".vcf.gz")):
        if not Path(genotype_path).exists():
            raise FileNotFoundError(f"VCF文件不存在:{genotype_path}")
        return "vcf"

    plink_bin_files = [f"{genotype_path}.bed", f"{genotype_path}.bim", f"{genotype_path}.fam"]
    if all(Path(f).exists() for f in plink_bin_files):
        return "plink_binary"

    plink_text_files = [f"{genotype_path}.map", f"{genotype_path}.ped"]
    if all(Path(f).exists() for f in plink_text_files):
        return "plink_text"

    raise ValueError(
        f"\n无法识别基因型格式:{genotype_path}\n"
        f"支持格式:VCF(.vcf/.vcf.gz)、PLINK二进制(.bed/.bim/.fam)、PLINK文本(.map/.ped)"
    )


def genotype_to_plink(genotype_path: str, output_prefix: str,) -> str:
    """通用基因型转PLINK二进制"""
    fmt = detect_genotype_format(genotype_path)
    validate_plink_files(genotype_path, fmt)

    if fmt == "vcf":
        plink_cmd = [
            PLINK_EXECUTABLE,
            "--vcf", genotype_path,
            "--make-bed",
            "--out", output_prefix,
            "--allow-extra-chr",
            "--set-missing-var-ids", "@:#",
            "--keep-allele-order"
        ]

        try:
            subprocess.run(plink_cmd, capture_output=True, text=True, check=True)
            if "temp_plink" in output_prefix:
                delete_temp_files(genotype_path, [".vcf", ".vcf.gz", ".log"])
            return output_prefix
        except subprocess.CalledProcessError as e:
            logger.error(f"PLINK执行失败:{e.stderr}")
            raise RuntimeError(f"VCF转PLINK失败: {e.stderr}")

    elif fmt == "plink_binary":
        return genotype_path
    elif fmt == "plink_text":
        return plink_text_to_binary(genotype_path, output_prefix)

def plink_text_to_binary(genotype_path: str, output_prefix: str, ) -> str:
    """
    将PLINK文本格式(.map/.ped)转换为PLINK二进制格式(.bed/.bim/.fam)
    
    :param genotype_path: PLINK文本格式文件前缀
    :param output_prefix: 输出PLINK二进制文件前缀
    :return: 输出PLINK二进制文件前缀
    """
    
    plink_cmd = [
        PLINK_EXECUTABLE,
        "--file", genotype_path,
        "--make-bed",
        "--out", output_prefix,
        "--allow-extra-chr",
        "--allow-no-sex",
        "--keep-allele-order"
    ]
    
    try:
        result = subprocess.run(
            plink_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "未知错误"
            logger.error(f"PLINK文本转二进制失败: {error_msg[:500]}")
            raise RuntimeError(f"PLINK文本转二进制失败: {error_msg[:500]}")
        
        return output_prefix
    except Exception as e:
        logger.error(f"PLINK文本转二进制异常: {str(e)}")
        raise


def normalize_chromosome_names(plink_prefix: str) -> None:
    """
    统一化.bim文件中的染色体名称格式为chr1, chr2, ...格式
    如果染色体名称不统一，修改为统一格式
    """
    bim_file = f"{Path(plink_prefix).absolute().as_posix()}.bim"
    if not Path(bim_file).exists():
        raise FileNotFoundError(f"PLINK .bim文件不存在:{bim_file}")
    
    # 读取.bim文件
    bim_df = pd.read_csv(
        bim_file,
        sep=r"\s+",
        header=None,
        dtype={0: str},
        engine='python'
    )
    
    original_chr_col = bim_df[0].copy()
    
    # 统一化规则:转换为chr1, chr2, ...格式
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

def auto_detect_chromosomes(plink_prefix: str) -> List[str]:
    """
    兼容所有PLINK版本:通过解析.bim文件自动识别染色体列表
    .bim文件格式:chr(第1列) snp_id genetic_dist physical_pos a1 a2
    """
    bim_file = f"{Path(plink_prefix).absolute().as_posix()}.bim"
    if not Path(bim_file).exists():
        raise FileNotFoundError(f"PLINK .bim文件不存在:{bim_file}")
    
    
    # 分块读取.bim文件，仅提取第一列（染色体）
    chr_series = pd.read_csv(
        bim_file,
        sep=r"\s+",
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
    
    # 排序:数字在前（升序），非数字在后
    chr_list = list(chr_set)
    num_chrs = [c for c in chr_list if c.isdigit()]
    non_num_chrs = [c for c in chr_list if not c.isdigit()]
    num_chrs_sorted = sorted(num_chrs, key=int)
    chr_list_sorted = num_chrs_sorted + non_num_chrs
    
    return chr_list_sorted


def load_phenotype(pheno_path: str, pheno_col: Optional[str] = None) -> pd.DataFrame:
    """加载表型文件，检查异常值"""
    if pheno_col is not None:
        logger.warning(f"--pheno-col参数无效（表型文件无表头）")

    # 自动识别分隔符（更稳健：多行投票 + 自动跳过疑似表头行）
    def detect_separator(file_path: str) -> str:
        """
        返回值用于 pandas.read_csv 的 sep 参数：
        - '\t'：TSV
        - ','：CSV
        - r'\s+'：空白分隔（一个或多个空格/Tab）

        说明：
        - 旧实现只看第一行，若第一行是带逗号的表头（如 Sample,Phenotype），但数据行用空白分隔，会误判 sep=','，
          导致大量 phenotype 读成缺失值并被当作 abnormal value 删除。
        - 新实现会读取多行，尽量避开表头，并按“多数行”选择能稳定切分出 >=2 列的分隔符。
        """
        import csv
        candidates = ["\t", ",", r"\s+"]

        def _is_header_like(line: str) -> bool:
            # 粗略判断：同时包含字母且不包含数字的行，很可能是表头
            s = line.strip()
            if not s:
                return False
            has_alpha = any(ch.isalpha() for ch in s)
            has_digit = any(ch.isdigit() for ch in s)
            return has_alpha and not has_digit

        # 读取若干行用于检测
        lines: List[str] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for _ in range(30):
                ln = f.readline()
                if not ln:
                    break
                ln = ln.strip("\n\r")
                if ln.strip() == "":
                    continue
                lines.append(ln)

        if not lines:
            return r"\s+"

        # 跳过疑似表头行（最多跳1行）
        work_lines = lines[:]
        if _is_header_like(work_lines[0]) and len(work_lines) > 1:
            work_lines = work_lines[1:]

        # 先尝试 csv.Sniffer（只对 \t 和 , 有意义；\s+ 属于正则分隔，不参与 Sniffer）
        try:
            sample_text = "\n".join(work_lines[:10])
            dialect = csv.Sniffer().sniff(sample_text, delimiters="\t,")
            sniffed = dialect.delimiter
            if sniffed in ("\t", ","):
                # 只有当“多数行”确实能切出 >=2 列时才采纳
                ok = 0
                for ln in work_lines[:20]:
                    if len(ln.split(sniffed)) >= 2:
                        ok += 1
                if ok >= max(1, int(0.6 * min(20, len(work_lines)))):
                    return sniffed
        except Exception:
            pass

        # 多数投票：统计各分隔符在多数行上是否能切分出>=2列
        best_sep = r"\s+"
        best_score = -1
        eval_lines = work_lines[:20]
        for sep in candidates:
            score = 0
            for ln in eval_lines:
                parts = re.split(sep, ln.strip()) if sep == r"\s+" else ln.split(sep)
                parts = [p for p in parts if p != ""]
                if len(parts) >= 2:
                    score += 1
            if score > best_score:
                best_score = score
                best_sep = sep

        return best_sep

    sep = detect_separator(pheno_path)

    # 读取表型（先以字符串读取，便于检查异常值）
    try:
        pheno_df = pd.read_csv(
            pheno_path,
            sep=sep,
            header=None,
            dtype=str,  # 先全读为字符串，便于检查异常值；同时兼容多列表型文件
            engine='python'
        )
    except Exception as e:
        raise RuntimeError(f"读取表型文件失败:{str(e)}")

    # 至少需要两列：sample + phenotype
    if pheno_df.shape[1] < 2:
        raise ValueError(f"Phenotype file column error: at least 2 columns required, found {pheno_df.shape[1]} columns")

    # 如果列数>2：默认使用第2列作为表型列，其余列忽略
    if pheno_df.shape[1] > 2:
        pheno_df = pheno_df.iloc[:, [0, 1]].copy()

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
        
        # 剔除异常值样本
        original_count = len(pheno_df)
        pheno_df = pheno_df[~abnormal_mask].reset_index(drop=True)
        remaining_count = len(pheno_df)
        
        # 默认只输出摘要，详细样例与异常值列表降级到 debug，避免污染日志
        logger.warning(
            "Phenotype abnormal values removed: "
            f"count={abnormal_count}, original={original_count}, remaining={remaining_count}"
        )
        
        # 检查剔除后是否还有有效数据
        if pheno_df.empty:
            raise ValueError("After removing abnormal values, phenotype file has no valid data, cannot continue processing")
    
    # 数据清洗:去重
    pheno_df = pheno_df.drop_duplicates(subset="sample", keep="last")
    
    # 检查样本ID列
    pheno_df = pheno_df[pheno_df["sample"].notna() & (pheno_df["sample"] != "")].reset_index(drop=True)

    if pheno_df.empty:
        raise ValueError("Phenotype file has no valid data (all missing values/empty IDs)")
    
    # 将表型值转换为数值类型（无法转换的值设为NaN，然后剔除）
    original_count_before_convert = len(pheno_df)
    pheno_df["phenotype"] = pd.to_numeric(pheno_df["phenotype"], errors='coerce')
    
    # 剔除转换后为NaN的样本
    nan_mask = pheno_df["phenotype"].isna()
    if nan_mask.any():
        nan_count = nan_mask.sum()
        nan_samples = pheno_df.loc[nan_mask, "sample"].tolist()[:10]  # 只显示前10个
        
        pheno_df = pheno_df[~nan_mask].reset_index(drop=True)
        remaining_count_after_convert = len(pheno_df)
        
        logger.warning(
            "Phenotype non-numeric values removed: "
            f"count={nan_count}, before={original_count_before_convert}, remaining={remaining_count_after_convert}"
        )
        
        if pheno_df.empty:
            raise ValueError("After removing non-convertible samples, phenotype file has no valid data, cannot continue processing")

    # 显式释放 + 强制GC
    gc.collect()
    return pheno_df

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
    
    unique_values = np.unique(phenotype_values)
    unique_count = len(unique_values)
    
    # 检查是否都是整数
    is_all_integer = np.all(phenotype_values == np.round(phenotype_values))
    
    # 判断规则:
    # - 唯一值 <= 10 且都是整数 -> 分类
    # - 唯一值 > 10 或包含小数 -> 回归
    if unique_count <= 10 and is_all_integer:
        task_type = "classification"
        logger.info(f"Phenotype type: classification ({unique_count} classes)")
    else:
        task_type = "regression"
        logger.info("Phenotype type: regression")
    
    # 绘制可视化图表（仅分类任务，回归任务在匹配后绘制）
    if MATPLOTLIB_AVAILABLE:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if task_type == "classification":
            # 分类任务:绘制饼图
            value_counts = pd.Series(phenotype_values).value_counts().sort_index()
            
            plt.figure(figsize=(10, 6))
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(f"Phenotype Distribution (Classification)\nUnique Values: {unique_count}", fontsize=14)
            plt.axis('equal')
            
            plot_file = output_dir / "phenotype_distribution_pie.png"
            plt.savefig(plot_file, dpi=600, bbox_inches='tight')
            plt.close()
        # 回归任务的绘图在匹配后统一进行
    
    return task_type


def convert_phenotype_dtype(pheno_df: pd.DataFrame, task_type: Literal["regression", "classification"]) -> pd.DataFrame:
    """
    根据任务类型转换表型值数据类型：
    - 分类任务：转为 np.int32（整数）
    - 回归任务：转为 np.float32（浮点数，精度足够且内存减半）
    """
    if task_type == "classification":
        # 分类任务：转为 np.int32
        pheno_df["phenotype"] = pheno_df["phenotype"].astype(np.int32)
    else:
        # 回归任务：转为 np.float32（精度足够且内存减半）
        pheno_df["phenotype"] = pheno_df["phenotype"].astype(np.float32)
    
    # 显式释放 + 强制GC
    gc.collect()
    return pheno_df

def plot_regression_phenotype_distribution(pheno_df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制回归任务的表型分布图（在剔除表型值为0的样本和匹配后）
    
    :param pheno_df: 匹配后的表型数据
    :param output_dir: 输出目录
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
        return
    
    if not MATPLOTLIB_AVAILABLE:
        return
    
    phenotype_values = pheno_df["phenotype"].values
    unique_values = np.unique(phenotype_values)
    unique_count = len(unique_values)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 回归任务:绘制密度分布直方图
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
    logger.info(f"回归任务表型分布图已保存: {plot_file}")

def filter_samples_by_phenotype(plink_prefix: str, pheno_df: pd.DataFrame, output_prefix: str) -> str:
    """按表型ID过滤PLINK样本"""
    output_prefix_abs = Path(output_prefix).absolute().as_posix()
    keep_file = f"{output_prefix_abs}_keep.txt"

    # 使用新的辅助函数生成keep文件
    create_keep_file(pheno_df, keep_file)

    filter_cmd = [
        PLINK_EXECUTABLE,
        "--bfile", Path(plink_prefix).absolute().as_posix(),
        "--keep", keep_file,
        "--make-bed",
        "--out", output_prefix_abs,
        "--allow-extra-chr",
        "--allow-no-sex",
    ]

    try:
        subprocess.run(filter_cmd, capture_output=True, text=True, check=True)
        delete_temp_files(output_prefix_abs, ["_keep.txt"])
        if "temp_plink" in plink_prefix:
            delete_temp_files(plink_prefix, [".bed", ".bim", ".fam", ".log", ".nosex"])
        return output_prefix_abs
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK样本过滤失败:{e.stderr}")
        raise RuntimeError(f"样本过滤失败: {e.stderr}")


def filter_snps_by_quality(
    plink_prefix: str, 
    output_prefix: str,
    maf: float = 0.05,
    geno: float = 0.2,
) -> str:
    """
    全局SNP质量过滤（MAF和缺失率）
    
    重要说明：
    - PLINK的--maf和--geno参数默认在整个数据集（所有染色体）上计算，不区分染色体
    - MAF（最小等位基因频率）应在整个样本集上计算，而不是分染色体计算
    - 缺失率（geno）也应在整个样本集上计算
    - 此函数在分染色体处理前先进行全局过滤，可以显著减少后续处理的SNP数量
    
    :param plink_prefix: 输入PLINK文件前缀
    :param output_prefix: 输出PLINK文件前缀
    :param maf: 最小等位基因频率阈值（在整个数据集上计算）
    :param geno: 最大缺失率阈值（在整个数据集上计算）
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
    ]
    
    try:
        result = subprocess.run(
            filter_cmd, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            logger.error(f"SNP quality filtering failed: {error_msg[:500]}")
            raise RuntimeError(f"SNP quality filtering failed: {error_msg[:500]}")
        
        # 统计过滤后的SNP数量
        bim_file = f"{output_prefix_abs}.bim"
        if Path(bim_file).exists():
            snp_count = sum(1 for _ in open(bim_file))
        else:
            logger.warning("Cannot count filtered SNPs")
        
        # 清理临时文件
        if "temp_plink" in plink_prefix:
            delete_temp_files(plink_prefix, [".bed", ".bim", ".fam", ".log", ".nosex"])
        
        return output_prefix_abs
    except Exception as e:
        logger.error(f"SNP quality filtering exception: {str(e)}")
        raise


def generate_gwas_phenotype_file_preprocess(
    pheno_df: pd.DataFrame,
    output_file: Union[str, Path]
) -> str:
    """
    在预处理阶段为GWAS生成表型文件（两列：IID PHENO）
    
    说明：
    - gemma_gwas.merge_phenotype_to_fam 支持两列表型格式：IID PHENO
    - 这里统一使用 pheno_df['sample'] 作为 IID 列，pheno_df['phenotype'] 作为 PHENO 列
    """
    output_path = Path(output_file).absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    

    with open(output_path, "w") as f:
        for sample_id, pheno in zip(pheno_df["sample"], pheno_df["phenotype"]):
            f.write(f"{sample_id}\t{pheno}\n")
    
    return output_path.as_posix()


def run_gwas_preprocess(
    plink_prefix: str,
    pheno_df: pd.DataFrame,
    output_dir: Union[str, Path],
    tmp_dir: Union[str, Path],
    pvalue_threshold: float = 0.01,
    perform_plink_qc: bool = True,
    keep_temp_files: bool = False,
) -> List[str]:
    """
    预处理阶段运行GWAS，返回显著SNP列表（使用原始SNP ID或rsID）
    
    流程：
    1. 在 tmp_dir/gwas 下生成两列表型文件（IID PHENO）
    2. 切换工作目录到 output_dir，调用 gemma_gwas.run_complete_gwas_pipeline
    3. 在 output_dir/output/ 目录下读取 {basename(output_prefix)}_gwas.assoc.txt
    4. 根据 p 值阈值筛选显著SNP
    """
    output_dir_path = Path(output_dir).absolute()
    tmp_dir_path = Path(tmp_dir).absolute()
    
    gwas_work_dir = tmp_dir_path / "gwas"
    gwas_work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成GWAS表型文件（两列：IID PHENO）
    gwas_pheno_file = gwas_work_dir / "pheno_for_gwas.txt"
    try:
        pheno_file_abs = generate_gwas_phenotype_file_preprocess(pheno_df, gwas_pheno_file)
        if not Path(pheno_file_abs).exists():
            raise FileNotFoundError(f"表型文件生成失败：{pheno_file_abs}")
    except Exception as e:
        raise RuntimeError(f"生成表型文件出错：{str(e)}") from e
    
    # 2. 修正：GWAS输出前缀直接放在output_dir下，避免路径混乱
    gwas_output_prefix = output_dir_path / "preprocess_gwas"  # 改为output_dir下
    gwas_output_prefix_abs = gwas_output_prefix.absolute().as_posix()
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    
    # 结果文件名与 gemma_gwas.run_gemma_gwas 中的 -o 前缀保持一致：
    # 在该函数中，-o 使用的是 `{basename(output_prefix)}_gwas` 作为前缀，
    # 因此这里预期的结果文件名为 `{base_prefix}_gwas.assoc.txt`
    base_prefix = gwas_output_prefix.name  # 例如 "preprocess_gwas"
    gwas_result_file = output_dir_path / "output" / f"{base_prefix}_gwas.assoc.txt"
    
    original_cwd = os.getcwd()
    try:
        # GEMMA 默认在当前工作目录下创建 output/ 目录
        os.chdir(output_dir_path)
        # 补充：捕获GEMMA执行异常
        logger.info(f"开始执行GWAS流程，输出前缀：{gwas_output_prefix_abs}，perform_plink_qc={perform_plink_qc}")
        gemma_gwas.run_complete_gwas_pipeline(
            input_plink_prefix=plink_prefix_abs,
            phenotype_file=pheno_file_abs,
            output_prefix=gwas_output_prefix_abs,
            perform_plink_qc=perform_plink_qc,
        )
    except Exception as e:
        raise RuntimeError(f"GEMMA GWAS流程执行失败：{str(e)}") from e
    finally:
        os.chdir(original_cwd)
    
    # 3. 检查GWAS结果文件是否存在
    logger.info(f"检查GWAS结果文件：{gwas_result_file}")
    if not gwas_result_file.exists():
        # 补充：打印目录下的所有文件，辅助排查
        output_subdir = output_dir_path / "output"
        if output_subdir.exists():
            files_in_output = os.listdir(output_subdir)
            logger.error(f"GWAS结果文件缺失，预期文件：{gwas_result_file.name}")
            logger.error(f"output目录下的文件：{files_in_output}")
            # 尝试查找类似的文件名
            matching_files = [f for f in files_in_output if "gwas" in f.lower() and "assoc" in f.lower()]
            if matching_files:
                logger.warning(f"找到可能的GWAS结果文件：{matching_files}")
        else:
            logger.error(f"GEMMA未创建output目录：{output_subdir}")
        raise FileNotFoundError(f"GWAS result file missing: {gwas_result_file.absolute()}")
    else:
        logger.info(f"GWAS结果文件已找到：{gwas_result_file}")
    
    # 4. 解析GWAS结果并筛选显著SNP
    try:
        gwas_df = pd.read_csv(gwas_result_file, sep="\t")
    except Exception as e:
        raise RuntimeError(f"读取GWAS结果文件失败：{str(e)}") from e
    
    # 查找P值列
    pvalue_col = None
    for col in ["p_wald", "p_lrt", "p_score"]:
        if col in gwas_df.columns:
            pvalue_col = col
            break
    if not pvalue_col:
        raise ValueError(f"Cannot identify P-value column in GWAS results: {gwas_result_file}\n可用列：{gwas_df.columns.tolist()}")
    
    # 查找SNP ID列
    snp_col = None
    for col in ["rs", "SNP"]:
        if col in gwas_df.columns:
            snp_col = col
            break
    if not snp_col:
        snp_col = gwas_df.columns[0]
        logger.warning(f"未找到rs/SNP列，使用第一列 {snp_col} 作为SNP ID列")
    
    # 筛选显著SNP（去重+去空值）
    significant_snps = gwas_df[gwas_df[pvalue_col] < pvalue_threshold][snp_col].dropna().astype(str).unique().tolist()
    logger.info(f"GWAS in preprocess completed: {len(significant_snps):,} significant SNPs (p < {pvalue_threshold})")

    # 5. 清理GWAS产生的所有临时文件和目录（包括output目录）
    if not keep_temp_files:
        try:
            logger.info("开始清理GWAS临时文件...")
            import shutil
            # 以 gwas_output_prefix 作为基础前缀
            gwas_prefix_base = str(Path(gwas_output_prefix_abs))
            # 1）PLINK 质控及过滤产生的中间PLINK文件
            delete_temp_files(f"{gwas_prefix_base}_clean_geno", [".bed", ".bim", ".fam", ".log", ".nosex"])
            delete_temp_files(f"{gwas_prefix_base}_clean_geno_filtered", [".bed", ".bim", ".fam", ".log", ".nosex"])
            # 2）PCA 相关文件
            delete_temp_files(f"{gwas_prefix_base}_geno_pca", [".eigenvec", ".eigenval", ".log"])
            delete_temp_files(f"{gwas_prefix_base}_geno_pca_filtered", [".eigenvec"])
            # 3）样本列表和协变量文件
            delete_temp_files(gwas_prefix_base, ["_valid_samples.txt"])
            delete_temp_files(gwas_prefix_base, ["_covariates_filtered.txt"])
            # 4）删除整个 output 目录（包含所有GWAS结果文件，包括assoc结果文件）
            output_subdir = output_dir_path / "output"
            if output_subdir.exists():
                try:
                    shutil.rmtree(output_subdir, ignore_errors=True)
                except Exception as e:
                    pass
            # 5）删除GWAS输出前缀相关的所有文件
            gwas_prefix_path = Path(gwas_output_prefix_abs)
            if gwas_prefix_path.parent.exists():
                for fp in gwas_prefix_path.parent.glob(f"{gwas_prefix_path.name}*"):
                    try:
                        if fp.is_file():
                            fp.unlink()
                    except Exception:
                        pass
            # 6）GWAS 工作子目录（仅存放表型临时文件）
            try:
                if gwas_work_dir.exists():
                    shutil.rmtree(gwas_work_dir, ignore_errors=True)
            except Exception:
                pass
            logger.info("GWAS临时文件清理完成")
        except Exception as e:
            logger.warning(f"清理GWAS中间文件时发生异常，但不影响主流程：{e}")

    return significant_snps

def extract_snps_with_plink(
    plink_prefix: str,
    snp_list: List[str],
    output_prefix: str,
) -> str:
    """
    使用PLINK --extract 从给定PLINK前缀中提取指定SNP，生成新的PLINK前缀
    """
    if not snp_list:
        raise ValueError("SNP list for extraction is empty")
    
    output_prefix_abs = Path(output_prefix).absolute().as_posix()
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    
    snp_file = f"{output_prefix_abs}_extract_snps.txt"
    with open(snp_file, "w") as f:
        for snp in snp_list:
            f.write(f"{snp}\n")
    
    cmd = [
        PLINK_EXECUTABLE,
        "--bfile", plink_prefix_abs,
        "--extract", snp_file,
        "--make-bed",
        "--out", output_prefix_abs,
        "--allow-extra-chr",
        "--allow-no-sex"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            logger.error(f"PLINK extract failed: {error_msg[:500]}")
            raise RuntimeError(f"PLINK extract failed: {error_msg[:500]}")
        
        # 验证生成的PLINK文件
        bed_file = f"{output_prefix_abs}.bed"
        bim_file = f"{output_prefix_abs}.bim"
        fam_file = f"{output_prefix_abs}.fam"
        for fp in [bed_file, bim_file, fam_file]:
            if not Path(fp).exists():
                raise FileNotFoundError(f"Expected PLINK file not generated: {fp}")
        
        logger.info(f"PLINK extract completed: {len(snp_list):,} SNPs retained")
        return output_prefix_abs
    finally:
        # 删除SNP列表临时文件
        try:
            Path(snp_file).unlink()
        except Exception:
            pass

def generate_metadata(
    output_prefix: str,
    plink_prefix: str,
    train_file: str,
    valid_samples: List[str],
    genotype_format: str,
    task_type: Optional[Literal["regression", "classification"]] = None,
    preprocess_tmp_dir: Optional[Union[str, Path]] = None,
    filter_snps: bool = True,
    feature_selection_mode: int = 1,
    gwas_pvalue: Optional[float] = None,
    ld_window_kb: Optional[int] = None,
    ld_window: Optional[int] = None,
    ld_window_r2: Optional[float] = None
) -> None:
    """
    生成元数据文件
    
    :param plink_prefix: PLINK二进制文件前缀（绝对路径），供GWAS/LD使用
    :param filter_snps: 是否进行了SNP质量过滤
    """
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
        "parallel_processes": 1,  # 单线程单染色体处理
        "processing_mode": "single_thread_sequential",  # 处理模式：单线程串行
        "plink_memory_limit_MB": "unlimited (no --memory passed to PLINK)",
        "snp_dtype": str(SNP_DTYPE),
        "snp_naming_rule": "染色体_物理位置（如'1_123456'）",
        "plink_executable": PLINK_EXECUTABLE,
        "plink_recode_param": "--recodeA",
        # 为简化后续流程并避免依赖 tmp 目录的 PLINK 中间文件，不再在元数据中记录 gwas_genotype_prefix
        # 如有需要，可在GWAS/LD模块内部根据实际路径重新指定
        "gwas_genotype_prefix": None,
        "preprocess_tmp_dir": str(Path(preprocess_tmp_dir).absolute()) if preprocess_tmp_dir else None,
        "snp_filtering": {
            "filtered": filter_snps,
            "maf_threshold": 0.05 if filter_snps else None,
            "geno_threshold": 0.2 if filter_snps else None
        },
        "feature_selection": {
            "mode": feature_selection_mode,
            "description": {
                1: "no GWAS/LD feature selection",
                2: "GWAS-based SNP selection",
                3: "LD-based SNP pruning",
                4: "GWAS + LD combined filtering (GWAS first, then LD)"
            }.get(feature_selection_mode, "unknown"),
            "gwas_pvalue_threshold": gwas_pvalue if feature_selection_mode in [2, 4] else None,
            "ld_window_kb": ld_window_kb if feature_selection_mode in [3, 4] else None,
            "ld_window": ld_window if feature_selection_mode in [3, 4] else None,
            "ld_window_r2": ld_window_r2 if feature_selection_mode in [3, 4] else None
        }
    }
    
    # 添加表型值类别（如果提供）
    if task_type:
        metadata["task_type"] = task_type
    

    metadata_file = f"{output_prefix}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"元数据保存完成")

def validate_plink_files(genotype_path: str, fmt: str) -> None:
    """校验PLINK文件完整性"""
    required = {
        "plink_binary": [".bed", ".bim", ".fam"],
        "plink_text": [".map", ".ped"]
    }.get(fmt, [])

    missing = [f"{genotype_path}{ext}" for ext in required if not Path(f"{genotype_path}{ext}").exists()]
    if missing:
        raise FileNotFoundError(f"\nPLINK文件缺失:{', '.join(missing)}")

def create_keep_file(pheno_df: pd.DataFrame, output_file: str) -> str:
    """
    创建PLINK --keep文件（包含有表型的样本ID）
    
    :param pheno_df: 表型DataFrame，必须包含'sample'列
    :param output_file: 输出keep文件路径
    :return: keep文件路径
    """
    output_file = Path(output_file).absolute().as_posix()
    
    # 生成keep文件（PLINK格式：FID IID，这里FID和IID都使用sample ID）
    # 如果pheno_df有索引，使用索引；否则使用sample列
    if 'sample' in pheno_df.columns:
        sample_ids = pheno_df["sample"].astype(str).unique()
    elif pheno_df.index.name == 'sample' or pheno_df.index.name is None:
        sample_ids = pheno_df.index.astype(str).unique()
    else:
        # 如果索引列名不是sample，尝试使用第一列
        sample_ids = pheno_df.iloc[:, 0].astype(str).unique()
    
    # PLINK keep文件格式：FID IID（两列，用空格或tab分隔）
    with open(output_file, 'w') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\t{sample_id}\n")
    
    return output_file

def get_cache_key(bim_file: str, chr_num: Optional[str] = None) -> str:
    """
    生成缓存键（基于文件路径和修改时间）
    
    :param bim_file: .bim文件路径
    :param chr_num: 染色体号
    :return: 缓存键字符串
    """
    try:
        file_mtime = os.path.getmtime(bim_file)
        key_parts = [bim_file, str(chr_num), str(file_mtime)]
        return hashlib.md5(str(key_parts).encode()).hexdigest()
    except Exception:
        # 如果无法获取文件修改时间，使用文件路径和染色体号
        return hashlib.md5(f"{bim_file}_{chr_num}".encode()).hexdigest()

def load_cached_mapping(cache_key: str) -> Optional[dict]:
    """
    从磁盘加载缓存的SNP映射
    
    :param cache_key: 缓存键
    :return: 缓存的映射字典，如果不存在则返回None
    """
    # 使用全局缓存系统，而不是单独的磁盘缓存
    # 这里返回None，让调用者使用GLOBAL_CACHE
    return None

def save_cached_mapping(cache_key: str, mapping: dict) -> None:
    """
    保存SNP映射到磁盘缓存
    
    :param cache_key: 缓存键
    :param mapping: 映射字典
    """
    # 使用全局缓存系统，不需要单独的磁盘缓存操作
    # 映射已经保存在_snp_mapping_cache中
    pass

@functools.lru_cache(maxsize=32)
def get_snp_chr_pos_mapping_cached(bim_file: str, chr_num: Optional[str] = None) -> dict:
    """
    带缓存的SNP映射函数（使用LRU缓存）
    注意：由于functools.lru_cache不支持可变参数，这里使用文件路径+染色体号作为键
    """
    # 检查内存缓存
    cache_key = f"{bim_file}_{chr_num}"
    if cache_key in _snp_mapping_cache:
        return _snp_mapping_cache[cache_key]
    
    # 检查全局缓存系统
    global_cache_key = f"snp_mapping_{cache_key}"
    cached_result = GLOBAL_CACHE.get(global_cache_key, check_file_mtime=True, file_path=bim_file)
    if cached_result is not None:
        _snp_mapping_cache[cache_key] = cached_result
        return cached_result
    
    # 计算映射
    mapping = get_snp_chr_pos_mapping(bim_file, chr_num)
    
    # 保存到缓存
    _snp_mapping_cache[cache_key] = mapping
    GLOBAL_CACHE.set(global_cache_key, mapping, file_path=bim_file, ttl_seconds=3600)
    
    return mapping


# ======================== 10. 命令行入口 ========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="优化的预处理模块（第一层优化）")
    parser.add_argument("-g", "--genotype", required=True, help="基因型文件（VCF/PLINK二进制/文本前缀）")
    parser.add_argument("-p", "--phenotype", required=True, help="表型文件（无表头2列）")
    parser.add_argument("-o", "--output", required=True, help="输出目录或文件路径")
    parser.add_argument("--mem-limit", type=int, default=10000, help="（兼容参数）PLINK内存限制")
    parser.add_argument("--pheno-col", help="兼容参数，无实际作用")
    parser.add_argument("--no-filter-snps", action="store_true", help="不进行SNP质量过滤")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument(
        "--feature-selection-mode",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="特征筛选模式: 1=不做GWAS/LD; 2=仅GWAS; 3=仅LD; 4=GWAS+LD(先GWAS再LD)"
    )
    parser.add_argument(
        "--gwas-pvalue",
        type=float,
        default=0.01,
        help="GWAS显著性阈值（用于模式2和4），默认0.01"
    )
    parser.add_argument(
        "--ld-config",
        type=str,
        default="50,5,0.2",
        help=(
            "LD三合一配置参数 (standardized): "
            "\"<window_kb>,<window_variants>,<r2_threshold>\", "
            "例如 \"50,5,0.2\" 表示窗口50KB、窗口内5个变体、r²阈值0.2；"
            "仅在特征筛选模式为3或4时使用"
        ),
    )

    
    args = parser.parse_args()
    
    # 执行优化的预处理
    sys.exit(run_preprocess(
        genotype_file=args.genotype,
        phenotype_file=args.phenotype,
        output_file=args.output,
        pheno_col=args.pheno_col,
        filter_snps=not args.no_filter_snps,
        use_cache=not args.no_cache,
        feature_selection_mode=args.feature_selection_mode,
        gwas_pvalue=args.gwas_pvalue,
        ld_config=args.ld_config,
    ))