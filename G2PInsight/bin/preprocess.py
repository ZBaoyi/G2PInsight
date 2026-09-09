#!/usr/bin/env python3
import argparse
import os
import sys
import json
import gc
import re
import time
import logging
import subprocess
import shutil
import warnings
import pandas as pd
import numpy as np
import mmap
import functools
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Optional, Dict, List, Literal, Tuple, Union, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from G2PInsight.bin import gemma_gwas, plink_ld
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug('psutil not installed, memory monitoring functionality limited')
try:
    from bed_reader import open_bed
    BED_READER_AVAILABLE = True
except ImportError:
    BED_READER_AVAILABLE = False
SNP_DTYPE = np.int8
PREPROCESS_SNP_WARN_THRESHOLD = 500_000
PREPROCESS_SNP_STRONG_WARN_THRESHOLD = 1_000_000
TRAIN_MATRIX_COL_WRITE_CHUNK = 50_000
TRAIN_TEST_HOLDOUT_RATIO = 0.20
DEFAULT_SPLIT_RANDOM_STATE = 42

@dataclass
class TrainingMatrixOutput:
    train_file: str
    sample_ids: List[str]
    n_snp_features: int
    includes_phenotype_column: bool = True

def _count_bim_snps(plink_prefix: str) -> int:
    bim_file = Path(plink_prefix).with_suffix('.bim')
    if not bim_file.exists():
        bim_file = Path(f'{plink_prefix}.bim')
    if not bim_file.exists():
        return 0
    count = 0
    with open(bim_file, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def _warn_large_snp_count(n_snps: int, feature_selection_mode: int, filter_snps: bool) -> None:
    if n_snps <= 0:
        return
    if not filter_snps:
        logger.debug(f'--no-filter-snps is enabled; {n_snps:,} SNP(s) may enter the training matrix. Use default SNP QC unless you intentionally need all variants.')
    if n_snps >= PREPROCESS_SNP_STRONG_WARN_THRESHOLD:
        logger.debug(f'Very large SNP count: {n_snps:,}. Preprocess and training may be slow or run out of memory. Use -f 2 or -f 4 to reduce features before training.')
    elif n_snps >= PREPROCESS_SNP_WARN_THRESHOLD and feature_selection_mode == 1:
        logger.debug(f'Large SNP count: {n_snps:,} with -f 1 (no GWAS/LD). Consider -f 4 (GWAS + LD) before training.')
GenotypeFormat = Literal['vcf', 'plink_binary', 'plink_text']
GenotypeBackend = Literal['auto', 'bed', 'raw']

class VectorizedProcessor:

    @staticmethod
    def batch_rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        if not mapping:
            return df
        existing_cols = [col for col in mapping.keys() if col in df.columns]
        if not existing_cols:
            return df
        rename_dict = {col: mapping[col] for col in existing_cols}
        return df.rename(columns=rename_dict)

    @staticmethod
    def _convert_chunk(chunk_data: np.ndarray, is_numeric: bool, target_dtype: type) -> np.ndarray:
        if is_numeric:
            return chunk_data.astype(target_dtype)
        else:
            try:
                return chunk_data.astype(float).astype(target_dtype)
            except (ValueError, TypeError):
                chunk_converted = pd.to_numeric(chunk_data, errors='coerce').values
                return np.nan_to_num(chunk_converted, nan=-1).astype(target_dtype)

    @staticmethod
    def vectorized_type_conversion(df: pd.DataFrame, exclude_cols: List[str]=None, target_dtype: type=np.int8) -> pd.DataFrame:
        if df.empty:
            return df
        if exclude_cols is None:
            exclude_cols = ['sample']
        cols_to_convert = [col for col in df.columns if col not in exclude_cols]
        if not cols_to_convert:
            return df
        data_to_convert = df[cols_to_convert]
        total_cols = len(cols_to_convert)
        arr = data_to_convert.values
        arr_dtype = arr.dtype
        is_numeric = np.issubdtype(arr_dtype, np.number)
        try:
            total_elements = arr.size
            if total_elements > 50000000:
                chunk_size = 10000000
                total_chunks = (total_elements + chunk_size - 1) // chunk_size
                chunk_ranges = []
                for chunk_start in range(0, total_elements, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_elements)
                    chunk_ranges.append((chunk_start, chunk_end))
                arr_converted_list = []
                for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                    if chunk_idx % 10 == 0:
                        pass
                    chunk_flat = arr.ravel()[chunk_start:chunk_end]
                    chunk_converted = VectorizedProcessor._convert_chunk(chunk_flat, is_numeric, target_dtype)
                    arr_converted_list.append(chunk_converted)
                arr_converted = np.concatenate(arr_converted_list).reshape(arr.shape)
            elif is_numeric:
                arr_converted = arr.astype(target_dtype)
            else:
                try:
                    arr_converted = arr.astype(float).astype(target_dtype)
                except (ValueError, TypeError):
                    arr_flat = pd.to_numeric(arr.ravel(), errors='coerce').values
                    arr_converted = arr_flat.reshape(arr.shape)
                    arr_converted = np.nan_to_num(arr_converted, nan=-1).astype(target_dtype)
            df[cols_to_convert] = arr_converted
        except Exception as e:
            logger.debug(f'Bulk vectorized conversion failed, falling back to column-wise conversion: {e}')
            if total_cols > 1000:
                batch_size = 1000
                for i in range(0, total_cols, batch_size):
                    batch_cols = cols_to_convert[i:i + batch_size]
                    if i // batch_size % 10 == 0:
                        pass
                    for col in batch_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(target_dtype)
            else:
                for col in cols_to_convert:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(target_dtype)
        return df

    @staticmethod
    def batch_parse_plink_columns(col_names: List[str]) -> Dict[str, str]:
        mapping = {}
        pattern = re.compile('^([^:]+):([^_]+)_[^_]+$')
        for col in col_names:
            match = pattern.match(str(col))
            if match:
                chr_part = match.group(1)
                pos_part = match.group(2)
                mapping[col] = f'{chr_part}_{pos_part}'
        return mapping

    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame, categorical_threshold: float=0.5) -> pd.DataFrame:
        if df.empty:
            return df
        optimized_df = df.copy()
        for col in optimized_df.columns:
            col_data = optimized_df[col]
            dtype = col_data.dtype
            if dtype.name == 'category' or dtype == object:
                continue
            if len(col_data) > 0:
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < categorical_threshold:
                    optimized_df[col] = col_data.astype('category')
                    continue
            if pd.api.types.is_integer_dtype(dtype):
                col_min = col_data.min()
                col_max = col_data.max()
                if col_min >= 0:
                    if col_max <= 255:
                        optimized_df[col] = col_data.astype(np.uint8)
                    elif col_max <= 65535:
                        optimized_df[col] = col_data.astype(np.uint16)
                    elif col_max <= 4294967295:
                        optimized_df[col] = col_data.astype(np.uint32)
                elif col_min >= -128 and col_max <= 127:
                    optimized_df[col] = col_data.astype(np.int8)
                elif col_min >= -32768 and col_max <= 32767:
                    optimized_df[col] = col_data.astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized_df[col] = col_data.astype(np.int32)
        return optimized_df

class SmartCache:

    def __init__(self, max_size_mb: int=512, max_items: int=1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_items = max_items
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        self.stats = {'total_requests': 0, 'hits': 0, 'misses': 0, 'evictions': 0, 'size_mb': 0}

    def _get_item_size(self, item: Any) -> int:
        try:
            if isinstance(item, (pd.DataFrame, pd.Series)):
                return item.memory_usage(deep=True).sum()
            elif isinstance(item, np.ndarray):
                return item.nbytes
            elif isinstance(item, dict):
                return sum((len(str(k)) + len(str(v)) for k, v in item.items())) * 2
            else:
                return sys.getsizeof(item)
        except:
            return 1024 * 1024

    def _evict_if_needed(self, new_item_size: int):
        with self.lock:
            while (self.current_size_bytes + new_item_size > self.max_size_bytes or len(self.cache) >= self.max_items) and self.cache:
                key, item = self.cache.popitem(last=False)
                self.current_size_bytes -= item['size']
                self.stats['evictions'] += 1

    def get(self, key: str, check_file_mtime: bool=False, file_path: Optional[str]=None) -> Optional[Any]:
        with self.lock:
            self.stats['total_requests'] += 1
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            cache_item = self.cache[key]
            if check_file_mtime and file_path:
                try:
                    file_mtime = os.path.getmtime(file_path)
                    cached_mtime = cache_item.get('file_mtime', 0)
                    if file_mtime > cached_mtime:
                        self._delete(key)
                        self.stats['misses'] += 1
                        return None
                except (OSError, FileNotFoundError) as e:
                    self._delete(key)
                    self.stats['misses'] += 1
                    return None
            self.cache.move_to_end(key)
            cache_item['last_access'] = time.time()
            cache_item['access_count'] = cache_item.get('access_count', 0) + 1
            self.stats['hits'] += 1
            return cache_item['data']

    def set(self, key: str, data: Any, file_path: Optional[str]=None, ttl_seconds: Optional[int]=None):
        with self.lock:
            data_size = self._get_item_size(data)
            self._evict_if_needed(data_size)
            file_mtime = 0
            if file_path:
                try:
                    file_mtime = os.path.getmtime(file_path)
                except (OSError, FileNotFoundError):
                    pass
            cache_item = {'data': data, 'size': data_size, 'created': time.time(), 'last_access': time.time(), 'access_count': 1, 'file_path': file_path, 'file_mtime': file_mtime, 'ttl': ttl_seconds}
            self.cache[key] = cache_item
            self.current_size_bytes += data_size
            self.stats['size_mb'] = self.current_size_bytes / (1024 * 1024)

    def _delete(self, key: str):
        if key in self.cache:
            item = self.cache.pop(key)
            self.current_size_bytes -= item['size']

    def delete(self, key: str):
        with self.lock:
            self._delete(key)

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.stats['size_mb'] = 0

    def cleanup_expired(self):
        with self.lock:
            current_time = time.time()
            expired_keys = []
            for key, item in self.cache.items():
                if item.get('ttl'):
                    if current_time - item['created'] > item['ttl']:
                        expired_keys.append(key)
                        continue
                file_path = item.get('file_path')
                if file_path:
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        if file_mtime > item.get('file_mtime', 0):
                            expired_keys.append(key)
                    except (OSError, FileNotFoundError):
                        expired_keys.append(key)
            for key in expired_keys:
                self._delete(key)

    def get_stats(self) -> Dict:
        with self.lock:
            stats = self.stats.copy()
            stats['current_items'] = len(self.cache)
            stats['hit_ratio'] = stats['hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            return stats

    def print_stats(self):
        stats = self.get_stats()
GLOBAL_CACHE = SmartCache(max_size_mb=1024 * 4)

def cached(key_func: Optional[Callable]=None, ttl_seconds: Optional[int]=None):

    def _make_hashable(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return tuple((_make_hashable(item) for item in obj))
        elif isinstance(obj, dict):
            return tuple(sorted(((k, _make_hashable(v)) for k, v in obj.items())))
        elif isinstance(obj, pd.DataFrame):
            try:
                data_hash = hashlib.md5(obj.values.tobytes() if hasattr(obj.values, 'tobytes') else str(obj.values).encode()).hexdigest()[:16]
                return f'df_{obj.shape}_{tuple(obj.columns)}_{data_hash}'
            except:
                return f'df_{id(obj)}'
        elif isinstance(obj, np.ndarray):
            try:
                data_hash = hashlib.md5(obj.tobytes()).hexdigest()[:16]
                return f'ndarray_{obj.shape}_{obj.dtype}_{data_hash}'
            except:
                return f'ndarray_{id(obj)}'
        elif isinstance(obj, Path):
            try:
                path_str = str(obj)
                if obj.exists():
                    mtime = os.path.getmtime(path_str)
                    return f'path_{path_str}_{mtime}'
                return f'path_{path_str}'
            except:
                return f'path_{str(obj)}'
        else:
            try:
                return str(obj)
            except:
                return f'obj_{id(obj)}'

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                hashable_args = tuple((_make_hashable(arg) for arg in args))
                hashable_kwargs = tuple(sorted(((k, _make_hashable(v)) for k, v in kwargs.items())))
                key_parts = (func.__name__, hashable_args, hashable_kwargs)
                cache_key = hashlib.md5(pickle.dumps(key_parts, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
            cached_result = GLOBAL_CACHE.get(cache_key)
            if cached_result is not None:
                return cached_result
            result = func(*args, **kwargs)
            GLOBAL_CACHE.set(cache_key, result, ttl_seconds=ttl_seconds)
            return result
        return wrapper
    return decorator

class PLINKOptimizer:

    @staticmethod
    def optimize_plink_parameters(available_memory_mb: Optional[float]=None) -> Dict[str, Any]:
        if available_memory_mb is None and PSUTIL_AVAILABLE:
            try:
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            except:
                available_memory_mb = 4096
        params = {'memory_mb': 4096, 'window_size': 50, 'window_kb': 1000}
        cpu_count = os.cpu_count() or 1
        params['threads'] = int(min(4, cpu_count))
        if available_memory_mb:
            safe_memory = available_memory_mb * 0.7
            params['memory_mb'] = int(min(safe_memory, 32768))
            params['memory_mb'] = max(1024, params['memory_mb'])
        params['performance_flags'] = ['--allow-extra-chr', '--allow-no-sex', '--keep-allele-order', '--nonfounders']
        return params

    @staticmethod
    def build_plink_command(base_cmd: List[str], optimized_params: Dict[str, Any], additional_args: Optional[List[str]]=None) -> List[str]:
        cmd = base_cmd.copy()
        if optimized_params.get('memory_mb'):
            cmd.extend(['--memory', str(int(optimized_params['memory_mb']))])
        if optimized_params.get('threads'):
            cmd.extend(['--threads', str(int(optimized_params['threads']))])
        if optimized_params.get('performance_flags'):
            cmd.extend(optimized_params['performance_flags'])
        if additional_args:
            cmd.extend(additional_args)
        return cmd

class BimIndex:

    _CACHE: Dict[str, 'BimIndex'] = {}
    _BIM_COLUMNS = ['chr', 'snp_id', 'genetic_dist', 'physical_pos', 'a1', 'a2']

    def __init__(self, bim_path: str):
        self.bim_path = str(Path(bim_path).absolute())
        self._df: Optional[pd.DataFrame] = None
        self._raw_mapping_cache: Dict[Optional[str], Dict[str, str]] = {}

    @classmethod
    def from_bim(cls, bim_path: str) -> 'BimIndex':
        key = str(Path(bim_path).absolute())
        if key not in cls._CACHE:
            cls._CACHE[key] = cls(bim_path)
        return cls._CACHE[key]

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._load_df()
        return self._df

    def _load_df(self) -> None:
        if not Path(self.bim_path).exists():
            raise FileNotFoundError(f'.bim file not found: {self.bim_path}')
        dtypes = {'chr': str, 'snp_id': str, 'physical_pos': str, 'a1': str, 'a2': str}
        file_size = Path(self.bim_path).stat().st_size
        if file_size < 100 * 1024 * 1024:
            df = pd.read_csv(self.bim_path, sep='\\s+', header=None, names=self._BIM_COLUMNS, dtype=dtypes)
        else:
            chunks = []
            for chunk in pd.read_csv(self.bim_path, sep='\\s+', header=None, names=self._BIM_COLUMNS, dtype=dtypes, chunksize=100000, engine='python'):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        df['column_name'] = df['chr'] + '_' + df['physical_pos'].astype(str)
        self._df = df

    @property
    def n_snps(self) -> int:
        return len(self.df)

    @property
    def column_names(self) -> List[str]:
        return self.df['column_name'].tolist()

    def ordered_chr_blocks(self, chr_list: List[str]) -> List[Tuple[str, np.ndarray]]:
        blocks: List[Tuple[str, np.ndarray]] = []
        chr_values = self.df['chr'].values
        for chr_num in chr_list:
            indices = np.flatnonzero(chr_values == chr_num)
            if len(indices) > 0:
                blocks.append((chr_num, indices))
        return blocks

    def column_names_for_chr_list(self, chr_list: List[str]) -> List[str]:
        names: List[str] = []
        for _, indices in self.ordered_chr_blocks(chr_list):
            names.extend(self.df['column_name'].values[indices].tolist())
        return names

    def raw_column_mapping(self, chr_num: Optional[str]=None) -> Dict[str, str]:
        if chr_num in self._raw_mapping_cache:
            return self._raw_mapping_cache[chr_num]
        df = self.df if chr_num is None else self.df[self.df['chr'] == chr_num]
        snp_mapping: Dict[str, str] = {}
        for row in df.itertuples(index=False):
            chr_val = str(row.chr)
            pos_val = str(row.physical_pos)
            a1_val = str(row.a1)
            a2_val = str(row.a2)
            snp_id = str(row.snp_id)
            new_name = f'{chr_val}_{pos_val}'
            snp_mapping[snp_id] = new_name
            plink_formats = [f'{chr_val}:{pos_val}_{a1_val}', f'{chr_val}:{pos_val}_{a2_val}', f'{chr_val}:{pos_val}']
            rs_allele_formats = [f'{snp_id}_{a1_val}', f'{snp_id}_{a2_val}', f'{snp_id}_{a1_val}/{a2_val}', f'{snp_id}_{a2_val}/{a1_val}']
            for fmt in plink_formats + rs_allele_formats:
                snp_mapping[fmt] = new_name
        self._raw_mapping_cache[chr_num] = snp_mapping
        return snp_mapping

def _resolve_genotype_backend(genotype_backend: GenotypeBackend, plink_prefix_abs: str) -> Literal['bed', 'raw']:
    bed_file = f'{plink_prefix_abs}.bed'
    if genotype_backend == 'raw':
        return 'raw'
    if genotype_backend == 'bed':
        if not BED_READER_AVAILABLE:
            raise RuntimeError('genotype backend "bed" requires bed-reader; install with: pip install bed-reader')
        if not Path(bed_file).exists():
            raise FileNotFoundError(f'genotype backend "bed" requires PLINK binary .bed file: {bed_file}')
        return 'bed'
    if BED_READER_AVAILABLE and Path(bed_file).exists():
        return 'bed'
    if not BED_READER_AVAILABLE:
        logger.debug('bed-reader not installed; falling back to PLINK --recodeA backend')
    else:
        logger.debug(f'PLINK .bed file not found ({bed_file}); falling back to PLINK --recodeA backend')
    return 'raw'

def _prepare_pheno_df_loaded(pheno_df: pd.DataFrame, tmp_base: Path) -> Tuple[pd.DataFrame, str]:
    temp_pheno_file = str(tmp_base / 'temp_pheno.feather')
    try:
        pheno_for_save = pheno_df.copy()
        if 'sample' not in pheno_for_save.columns:
            pheno_for_save = pheno_for_save.reset_index()
            if 'sample' not in pheno_for_save.columns and 'index' in pheno_for_save.columns:
                pheno_for_save = pheno_for_save.rename(columns={'index': 'sample'})
        else:
            pheno_for_save = pheno_for_save.reset_index(drop=True)
        pheno_for_save.to_feather(temp_pheno_file)
        pheno_df_loaded = pd.read_feather(temp_pheno_file)
        if 'sample' not in pheno_df_loaded.columns:
            raise ValueError("Phenotype data missing required 'sample' column after serialization")
        pheno_df_loaded = pheno_df_loaded.set_index('sample')
        return pheno_df_loaded, temp_pheno_file
    except Exception as e:
        logger.debug(f'Cannot use feather format, using pickle: {e}')
        temp_pheno_file = str(tmp_base / 'temp_pheno.pkl')
        pheno_df.to_pickle(temp_pheno_file)
        pheno_df_loaded = pd.read_pickle(temp_pheno_file)
        if 'sample' in pheno_df_loaded.columns:
            pheno_df_loaded = pheno_df_loaded.set_index('sample')
        return pheno_df_loaded, temp_pheno_file

def _align_pheno_to_bed_samples(pheno_df_loaded: pd.DataFrame, plink_prefix_abs: str) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    fam_file = f'{plink_prefix_abs}.fam'
    if not Path(fam_file).exists():
        raise FileNotFoundError(f'PLINK .fam file not found: {fam_file}')
    fam_df = pd.read_csv(fam_file, sep='\\s+', header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], dtype=str, engine='python')
    iid_to_row = {str(iid): i for i, iid in enumerate(fam_df['IID'].tolist())}
    sample_ids = [str(s) for s in pheno_df_loaded.index.astype(str) if str(s) in iid_to_row]
    if not sample_ids:
        raise ValueError('No overlapping samples between phenotype and genotype')
    bed_row_idx = np.array([iid_to_row[s] for s in sample_ids], dtype=np.int64)
    pheno_aligned = pheno_df_loaded.loc[sample_ids]
    return sample_ids, bed_row_idx, pheno_aligned

def _normalize_output_train_path(output_file_abs: str, n_cells: int) -> Tuple[str, Optional[str]]:
    if not output_file_abs.endswith('.txt') and (not output_file_abs.endswith('.txt.gz')):
        if output_file_abs.endswith('.gz'):
            output_file_abs = output_file_abs.replace('.gz', '.txt.gz')
        else:
            output_file_abs = output_file_abs + '.txt'
    use_gzip = n_cells > 1000000
    if use_gzip and (not output_file_abs.endswith('.gz')):
        output_file_abs = output_file_abs + '.gz'
    compression = 'gzip' if use_gzip else None
    return output_file_abs, compression

def _validate_train_matrix_file(path: str, expected_n_samples: Optional[int]=None) -> None:
    """Verify training matrix exists and gzip stream is complete (not truncated)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Training matrix file not found: {path}')
    size_bytes = p.stat().st_size
    if size_bytes == 0:
        raise RuntimeError(f'Training matrix file is empty: {path}')
    is_gz = str(path).endswith('.gz')
    newline_count = 0
    try:
        if is_gz:
            import gzip
            with gzip.open(path, 'rb') as f:
                while True:
                    chunk = f.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    newline_count += chunk.count(b'\n')
        else:
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    newline_count += chunk.count(b'\n')
    except EOFError as e:
        raise RuntimeError(f'Training matrix gzip file is truncated or corrupt: {path} ({size_bytes / (1024 ** 3):.2f} GB on disk)\nPreprocess was likely interrupted (OOM, disk full, or job kill) before gzip finished writing.\nDelete this file and re-run preprocess. For very large SNP counts, use -f 2 or -f 4 to reduce features before export.') from e
    if expected_n_samples is not None and expected_n_samples > 0:
        data_rows = newline_count - 1
        if data_rows != expected_n_samples:
            raise RuntimeError(f'Training matrix row count mismatch: file has {data_rows} data row(s), expected {expected_n_samples} (path={path}).\nThe export is incomplete; re-run preprocess.')

def _genotypes_bed_to_int8(geno: np.ndarray) -> np.ndarray:
    return np.where(geno == -127, -1, geno).astype(np.int8)

def _write_train_matrix_to_csv(output_file_abs: str, sample_ids: List[str], column_names: List[str], genotype_matrix: np.ndarray, phenotype_values: np.ndarray, compression: Optional[str]) -> None:
    if compression == 'gzip':
        import gzip
        fopen = lambda path, mode, **kw: gzip.open(path, mode, compresslevel=1, **kw)
        mode = 'wt'
    else:
        fopen = open
        mode = 'w'
    n_samples = len(sample_ids)
    n_snps = len(column_names)
    chunk_cols = TRAIN_MATRIX_COL_WRITE_CHUNK
    pheno_str = phenotype_values.astype(str)
    n_cells = n_samples * max(n_snps, 1)
    if n_cells > 1_000_000:
        logger.debug(f'Writing training matrix: {n_samples:,} samples × {n_snps:,} SNPs (compression={compression or "none"})')
    log_every = max(1, n_samples // 20) if n_samples > 50 else 0
    with fopen(output_file_abs, mode, encoding='utf-8') as f:
        f.write('sample')
        for start in range(0, n_snps, chunk_cols):
            f.write('\t')
            f.write('\t'.join(column_names[start:start + chunk_cols]))
        f.write('\tphenotype\n')
        for i in range(n_samples):
            row = genotype_matrix[i]
            f.write(sample_ids[i])
            for start in range(0, n_snps, chunk_cols):
                f.write('\t')
                f.write('\t'.join(row[start:start + chunk_cols].astype(str)))
            f.write('\t')
            f.write(pheno_str[i])
            f.write('\n')
            if log_every and i > 0 and (i % log_every == 0 or i == n_samples - 1):
                logger.debug(f'  Training matrix write progress: {i + 1:,}/{n_samples:,} samples ({100 * (i + 1) / n_samples:.0f}%)')
    _validate_train_matrix_file(output_file_abs, expected_n_samples=n_samples)
    logger.debug(f'Training matrix file validated: {output_file_abs}')

def _plink_to_training_data_from_bed(plink_prefix: str, pheno_df: pd.DataFrame, output_file: str, tmp_dir: Optional[Union[str, Path]]=None, cleanup_registry: Optional[List[str]]=None) -> TrainingMatrixOutput:
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    output_file_path = Path(output_file).absolute()
    output_file_abs = output_file_path.as_posix()
    if tmp_dir is None:
        tmp_base = output_file_path.parent / 'tmp_p'
    else:
        tmp_base = Path(tmp_dir).absolute()
    tmp_base.mkdir(parents=True, exist_ok=True)
    bed_file = f'{plink_prefix_abs}.bed'
    bim_file = f'{plink_prefix_abs}.bim'
    fam_file = f'{plink_prefix_abs}.fam'
    if not Path(bed_file).exists():
        raise FileNotFoundError(f'PLINK .bed file not found: {bed_file}')
    chr_list = auto_detect_chromosomes(plink_prefix_abs)
    if not chr_list:
        raise RuntimeError('No chromosomes detected')
    logger.debug(f'Chromosome scan completed: {len(chr_list)} chromosome(s) detected')
    pheno_df_loaded, temp_pheno_file = _prepare_pheno_df_loaded(pheno_df, tmp_base)
    mmap_path = tmp_base / 'train_matrix.mmap'
    if cleanup_registry is not None:
        cleanup_registry.append(str(mmap_path))
    try:
        if 'phenotype' not in pheno_df_loaded.columns:
            raise ValueError("Phenotype column 'phenotype' not found in phenotype data")
        sample_ids, bed_row_idx, pheno_aligned = _align_pheno_to_bed_samples(pheno_df_loaded, plink_prefix_abs)
        bim_index = BimIndex.from_bim(bim_file)
        column_names = bim_index.column_names_for_chr_list(chr_list)
        n_samples = len(sample_ids)
        n_snps = len(column_names)
        if n_snps == 0:
            raise RuntimeError('No SNPs found in .bim for detected chromosomes')
        logger.debug(f'Reading genotypes from PLINK bed (direct): {n_samples:,} samples × {n_snps:,} SNPs')
        genotype_mm = np.memmap(str(mmap_path), dtype=np.int8, mode='w+', shape=(n_samples, n_snps))
        col_offset = 0
        cpu = os.cpu_count() or 1
        bed_threads = min(8, cpu)
        with open_bed(bed_file, fam_location=fam_file, bim_location=bim_file, count_A1=True, num_threads=bed_threads) as bed:
            for chr_idx, (chr_num, snp_indices) in enumerate(bim_index.ordered_chr_blocks(chr_list), 1):
                n_chr_snps = len(snp_indices)
                logger.debug(f'  Chromosome {chr_num}: reading {n_chr_snps:,} SNP(s) ({chr_idx}/{len(chr_list)})')
                geno = bed.read(np.s_[bed_row_idx, snp_indices], dtype='int8', order='C')
                genotype_mm[:, col_offset:col_offset + n_chr_snps] = _genotypes_bed_to_int8(geno)
                col_offset += n_chr_snps
                del geno
                gc.collect()
        genotype_mm.flush()
        output_file_abs, compression = _normalize_output_train_path(output_file_abs, n_samples * (n_snps + 1))
        _write_train_matrix_to_csv(output_file_abs, sample_ids, column_names, genotype_mm, pheno_aligned['phenotype'].values, compression)
        del genotype_mm
        try:
            os.remove(mmap_path)
        except Exception:
            pass
        logger.debug(f'Final training dataset (bed backend): {n_samples:,} samples, {n_snps:,} feature SNPs')
        return TrainingMatrixOutput(train_file=output_file_abs, sample_ids=sample_ids, n_snp_features=n_snps, includes_phenotype_column=True)
    finally:
        try:
            os.remove(temp_pheno_file)
        except Exception:
            pass

def _make_cache_key_for_bim(bim_file: str, chr_num: Optional[str]=None) -> str:
    bim_path = Path(bim_file).absolute()
    try:
        mtime = os.path.getmtime(str(bim_path))
        key_parts = (str(bim_path), str(chr_num), mtime)
    except:
        key_parts = (str(bim_path), str(chr_num))
    return hashlib.md5(pickle.dumps(key_parts, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()

@cached(key_func=_make_cache_key_for_bim, ttl_seconds=3600)
def get_snp_chr_pos_mapping_optimized(bim_file: str, chr_num: Optional[str]=None) -> dict:
    return BimIndex.from_bim(str(Path(bim_file).absolute())).raw_column_mapping(chr_num)

def get_snp_chr_pos_mapping(bim_file: str, chr_num: Optional[str]=None) -> dict:
    return BimIndex.from_bim(str(Path(bim_file).absolute())).raw_column_mapping(chr_num)

def standardize_snp_column_names(df: pd.DataFrame, bim_file: str, pheno_columns: Optional[List[str]]=None) -> pd.DataFrame:
    if df.empty:
        return df
    bim_file = str(Path(bim_file).absolute())
    if not Path(bim_file).exists():
        logger.debug(f'standardize_snp_column_names: .bim file not found, skip renaming: {bim_file}')
        return df
    pheno_cols: set = set(pheno_columns or [])
    pheno_cols.update({'phenotype'})
    try:
        snp_mapping = get_snp_chr_pos_mapping_optimized(bim_file, chr_num=None)
    except Exception as e:
        logger.debug(f'standardize_snp_column_names: failed to build SNP mapping from .bim: {e}')
        return df
    rename_dict: Dict[str, str] = {}
    pattern_plink = re.compile('^([^:]+):([^_]+)_[^_]+$')
    for col in df.columns:
        if col in pheno_cols or any((str(col).lower().startswith(str(p).lower()) for p in pheno_cols)):
            continue
        col_str = str(col)
        new_name = None
        if col_str in snp_mapping:
            new_name = snp_mapping[col_str]
        else:
            m = pattern_plink.match(col_str)
            if m:
                chr_part, pos_part = (m.group(1), m.group(2))
                new_name = f'{chr_part}_{pos_part}'
        if new_name and new_name != col_str:
            rename_dict[col] = new_name
    if rename_dict:
        df = df.rename(columns=rename_dict)
    else:
        pass
    return df

def get_plink_path() -> str:
    current_file = Path(__file__).absolute()
    software_dir = current_file.parent / 'software'
    plink_candidates = [software_dir / 'plink']
    return str(plink_candidates[0])
PLINK_EXECUTABLE = get_plink_path()

def _process_chunk_parallel(chunk_data: Tuple[int, pd.DataFrame, Dict[str, str], Optional[int], List[str], type]) -> Tuple[int, pd.DataFrame]:
    chunk_idx, chunk_df, batch_mapping, sample_col_idx, original_columns, dtype_obj = chunk_data
    try:
        chunk_df = VectorizedProcessor.batch_rename_columns(chunk_df, batch_mapping)
        if sample_col_idx is not None:
            sample_col_name = original_columns[sample_col_idx]
            if sample_col_name in chunk_df.columns:
                chunk_df = chunk_df.rename(columns={sample_col_name: 'sample'})
        chunk_df = VectorizedProcessor.vectorized_type_conversion(chunk_df, ['sample'], dtype_obj)
        if 'sample' in chunk_df.columns:
            chunk_df = chunk_df.set_index('sample')
        return (chunk_idx, chunk_df)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f'Chunk {chunk_idx} processing failed: {e}')
        raise

def parse_raw_file_optimized_v2(raw_file: str, chr_snp_mapping: Dict[str, str], dtype_obj: type=np.int8, chunk_rows: int=10000, use_cache: bool=True, max_workers: Optional[int]=None, use_parallel: bool=False) -> pd.DataFrame:
    raw_file_abs = str(Path(raw_file).absolute())
    try:
        sorted_items = sorted(chr_snp_mapping.items())
        mapping_hash = hashlib.md5(pickle.dumps(sorted_items, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()[:16]
    except Exception:
        mapping_hash = hashlib.md5(str(sorted(chr_snp_mapping.items())).encode()).hexdigest()[:16]
    cache_key = f'raw_parse_{raw_file_abs}_{mapping_hash}'
    if use_cache:
        cached_result = GLOBAL_CACHE.get(cache_key, check_file_mtime=True, file_path=raw_file_abs)
        if cached_result is not None:
            return cached_result
    with open(raw_file, 'rb') as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            header_end = mmapped_file.find(b'\n')
            if header_end == -1:
                raise ValueError('Cannot find header line')
            header_line = mmapped_file[:header_end].decode('utf-8').strip()
            original_columns = header_line.split()
            sample_col_idx = None
            snp_col_indices = []
            for i, col in enumerate(original_columns):
                if col == 'IID':
                    sample_col_idx = i
                elif col not in ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']:
                    snp_col_indices.append(i)
            batch_mapping = VectorizedProcessor.batch_parse_plink_columns(original_columns)
            batch_mapping.update(chr_snp_mapping)
            usecols_list = None
            if sample_col_idx is not None:
                usecols_list = [sample_col_idx] + snp_col_indices
            data_start = header_end + 1
            import io
            mmap_file_obj = io.BytesIO(mmapped_file[data_start:])
            if max_workers is None:
                max_workers = 6
            if chunk_rows < 5000:
                chunk_rows = 5000
            elif chunk_rows > 50000:
                chunk_rows = 50000
            raw_chunks = []
            chunk_count = 0
            for chunk in pd.read_csv(mmap_file_obj, sep='\\s+', header=None, names=original_columns, dtype=str, engine='c', chunksize=chunk_rows, usecols=usecols_list):
                raw_chunks.append((chunk_count, chunk))
                chunk_count += 1
                if chunk_count % 10 == 0:
                    pass
            if not raw_chunks:
                raise ValueError('No data read from file')
            processed_chunks = []
            if use_parallel and max_workers > 1 and (chunk_count > 1):
                chunk_data_list = [(idx, chunk, batch_mapping, sample_col_idx, original_columns, dtype_obj) for idx, chunk in raw_chunks]
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_process_chunk_parallel, chunk_data): chunk_data[0] for chunk_data in chunk_data_list}
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            chunk_idx, processed_chunk = future.result()
                            processed_chunks.append((chunk_idx, processed_chunk))
                            completed += 1
                            if completed % max(1, chunk_count // 10) == 0:
                                pass
                        except Exception as e:
                            chunk_idx = futures[future]
                            logger.error(f'Failed to process chunk {chunk_idx}: {e}')
                            raise
                processed_chunks.sort(key=lambda x: x[0])
                chunks = [chunk for _, chunk in processed_chunks]
            else:
                chunks = []
                for chunk_idx, chunk in raw_chunks:
                    chunk_data = (chunk_idx, chunk, batch_mapping, sample_col_idx, original_columns, dtype_obj)
                    _, processed_chunk = _process_chunk_parallel(chunk_data)
                    chunks.append(processed_chunk)
                    if (chunk_idx + 1) % 10 == 0:
                        pass
            if len(chunks) == 1:
                result_df = chunks[0]
            else:
                batch_size = max(10, max_workers * 2) if use_parallel else 10
                merged_batches = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    if len(batch) == 1:
                        merged_batches.append(batch[0])
                    else:
                        merged_batch = pd.concat(batch, axis=0)
                        merged_batches.append(merged_batch)
                        del batch
                        gc.collect()
                if len(merged_batches) == 1:
                    result_df = merged_batches[0]
                else:
                    result_df = pd.concat(merged_batches, axis=0)
                    del merged_batches
                    gc.collect()
            result_df = VectorizedProcessor.optimize_dataframe_memory(result_df)
            if use_cache:
                GLOBAL_CACHE.set(cache_key, result_df, file_path=raw_file_abs, ttl_seconds=1800)
            logger.debug(f'.raw file parsing completed: shape={result_df.shape[0]:,} samples × {result_df.shape[1]:,} features')
            return result_df
        finally:
            mmapped_file.close()

def process_single_chromosome_optimized(chr_num: str, plink_prefix_abs: str, pheno_df: pd.DataFrame, plink_executable: str, use_cache: bool=True, skip_recode: bool=False, chr_recode_prefix: Optional[str]=None, cleanup_recode_files: bool=True) -> Tuple[str, Optional[pd.DataFrame]]:
    try:
        try:
            pheno_sorted = pheno_df.sort_index()
            pheno_hash = hashlib.md5(pheno_sorted.values.tobytes() if hasattr(pheno_sorted.values, 'tobytes') else pickle.dumps(pheno_sorted.values, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()[:16]
        except Exception:
            pheno_hash = hashlib.md5(f'{pheno_df.shape}_{tuple(pheno_df.columns)}'.encode()).hexdigest()[:16]
        plink_prefix_abs = str(Path(plink_prefix_abs).absolute())
        cache_key = f'chr_{chr_num}_{plink_prefix_abs}_{pheno_hash}'
        if use_cache:
            cached_result = GLOBAL_CACHE.get(cache_key)
            if cached_result is not None:
                try:
                    df_cached = cached_result
                    if 'sample' in pheno_df.columns and pheno_df.index.name != 'sample':
                        pheno_df = pheno_df.set_index('sample')
                    elif pheno_df.index.name is None and 'sample' not in pheno_df.columns:
                        pheno_df = pheno_df.copy()
                        pheno_df.index.name = 'sample'
                    df_cached = df_cached.loc[df_cached.index.intersection(pheno_df.index)]
                    return (chr_num, df_cached)
                except Exception:
                    return (chr_num, cached_result)
        if chr_recode_prefix is None:
            chr_recode_prefix = f'{plink_prefix_abs}_chr{chr_num}'
        if not skip_recode:
            optimized_params = PLINKOptimizer.optimize_plink_parameters()
            base_cmd = [plink_executable, '--bfile', plink_prefix_abs, '--chr', chr_num, '--recodeA', '--out', chr_recode_prefix]
            plink_cmd = PLINKOptimizer.build_plink_command(base_cmd, optimized_params)
            try:
                result = subprocess.run(plink_cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or 'Unknown error'
                    logger.error(f'Chromosome {chr_num}: PLINK execution failed: {error_msg[:500]}')
                    return (chr_num, None)
            except Exception as e:
                logger.error(f'Chromosome {chr_num}: PLINK execution exception: {str(e)}')
                return (chr_num, None)
        chr_geno_file = f'{chr_recode_prefix}.raw'
        if not Path(chr_geno_file).exists():
            logger.error(f'Chromosome {chr_num}: .raw file was not generated')
            return (chr_num, None)
        chr_bim_file = f'{plink_prefix_abs}.bim'
        chr_snp_mapping = get_snp_chr_pos_mapping_optimized(chr_bim_file, chr_num=chr_num)
        chr_geno_df = parse_raw_file_optimized_v2(chr_geno_file, chr_snp_mapping, dtype_obj=np.int8, use_cache=False)
        if chr_geno_df is None or chr_geno_df.empty:
            logger.debug(f'Chromosome {chr_num}: No valid data read')
            return (chr_num, None)
        if 'sample' in pheno_df.columns and pheno_df.index.name != 'sample':
            pheno_df = pheno_df.set_index('sample')
        elif pheno_df.index.name is None and 'sample' not in pheno_df.columns:
            pheno_df = pheno_df.copy()
            pheno_df.index.name = 'sample'
        common_index = chr_geno_df.index.intersection(pheno_df.index)
        chr_geno_df = chr_geno_df.loc[common_index]
        if cleanup_recode_files:
            delete_temp_files(chr_recode_prefix, ['.raw', '.log', '.nosex'])
        if use_cache:
            GLOBAL_CACHE.set(cache_key, chr_geno_df, ttl_seconds=7200)
        return (chr_num, chr_geno_df)
    except Exception as e:
        logger.error(f'Chromosome {chr_num} processing failed: {str(e)}', exc_info=True)
        return (chr_num, None)

def _plink_recodeA_single_chr_worker(chr_num: str, plink_prefix_abs: str, plink_executable: str, optimized_params: Dict[str, Any]) -> Tuple[str, int, str, str]:
    chr_recode_prefix = f'{plink_prefix_abs}_chr{chr_num}'
    base_cmd = [plink_executable, '--bfile', plink_prefix_abs, '--chr', chr_num, '--recodeA', '--out', chr_recode_prefix]
    plink_cmd = PLINKOptimizer.build_plink_command(base_cmd, optimized_params)
    try:
        result = subprocess.run(plink_cmd, capture_output=True, text=True, check=False)
        msg = (result.stderr or result.stdout or '')[:800]
        return (chr_num, int(result.returncode), chr_recode_prefix, msg)
    except Exception as e:
        return (chr_num, 1, chr_recode_prefix, str(e)[:800])

def plink_to_training_data_optimized(plink_prefix: str, pheno_df: pd.DataFrame, output_file: str, tmp_dir: Optional[Union[str, Path]]=None, cleanup_registry: Optional[List[str]]=None, use_cache: bool=True, parallel_chr_recode: bool=False, chr_recode_workers: Optional[int]=None, genotype_backend: GenotypeBackend='auto') -> TrainingMatrixOutput:
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    backend = _resolve_genotype_backend(genotype_backend, plink_prefix_abs)
    if backend == 'bed':
        logger.debug('Genotype conversion backend: bed (direct .bed read)')
        return _plink_to_training_data_from_bed(plink_prefix, pheno_df, output_file, tmp_dir=tmp_dir, cleanup_registry=cleanup_registry)
    logger.debug('Genotype conversion backend: raw (PLINK --recodeA)')
    return _plink_to_training_data_via_recodeA(plink_prefix, pheno_df, output_file, tmp_dir=tmp_dir, cleanup_registry=cleanup_registry, use_cache=use_cache, parallel_chr_recode=parallel_chr_recode, chr_recode_workers=chr_recode_workers)

def _plink_to_training_data_via_recodeA(plink_prefix: str, pheno_df: pd.DataFrame, output_file: str, tmp_dir: Optional[Union[str, Path]]=None, cleanup_registry: Optional[List[str]]=None, use_cache: bool=True, parallel_chr_recode: bool=False, chr_recode_workers: Optional[int]=None) -> TrainingMatrixOutput:
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    output_file_path = Path(output_file).absolute()
    output_file_abs = output_file_path.as_posix()
    if use_cache:
        GLOBAL_CACHE.cleanup_expired()
    if tmp_dir is None:
        tmp_base = output_file_path.parent / 'tmp_p'
    else:
        tmp_base = Path(tmp_dir).absolute()
    tmp_base.mkdir(parents=True, exist_ok=True)
    chr_list = auto_detect_chromosomes(plink_prefix_abs)
    if not chr_list:
        raise RuntimeError('No chromosomes detected')
    logger.debug(f'Chromosome scan completed: {len(chr_list)} chromosome(s) detected')
    chr_result_files = []
    pheno_df_loaded, temp_pheno_file = _prepare_pheno_df_loaded(pheno_df, tmp_base)
    try:
        chr_to_recode_prefix: Dict[str, str] = {}
        if parallel_chr_recode and len(chr_list) > 1:
            recode_workers = chr_recode_workers
            if recode_workers is None:
                cpu = os.cpu_count() or 2
                recode_workers = min(4, len(chr_list), max(1, cpu // 2))
            base_params = PLINKOptimizer.optimize_plink_parameters()
            per_plink_threads = max(1, int((os.cpu_count() or 1) // max(1, recode_workers)))
            per_plink_mem = base_params.get('memory_mb')
            if isinstance(per_plink_mem, (int, float)) and per_plink_mem > 0:
                per_plink_mem = max(1024, int(per_plink_mem // max(1, recode_workers)))
            optimized_params = dict(base_params)
            optimized_params['threads'] = per_plink_threads
            if per_plink_mem:
                optimized_params['memory_mb'] = per_plink_mem
            logger.debug('Genotype recode started (parallel mode)')
            with ProcessPoolExecutor(max_workers=recode_workers) as executor:
                futures = {executor.submit(_plink_recodeA_single_chr_worker, chr_num, plink_prefix_abs, PLINK_EXECUTABLE, optimized_params): chr_num for chr_num in chr_list}
                completed = 0
                for future in as_completed(futures):
                    chr_num = futures[future]
                    try:
                        chr_num_ret, rc, recode_prefix, msg = future.result()
                    except Exception as e:
                        logger.error(f'Chromosome {chr_num}: recodeA worker crashed: {e}')
                        continue
                    completed += 1
                    if rc != 0:
                        logger.error(f'Chromosome {chr_num_ret}: PLINK recodeA failed: {msg}')
                        continue
                    raw_fp = f'{recode_prefix}.raw'
                    if not Path(raw_fp).exists():
                        logger.error(f'Chromosome {chr_num_ret}: .raw not found after recodeA: {raw_fp}')
                        continue
                    chr_to_recode_prefix[chr_num_ret] = recode_prefix
            ok = len(chr_to_recode_prefix)
            if ok == 0:
                raise RuntimeError('All chromosome recodeA tasks failed; no .raw generated')
            logger.debug(f'Genotype recode completed: {ok}/{len(chr_list)} chromosome(s) succeeded')
        else:
            logger.debug('Genotype recode started (serial mode)')
            for chr_num in chr_list:
                chr_recode_prefix = f'{plink_prefix_abs}_chr{chr_num}'
                base_params = PLINKOptimizer.optimize_plink_parameters()
                base_params = dict(base_params)
                base_params['threads'] = 1
                chr_num_ret, rc, recode_prefix, msg = _plink_recodeA_single_chr_worker(chr_num, plink_prefix_abs, PLINK_EXECUTABLE, base_params)
                if rc != 0:
                    logger.error(f'Chromosome {chr_num_ret}: PLINK recodeA failed: {msg}')
                    continue
                if Path(f'{recode_prefix}.raw').exists():
                    chr_to_recode_prefix[chr_num_ret] = recode_prefix
        for chr_idx, chr_num in enumerate(chr_list, 1):
            if chr_num not in chr_to_recode_prefix:
                logger.debug(f'Chromosome {chr_num}: .raw not generated, skipping parsing')
                continue
            _, chr_df = process_single_chromosome_optimized(chr_num, plink_prefix_abs, pheno_df_loaded, PLINK_EXECUTABLE, use_cache=use_cache, skip_recode=True, chr_recode_prefix=chr_to_recode_prefix[chr_num], cleanup_recode_files=True)
            if chr_df is not None and (not chr_df.empty):
                chr_temp_file = str(tmp_base / f'chr_{chr_num}_temp.feather')
                try:
                    chr_df.reset_index().to_feather(chr_temp_file)
                except Exception as e:
                    logger.debug(f'Cannot use feather format for chromosome {chr_num}, using pickle: {e}')
                    chr_temp_file = str(tmp_base / f'chr_{chr_num}_temp.pkl')
                    chr_df.to_pickle(chr_temp_file)
                chr_result_files.append((chr_num, chr_temp_file))
                if cleanup_registry is not None:
                    cleanup_registry.append(chr_temp_file)
                del chr_df
                gc.collect()
            else:
                logger.debug(f'Chromosome {chr_num} has no valid data, skipping')
        if not chr_result_files:
            raise RuntimeError('All chromosomes have no valid data')
        logger.debug(f'Merging chromosome results: {len(chr_result_files)} chunk(s)')
        batch_size = min(5, len(chr_result_files))
        all_batches = []
        for i in range(0, len(chr_result_files), batch_size):
            batch = chr_result_files[i:i + batch_size]
            batch_dfs = []
            for chr_num, chr_file in batch:
                try:
                    if chr_file.endswith('.feather'):
                        chr_df = pd.read_feather(chr_file).set_index('sample')
                    else:
                        chr_df = pd.read_pickle(chr_file)
                    batch_dfs.append(chr_df)
                except Exception as e:
                    logger.debug(f'Failed to load chromosome {chr_num} data: {e}')
            if batch_dfs:
                if len(batch_dfs) == 1:
                    batch_merged = batch_dfs[0]
                else:
                    common_index = batch_dfs[0].index
                    for df in batch_dfs[1:]:
                        common_index = common_index.intersection(df.index)
                    batch_dfs = [df.loc[common_index] for df in batch_dfs]
                    batch_merged = pd.concat(batch_dfs, axis=1, join='inner')
                all_batches.append(batch_merged)
                del batch_dfs, batch_merged
                gc.collect()
        if len(all_batches) == 1:
            final_train_df = all_batches[0]
        else:
            common_index = all_batches[0].index
            for batch in all_batches[1:]:
                common_index = common_index.intersection(batch.index)
            all_batches = [batch.loc[common_index] for batch in all_batches]
            final_train_df = pd.concat(all_batches, axis=1, join='inner')
        if 'phenotype' in pheno_df_loaded.columns:
            final_train_df = final_train_df.merge(pheno_df_loaded[['phenotype']], left_index=True, right_index=True, how='inner')
        else:
            raise ValueError("Phenotype column 'phenotype' not found in phenotype data")
        try:
            bim_file = f'{plink_prefix_abs}.bim'
            pheno_columns = list(pheno_df_loaded.columns) if 'pheno_df_loaded' in locals() else None
            final_train_df = standardize_snp_column_names(final_train_df, bim_file=bim_file, pheno_columns=pheno_columns)
        except Exception:
            logger.debug('SNP name standardization skipped due to format conversion issue')
        output_file_abs, compression = _normalize_output_train_path(output_file_abs, len(final_train_df) * len(final_train_df.columns))
        snp_cols = [c for c in final_train_df.columns if c != 'phenotype']
        sample_ids = final_train_df.index.astype(str).tolist()
        column_names = [str(c) for c in snp_cols]
        genotype_matrix = final_train_df[snp_cols].to_numpy(dtype=np.int8, copy=False)
        pheno_values = final_train_df['phenotype'].values
        _write_train_matrix_to_csv(output_file_abs, sample_ids, column_names, genotype_matrix, pheno_values, compression)
        del genotype_matrix
        sample_ids = final_train_df.index.astype(str).tolist()
        includes_pheno = 'phenotype' in final_train_df.columns
        n_snp_features = final_train_df.shape[1] - (1 if includes_pheno else 0)
        logger.debug(f'Final training dataset (after GWAS/LD/combined strategy if applied): {len(sample_ids):,} samples, {n_snp_features:,} feature SNPs')
        if use_cache:
            GLOBAL_CACHE.print_stats()
        del final_train_df
        gc.collect()
        return TrainingMatrixOutput(train_file=output_file_abs, sample_ids=sample_ids, n_snp_features=n_snp_features, includes_phenotype_column=includes_pheno)
    finally:
        try:
            os.remove(temp_pheno_file)
        except:
            pass
        for _, chr_file in chr_result_files:
            try:
                os.remove(chr_file)
            except:
                pass

def load_sample_id_file(sample_id_file: str) -> List[str]:
    sample_path = Path(sample_id_file)
    if not sample_path.exists():
        raise FileNotFoundError(f'Sample ID file not found: {sample_id_file}')
    sample_ids: List[str] = []
    with open(sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            sample_ids.append(str(stripped.split()[0]))
    if not sample_ids:
        raise ValueError(f'Sample ID file is empty: {sample_id_file}')
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f'Sample ID file contains duplicate sample IDs: {sample_id_file}')
    return sample_ids

def write_sample_id_list(sample_ids: List[str], output_file: Union[str, Path]) -> str:
    out_path = Path(output_file).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for sid in sample_ids:
            f.write(f'{sid}\n')
    return str(out_path)

def write_plink_keep_file(sample_ids: List[str], output_file: Union[str, Path]) -> str:
    out_path = Path(output_file).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for sid in sample_ids:
            f.write(f'{sid}\t{sid}\n')
    return str(out_path)

def _validate_ids_file_pair(train_ids_file: Optional[str], test_ids_file: Optional[str]) -> None:
    has_train = train_ids_file is not None
    has_test = test_ids_file is not None
    if has_train ^ has_test:
        raise ValueError('Both --train-ids-file and --test-ids-file must be provided together')

def resolve_sample_split(sample_ids: List[str], pheno_df: pd.DataFrame, task_type: str, train_ids_file: Optional[str]=None, test_ids_file: Optional[str]=None, test_ratio: float=TRAIN_TEST_HOLDOUT_RATIO, random_state: int=DEFAULT_SPLIT_RANDOM_STATE) -> Dict[str, Any]:
    """Resolve train/test IDs. Custom files if given; else reproducible 80/20 after sorting sample IDs."""
    _validate_ids_file_pair(train_ids_file, test_ids_file)
    sample_ids_str = [str(s) for s in sample_ids]
    sample_set = set(sample_ids_str)
    if not sample_ids_str:
        raise ValueError('No samples available for train/test split')
    if train_ids_file and test_ids_file:
        train_ids = load_sample_id_file(train_ids_file)
        test_ids = load_sample_id_file(test_ids_file)
        train_set = set(train_ids)
        test_set = set(test_ids)
        overlap = train_set & test_set
        if overlap:
            preview = ', '.join(sorted(overlap)[:10])
            suffix = f' ... (+{len(overlap) - 10} more)' if len(overlap) > 10 else ''
            raise ValueError(f'Train and test sample ID files overlap on {len(overlap)} sample(s): {preview}{suffix}')
        missing_train = sorted(train_set - sample_set)
        missing_test = sorted(test_set - sample_set)
        if missing_train:
            preview = ', '.join(missing_train[:10])
            suffix = f' ... (+{len(missing_train) - 10} more)' if len(missing_train) > 10 else ''
            raise ValueError(f'{len(missing_train)} train sample ID(s) not found in dataset: {preview}{suffix}')
        if missing_test:
            preview = ', '.join(missing_test[:10])
            suffix = f' ... (+{len(missing_test) - 10} more)' if len(missing_test) > 10 else ''
            raise ValueError(f'{len(missing_test)} test sample ID(s) not found in dataset: {preview}{suffix}')
        train_ids_final = sorted(train_set)
        test_ids_final = sorted(test_set)
        unassigned = sorted(sample_set - train_set - test_set)
        if unassigned:
            logger.warning(f'{len(unassigned)} sample(s) not listed in train/test ID files and will be excluded from feature selection and training')
        mode = 'custom_sample_ids'
        logger.debug(f'Custom train/test split from ID files: train={len(train_ids_final)}, test={len(test_ids_final)}')
        return {'mode': mode, 'train_ids': train_ids_final, 'test_ids': test_ids_final, 'n_train': len(train_ids_final), 'n_test': len(test_ids_final), 'n_unassigned': len(unassigned), 'train_ratio': None, 'test_ratio': None, 'random_state': None, 'train_ids_file': str(Path(train_ids_file).absolute()), 'test_ids_file': str(Path(test_ids_file).absolute())}
    if len(sample_ids_str) < 5:
        raise ValueError(f'Need at least 5 samples for train/test split, got {len(sample_ids_str)}')
    from sklearn.model_selection import train_test_split
    ordered_ids = sorted(sample_ids_str, key=str)
    pheno_by_sample = pheno_df.set_index(pheno_df['sample'].astype(str))['phenotype'] if 'sample' in pheno_df.columns else pheno_df['phenotype']
    stratify = None
    if task_type == 'classification':
        try:
            y_ordered = pheno_by_sample.reindex(ordered_ids)
            if y_ordered.isna().any():
                raise ValueError('Missing phenotype for stratification')
            stratify = y_ordered.values
        except Exception as e:
            logger.debug(f'Stratified split unavailable ({e}); using non-stratified reproducible split')
            stratify = None
    try:
        train_ids_arr, test_ids_arr = train_test_split(ordered_ids, test_size=test_ratio, random_state=random_state, stratify=stratify)
    except ValueError as e:
        if stratify is not None:
            logger.debug(f'Stratified train/test split failed ({e}); falling back to random split')
            train_ids_arr, test_ids_arr = train_test_split(ordered_ids, test_size=test_ratio, random_state=random_state, stratify=None)
        else:
            raise
    train_ids_final = sorted((str(x) for x in train_ids_arr), key=str)
    test_ids_final = sorted((str(x) for x in test_ids_arr), key=str)
    train_ratio = 1.0 - float(test_ratio)
    logger.debug(f'Reproducible random hold-out split (sorted IDs, random_state={random_state}): train={len(train_ids_final)}, test={len(test_ids_final)} ({train_ratio:.0%}/{float(test_ratio):.0%})')
    return {'mode': 'random_holdout', 'train_ids': train_ids_final, 'test_ids': test_ids_final, 'n_train': len(train_ids_final), 'n_test': len(test_ids_final), 'n_unassigned': 0, 'train_ratio': train_ratio, 'test_ratio': float(test_ratio), 'random_state': int(random_state), 'train_ids_file': None, 'test_ids_file': None}

def run_preprocess(genotype_path: str, genotype_format: GenotypeFormat, phenotype_file: str, output_file: str, filter_snps: bool=True, feature_selection_mode: int=1, gwas_pvalue: float=0.01, ld_config: str='50,5,0.2', train_ids_file: Optional[str]=None, test_ids_file: Optional[str]=None, random_state: int=DEFAULT_SPLIT_RANDOM_STATE) -> int:
    cleanup_registry: List[str] = []
    temp_prefixes: List[str] = []

    def cleanup_current_run():
        for prefix in temp_prefixes:
            delete_temp_files(prefix, ['.bed', '.bim', '.fam', '.log', '.nosex', '.raw', '.map', '.ped'])
        for fp in cleanup_registry:
            try:
                Path(fp).unlink()
            except Exception:
                pass
        try:
            import shutil
            if 'stage_output_dir' in locals() or 'stage_output_dir' in globals():
                stage_output_dir_path = Path(stage_output_dir)
                gwas_output_dir = stage_output_dir_path / 'output'
                if gwas_output_dir.exists():
                    shutil.rmtree(gwas_output_dir, ignore_errors=True)
                gwas_prefix_pattern = stage_output_dir_path / 'preprocess_gwas*'
                for gwas_file in stage_output_dir_path.glob('preprocess_gwas*'):
                    try:
                        if gwas_file.is_file():
                            gwas_file.unlink()
                    except Exception:
                        pass
        except Exception as e:
            pass
        try:
            import shutil
            if 'base_output_dir' in locals() or 'base_output_dir' in globals():
                tmp_root_dir = base_output_dir / 'tmp'
                if tmp_root_dir.exists():
                    shutil.rmtree(tmp_root_dir, ignore_errors=True)
            elif 'tmp_dir' in locals() or 'tmp_dir' in globals():
                tmp_dir_path = Path(tmp_dir)
                tmp_root_dir = tmp_dir_path.parent
                if tmp_root_dir.exists() and tmp_root_dir.name == 'tmp':
                    shutil.rmtree(tmp_root_dir, ignore_errors=True)
        except Exception as e:
            pass
    try:
        if not Path(phenotype_file).exists():
            raise FileNotFoundError(f'Phenotype file not found: {phenotype_file}')
        try:
            ld_parts = [p.strip() for p in str(ld_config).split(',') if p.strip() != '']
            if len(ld_parts) != 3:
                raise ValueError
            ld_window_kb = int(ld_parts[0])
            ld_window = int(ld_parts[1])
            ld_window_r2 = float(ld_parts[2])
        except Exception:
            raise ValueError(f'Invalid LD three-in-one configuration ld_config={ld_config!r}; Expected format is "<window_kb>,<window_variants>,<r2_threshold>", e.g., "50,5,0.2"')
        output_path = Path(output_file).absolute()
        if output_path.is_dir() or (not output_path.suffix and (not output_path.exists())):
            base_output_dir = output_path
        else:
            base_output_dir = output_path.parent
        base_output_dir.mkdir(parents=True, exist_ok=True)
        stage_output_dir = base_output_dir / 'preprocess'
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = base_output_dir / 'tmp' / 'preprocess'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir() or (not output_path.suffix and (not output_path.exists())):
            output_prefix = base_output_dir.name
        else:
            output_prefix = output_path.stem
        output_file_name = f'{output_prefix}_train_data.txt'
        output_file_path = stage_output_dir / output_file_name
        temp_plink_prefix = str(tmp_dir / 'temp_plink')
        temp_prefixes.append(temp_plink_prefix)
        plink_prefix = genotype_to_plink(genotype_path, temp_plink_prefix, fmt=genotype_format)
        normalize_chromosome_names(plink_prefix)
        pheno_df = load_phenotype(phenotype_file)
        task_type = determine_phenotype_type(pheno_df, stage_output_dir)
        pheno_df = convert_phenotype_dtype(pheno_df, task_type)
        if task_type == 'regression':
            original_count = len(pheno_df)
            pheno_df = pheno_df[pheno_df['phenotype'] != 0].reset_index(drop=True)
            removed_count = original_count - len(pheno_df)
            if removed_count > 0:
                logger.debug(f'Regression task: removed {removed_count} samples with phenotype value = 0')
        fam_file = f'{plink_prefix}.fam'
        if Path(fam_file).exists():
            try:
                fam_df = pd.read_csv(fam_file, sep='\\s+', header=None, dtype=str, engine='python')
                if fam_df.shape[1] >= 2:
                    geno_samples = set(fam_df.iloc[:, 1].astype(str))
                    original_pheno_n = len(pheno_df)
                    pheno_df = pheno_df[pheno_df['sample'].isin(geno_samples)].reset_index(drop=True)
                    if pheno_df.empty:
                        raise ValueError('No intersection between phenotype samples and genotype samples')
                    logger.debug(f'Sample matching completed: {len(pheno_df):,} samples remaining after matching')
            except Exception as e:
                logger.debug(f'Sample matching failed: {e}')
        if task_type == 'regression':
            plot_regression_phenotype_distribution(pheno_df, stage_output_dir)
        sample_filtered_prefix = str(tmp_dir / 'sample_filtered_plink')
        temp_prefixes.append(sample_filtered_prefix)
        sample_filtered_plink = filter_samples_by_phenotype(plink_prefix, pheno_df, sample_filtered_prefix)
        logger.debug(f'SNP quality filter setting: filter_snps={filter_snps}')
        if filter_snps:
            filtered_plink_prefix = str(tmp_dir / 'filtered_plink')
            temp_prefixes.append(filtered_plink_prefix)
            final_plink_prefix = filter_snps_by_quality(sample_filtered_plink, filtered_plink_prefix, maf=0.05, geno=0.2)
            gwas_genotype_prefix = final_plink_prefix
            logger.debug('SNP quality filtering completed')
        else:
            final_plink_prefix = sample_filtered_plink
            gwas_genotype_prefix = sample_filtered_plink
            logger.debug('SNP filtering skipped')
        post_qc_snp_count = _count_bim_snps(final_plink_prefix)
        if post_qc_snp_count > 0:
            logger.debug(f'SNP count after sample/QC steps: {post_qc_snp_count:,}')
            _warn_large_snp_count(post_qc_snp_count, feature_selection_mode, filter_snps)
        sample_split = resolve_sample_split(sample_ids=pheno_df['sample'].astype(str).tolist(), pheno_df=pheno_df, task_type=task_type, train_ids_file=train_ids_file, test_ids_file=test_ids_file, test_ratio=TRAIN_TEST_HOLDOUT_RATIO, random_state=random_state)
        train_id_set = set(sample_split['train_ids'])
        pheno_df_train = pheno_df[pheno_df['sample'].astype(str).isin(train_id_set)].reset_index(drop=True)
        if pheno_df_train.empty:
            raise ValueError('No training samples remain after train/test split')
        train_ids_out = write_sample_id_list(sample_split['train_ids'], stage_output_dir / f'{output_prefix}_train_ids.txt')
        test_ids_out = write_sample_id_list(sample_split['test_ids'], stage_output_dir / f'{output_prefix}_test_ids.txt')
        sample_split['train_ids_output'] = train_ids_out
        sample_split['test_ids_output'] = test_ids_out
        logger.debug(f'Train/test split fixed for feature selection and training: train={sample_split["n_train"]}, test={sample_split["n_test"]} (mode={sample_split["mode"]})')
        logger.debug('Feature selection (GWAS/LD) will use training samples only; test samples are held out for model evaluation')
        train_keep_file = write_plink_keep_file(sample_split['train_ids'], tmp_dir / 'train_samples_keep.txt')
        cleanup_registry.append(train_keep_file)
        if feature_selection_mode not in [1, 2, 3, 4]:
            raise ValueError(f'feature_selection_mode must be 1, 2, 3, or 4, got: {feature_selection_mode}')
        if feature_selection_mode != 1:
            logger.debug(f'Feature selection started (mode={feature_selection_mode})')
        if feature_selection_mode in [2, 3, 4]:
            significant_snps: Optional[List[str]] = None
            if feature_selection_mode in [2, 4]:
                try:
                    logger.debug(f'GWAS analysis started (training samples only: {len(pheno_df_train):,})')
                    significant_snps = run_gwas_preprocess(plink_prefix=final_plink_prefix, pheno_df=pheno_df_train, output_dir=stage_output_dir, tmp_dir=tmp_dir, pvalue_threshold=gwas_pvalue, perform_plink_qc=filter_snps)
                    if not significant_snps:
                        logger.warning('No significant SNPs found by GWAS; feature selection will be skipped.')
                        feature_selection_mode = 1
                    else:
                        logger.info(f'Feature selection (GWAS): {len(significant_snps):,} SNPs retained')
                except Exception as e:
                    logger.error(f'GWAS analysis failed: {e}')
                    feature_selection_mode = 1
            if feature_selection_mode in [3, 4]:
                if feature_selection_mode == 4 and (not significant_snps):
                    logger.warning('GWAS failed or no significant SNPs; skip LD filtering.')
                    feature_selection_mode = 1
                else:
                    logger.debug(f'LD analysis started (training samples only: {len(sample_split["train_ids"]):,})')
                    ld_output_prefix = str(Path(tmp_dir / 'ld_filtered_plink').absolute())
                    extract_snps_file = None
                    if feature_selection_mode == 4 and significant_snps:
                        extract_snps_file = str(Path(tmp_dir / 'gwas_significant_snps_for_ld.txt').absolute())
                        with open(extract_snps_file, 'w') as f:
                            for snp in significant_snps:
                                f.write(f'{snp}\n')
                        logger.debug(f'LD analysis scope: {len(significant_snps):,} SNP(s) from GWAS')
                    try:
                        final_plink_prefix_abs = str(Path(final_plink_prefix).absolute())
                        ld_ret = plink_ld.run_ld_filtering(input_path=final_plink_prefix_abs, output_prefix=ld_output_prefix, ld_window_kb=ld_window_kb, ld_window=ld_window, ld_window_r2=ld_window_r2, keep_intermediate=False, keep_samples_file=train_keep_file, extract_snps_file=extract_snps_file)
                        if ld_ret != 0:
                            logger.error('LD analysis failed, fallback to no feature selection.')
                            feature_selection_mode = 1
                        else:
                            final_plink_prefix = ld_output_prefix
                            gwas_genotype_prefix = ld_output_prefix
                            ld_snp_count = _count_bim_snps(ld_output_prefix)
                            logger.info(f'Feature selection (LD): {ld_snp_count:,} SNPs retained')
                    finally:
                        if extract_snps_file:
                            try:
                                Path(extract_snps_file).unlink()
                            except Exception:
                                pass
            elif feature_selection_mode == 2 and significant_snps:
                logger.debug('Applying GWAS-selected SNP extraction')
                gwas_only_prefix = str(tmp_dir / 'gwas_filtered_plink')
                try:
                    final_plink_prefix = extract_snps_with_plink(plink_prefix=final_plink_prefix, snp_list=significant_snps, output_prefix=gwas_only_prefix)
                    gwas_genotype_prefix = final_plink_prefix
                except Exception as e:
                    logger.error(f'GWAS SNP extraction failed, fallback to no feature selection: {e}')
                    feature_selection_mode = 1
        # Fixed defaults: auto backend (bed when available), in-process cache on, serial chr recode
        matrix_output = plink_to_training_data_optimized(final_plink_prefix, pheno_df, str(output_file_path), tmp_dir=tmp_dir, cleanup_registry=cleanup_registry, use_cache=True, parallel_chr_recode=False, chr_recode_workers=None, genotype_backend='auto')
        actual_train_file = matrix_output.train_file
        includes_phenotype_column = matrix_output.includes_phenotype_column
        n_snp_features = matrix_output.n_snp_features
        _warn_large_snp_count(n_snp_features, feature_selection_mode, filter_snps)
        generate_metadata(output_prefix=str(stage_output_dir / output_prefix), plink_prefix=gwas_genotype_prefix, train_file=actual_train_file, valid_samples=matrix_output.sample_ids, genotype_format=genotype_format, task_type=task_type, preprocess_tmp_dir=tmp_dir, filter_snps=filter_snps, feature_selection_mode=feature_selection_mode, gwas_pvalue=gwas_pvalue, ld_window_kb=ld_window_kb, ld_window=ld_window, ld_window_r2=ld_window_r2, n_feature_columns=n_snp_features, includes_phenotype_column=includes_phenotype_column, sample_split=sample_split)
        logger.info(f'Preprocess completed: {len(matrix_output.sample_ids):,} samples × {n_snp_features:,} SNPs (mode={feature_selection_mode}, split={sample_split["mode"]})')
        gc.collect()
        cleanup_current_run()
        return 0
    except Exception as e:
        logger.error(f'Preprocessing failed: {str(e)}', exc_info=True)
        cleanup_current_run()
        return 1

def delete_temp_files(prefix: str, exts: list) -> None:
    for ext in exts:
        temp_file = f'{prefix}{ext}'
        if Path(temp_file).exists():
            try:
                os.remove(temp_file)
            except:
                pass

def add_genotype_input_arguments(parser: argparse.ArgumentParser, *, required: bool=True) -> argparse._MutuallyExclusiveGroup:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--bfile', metavar='PREFIX', dest='bfile', help='PLINK binary genotype prefix (.bed/.bim/.fam)')
    group.add_argument('--file', metavar='PREFIX', dest='file', help='PLINK text genotype prefix (.map/.ped)')
    group.add_argument('--vcf', metavar='PATH', dest='vcf', help='VCF genotype file (.vcf or .vcf.gz)')
    return group

def resolve_genotype_input(args: argparse.Namespace) -> Tuple[str, GenotypeFormat]:
    options: List[Tuple[str, str, GenotypeFormat]] = []
    if getattr(args, 'bfile', None):
        options.append(('--bfile', args.bfile, 'plink_binary'))
    if getattr(args, 'file', None):
        options.append(('--file', args.file, 'plink_text'))
    if getattr(args, 'vcf', None):
        options.append(('--vcf', args.vcf, 'vcf'))
    if len(options) == 0:
        raise ValueError('Genotype input required: specify exactly one of --bfile, --file, or --vcf')
    if len(options) > 1:
        flags = ', '.join((o[0] for o in options))
        raise ValueError(f'Genotype input ambiguous: only one of --bfile, --file, or --vcf allowed (got: {flags})')
    _, genotype_path, genotype_format = options[0]
    validated_path = validate_genotype_input(genotype_path, genotype_format)
    return validated_path, genotype_format

def build_preprocess_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='G2PInsight data preprocessing (optional GWAS/LD feature selection)')
    add_genotype_input_arguments(parser, required=True)
    parser.add_argument('-p', '--phenotype', required=True, help='Phenotype file path')
    parser.add_argument('-o', '--output', required=True, help='Output file path or directory prefix')
    parser.add_argument('--no-filter-snps', action='store_true', help='Disable SNP quality filtering')
    parser.add_argument('-f', '--feature_selection_mode', type=int, default=1, choices=[1, 2, 3, 4], help='Feature selection: 1=none, 2=GWAS, 3=LD, 4=GWAS+LD (default: 1)')
    parser.add_argument('--gwas_pvalue', type=float, default=0.01, help='GWAS P-value threshold for modes 2 and 4 (default: 0.01)')
    parser.add_argument('--ld-config', type=str, default='50,5,0.2', help='LD config "<window_kb>,<window_variants>,<r2>" (default: 50,5,0.2); modes 3 and 4')
    parser.add_argument('--train-ids-file', dest='train_ids_file', default=None, help='Optional one-column sample ID file for the training set (must be used with --test-ids-file)')
    parser.add_argument('--test-ids-file', dest='test_ids_file', default=None, help='Optional one-column sample ID file for the test set (must be used with --train-ids-file)')
    parser.add_argument('--random_state', type=int, default=DEFAULT_SPLIT_RANDOM_STATE, help=f'Random seed for default 80/20 train/test split after sorting sample IDs (default: {DEFAULT_SPLIT_RANDOM_STATE}; ignored when ID files are provided)')
    return parser

def validate_genotype_input(genotype_path: str, fmt: GenotypeFormat) -> str:
    genotype_path = Path(genotype_path).absolute().as_posix()
    if fmt == 'vcf':
        if not Path(genotype_path).exists():
            raise FileNotFoundError(f'VCF file not found: {genotype_path}')
        return genotype_path
    try:
        validate_plink_files(genotype_path, fmt)
    except FileNotFoundError as exc:
        if fmt == 'plink_binary':
            text_files = [f'{genotype_path}{ext}' for ext in ['.map', '.ped']]
            if all((Path(f).exists() for f in text_files)):
                raise FileNotFoundError(f'{exc}\nHint: files look like PLINK text format; use --file instead of --bfile.') from exc
        elif fmt == 'plink_text':
            bin_files = [f'{genotype_path}{ext}' for ext in ['.bed', '.bim', '.fam']]
            if all((Path(f).exists() for f in bin_files)):
                raise FileNotFoundError(f'{exc}\nHint: files look like PLINK binary format; use --bfile instead of --file.') from exc
        raise
    return genotype_path

def detect_genotype_format(genotype_path: str) -> Literal['vcf', 'plink_binary', 'plink_text']:
    genotype_path = Path(genotype_path).absolute().as_posix()
    if genotype_path.endswith(('.vcf', '.vcf.gz')):
        if not Path(genotype_path).exists():
            raise FileNotFoundError(f'VCF file not found: {genotype_path}')
        return 'vcf'
    plink_bin_files = [f'{genotype_path}.bed', f'{genotype_path}.bim', f'{genotype_path}.fam']
    if all((Path(f).exists() for f in plink_bin_files)):
        return 'plink_binary'
    plink_text_files = [f'{genotype_path}.map', f'{genotype_path}.ped']
    if all((Path(f).exists() for f in plink_text_files)):
        return 'plink_text'
    raise ValueError(f'\nUnable to detect genotype format: {genotype_path}\nSupported formats: VCF(.vcf/.vcf.gz), PLINK binary(.bed/.bim/.fam), PLINK text(.map/.ped)')

def _copy_plink_binary_to_prefix(genotype_path: str, output_prefix: str) -> str:
    """Copy .bed/.bim/.fam to output_prefix so subsequent edits never touch the originals."""
    src_prefix = Path(genotype_path).absolute().as_posix()
    dst_prefix = Path(output_prefix).absolute().as_posix()
    if src_prefix == dst_prefix:
        return dst_prefix
    Path(dst_prefix).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.bed', '.bim', '.fam'):
        src_file = Path(f'{src_prefix}{ext}')
        dst_file = Path(f'{dst_prefix}{ext}')
        if not src_file.exists():
            raise FileNotFoundError(f'Missing PLINK binary file: {src_file}')
        shutil.copy2(src_file, dst_file)
    logger.debug(f'Copied PLINK binary genotype to temporary prefix: {dst_prefix}')
    return dst_prefix

def genotype_to_plink(genotype_path: str, output_prefix: str, fmt: Optional[Literal['vcf', 'plink_binary', 'plink_text']]=None) -> str:
    if fmt is None:
        fmt = detect_genotype_format(genotype_path)
    else:
        genotype_path = validate_genotype_input(genotype_path, fmt)
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'vcf':
        plink_cmd = [PLINK_EXECUTABLE, '--vcf', genotype_path, '--make-bed', '--out', output_prefix, '--allow-extra-chr', '--set-missing-var-ids', '@:#', '--keep-allele-order']
        try:
            subprocess.run(plink_cmd, capture_output=True, text=True, check=True)
            return output_prefix
        except subprocess.CalledProcessError as e:
            logger.error(f'PLINK execution failed: {e.stderr}')
            raise RuntimeError(f'VCF-to-PLINK conversion failed: {e.stderr}')
    elif fmt == 'plink_binary':
        return _copy_plink_binary_to_prefix(genotype_path, output_prefix)
    elif fmt == 'plink_text':
        return plink_text_to_binary(genotype_path, output_prefix)

def plink_text_to_binary(genotype_path: str, output_prefix: str) -> str:
    plink_cmd = [PLINK_EXECUTABLE, '--file', genotype_path, '--make-bed', '--out', output_prefix, '--allow-extra-chr', '--allow-no-sex', '--keep-allele-order']
    try:
        result = subprocess.run(plink_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or 'Unknown error'
            logger.error(f'PLINK text-to-binary conversion failed: {error_msg[:500]}')
            raise RuntimeError(f'PLINK text-to-binary conversion failed: {error_msg[:500]}')
        return output_prefix
    except Exception as e:
        logger.error(f'PLINK text-to-binary conversion exception: {str(e)}')
        raise

def normalize_chromosome_names(plink_prefix: str) -> None:
    bim_file = f'{Path(plink_prefix).absolute().as_posix()}.bim'
    if not Path(bim_file).exists():
        raise FileNotFoundError(f'PLINK .bim file not found: {bim_file}')
    bim_df = pd.read_csv(bim_file, sep='\\s+', header=None, dtype={0: str}, engine='python')
    original_chr_col = bim_df[0].copy()

    def normalize_chr(chr_name: str) -> str:
        chr_str = str(chr_name).strip()
        if chr_str.lower().startswith('chr'):
            chr_num = chr_str[3:].strip()
            if chr_num.isdigit():
                return f'chr{chr_num}'
            return f'chr{chr_num}'
        elif chr_str.isdigit():
            return f'chr{chr_str}'
        elif chr_str.upper() in ['X', 'Y', 'MT', 'M']:
            return f'chr{chr_str.upper()}'
        else:
            import re
            match = re.search('\\d+', chr_str)
            if match:
                return f'chr{match.group()}'
            return f'chr{chr_str}'
    normalized_chr_col = original_chr_col.apply(normalize_chr)
    if not normalized_chr_col.equals(original_chr_col):
        bim_df[0] = normalized_chr_col
        bim_df.to_csv(bim_file, sep='\t', header=False, index=False)

def auto_detect_chromosomes(plink_prefix: str) -> List[str]:
    bim_file = f'{Path(plink_prefix).absolute().as_posix()}.bim'
    if not Path(bim_file).exists():
        raise FileNotFoundError(f'PLINK .bim file not found: {bim_file}')
    chr_series = pd.read_csv(bim_file, sep='\\s+', header=None, usecols=[0], dtype={0: str}, chunksize=100000, engine='python')
    chr_set = set()
    for chunk in chr_series:
        chr_set.update(chunk[0].unique())
    chr_list = list(chr_set)
    num_chrs = [c for c in chr_list if c.isdigit()]
    non_num_chrs = [c for c in chr_list if not c.isdigit()]
    num_chrs_sorted = sorted(num_chrs, key=int)
    chr_list_sorted = num_chrs_sorted + non_num_chrs
    return chr_list_sorted

def load_phenotype(pheno_path: str) -> pd.DataFrame:

    def detect_separator(file_path: str) -> str:
        import csv
        candidates = ['\t', ',', '\\s+']

        def _is_header_like(line: str) -> bool:
            s = line.strip()
            if not s:
                return False
            has_alpha = any((ch.isalpha() for ch in s))
            has_digit = any((ch.isdigit() for ch in s))
            return has_alpha and (not has_digit)
        lines: List[str] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(30):
                ln = f.readline()
                if not ln:
                    break
                ln = ln.strip('\n\r')
                if ln.strip() == '':
                    continue
                lines.append(ln)
        if not lines:
            return '\\s+'
        work_lines = lines[:]
        if _is_header_like(work_lines[0]) and len(work_lines) > 1:
            work_lines = work_lines[1:]
        try:
            sample_text = '\n'.join(work_lines[:10])
            dialect = csv.Sniffer().sniff(sample_text, delimiters='\t,')
            sniffed = dialect.delimiter
            if sniffed in ('\t', ','):
                ok = 0
                for ln in work_lines[:20]:
                    if len(ln.split(sniffed)) >= 2:
                        ok += 1
                if ok >= max(1, int(0.6 * min(20, len(work_lines)))):
                    return sniffed
        except Exception:
            pass
        best_sep = '\\s+'
        best_score = -1
        eval_lines = work_lines[:20]
        for sep in candidates:
            score = 0
            for ln in eval_lines:
                parts = re.split(sep, ln.strip()) if sep == '\\s+' else ln.split(sep)
                parts = [p for p in parts if p != '']
                if len(parts) >= 2:
                    score += 1
            if score > best_score:
                best_score = score
                best_sep = sep
        return best_sep
    sep = detect_separator(pheno_path)
    try:
        pheno_df = pd.read_csv(pheno_path, sep=sep, header=None, dtype=str, engine='python')
    except Exception as e:
        raise RuntimeError(f'Failed to read phenotype file: {str(e)}')
    if pheno_df.shape[1] < 2:
        raise ValueError(f'Phenotype file column error: at least 2 columns required, found {pheno_df.shape[1]} columns')
    if pheno_df.shape[1] > 2:
        pheno_df = pheno_df.iloc[:, [0, 1]].copy()
    pheno_df.columns = ['sample', 'phenotype']
    abnormal_values = ['na', 'NA', 'nan', 'NAN', 'NaN', 'null', 'NULL', 'None', 'NONE', '', '.', '-9', '-999', 'missing', 'MISSING', 'NULL', 'null']
    phenotype_str = pheno_df['phenotype'].astype(str).str.strip().str.lower()
    abnormal_mask = phenotype_str.isin([v.lower() for v in abnormal_values]) | phenotype_str.isna()
    if abnormal_mask.any():
        abnormal_count = abnormal_mask.sum()
        abnormal_samples = pheno_df.loc[abnormal_mask, 'sample'].tolist()[:10]
        abnormal_pheno_values = pheno_df.loc[abnormal_mask, 'phenotype'].tolist()[:10]
        original_count = len(pheno_df)
        pheno_df = pheno_df[~abnormal_mask].reset_index(drop=True)
        remaining_count = len(pheno_df)
        logger.debug(f'Phenotype abnormal values removed: count={abnormal_count}, original={original_count}, remaining={remaining_count}')
        if pheno_df.empty:
            raise ValueError('After removing abnormal values, phenotype file has no valid data, cannot continue processing')
    pheno_df = pheno_df.drop_duplicates(subset='sample', keep='last')
    pheno_df = pheno_df[pheno_df['sample'].notna() & (pheno_df['sample'] != '')].reset_index(drop=True)
    if pheno_df.empty:
        raise ValueError('Phenotype file has no valid data (all missing values/empty IDs)')
    original_count_before_convert = len(pheno_df)
    pheno_df['phenotype'] = pd.to_numeric(pheno_df['phenotype'], errors='coerce')
    nan_mask = pheno_df['phenotype'].isna()
    if nan_mask.any():
        nan_count = nan_mask.sum()
        nan_samples = pheno_df.loc[nan_mask, 'sample'].tolist()[:10]
        pheno_df = pheno_df[~nan_mask].reset_index(drop=True)
        remaining_count_after_convert = len(pheno_df)
        logger.debug(f'Phenotype non-numeric values removed: count={nan_count}, before={original_count_before_convert}, remaining={remaining_count_after_convert}')
        if pheno_df.empty:
            raise ValueError('After removing non-convertible samples, phenotype file has no valid data, cannot continue processing')
    gc.collect()
    return pheno_df

def determine_phenotype_type(pheno_df: pd.DataFrame, output_dir: Path) -> Literal['regression', 'classification']:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        try:
            from G2PInsight.bin.font_utils import setup_matplotlib_font
            setup_matplotlib_font()
        except ImportError:
            pass
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        logger.debug('matplotlib is not installed; phenotype visualization will be skipped')
    phenotype_values = pheno_df['phenotype'].values
    unique_values = np.unique(phenotype_values)
    unique_count = len(unique_values)
    is_all_integer = np.all(phenotype_values == np.round(phenotype_values))
    if unique_count <= 10 and is_all_integer:
        task_type = 'classification'
        logger.info(f'Phenotype type: classification ({unique_count} classes)')
    else:
        task_type = 'regression'
        logger.info('Phenotype type: regression')
    if MATPLOTLIB_AVAILABLE:
        output_dir.mkdir(parents=True, exist_ok=True)
        if task_type == 'classification':
            value_counts = pd.Series(phenotype_values).value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(f'Phenotype Distribution (Classification)\nUnique Values: {unique_count}', fontsize=14)
            plt.axis('equal')
            plot_file = output_dir / 'phenotype_distribution_pie.png'
            plt.savefig(plot_file, dpi=600, bbox_inches='tight')
            plt.close()
    return task_type

def convert_phenotype_dtype(pheno_df: pd.DataFrame, task_type: Literal['regression', 'classification']) -> pd.DataFrame:
    if task_type == 'classification':
        pheno_df['phenotype'] = pheno_df['phenotype'].astype(np.int32)
    else:
        pheno_df['phenotype'] = pheno_df['phenotype'].astype(np.float32)
    gc.collect()
    return pheno_df

def plot_regression_phenotype_distribution(pheno_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        try:
            from G2PInsight.bin.font_utils import setup_matplotlib_font
            setup_matplotlib_font()
        except ImportError:
            pass
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        logger.debug('matplotlib is not installed; phenotype visualization will be skipped')
        return
    if not MATPLOTLIB_AVAILABLE:
        return
    phenotype_values = pheno_df['phenotype'].values
    unique_values = np.unique(phenotype_values)
    unique_count = len(unique_values)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(phenotype_values, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('Phenotype Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Phenotype Distribution (Regression)\nSamples: {len(phenotype_values)}, Unique Values: {unique_count}', fontsize=14)
    plt.grid(True, alpha=0.3)
    mean_val = np.mean(phenotype_values)
    std_val = np.std(phenotype_values)
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1SD: {std_val:.2f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)
    plot_file = output_dir / 'phenotype_distribution_histogram.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug(f'Regression phenotype distribution plot saved: {plot_file}')

def filter_samples_by_phenotype(plink_prefix: str, pheno_df: pd.DataFrame, output_prefix: str) -> str:
    output_prefix_abs = Path(output_prefix).absolute().as_posix()
    keep_file = f'{output_prefix_abs}_keep.txt'
    create_keep_file(pheno_df, keep_file)
    filter_cmd = [PLINK_EXECUTABLE, '--bfile', Path(plink_prefix).absolute().as_posix(), '--keep', keep_file, '--make-bed', '--out', output_prefix_abs, '--allow-extra-chr', '--allow-no-sex']
    try:
        subprocess.run(filter_cmd, capture_output=True, text=True, check=True)
        delete_temp_files(output_prefix_abs, ['_keep.txt'])
        if 'temp_plink' in plink_prefix:
            delete_temp_files(plink_prefix, ['.bed', '.bim', '.fam', '.log', '.nosex'])
        return output_prefix_abs
    except subprocess.CalledProcessError as e:
        logger.error(f'PLINK sample filtering failed: {e.stderr}')
        raise RuntimeError(f'Sample filtering failed: {e.stderr}')

def filter_snps_by_quality(plink_prefix: str, output_prefix: str, maf: float=0.05, geno: float=0.2) -> str:
    output_prefix_abs = Path(output_prefix).absolute().as_posix()
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    filter_cmd = [PLINK_EXECUTABLE, '--bfile', plink_prefix_abs, '--maf', str(maf), '--geno', str(geno), '--make-bed', '--out', output_prefix_abs, '--allow-extra-chr', '--allow-no-sex']
    try:
        result = subprocess.run(filter_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or 'Unknown error'
            logger.error(f'SNP quality filtering failed: {error_msg[:500]}')
            raise RuntimeError(f'SNP quality filtering failed: {error_msg[:500]}')
        bim_file = f'{output_prefix_abs}.bim'
        if Path(bim_file).exists():
            snp_count = sum((1 for _ in open(bim_file)))
        else:
            logger.debug('Cannot count filtered SNPs')
        if 'temp_plink' in plink_prefix:
            delete_temp_files(plink_prefix, ['.bed', '.bim', '.fam', '.log', '.nosex'])
        return output_prefix_abs
    except Exception as e:
        logger.error(f'SNP quality filtering exception: {str(e)}')
        raise

def generate_gwas_phenotype_file_preprocess(pheno_df: pd.DataFrame, output_file: Union[str, Path]) -> str:
    output_path = Path(output_file).absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample_id, pheno in zip(pheno_df['sample'], pheno_df['phenotype']):
            f.write(f'{sample_id}\t{pheno}\n')
    return output_path.as_posix()

def run_gwas_preprocess(plink_prefix: str, pheno_df: pd.DataFrame, output_dir: Union[str, Path], tmp_dir: Union[str, Path], pvalue_threshold: float=0.01, perform_plink_qc: bool=True) -> List[str]:
    output_dir_path = Path(output_dir).absolute()
    tmp_dir_path = Path(tmp_dir).absolute()
    gwas_work_dir = tmp_dir_path / 'gwas'
    gwas_work_dir.mkdir(parents=True, exist_ok=True)
    gwas_pheno_file = gwas_work_dir / 'pheno_for_gwas.txt'
    try:
        pheno_file_abs = generate_gwas_phenotype_file_preprocess(pheno_df, gwas_pheno_file)
        if not Path(pheno_file_abs).exists():
            raise FileNotFoundError(f'Failed to generate phenotype file: {pheno_file_abs}')
    except Exception as e:
        raise RuntimeError(f'Error while generating phenotype file: {str(e)}') from e
    gwas_output_prefix = output_dir_path / 'preprocess_gwas'
    gwas_output_prefix_abs = gwas_output_prefix.absolute().as_posix()
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    base_prefix = gwas_output_prefix.name
    gwas_result_file = output_dir_path / 'output' / f'{base_prefix}_gwas.assoc.txt'
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir_path)
        logger.debug('GWAS workflow running')
        gemma_gwas.run_complete_gwas_pipeline(input_plink_prefix=plink_prefix_abs, phenotype_file=pheno_file_abs, output_prefix=gwas_output_prefix_abs, perform_plink_qc=perform_plink_qc)
    except Exception as e:
        raise RuntimeError(f'GEMMA GWAS pipeline failed: {str(e)}') from e
    finally:
        os.chdir(original_cwd)
    if not gwas_result_file.exists():
        logger.error('GWAS result file missing')
        raise FileNotFoundError(f'GWAS result file missing: {gwas_result_file.absolute()}')
    try:
        gwas_df = pd.read_csv(gwas_result_file, sep='\t')
    except Exception as e:
        raise RuntimeError(f'Failed to read GWAS result file: {str(e)}') from e
    pvalue_col = None
    for col in ['p_wald', 'p_lrt', 'p_score']:
        if col in gwas_df.columns:
            pvalue_col = col
            break
    if not pvalue_col:
        raise ValueError(f'Cannot identify P-value column in GWAS results: {gwas_result_file}\nAvailable columns: {gwas_df.columns.tolist()}')
    snp_col = None
    for col in ['rs', 'SNP']:
        if col in gwas_df.columns:
            snp_col = col
            break
    if not snp_col:
        snp_col = gwas_df.columns[0]
        logger.debug(f"'rs'/'SNP' column not found; using the first column '{snp_col}' as SNP ID")
    significant_snps = gwas_df[gwas_df[pvalue_col] < pvalue_threshold][snp_col].dropna().astype(str).unique().tolist()
    logger.debug(f'GWAS result summary: {len(significant_snps):,} significant SNP(s)')
    try:
        logger.debug('GWAS temporary files cleanup started')
        import shutil
        gwas_prefix_base = str(Path(gwas_output_prefix_abs))
        delete_temp_files(f'{gwas_prefix_base}_clean_geno', ['.bed', '.bim', '.fam', '.log', '.nosex'])
        delete_temp_files(f'{gwas_prefix_base}_clean_geno_filtered', ['.bed', '.bim', '.fam', '.log', '.nosex'])
        delete_temp_files(f'{gwas_prefix_base}_geno_pca', ['.eigenvec', '.eigenval', '.log'])
        delete_temp_files(f'{gwas_prefix_base}_geno_pca_filtered', ['.eigenvec'])
        delete_temp_files(gwas_prefix_base, ['_valid_samples.txt'])
        delete_temp_files(gwas_prefix_base, ['_covariates_filtered.txt'])
        output_subdir = output_dir_path / 'output'
        if output_subdir.exists():
            try:
                shutil.rmtree(output_subdir, ignore_errors=True)
            except Exception:
                pass
        gwas_prefix_path = Path(gwas_output_prefix_abs)
        if gwas_prefix_path.parent.exists():
            for fp in gwas_prefix_path.parent.glob(f'{gwas_prefix_path.name}*'):
                try:
                    if fp.is_file():
                        fp.unlink()
                except Exception:
                    pass
        try:
            if gwas_work_dir.exists():
                shutil.rmtree(gwas_work_dir, ignore_errors=True)
        except Exception:
            pass
        logger.debug('GWAS temporary files cleanup completed')
    except Exception as e:
        logger.debug('GWAS temporary files cleanup partially failed (ignored)')
    return significant_snps

def chr_pos_match_variants(key: str) -> set:
    key = str(key).strip()
    parts = key.split('_', 1)
    if len(parts) != 2 or not str(parts[1]).isdigit():
        return {key}
    chr_part, pos_part = parts[0], parts[1]
    bare = chr_part[3:] if chr_part.lower().startswith('chr') else chr_part
    variants = {f'{chr_part}_{pos_part}', f'{bare}_{pos_part}', f'chr{bare}_{pos_part}'}
    return {v for v in variants if v}

def resolve_bim_snp_ids_for_chr_pos_keys(plink_prefix: str, chr_pos_keys: List[str]) -> List[str]:
    if not chr_pos_keys:
        return []
    requested: set = set()
    for key in chr_pos_keys:
        requested |= chr_pos_match_variants(key)
    bim_file = f'{Path(plink_prefix).absolute().as_posix()}.bim'
    if not Path(bim_file).exists():
        raise FileNotFoundError(f'PLINK .bim file not found: {bim_file}')
    snp_ids: List[str] = []
    seen: set = set()
    for chunk in pd.read_csv(bim_file, sep='\\s+', header=None, names=['chr', 'snp_id', 'genetic_dist', 'physical_pos', 'a1', 'a2'], dtype=str, chunksize=100000, engine='python'):
        for _, row in chunk.iterrows():
            sid = str(row['snp_id'])
            row_variants = chr_pos_match_variants(f'{row["chr"]}_{row["physical_pos"]}')
            if (row_variants & requested) or sid in requested:
                if sid not in seen:
                    snp_ids.append(sid)
                    seen.add(sid)
    return snp_ids

def extract_snps_by_chr_pos_with_plink(plink_prefix: str, chr_pos_keys: List[str], output_prefix: str) -> str:
    if not chr_pos_keys:
        raise ValueError('chr_pos_keys list is empty')
    snp_ids = resolve_bim_snp_ids_for_chr_pos_keys(plink_prefix, chr_pos_keys)
    if not snp_ids:
        logger.debug(f'No PLINK SNP IDs matched {len(chr_pos_keys):,} training chr_pos key(s); skipping --extract')
        return Path(plink_prefix).absolute().as_posix()
    logger.debug(f'PLINK --extract: {len(snp_ids):,} SNP ID(s) resolved from {len(chr_pos_keys):,} training chr_pos key(s)')
    return extract_snps_with_plink(plink_prefix, snp_ids, output_prefix)

def extract_snps_with_plink(plink_prefix: str, snp_list: List[str], output_prefix: str) -> str:
    if not snp_list:
        raise ValueError('SNP list for extraction is empty')
    output_prefix_abs = Path(output_prefix).absolute().as_posix()
    plink_prefix_abs = Path(plink_prefix).absolute().as_posix()
    snp_file = f'{output_prefix_abs}_extract_snps.txt'
    with open(snp_file, 'w') as f:
        for snp in snp_list:
            f.write(f'{snp}\n')
    cmd = [PLINK_EXECUTABLE, '--bfile', plink_prefix_abs, '--extract', snp_file, '--make-bed', '--out', output_prefix_abs, '--allow-extra-chr', '--allow-no-sex']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or 'Unknown error'
            logger.error(f'PLINK extract failed: {error_msg[:500]}')
            raise RuntimeError(f'PLINK extract failed: {error_msg[:500]}')
        bed_file = f'{output_prefix_abs}.bed'
        bim_file = f'{output_prefix_abs}.bim'
        fam_file = f'{output_prefix_abs}.fam'
        for fp in [bed_file, bim_file, fam_file]:
            if not Path(fp).exists():
                raise FileNotFoundError(f'Expected PLINK file not generated: {fp}')
        logger.debug(f'PLINK extract completed: {len(snp_list):,} SNPs retained')
        return output_prefix_abs
    finally:
        try:
            Path(snp_file).unlink()
        except Exception:
            pass

def generate_metadata(output_prefix: str, plink_prefix: str, train_file: str, valid_samples: List[str], genotype_format: str, task_type: Optional[Literal['regression', 'classification']]=None, preprocess_tmp_dir: Optional[Union[str, Path]]=None, filter_snps: bool=True, feature_selection_mode: int=1, gwas_pvalue: Optional[float]=None, ld_window_kb: Optional[int]=None, ld_window: Optional[int]=None, ld_window_r2: Optional[float]=None, n_feature_columns: Optional[int]=None, includes_phenotype_column: bool=True, sample_split: Optional[Dict[str, Any]]=None) -> None:
    chr_list = auto_detect_chromosomes(plink_prefix)
    metadata = {'preprocess_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), 'genotype_format': genotype_format, 'phenotype_format': 'Two-column, no-header format (sample ID + phenotype value)', 'output_train_file': train_file, 'valid_samples': valid_samples, 'sample_count': len(valid_samples), 'chromosome_list': chr_list, 'chromosome_count': len(chr_list), 'parallel_processes': 1, 'processing_mode': 'single_thread_sequential', 'plink_memory_limit_MB': 'unlimited (no --memory passed to PLINK)', 'snp_dtype': 'int8', 'snp_naming_rule': "chromosome_physicalPosition (e.g., '1_123456')", 'plink_executable': PLINK_EXECUTABLE, 'plink_recode_param': '--recodeA', 'gwas_genotype_prefix': None, 'preprocess_tmp_dir': str(Path(preprocess_tmp_dir).absolute()) if preprocess_tmp_dir else None, 'snp_filtering': {'filtered': filter_snps, 'maf_threshold': 0.05 if filter_snps else None, 'geno_threshold': 0.2 if filter_snps else None}, 'feature_selection': {'mode': feature_selection_mode, 'description': {1: 'no GWAS/LD feature selection', 2: 'GWAS-based SNP selection', 3: 'LD-based SNP pruning', 4: 'GWAS + LD combined filtering (GWAS first, then LD)'}.get(feature_selection_mode, 'unknown'), 'gwas_pvalue_threshold': gwas_pvalue if feature_selection_mode in [2, 4] else None, 'ld_window_kb': ld_window_kb if feature_selection_mode in [3, 4] else None, 'ld_window': ld_window if feature_selection_mode in [3, 4] else None, 'ld_window_r2': ld_window_r2 if feature_selection_mode in [3, 4] else None, 'samples_used': 'training_split_only' if feature_selection_mode in [2, 3, 4] else None}}
    if task_type:
        metadata['task_type'] = task_type
    if sample_split:
        metadata['sample_split'] = sample_split
    metadata['train_file_format'] = {'delimiter': '\t', 'index_column': 'sample', 'n_feature_columns': int(n_feature_columns) if n_feature_columns is not None else None, 'includes_phenotype_column': bool(includes_phenotype_column), 'compression': 'gzip' if str(train_file).endswith('.gz') else None, 'snp_dtype': 'int8'}
    metadata_file = f'{output_prefix}_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.debug('Metadata file generated successfully')

def validate_plink_files(genotype_path: str, fmt: str) -> None:
    required = {'plink_binary': ['.bed', '.bim', '.fam'], 'plink_text': ['.map', '.ped']}.get(fmt, [])
    missing = [f'{genotype_path}{ext}' for ext in required if not Path(f'{genotype_path}{ext}').exists()]
    if missing:
        raise FileNotFoundError(f"\nMissing PLINK files: {', '.join(missing)}")

def create_keep_file(pheno_df: pd.DataFrame, output_file: str) -> str:
    output_file = Path(output_file).absolute().as_posix()
    if 'sample' in pheno_df.columns:
        sample_ids = pheno_df['sample'].astype(str).unique()
    elif pheno_df.index.name == 'sample' or pheno_df.index.name is None:
        sample_ids = pheno_df.index.astype(str).unique()
    else:
        sample_ids = pheno_df.iloc[:, 0].astype(str).unique()
    with open(output_file, 'w') as f:
        for sample_id in sample_ids:
            f.write(f'{sample_id}\t{sample_id}\n')
    return output_file

def main(argv: Optional[List[str]]=None) -> int:
    parser = build_preprocess_argument_parser()
    args = parser.parse_args(argv)
    try:
        genotype_path, genotype_format = resolve_genotype_input(args)
        logger.debug('Starting preprocessing')
        return run_preprocess(genotype_path=genotype_path, genotype_format=genotype_format, phenotype_file=args.phenotype, output_file=args.output, filter_snps=not args.no_filter_snps, feature_selection_mode=args.feature_selection_mode, gwas_pvalue=args.gwas_pvalue, ld_config=args.ld_config, train_ids_file=getattr(args, 'train_ids_file', None), test_ids_file=getattr(args, 'test_ids_file', None), random_state=getattr(args, 'random_state', DEFAULT_SPLIT_RANDOM_STATE))
    except Exception as e:
        logger.error(f'Preprocessing failed: {e}')
        return 1
if __name__ == '__main__':
    sys.exit(main())
