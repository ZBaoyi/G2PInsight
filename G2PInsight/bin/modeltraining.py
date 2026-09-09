#!/usr/bin/env python3
import os
import sys
import json
import logging
import subprocess
import time
import shutil
import gc
import atexit
import signal
import threading
import resource
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, PredefinedSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class PeakMemoryTracker:
    """Poll RSS and track peak during start()..stop().

    With include_children=True, sums this process + descendants (process tree).

    Important: do NOT treat resource.ru_maxrss as the window peak by itself.
    ru_maxrss is a process-lifetime high-water mark and never decreases, so in
    serial train-all (multiple models in one process) every later model would
    incorrectly report the same lifetime max. We track instantaneous RSS over
    the window; ru_maxrss is only used when it rises above the start baseline
    (captures a new lifetime high that a coarse poll might miss).
    """

    def __init__(self, poll_interval_sec: float=0.5, include_children: bool=False):
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._poll_interval = poll_interval_sec
        self._include_children = bool(include_children)
        self._ru_baseline = 0

    @staticmethod
    def _ru_maxrss_bytes() -> int:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return int(rss)
        return int(rss) * 1024

    @staticmethod
    def _proc_self_rss_bytes() -> Optional[int]:
        """Linux current RSS via /proc (no psutil required)."""
        try:
            with open('/proc/self/status', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except (OSError, ValueError):
            return None
        return None

    def _self_rss_bytes(self) -> int:
        if PSUTIL_AVAILABLE:
            try:
                return int(psutil.Process(os.getpid()).memory_info().rss)
            except (psutil.Error, OSError):
                pass
        proc_rss = self._proc_self_rss_bytes()
        if proc_rss is not None:
            return proc_rss
        return self._ru_maxrss_bytes()

    def _sample_rss_bytes(self) -> int:
        rss = self._self_rss_bytes()
        if not self._include_children or not PSUTIL_AVAILABLE:
            return rss
        try:
            proc = psutil.Process(os.getpid())
            for child in proc.children(recursive=True):
                try:
                    rss += int(child.memory_info().rss)
                except (psutil.Error, OSError):
                    continue
        except (psutil.Error, OSError):
            pass
        return rss

    def _can_poll(self) -> bool:
        # Poll whenever we can read current RSS (not lifetime-only ru_maxrss).
        if PSUTIL_AVAILABLE:
            return True
        return self._proc_self_rss_bytes() is not None

    def _update_peak(self) -> None:
        rss = self._sample_rss_bytes()
        self._peak_bytes = max(self._peak_bytes, rss)
        # Only credit ru_maxrss when it advances past the value at start().
        ru_now = self._ru_maxrss_bytes()
        if ru_now > self._ru_baseline:
            self._peak_bytes = max(self._peak_bytes, ru_now)

    def start(self) -> 'PeakMemoryTracker':
        self._ru_baseline = self._ru_maxrss_bytes()
        self._peak_bytes = self._sample_rss_bytes()
        if self._can_poll():
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        return self

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            try:
                self._update_peak()
            except Exception:
                pass

    def stop(self) -> float:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._update_peak()
        return round(self._peak_bytes / (1024 * 1024), 2)
PUB_QUALITY_MODE: bool = True
CLEANUP_TEMP_FILES = True
DEFAULT_N_JOBS = 1
TRAIN_ALL_MAX_AUTO_WORKERS = 4  # upper cap when user explicitly sets --parallel_models > 1
_MAIN_PROCESS_PID = os.getpid()

def _is_main_process() -> bool:
    try:
        return mp.current_process().name == 'MainProcess'
    except Exception:
        return os.getpid() == _MAIN_PROCESS_PID

def _reset_worker_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (ValueError, OSError):
            pass

def _resolve_train_all_parallelism(num_models: int, parallel_models: Optional[int]=None, threads: Optional[int]=None) -> Tuple[int, int]:
    if parallel_models is None:
        parallel_model_count = 1
    else:
        parallel_model_count = max(1, min(int(parallel_models), num_models))
    cpu_cores_per_worker = DEFAULT_N_JOBS if threads is None else max(1, int(threads))
    return (max(1, parallel_model_count), cpu_cores_per_worker)

def _log_worker_crash_hint(*, parallel_mode: bool) -> None:
    if parallel_mode:
        logger.error('[TRAIN_ALL] Parallel training aborted: a worker process was terminated abruptly (likely insufficient memory or CPU).')
        logger.error('[TRAIN_ALL] Increase job memory/CPU limits, or reduce --parallel_models.')
    else:
        logger.error('[TRAIN_ALL] A model worker was terminated abruptly (likely OOM or native-library crash).')
        logger.error('[TRAIN_ALL] Increase SLURM --mem, or train the failed model alone: G2PInsight train -m <Model> ...')

MODEL_HP_SEARCH_N_ITER: Dict[str, int] = {'RandomForest': 15, 'SVM': 20}
HIGH_DIM_FEATURE_RF_THRESHOLD = 1000
HIGH_DIM_SHAP_DEPENDENCE_THRESHOLD = 1000
SHAP_DEPENDENCE_MAX_FEATURES = 50
TRAIN_SNP_WARN_THRESHOLD = 500_000
TRAIN_SNP_STRONG_WARN_THRESHOLD = 1_000_000
TRAIN_SVM_IMPRACTICAL_THRESHOLD = 10_000

def _resolve_shap_dependence_max_features(value: Optional[int]) -> int:
    if value is None:
        return SHAP_DEPENDENCE_MAX_FEATURES
    return max(0, int(value))

def _resolve_hp_search_n_iter(model_type: str) -> int:
    return MODEL_HP_SEARCH_N_ITER.get(model_type, 30)

class TempFileManager:

    def __init__(self):
        self.temp_files: List[Path] = []
        self.temp_dirs: List[Path] = []
        self.preprocess_tmp_dirs: List[Path] = []
        self._cleanup_registered = False
        self._cleaned = False
        self._register_exit_handlers()

    def _register_exit_handlers(self):
        if self._cleanup_registered:
            return
        if not _is_main_process():
            return
        atexit.register(self.cleanup_on_exit)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            pass
        self._cleanup_registered = True

    def _signal_handler(self, signum, frame):
        logger.debug(f'Received signal {signum}, cleaning up temporary files...')
        self.cleanup_on_exit()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def register_file(self, file_path: Path) -> Path:
        file_path = Path(file_path).absolute()
        if file_path not in self.temp_files:
            self.temp_files.append(file_path)
        return file_path

    def register_dir(self, dir_path: Path) -> Path:
        dir_path = Path(dir_path).absolute()
        dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path not in self.temp_dirs:
            self.temp_dirs.append(dir_path)
        return dir_path

    def register_preprocess_tmp_dir(self, dir_path: Path) -> Path:
        dir_path = Path(dir_path).absolute()
        if dir_path not in self.preprocess_tmp_dirs:
            self.preprocess_tmp_dirs.append(dir_path)
        return dir_path

    def cleanup_on_exit(self, cleanup_preprocess: bool=False):
        if not CLEANUP_TEMP_FILES:
            return
        if self._cleaned and (not cleanup_preprocess):
            return
        try:
            if not self._cleaned:
                for fp in self.temp_files:
                    try:
                        if fp.exists():
                            fp.unlink()
                    except Exception as e:
                        pass
                for dp in reversed(self.temp_dirs):
                    try:
                        if dp.exists():
                            shutil.rmtree(dp, ignore_errors=True)
                    except Exception as e:
                        pass
                self.temp_files.clear()
                self.temp_dirs.clear()
                self._cleaned = True
            if cleanup_preprocess:
                for dp in reversed(self.preprocess_tmp_dirs):
                    try:
                        if dp.exists():
                            shutil.rmtree(dp, ignore_errors=True)
                    except Exception as e:
                        pass
                self.preprocess_tmp_dirs.clear()
        except Exception as e:
            logger.debug(f'Error during temporary file cleanup (ignored): {e}')

    def clear(self):
        self.temp_files.clear()
        self.temp_dirs.clear()
        self.preprocess_tmp_dirs.clear()
        self._cleaned = False
_temp_file_manager = TempFileManager()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout)

def load_preprocess_metadata(metadata_file: str) -> Dict:
    if not Path(metadata_file).exists():
        raise FileNotFoundError(f'Preprocess metadata file not found: {metadata_file}')
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid metadata JSON format: {metadata_file}, error: {str(e)}')
    required_fields = ['valid_samples', 'output_train_file']
    missing_fields = [f for f in required_fields if f not in metadata]
    if missing_fields:
        raise ValueError(f"Metadata missing required fields: {', '.join(missing_fields)}")
    return metadata

def _training_dimensions_from_metadata(metadata: Optional[Dict]) -> Tuple[Optional[int], Optional[int]]:
    if not metadata:
        return (None, None)
    file_format = metadata.get('train_file_format') or {}
    n_features = file_format.get('n_feature_columns')
    n_samples = metadata.get('sample_count')
    if n_features is not None:
        n_features = int(n_features)
    if n_samples is not None:
        n_samples = int(n_samples)
    return (n_features, n_samples)

def _estimate_matrix_ram_gb(n_samples: int, n_features: int, dtype_bytes: int=1) -> float:
    return n_samples * n_features * dtype_bytes / 1024 ** 3

def _verify_compressed_train_file(train_file: str, expected_n_samples: Optional[int]=None) -> None:
    """Fail fast when a gzip training matrix is truncated before pandas load."""
    if not (str(train_file).endswith('.gz')):
        return
    path = Path(train_file)
    if not path.exists():
        raise FileNotFoundError(f'Training data file not found: {train_file}')
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f'Training matrix file is empty: {train_file}')
    import gzip
    newline_count = 0
    try:
        with gzip.open(train_file, 'rb') as f:
            while True:
                chunk = f.read(64 * 1024 * 1024)
                if not chunk:
                    break
                newline_count += chunk.count(b'\n')
    except EOFError as e:
        raise ValueError(f'Training matrix gzip file is truncated or corrupt: {train_file} ({size_bytes / (1024 ** 3):.2f} GB on disk).\nThis usually means preprocess was interrupted (OOM, disk full, or job kill) before the file finished writing.\nDelete the corrupt file and re-run preprocess with SNP filtering (-f 2 or -f 4). With ~4.7M SNPs and 528 samples, training also requires far more RAM than most jobs provide even if the file were valid.') from e
    if expected_n_samples is not None and expected_n_samples > 0:
        data_rows = newline_count - 1
        if data_rows != expected_n_samples:
            raise ValueError(f'Training matrix row count mismatch: file has {data_rows} data row(s), metadata expects {expected_n_samples} ({train_file}).\nRe-run preprocess to regenerate a complete training matrix.')

def check_training_metadata_guardrails(metadata: Optional[Dict], *, model_type: Optional[str]=None, task_type: Optional[str]=None, parallel_models: Optional[int]=None, enable_hyperparameter_search: bool=False, n_folds: int=5, threads: Optional[int]=None) -> None:
    n_features, n_samples = _training_dimensions_from_metadata(metadata)
    file_format = (metadata or {}).get('train_file_format') or {}
    is_gzip = file_format.get('compression') == 'gzip' or str((metadata or {}).get('output_train_file', '')).endswith('.gz')
    if n_features is None:
        logger.debug('[TRAIN] Metadata has no n_feature_columns; cannot pre-check matrix size before load')
        return
    logger.debug(f'[TRAIN] Metadata guardrail: {n_samples or "?"} samples × {n_features:,} SNP features')
    if n_samples is not None:
        est_gb = _estimate_matrix_ram_gb(n_samples, n_features, dtype_bytes=1)
        logger.debug(f'[TRAIN] Estimated feature matrix size (int8): ~{est_gb:.2f} GB')
        if est_gb >= 2.0:
            logger.debug(f'[TRAIN] Large feature matrix (~{est_gb:.1f} GB int8). Ensure job memory is well above this (CV/SHAP/workers add overhead).')
    train_file = (metadata or {}).get('output_train_file')
    if train_file and str(train_file).endswith('.gz'):
        _verify_compressed_train_file(str(train_file), expected_n_samples=n_samples)
    if n_features >= TRAIN_SNP_STRONG_WARN_THRESHOLD:
        logger.debug(f'[TRAIN] Very high SNP count ({n_features:,}). Training will be slow and memory-heavy; consider stronger SNP filtering in preprocess (-f 2 or -f 4).')
    elif n_features >= TRAIN_SNP_WARN_THRESHOLD:
        logger.debug(f'[TRAIN] High SNP count ({n_features:,}). Prefer tree models (LightGBM/XGBoost/CatBoost) and conservative parallelism.')
    if is_gzip:
        logger.debug('[TRAIN] Training matrix is gzip-compressed; load is slower than uncompressed .txt. Re-run preprocess without compression if you train repeatedly.')
    parallel_count = 1 if parallel_models is None else max(1, int(parallel_models))
    if parallel_count > 1 and n_features >= HIGH_DIM_FEATURE_RF_THRESHOLD:
        logger.debug(f'[TRAIN] parallel_models={parallel_count} with {n_features:,} features: each worker holds a full matrix copy; ensure job memory is large enough or use --parallel_models 1.')
    if model_type == 'SVM' and n_features > TRAIN_SVM_IMPRACTICAL_THRESHOLD:
        logger.debug(f'[TRAIN] SVM on {n_features:,} features is usually impractical (time/memory). Prefer LightGBM for high-dimensional GWAS matrices.')
    if model_type == 'RandomForest' and n_features > HIGH_DIM_FEATURE_RF_THRESHOLD:
        logger.debug(f'[TRAIN] RandomForest on {n_features:,} features is slow at this scale; LightGBM/XGBoost are usually faster.')
    if enable_hyperparameter_search and n_features >= TRAIN_SNP_WARN_THRESHOLD:
        logger.debug('[TRAIN] Hyperparameter search on a very wide matrix is expensive; omit --hyperparameter_search for a faster first run.')
    if n_folds > 3 and n_features >= TRAIN_SNP_WARN_THRESHOLD:
        logger.debug(f'[TRAIN] {n_folds}-fold CV multiplies full-matrix fits on high-dimensional data; expect longer runtime and higher peak RAM.')
    if threads is not None and int(threads) > 4 and n_features >= TRAIN_SNP_WARN_THRESHOLD:
        logger.debug(f'[TRAIN] --threads {int(threads)} increases per-model CPU/memory pressure on wide matrices; --threads 4 is often enough.')
    if model_type is None and task_type == 'regression' and n_features > TRAIN_SVM_IMPRACTICAL_THRESHOLD:
        logger.debug('[TRAIN] train-all will include SVM on a wide matrix; expect SVM to fail or run extremely long. Train LightGBM alone first: G2PInsight train -m LightGBM ...')

def parse_train_input_path(input_path: str) -> Dict:
    input_path = Path(input_path).absolute().as_posix()
    result = {'train_file': None, 'metadata': None, 'gwas_genotype_prefix': None, 'task_type': None, 'preprocess_tmp_dir': None, '_metadata_file_path': None}
    if input_path.endswith('_metadata.json'):
        metadata = load_preprocess_metadata(input_path)
        result['metadata'] = metadata
        result['train_file'] = metadata['output_train_file']
        result['gwas_genotype_prefix'] = metadata.get('gwas_genotype_prefix')
        result['task_type'] = metadata.get('task_type')
        result['preprocess_tmp_dir'] = metadata.get('preprocess_tmp_dir')
        result['_metadata_file_path'] = input_path
    else:
        result['train_file'] = input_path
        train_file_path = Path(input_path)
        possible_metadata_paths = [f'{os.path.splitext(input_path)[0]}_metadata.json', str(train_file_path.parent / f'{train_file_path.stem}_metadata.json'), str(train_file_path.parent / f'{train_file_path.parent.name}_metadata.json')]
        if train_file_path.parent.exists():
            for metadata_file in train_file_path.parent.glob('*_metadata.json'):
                possible_metadata_paths.append(str(metadata_file))
        metadata_found = False
        for metadata_path in possible_metadata_paths:
            if Path(metadata_path).exists():
                try:
                    metadata = load_preprocess_metadata(metadata_path)
                    result['metadata'] = metadata
                    result['gwas_genotype_prefix'] = metadata.get('gwas_genotype_prefix')
                    result['task_type'] = metadata.get('task_type')
                    result['preprocess_tmp_dir'] = metadata.get('preprocess_tmp_dir')
                    result['_metadata_file_path'] = metadata_path
                    metadata_found = True
                    break
                except Exception as e:
                    continue
        if not metadata_found:
            pass
    if not Path(result['train_file']).exists():
        raise FileNotFoundError(f"Training file does not exist: {result['train_file']}")
    return result

def clean_feature_names(feature_names: List[str]) -> List[str]:
    import re
    special_chars = '[{}\\[\\]\\,"\\\':]'
    cleaned_names = []
    for name in feature_names:
        cleaned = re.sub(special_chars, '_', str(name))
        cleaned = re.sub('_+', '_', cleaned)
        cleaned = cleaned.strip('_')
        if not cleaned:
            cleaned = f'feature_{hash(name) % 1000000}'
        cleaned_names.append(cleaned)
    return cleaned_names

def column_to_chr_pos(col_name: str) -> Optional[str]:
    import re
    col_str = str(col_name)
    pattern_colon = re.compile('^([^:]+):([^_]+)_[^_]+$')
    pattern_under = re.compile('^([^_]+)_([^_]+)_[^_]+$')
    match = pattern_colon.match(col_str)
    if match:
        return f'{match.group(1)}_{match.group(2)}'
    match = pattern_under.match(col_str)
    if match:
        return f'{match.group(1)}_{match.group(2)}'
    cleaned = clean_feature_names([col_str])[0]
    parts = cleaned.split('_')
    if len(parts) >= 2 and parts[1].isdigit():
        return f'{parts[0]}_{parts[1]}'
    if '_' in cleaned and all((p.isdigit() for p in parts[1:2])):
        return cleaned
    return None

def build_feature_chr_pos_keys(feature_names: List[str]) -> List[str]:
    keys: List[str] = []
    for name in feature_names:
        key = column_to_chr_pos(name)
        keys.append(key if key else str(name))
    return keys

def _rename_columns_to_chr_pos(columns: List[str]) -> List[str]:
    return [column_to_chr_pos(col) or str(col) for col in columns]

def _apply_feature_column_names(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    renamed_cols = _rename_columns_to_chr_pos(X.columns.tolist())
    X = X.copy()
    X.columns = renamed_cols
    original_cols = X.columns.tolist()
    cleaned_cols = clean_feature_names(original_cols)
    snp_name_mapping: Dict[str, str] = {}
    if len(cleaned_cols) != len(set(cleaned_cols)):
        from collections import Counter, defaultdict
        col_counts = Counter(cleaned_cols)
        col_indices = defaultdict(int)
        final_cols = []
        for i, cleaned_col in enumerate(cleaned_cols):
            if col_counts[cleaned_col] > 1:
                col_indices[cleaned_col] += 1
                final_col = f'{cleaned_col}_{col_indices[cleaned_col]}'
            else:
                final_col = cleaned_col
            final_cols.append(final_col)
            snp_name_mapping[original_cols[i]] = final_col
        X.columns = final_cols
    else:
        X.columns = cleaned_cols
        for orig, cleaned in zip(original_cols, cleaned_cols):
            snp_name_mapping[orig] = cleaned
    if X.columns.duplicated().any():
        dup_count = int(X.columns.duplicated().sum())
        logger.debug(f'  {dup_count} duplicate chr_pos column(s) after rename; keeping the first occurrence')
        X = X.loc[:, ~X.columns.duplicated(keep='first')]
    return X, snp_name_mapping

def _read_header_columns(train_file: str, delimiter: str, is_compressed: bool) -> List[str]:
    import gzip
    if is_compressed:
        with gzip.open(train_file, 'rt', encoding='utf-8') as f:
            header_line = f.readline().strip()
    else:
        with open(train_file, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
    return header_line.split(delimiter)

def _usecols_for_chr_pos_keys(header_columns: List[str], chr_pos_keys: set) -> List[str]:
    usecols: List[str] = []
    seen_keys: set = set()
    if 'sample' in header_columns:
        usecols.append('sample')
    for col in header_columns:
        if col == 'sample':
            continue
        key = column_to_chr_pos(col) or col
        if key in chr_pos_keys and key not in seen_keys:
            usecols.append(col)
            seen_keys.add(key)
    return usecols

def align_features_by_chr_pos(X: pd.DataFrame, train_features: List[str], train_chr_pos_keys: List[str]) -> Tuple[pd.DataFrame, int, int]:
    if len(train_features) != len(train_chr_pos_keys):
        raise ValueError(f'feature_names ({len(train_features)}) and feature_chr_pos_keys ({len(train_chr_pos_keys)}) length mismatch')
    X_by_key: Dict[str, pd.Series] = {}
    for col in X.columns:
        key = column_to_chr_pos(str(col)) or str(col)
        if key not in X_by_key:
            X_by_key[key] = X[col]
    X_aligned = pd.DataFrame(index=X.index)
    matched = 0
    for feat_name, chr_pos_key in zip(train_features, train_chr_pos_keys):
        if chr_pos_key in X_by_key:
            X_aligned[feat_name] = X_by_key[chr_pos_key]
            matched += 1
        else:
            X_aligned[feat_name] = 0
    missing = len(train_features) - matched
    return X_aligned, matched, missing

def _delimiter_from_metadata(file_format: Optional[Dict]) -> Optional[str]:
    if not file_format:
        return None
    delim = file_format.get('delimiter')
    if delim is None:
        return None
    if delim == '\\t':
        return '\t'
    return str(delim)

def _resolve_snp_storage_dtype(file_format: Optional[Dict]) -> np.dtype:
    default = np.dtype('int8')
    if not file_format:
        return default
    dtype_name = file_format.get('snp_dtype')
    if dtype_name is None:
        return default
    if isinstance(dtype_name, np.dtype):
        return dtype_name if dtype_name == default else default
    name = str(dtype_name).strip()
    name_lower = name.lower()
    if name_lower in ('int8', 'i1', 'numpy.int8', 'byte') or 'int8' in name_lower:
        return default
    try:
        return np.dtype(name)
    except (TypeError, ValueError):
        logger.debug(f'Unrecognized snp_dtype={dtype_name!r} in metadata; defaulting to int8')
        return default

def _is_int8_dtype(dtype: Any) -> bool:
    try:
        return np.dtype(dtype) == np.dtype('int8')
    except (TypeError, ValueError):
        return False

def _convert_feature_matrix_dtype(X: pd.DataFrame, target_dtype: np.dtype) -> pd.DataFrame:
    if X.empty:
        return X
    if _is_int8_dtype(target_dtype):
        arr = X.to_numpy(copy=False)
        if arr.dtype == np.int8:
            return X
        flat = pd.to_numeric(arr.ravel(), errors='coerce')
        converted = np.nan_to_num(flat, nan=-1).astype(np.int8).reshape(arr.shape)
        return pd.DataFrame(converted, index=X.index, columns=X.columns)
    try:
        return X.astype(target_dtype, copy=False)
    except (TypeError, ValueError):
        return X.astype(str(np.dtype(target_dtype)), copy=False)

def _training_file_read_engine(is_compressed: bool) -> str:
    return 'python' if is_compressed else 'c'

def _merge_oof_probabilities(prob_frames: List[pd.DataFrame], y_test_index: pd.Index) -> Optional[np.ndarray]:
    if not prob_frames:
        return None
    y_prob_df = pd.concat(prob_frames, axis=0)
    if y_prob_df.index.duplicated().any():
        logger.debug('  Duplicate sample indices in OOF probabilities; keeping the last occurrence per sample')
        y_prob_df = y_prob_df[~y_prob_df.index.duplicated(keep='last')]
    missing = y_test_index.difference(y_prob_df.index)
    if len(missing) > 0:
        logger.debug(f'  {len(missing)} sample(s) missing OOF probabilities after index alignment')
    aligned = y_prob_df.reindex(y_test_index)
    if aligned.isna().any().any():
        logger.debug('  NaN values in aligned OOF probabilities (missing folds/samples)')
    return aligned.to_numpy()

def load_training_data(train_file: str, valid_samples: List[str]=None, file_format: Optional[Dict]=None, chr_pos_filter: Optional[set]=None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    train_file_path = Path(train_file)
    if not train_file_path.exists():
        raise FileNotFoundError(f'Training data file not found: {train_file}')
    file_ext = train_file_path.suffix.lower()
    if file_ext in ['.vcf', '.vcf.gz']:
        raise ValueError(f"VCF file detected: {train_file}\nError: Model training/prediction requires preprocessed training data format (tab-separated .txt file with 'sample' column as index).\nVCF files must be preprocessed first using the 'preprocess' command.\nPlease run: G2PInsight preprocess --vcf|--bfile|--file <genotype> -p <phenotype_file> -o <output_dir>\nThen use the preprocessed output file for training/prediction.")
    is_compressed = file_ext == '.gz' or train_file.endswith('.gz')
    expected_n_samples = None
    if file_format and file_format.get('sample_count') is not None:
        expected_n_samples = int(file_format['sample_count'])
    elif valid_samples:
        expected_n_samples = len(valid_samples)
    if is_compressed:
        _verify_compressed_train_file(train_file, expected_n_samples=expected_n_samples)
    read_t0 = time.perf_counter()
    read_engine = _training_file_read_engine(is_compressed)
    if is_compressed:
        logger.debug('Training file is compressed (.gz); using python CSV engine for load')
    else:
        logger.debug('Training file is uncompressed; using C CSV engine for load')
    snp_storage_dtype = _resolve_snp_storage_dtype(file_format)
    try:
        import csv
        import gzip
        if is_compressed:
            with gzip.open(train_file, 'rt', encoding='utf-8') as f:
                sample_lines = []
                for _ in range(10):
                    line = f.readline()
                    if line.strip():
                        sample_lines.append(line)
                    if not line:
                        break
        else:
            with open(train_file, 'r', encoding='utf-8') as f:
                sample_lines = []
                for _ in range(10):
                    line = f.readline()
                    if line.strip():
                        sample_lines.append(line)
                    if not line:
                        break
                f.seek(0)
        if not sample_lines:
            raise ValueError(f'File {train_file} appears to be empty or contains only empty lines')
        metadata_delimiter = _delimiter_from_metadata(file_format)
        if metadata_delimiter is not None:
            delimiter = metadata_delimiter
            logger.debug(f'Using training file delimiter from metadata: {repr(delimiter)}')
        else:
            delimiter = '\t'
            try:
                sample_text = ''.join(sample_lines[:5])
                sniffer = csv.Sniffer()
                detected_delimiter = sniffer.sniff(sample_text, delimiters='\t, ').delimiter
                if detected_delimiter:
                    delimiter = detected_delimiter
            except Exception:
                delimiter_counts = {'\t': [], ',': [], ' ': []}
                for line in sample_lines:
                    if not line.strip():
                        continue
                    delimiter_counts['\t'].append(line.count('\t'))
                    delimiter_counts[','].append(line.count(','))
                    delimiter_counts[' '].append(line.count(' '))
                delimiter_avg = {}
                for sep, counts in delimiter_counts.items():
                    non_zero_counts = [c for c in counts if c > 0]
                    delimiter_avg[sep] = sum(non_zero_counts) / len(non_zero_counts) if non_zero_counts else 0
                max_avg = max(delimiter_avg.values())
                if max_avg > 0:
                    for sep, avg_count in delimiter_avg.items():
                        if avg_count == max_avg and avg_count > 0:
                            delimiter = sep
                            break
            if sample_lines:
                first_line = sample_lines[0].strip()
                field_count = len(first_line.split(delimiter))
                if field_count < 2:
                    logger.debug(f'Detected field count ({field_count}) is too low, trying alternative delimiters')
                    for alt_sep in ['\t', ',', ' ']:
                        if alt_sep != delimiter:
                            alt_count = len(first_line.split(alt_sep))
                            if alt_count >= 2:
                                logger.debug(f'Switching to delimiter {repr(alt_sep)} (field count: {alt_count})')
                                delimiter = alt_sep
                                break
        usecols = None
        if chr_pos_filter:
            header_columns = _read_header_columns(train_file, delimiter, is_compressed)
            usecols = _usecols_for_chr_pos_keys(header_columns, chr_pos_filter)
            n_snp_cols = max(0, len(usecols) - (1 if 'sample' in usecols else 0))
            logger.debug(f'  chr_pos filter: loading {n_snp_cols} of {max(0, len(header_columns) - 1)} SNP column(s) from file header')
        df = None
        read_errors = []
        read_csv_kwargs = {'sep': delimiter, 'index_col': 'sample', 'na_filter': False, 'engine': read_engine, 'compression': 'gzip' if is_compressed else 'infer'}
        if read_engine == 'c':
            read_csv_kwargs['low_memory'] = False
        if usecols is not None:
            read_csv_kwargs['usecols'] = usecols
        try:
            df = pd.read_csv(train_file, on_bad_lines='skip', **read_csv_kwargs)
        except TypeError as e:
            read_errors.append(f"Strategy 1 (on_bad_lines='skip'): {str(e)}")
            try:
                df = pd.read_csv(train_file, error_bad_lines=False, warn_bad_lines=True, **read_csv_kwargs)
            except (TypeError, ValueError) as e2:
                read_errors.append(f'Strategy 2 (error_bad_lines=False): {str(e2)}')
                try:
                    df = pd.read_csv(train_file, **read_csv_kwargs)
                except Exception as e3:
                    read_errors.append(f'Strategy 3 (basic mode): {str(e3)}')
                    if not is_compressed and usecols is None:
                        try:
                            df = pd.read_csv(train_file, sep=delimiter, index_col='sample', na_filter=False, engine='c', low_memory=False)
                        except Exception as e4:
                            read_errors.append(f'Strategy 4 (C engine): {str(e4)}')
                    else:
                        read_errors.append(f'Strategy 4 (C engine): Skipped (compressed file or usecols filter active)')
                        error_msg = f'Failed to read file {train_file} with delimiter {repr(delimiter)}. Tried strategies:\n'
                        error_msg += '\n'.join((f'  - {err}' for err in read_errors))
                        if any(('Index sample invalid' in err or 'sample' in err.lower() for err in read_errors)):
                            try:
                                with open(train_file, 'r', encoding='utf-8') as f:
                                    header_line = f.readline().strip()
                                    if delimiter:
                                        columns = header_line.split(delimiter)
                                        if 'sample' not in columns:
                                            error_msg += f"\n\nError: File does not contain 'sample' column as required.\n"
                                            cols_suffix = '...' if len(columns) > 10 else ''
                                            error_msg += f'Found columns: {columns[:10]}{cols_suffix}\n'
                                            error_msg += f'This file format is not compatible with model training/prediction.\n'
                                            error_msg += f"Expected format: Tab-separated file with 'sample' column as index (from preprocess module output).\n"
                                            if file_ext in ['.vcf', '.vcf.gz']:
                                                error_msg += 'Note: VCF files must be preprocessed first using: G2PInsight preprocess --vcf <vcf_file> -p <phenotype_file> -o <output_dir>'
                            except Exception:
                                pass
                        error_msg += f'\n\nPlease check the file format. Expected delimiter: {repr(delimiter)}'
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e3
        except ValueError as e:
            read_errors.append(f"Strategy 1 (on_bad_lines='skip'): {str(e)}")
            try:
                df = pd.read_csv(train_file, error_bad_lines=False, warn_bad_lines=True, **read_csv_kwargs)
            except (TypeError, ValueError) as e2:
                read_errors.append(f'Strategy 2 (error_bad_lines=False): {str(e2)}')
                try:
                    df = pd.read_csv(train_file, **read_csv_kwargs)
                except Exception as e3:
                    read_errors.append(f'Strategy 3 (basic mode): {str(e3)}')
                    if not is_compressed and usecols is None:
                        try:
                            df = pd.read_csv(train_file, sep=delimiter, index_col='sample', na_filter=False, engine='c', low_memory=False)
                        except Exception as e4:
                            read_errors.append(f'Strategy 4 (C engine): {str(e4)}')
                    else:
                        read_errors.append(f'Strategy 4 (C engine): Skipped (compressed file or usecols filter active)')
                        error_msg = f'Failed to read file {train_file} with delimiter {repr(delimiter)}. Tried strategies:\n'
                        error_msg += '\n'.join((f'  - {err}' for err in read_errors))
                        if any(('Index sample invalid' in err or 'sample' in err.lower() for err in read_errors)):
                            try:
                                with open(train_file, 'r', encoding='utf-8') as f:
                                    header_line = f.readline().strip()
                                    if delimiter:
                                        columns = header_line.split(delimiter)
                                        if 'sample' not in columns:
                                            error_msg += f"\n\nError: File does not contain 'sample' column as required.\n"
                                            cols_suffix = '...' if len(columns) > 10 else ''
                                            error_msg += f'Found columns: {columns[:10]}{cols_suffix}\n'
                                            error_msg += f'This file format is not compatible with model training/prediction.\n'
                                            error_msg += f"Expected format: Tab-separated file with 'sample' column as index (from preprocess module output).\n"
                                            if file_ext in ['.vcf', '.vcf.gz']:
                                                error_msg += 'Note: VCF files must be preprocessed first using: G2PInsight preprocess --vcf <vcf_file> -p <phenotype_file> -o <output_dir>'
                            except Exception:
                                pass
                        error_msg += f'\n\nPlease check the file format. Expected delimiter: {repr(delimiter)}'
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e4
        if df is None or df.empty:
            raise ValueError(f'Failed to read data from {train_file} or file is empty')
        if file_format:
            expected_snp = file_format.get('n_feature_columns')
            includes_pheno = bool(file_format.get('includes_phenotype_column', False))
            if expected_snp is not None:
                expected_cols = int(expected_snp) + (1 if includes_pheno else 0)
                actual_cols = len(df.columns)
                if actual_cols != expected_cols:
                    logger.debug(f'Training matrix column count differs from metadata: read {actual_cols} column(s), metadata expects {expected_cols} (n_feature_columns={expected_snp}, includes_phenotype_column={includes_pheno})')
                else:
                    logger.debug(f'Training matrix column count matches metadata ({actual_cols} columns)')
        if df.index.name != 'sample' and 'sample' not in str(df.index.name).lower():
            if 'sample' in df.columns:
                logger.debug("Found 'sample' column but not as index. Attempting to set it as index...")
                df = df.set_index('sample')
            else:
                raise ValueError(f"File {train_file} does not have 'sample' column/index as required.\nCurrent index name: {df.index.name}\nCurrent columns: {list(df.columns[:10])}{('...' if len(df.columns) > 10 else '')}\nThis file format is not compatible with model training/prediction.\nExpected format: Tab-separated file with 'sample' column as index (from preprocess module output).")
        if valid_samples:
            df = df.loc[df.index.isin(valid_samples)]
        if 'phenotype' in df.columns:
            X = df.drop(columns=['phenotype'])
            y = df['phenotype']
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        X = filter_phenotype_from_dataframe(X)
        X, snp_name_mapping = _apply_feature_column_names(X)
        if X.empty or y.empty:
            raise ValueError('Loaded training data is empty')
        if X.isnull().any().any():
            X = X.fillna(-1)
        X = _convert_feature_matrix_dtype(X, snp_storage_dtype)
        read_elapsed = time.perf_counter() - read_t0
        feature_dtype = X.dtypes.iloc[0] if len(X.columns) > 0 else 'n/a'
        logger.debug(f'Training matrix loaded: {X.shape[0]:,} samples × {X.shape[1]:,} features, feature_dtype={feature_dtype}, read_engine={read_engine}, elapsed={read_elapsed:.2f}s')
        return (X, y, snp_name_mapping)
    except EOFError as e:
        msg = (f'Compressed training matrix ended unexpectedly: {train_file}\nThe gzip file is truncated or corrupt. Delete it and re-run preprocess (prefer -f 4 for GWAS+LD filtering).\nWith millions of SNP columns, also ensure sufficient disk space and memory during preprocess export.')
        logger.error(msg)
        raise ValueError(msg) from e
    except Exception as e:
        logger.error(f'Failed to load training data: {str(e)}')
        raise

def _persist_training_matrix_for_workers(X: pd.DataFrame, y: pd.Series, cache_dir: Path) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_samples, n_features = X.shape
    x_path = cache_dir / 'train_X.int8.mmap'
    x_arr = np.ascontiguousarray(X.to_numpy(copy=False), dtype=np.int8)
    mm = np.memmap(x_path, dtype=np.int8, mode='w+', shape=(n_samples, n_features))
    mm[:] = x_arr
    mm.flush()
    del mm
    index_path = cache_dir / 'train_index.txt'
    with open(index_path, 'w', encoding='utf-8') as f:
        for sid in X.index:
            f.write(f'{sid}\n')
    columns_path = cache_dir / 'train_columns.txt'
    with open(columns_path, 'w', encoding='utf-8') as f:
        for col in X.columns:
            f.write(f'{col}\n')
    y_path = cache_dir / 'train_y.npy'
    np.save(y_path, np.asarray(y.values))
    bundle = {'_shared_matrix': True, 'shared_matrix_dir': str(cache_dir.absolute()), 'x_mmap_path': str(x_path.absolute()), 'x_shape': [int(n_samples), int(n_features)], 'index_path': str(index_path.absolute()), 'columns_path': str(columns_path.absolute()), 'y_path': str(y_path.absolute())}
    logger.debug(f'[TRAIN_ALL] Shared training matrix persisted: {n_samples:,} samples × {n_features:,} features ({x_path.name})')
    return bundle

def _load_training_data_from_shared(bundle: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    n_samples, n_features = (int(bundle['x_shape'][0]), int(bundle['x_shape'][1]))
    x_arr = np.memmap(bundle['x_mmap_path'], dtype=np.int8, mode='r', shape=(n_samples, n_features))
    with open(bundle['index_path'], 'r', encoding='utf-8') as f:
        index = [line.strip() for line in f if line.strip()]
    with open(bundle['columns_path'], 'r', encoding='utf-8') as f:
        columns = [line.strip() for line in f if line.strip()]
    if len(index) != n_samples:
        raise ValueError(f'Shared matrix index length mismatch: expected {n_samples}, got {len(index)}')
    if len(columns) != n_features:
        raise ValueError(f'Shared matrix column length mismatch: expected {n_features}, got {len(columns)}')
    X = pd.DataFrame(np.asarray(x_arr), index=index, columns=columns)
    y_values = np.load(bundle['y_path'])
    y = pd.Series(y_values, index=index, name='phenotype')
    logger.debug(f'[TRAIN_ALL] Worker attached to shared training matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features')
    return (X, y)

def _cleanup_shared_training_matrix(bundle: Optional[Dict[str, Any]]) -> None:
    if not bundle or not bundle.get('_shared_matrix'):
        return
    for key in ('x_mmap_path', 'index_path', 'columns_path', 'y_path'):
        path = bundle.get(key)
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except TypeError:
                fp = Path(path)
                if fp.exists():
                    fp.unlink()
            except Exception:
                pass
    shared_dir = bundle.get('shared_matrix_dir')
    if shared_dir:
        try:
            Path(shared_dir).rmdir()
        except Exception:
            pass

def _cleanup_train_all_temp_dirs(base_output_dir: Optional[Path], *, shared_matrix_bundle: Optional[Dict[str, Any]]=None, preprocess_tmp_dir: Optional[str]=None, output_dir: Optional[Path]=None) -> None:
    """Best-effort cleanup of train-all temp dirs (shared memmap, preprocess tmp, GWAS output). Safe to call multiple times."""
    if not CLEANUP_TEMP_FILES:
        return
    try:
        _cleanup_shared_training_matrix(shared_matrix_bundle)
    except Exception:
        pass
    if base_output_dir is not None:
        try:
            tmp_root_dir = Path(base_output_dir) / 'tmp'
            if tmp_root_dir.exists():
                shutil.rmtree(tmp_root_dir, ignore_errors=True)
                logger.debug(f'[TRAIN_ALL] Cleaned temporary directory: {tmp_root_dir}')
        except Exception as e:
            logger.debug(f'[TRAIN_ALL] Failed to clean temporary directory (ignored): {e}')
    if preprocess_tmp_dir:
        try:
            preprocess_tmp_dir_path = Path(preprocess_tmp_dir).absolute()
            if preprocess_tmp_dir_path.exists():
                shutil.rmtree(preprocess_tmp_dir_path, ignore_errors=True)
                logger.debug(f'[TRAIN_ALL] Deleted preprocess temporary directory: {preprocess_tmp_dir_path}')
        except Exception as e:
            logger.debug(f'[TRAIN_ALL] Failed to delete preprocess temporary directory (ignored): {e}')
    if output_dir is not None:
        gwas_output_dir = Path(output_dir) / 'output'
        if gwas_output_dir.exists():
            try:
                shutil.rmtree(gwas_output_dir, ignore_errors=True)
            except Exception as e:
                logger.debug(f'[TRAIN_ALL] Failed to delete GWAS output directory (ignored): {e}')

ALL_MODEL_TYPES = ['LightGBM', 'RandomForest', 'XGBoost', 'SVM', 'CatBoost', 'Logistic']
CLASSIFICATION_ONLY_MODELS = frozenset[str]({'Logistic'})

def get_supported_models(task_type: str) -> List[str]:
    if task_type == 'classification':
        return list(ALL_MODEL_TYPES)
    if task_type == 'regression':
        return [m for m in ALL_MODEL_TYPES if m not in CLASSIFICATION_ONLY_MODELS]
    raise ValueError(f'Unsupported task type: {task_type}; available: classification, regression')

def validate_model_for_task(model_type: str, task_type: str) -> None:
    if model_type in CLASSIFICATION_ONLY_MODELS and task_type != 'classification':
        raise ValueError(f"Model '{model_type}' only supports classification tasks, not '{task_type}'.")
    if model_type not in get_supported_models(task_type):
        raise ValueError(f"Unsupported model type '{model_type}' for task '{task_type}'; available: {get_supported_models(task_type)}")

def init_model(model_type: str, task_type: str, random_state: int=42, cpu_cores: Optional[int]=None) -> Any:
    common_params = {'random_state': random_state}
    if cpu_cores is None:
        cpu_cores_value = None
    else:
        cpu_cores_value = max(1, int(cpu_cores))
    model_classes = {'classification': {'LightGBM': lgb.LGBMClassifier, 'RandomForest': RandomForestClassifier, 'XGBoost': xgb.XGBClassifier, 'SVM': SVC, 'CatBoost': CatBoostClassifier, 'Logistic': LogisticRegression}, 'regression': {'LightGBM': lgb.LGBMRegressor, 'RandomForest': RandomForestRegressor, 'XGBoost': xgb.XGBRegressor, 'SVM': SVR, 'CatBoost': CatBoostRegressor}}
    model_params = {'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': cpu_cores_value if cpu_cores_value is not None else DEFAULT_N_JOBS, **common_params}, 'RandomForest': {'n_estimators': 100, 'max_depth': None, 'n_jobs': cpu_cores_value if cpu_cores_value is not None else DEFAULT_N_JOBS, **common_params}, 'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'verbosity': 0, 'n_jobs': cpu_cores_value if cpu_cores_value is not None else DEFAULT_N_JOBS, **common_params}, 'SVM': {'kernel': 'rbf', **({'probability': True} if task_type == 'classification' else {})}, 'CatBoost': {'iterations': 100, 'learning_rate': 0.1, 'verbose': 0, 'allow_writing_files': False, 'thread_count': cpu_cores_value if cpu_cores_value is not None else DEFAULT_N_JOBS, **common_params}, 'Logistic': {'n_jobs': cpu_cores_value if cpu_cores_value is not None else DEFAULT_N_JOBS, 'max_iter': 1000, **common_params}}
    if task_type not in model_classes:
        raise ValueError(f'Unsupported task type: {task_type}; available: {list(model_classes.keys())}')
    if model_type not in model_classes[task_type]:
        raise ValueError(f'Unsupported model type: {model_type}; available: {list(model_classes[task_type].keys())}')
    model_class = model_classes[task_type][model_type]
    params = model_params[model_type]
    model = model_class(**params)
    try:
        if model_type == 'CatBoost' and hasattr(model, 'set_params'):
            model.set_params(allow_writing_files=False)
    except Exception:
        pass
    return model

def get_param_grid(model_type: str, task_type: str, n_features: Optional[int]=None) -> Dict:
    common_param_grids = {'LightGBM': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'num_leaves': [15, 31, 63], 'max_depth': [3, 5, 7, -1], 'min_child_samples': [10, 20, 30], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]}, 'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}, 'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0], 'gamma': [0, 0.1, 0.2]}, 'CatBoost': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [4, 6, 8], 'l2_leaf_reg': [1, 3, 5], 'border_count': [32, 64, 128]}}
    if task_type == 'classification':
        task_specific = {'SVM': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'kernel': ['rbf', 'poly', 'sigmoid']}, 'Logistic': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2', 'elasticnet'], 'solver': ['liblinear', 'lbfgs', 'saga'], 'max_iter': [500, 1000, 2000]}}
    else:
        task_specific = {'SVM': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'kernel': ['rbf', 'poly', 'sigmoid'], 'epsilon': [0.01, 0.1, 0.2]}}
    param_grids = {**common_param_grids, **task_specific}
    if model_type not in param_grids:
        raise ValueError(f'Unsupported model type: {model_type}; available: {list(param_grids.keys())}')
    grid = dict(param_grids[model_type])
    if model_type == 'RandomForest' and n_features is not None and n_features > HIGH_DIM_FEATURE_RF_THRESHOLD:
        grid['max_depth'] = [5, 10, 20]
        grid['max_features'] = ['sqrt', 'log2']
        grid['n_estimators'] = [50, 100]
        logger.debug(f'  RandomForest search grid capped for high-dimensional data ({n_features:,} features)')
    return grid

def _make_cv_splitter(task_type: str, n_folds: int, random_state: int):
    if task_type == 'classification':
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

def perform_grid_search(model_type: str, task_type: str, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict, n_iter: int=20, cv: int=3, random_state: int=42, scoring: Optional[str]=None, cpu_cores: Optional[int]=None) -> RandomizedSearchCV:
    base_model = init_model(model_type, task_type, random_state=random_state, cpu_cores=cpu_cores)
    if scoring is None:
        if task_type == 'classification':
            scoring = 'roc_auc' if len(y_train.unique()) == 2 else 'f1_weighted'
        else:
            scoring = 'neg_mean_squared_error'
    cv_splitter = cv if hasattr(cv, 'split') else _make_cv_splitter(task_type, int(cv), random_state)
    search = RandomizedSearchCV(estimator=base_model, param_distributions=param_grid, n_iter=n_iter, cv=cv_splitter, scoring=scoring, n_jobs=DEFAULT_N_JOBS, random_state=random_state, verbose=0)
    search.fit(X_train, y_train)
    return search

def fit_model_on_data(model_type: str, task_type: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int=42, cpu_cores: Optional[int]=None) -> Any:
    model = init_model(model_type, task_type, random_state=random_state, cpu_cores=cpu_cores)
    model.fit(X_train, y_train)
    return model

def resolve_training_hyperparameters(model_type: str, task_type: str, X_train: pd.DataFrame, y_train: pd.Series, enable_hyperparameter_search: bool, n_folds: int, random_state: int, cpu_cores: Optional[int]=None, *, search_n_iter: Optional[int]=None, search_cv: Optional[int]=None) -> Tuple[Any, Dict[str, Any]]:
    default_model = init_model(model_type, task_type, random_state=random_state, cpu_cores=cpu_cores)
    if not enable_hyperparameter_search:
        return (default_model, default_model.get_params())
    if len(X_train) < 10:
        logger.debug('Training set too small for hold-out hyperparameter tuning; using default parameters')
        return (default_model, default_model.get_params())
    n_iter = _resolve_hp_search_n_iter(model_type) if search_n_iter is None else search_n_iter
    param_grid = get_param_grid(model_type, task_type, n_features=X_train.shape[1])
    position_idx = np.arange(len(X_train))
    stratify = y_train if task_type == 'classification' else None
    try:
        inner_train_idx, inner_val_idx = train_test_split(position_idx, test_size=0.2, random_state=random_state, stratify=stratify)
    except ValueError as e:
        logger.debug(f'Hold-out split for hyperparameter tuning failed ({e}); using default parameters')
        return (default_model, default_model.get_params())
    ordered_idx = np.concatenate([inner_train_idx, inner_val_idx])
    X_search = X_train.iloc[ordered_idx]
    y_search = y_train.iloc[ordered_idx]
    test_fold = np.full(len(X_search), -1, dtype=int)
    test_fold[len(inner_train_idx):] = 0
    cv_splitter = PredefinedSplit(test_fold)
    if task_type == 'classification':
        scoring = 'roc_auc' if len(y_train.unique()) == 2 else 'f1_weighted'
    else:
        scoring = 'neg_mean_squared_error'
    base_model = init_model(model_type, task_type, random_state=random_state, cpu_cores=cpu_cores)
    logger.debug(f'Hyperparameter tuning on training-set hold-out ({len(inner_train_idx)}/{len(inner_val_idx)} train/val); {n_iter} candidate(s), then {n_folds}-fold CV for reporting')
    search = RandomizedSearchCV(estimator=base_model, param_distributions=param_grid, n_iter=n_iter, cv=cv_splitter, scoring=scoring, n_jobs=DEFAULT_N_JOBS, random_state=random_state, verbose=0)
    search.fit(X_search, y_search)
    tuned_model = search.best_estimator_
    return (tuned_model, tuned_model.get_params())

TRAIN_TEST_HOLDOUT_RATIO = 0.20

def split_train_test_holdout(X: pd.DataFrame, y: pd.Series, task_type: str, test_ratio: float=TRAIN_TEST_HOLDOUT_RATIO, random_state: int=42) -> Dict[str, Any]:
    """Fallback hold-out when metadata lacks sample_split. Sorts IDs for cross-run reproducibility."""
    if len(X) != len(y):
        raise ValueError('X and y must have the same number of samples')
    if len(X) < 5:
        raise ValueError(f'Need at least 5 samples for train/test split, got {len(X)}')
    ordered_index = sorted(X.index.tolist(), key=lambda v: str(v))
    X_ordered = X.loc[ordered_index]
    y_ordered = y.loc[ordered_index]
    stratify = y_ordered if task_type == 'classification' else None
    try:
        train_idx, test_idx = train_test_split(X_ordered.index, test_size=test_ratio, random_state=random_state, stratify=stratify)
    except ValueError as e:
        if stratify is not None:
            logger.debug(f'Stratified train/test split failed ({e}); falling back to random split')
            train_idx, test_idx = train_test_split(X_ordered.index, test_size=test_ratio, random_state=random_state, stratify=None)
        else:
            raise
    train_idx_list = list(train_idx)
    test_idx_list = list(test_idx)
    train_ratio = 1.0 - test_ratio
    logger.debug(f'Train/test hold-out split: train={len(train_idx_list)}, test={len(test_idx_list)} ({train_ratio:.0%} train / {test_ratio:.0%} test)')
    return {'train_idx': train_idx_list, 'test_idx': test_idx_list, 'random_state': random_state, 'train_ratio': train_ratio, 'test_ratio': test_ratio, 'n_train': len(train_idx_list), 'n_test': len(test_idx_list), 'split_source': 'fallback_random_holdout'}

def holdout_from_sample_split(X: pd.DataFrame, y: pd.Series, sample_split: Dict[str, Any]) -> Dict[str, Any]:
    """Build train/test index lists from preprocess metadata sample_split."""
    if len(X) != len(y):
        raise ValueError('X and y must have the same number of samples')
    train_ids = {str(s) for s in sample_split.get('train_ids') or []}
    test_ids = {str(s) for s in sample_split.get('test_ids') or []}
    if not train_ids or not test_ids:
        raise ValueError('metadata sample_split must contain non-empty train_ids and test_ids')
    overlap = train_ids & test_ids
    if overlap:
        preview = ', '.join(sorted(overlap)[:10])
        raise ValueError(f'metadata sample_split train/test IDs overlap on {len(overlap)} sample(s): {preview}')
    train_idx = [idx for idx in X.index if str(idx) in train_ids]
    test_idx = [idx for idx in X.index if str(idx) in test_ids]
    missing_train = sorted(train_ids - {str(i) for i in train_idx})
    missing_test = sorted(test_ids - {str(i) for i in test_idx})
    if missing_train:
        preview = ', '.join(missing_train[:10])
        suffix = f' ... (+{len(missing_train) - 10} more)' if len(missing_train) > 10 else ''
        raise ValueError(f'{len(missing_train)} train ID(s) from preprocess metadata missing in matrix: {preview}{suffix}')
    if missing_test:
        preview = ', '.join(missing_test[:10])
        suffix = f' ... (+{len(missing_test) - 10} more)' if len(missing_test) > 10 else ''
        raise ValueError(f'{len(missing_test)} test ID(s) from preprocess metadata missing in matrix: {preview}{suffix}')
    if not train_idx or not test_idx:
        raise ValueError('preprocess sample_split matched empty train or test set in the training matrix')
    unassigned = len(X) - len(train_idx) - len(test_idx)
    if unassigned > 0:
        logger.warning(f'{unassigned} sample(s) in matrix are not in preprocess train/test split and will be excluded')
    logger.debug(f'Using preprocess sample_split (mode={sample_split.get("mode")}): train={len(train_idx)}, test={len(test_idx)}')
    return {'train_idx': train_idx, 'test_idx': test_idx, 'split_source': 'preprocess_metadata', 'mode': sample_split.get('mode'), 'random_state': sample_split.get('random_state'), 'train_ratio': sample_split.get('train_ratio'), 'test_ratio': sample_split.get('test_ratio'), 'n_train': len(train_idx), 'n_test': len(test_idx), 'n_unassigned': unassigned}

def resolve_holdout_split_from_metadata(X: pd.DataFrame, y: pd.Series, metadata: Optional[Dict[str, Any]], task_type: str, random_state: int=42) -> Dict[str, Any]:
    sample_split = (metadata or {}).get('sample_split') if metadata else None
    if sample_split and sample_split.get('train_ids') and sample_split.get('test_ids'):
        return holdout_from_sample_split(X, y, sample_split)
    logger.warning('Metadata missing sample_split; falling back to local reproducible 80/20 split. Re-run preprocess to fix the split at feature-selection time.')
    return split_train_test_holdout(X, y, task_type=task_type, test_ratio=TRAIN_TEST_HOLDOUT_RATIO, random_state=random_state)

def _predict_with_optional_proba(model: Any, X: pd.DataFrame, task_type: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    y_pred = model.predict(X)
    y_prob = None
    if task_type != 'classification':
        return (y_pred, None)
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X)
            from sklearn.utils.extmath import softmax
            if len(decision_scores.shape) == 1:
                prob_neg = 1 / (1 + np.exp(decision_scores))
                prob_pos = 1 - prob_neg
                y_prob = np.column_stack([prob_neg, prob_pos])
            else:
                y_prob = softmax(decision_scores)
    except Exception as e:
        logger.debug(f'  Failed to obtain prediction probabilities: {e}')
    return (y_pred, y_prob)

def _fit_model_with_params(model_type: str, task_type: str, X_train: pd.DataFrame, y_train: pd.Series, best_params: Dict[str, Any], random_state: int, cpu_cores: Optional[int]) -> Any:
    model = init_model(model_type, task_type, random_state=random_state, cpu_cores=cpu_cores)
    try:
        model.set_params(**best_params)
    except ValueError as e:
        logger.debug(f'  Failed to apply best_params to model: {e}')
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, task_type: str) -> Dict:
    metrics = {}
    if task_type == 'classification':
        try:
            y_true_array = _ensure_numpy_array(y_true)
            y_pred_array = _ensure_numpy_array(y_pred)
            if len(y_true_array) == 0 or len(y_pred_array) == 0:
                logger.debug('  True values or predictions are empty, cannot calculate metrics')
                metrics['accuracy'] = 'N/A'
                metrics['recall'] = 'N/A'
                metrics['f1'] = 'N/A'
            elif len(y_true_array) != len(y_pred_array):
                logger.debug(f'  True values and predictions length mismatch: {len(y_true_array)} vs {len(y_pred_array)}')
                metrics['accuracy'] = 'N/A'
                metrics['recall'] = 'N/A'
                metrics['f1'] = 'N/A'
            else:
                metrics['accuracy'] = round(accuracy_score(y_true_array, y_pred_array), 4)
                metrics['recall'] = round(recall_score(y_true_array, y_pred_array, average='weighted'), 4)
                metrics['f1'] = round(f1_score(y_true_array, y_pred_array, average='weighted'), 4)
        except Exception as e:
            logger.debug(f'  Failed to calculate classification metrics: {str(e)}')
            metrics['accuracy'] = 'N/A'
            metrics['recall'] = 'N/A'
            metrics['f1'] = 'N/A'
        try:
            if y_prob is not None and len(y_prob) > 0:
                from sklearn.metrics import roc_auc_score
                y_true_array = _ensure_numpy_array(y_true)
                y_prob_array = _ensure_numpy_array(y_prob)
                if len(y_prob_array.shape) == 1:
                    y_prob_array = y_prob_array.reshape(-1, 1)
                if len(y_true_array) != len(y_prob_array):
                    logger.debug(f'  True values and probabilities length mismatch: {len(y_true_array)} vs {len(y_prob_array)}')
                    metrics['auc'] = 'N/A'
                else:
                    unique_labels = np.unique(y_true_array)
                    n_unique = len(unique_labels)
                    if n_unique < 2:
                        metrics['auc'] = 'N/A'
                    elif n_unique == 2:
                        try:
                            if y_prob_array.shape[1] >= 2:
                                metrics['auc'] = round(roc_auc_score(y_true_array, y_prob_array[:, 1]), 4)
                            elif y_prob_array.shape[1] == 1:
                                metrics['auc'] = round(roc_auc_score(y_true_array, y_prob_array[:, 0]), 4)
                            else:
                                metrics['auc'] = 'N/A'
                        except Exception as e:
                            logger.debug(f'  Binary classification AUC calculation failed: {str(e)}')
                            metrics['auc'] = 'N/A'
                    else:
                        try:
                            from sklearn.preprocessing import label_binarize
                            n_classes = y_prob_array.shape[1]
                            y_true_binarized = label_binarize(y_true_array, classes=range(n_classes))
                            if y_true_binarized.shape[1] == 1:
                                metrics['auc'] = round(roc_auc_score(y_true_array, y_prob_array[:, 1] if y_prob_array.shape[1] > 1 else y_prob_array[:, 0]), 4)
                            else:
                                metrics['auc'] = round(roc_auc_score(y_true_binarized, y_prob_array, average='macro', multi_class='ovr'), 4)
                        except Exception as e:
                            logger.debug(f'  Multi-class AUC calculation failed: {str(e)}')
                            metrics['auc'] = 'N/A'
            else:
                metrics['auc'] = 'N/A'
                if task_type == 'classification':
                    pass
        except Exception as e:
            logger.debug(f'  Failed to calculate AUC: {str(e)}')
            metrics['auc'] = 'N/A'
    else:
        try:
            y_true_array = _ensure_numpy_array(y_true)
            y_pred_array = _ensure_numpy_array(y_pred)
            if len(y_true_array) == 0 or len(y_pred_array) == 0:
                logger.debug('  True values or predictions are empty, cannot calculate Pearson correlation')
                metrics['pearson_correlation'] = 'N/A'
                metrics['pearson_p_value'] = 'N/A'
            elif len(y_true_array) != len(y_pred_array):
                logger.debug(f'  True values and predictions length mismatch: {len(y_true_array)} vs {len(y_pred_array)}')
                metrics['pearson_correlation'] = 'N/A'
                metrics['pearson_p_value'] = 'N/A'
            else:
                from scipy.stats import pearsonr
                r, p = pearsonr(y_true_array, y_pred_array)
                if np.isnan(r):
                    metrics['pearson_correlation'] = 'N/A'
                    metrics['pearson_p_value'] = 'N/A'
                else:
                    metrics['pearson_correlation'] = round(float(r), 4)
                    metrics['pearson_p_value'] = 'N/A' if np.isnan(p) else float(p)
                try:
                    metrics['r2'] = round(float(r2_score(y_true_array, y_pred_array)), 4)
                    metrics['rmse'] = round(float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))), 4)
                    metrics['mae'] = round(float(mean_absolute_error(y_true_array, y_pred_array)), 4)
                except Exception as e:
                    logger.debug(f'  Failed to calculate regression error metrics: {str(e)}')
                    metrics.setdefault('r2', 'N/A')
                    metrics.setdefault('rmse', 'N/A')
                    metrics.setdefault('mae', 'N/A')
        except Exception as e:
            logger.debug(f'  Failed to calculate Pearson correlation coefficient: {str(e)}')
            metrics['pearson_correlation'] = 'N/A'
            metrics['pearson_p_value'] = 'N/A'
    return metrics

def _aggregate_cv_fold_metrics(cv_fold_metrics: List[Dict], task_type: str) -> Dict[str, float]:
    avg_metrics: Dict[str, float] = {}
    if task_type == 'regression':
        pearson_corrs = [m.get('pearson_correlation', 0) for m in cv_fold_metrics if isinstance(m.get('pearson_correlation'), (int, float))]
        if pearson_corrs:
            avg_metrics['pearson_correlation'] = float(round(np.mean(pearson_corrs), 4))
            avg_metrics['pearson_correlation_std'] = float(round(np.std(pearson_corrs), 4))
    else:
        accuracies = [m.get('accuracy', 0) for m in cv_fold_metrics if isinstance(m.get('accuracy'), (int, float))]
        recalls = [m.get('recall', 0) for m in cv_fold_metrics if isinstance(m.get('recall'), (int, float))]
        f1_scores = [m.get('f1', 0) for m in cv_fold_metrics if isinstance(m.get('f1'), (int, float))]
        aucs = [m.get('auc', 0) for m in cv_fold_metrics if isinstance(m.get('auc'), (int, float)) and m.get('auc') != 'N/A']
        if accuracies:
            avg_metrics['accuracy'] = float(round(np.mean(accuracies), 4))
            avg_metrics['accuracy_std'] = float(round(np.std(accuracies), 4))
        if recalls:
            avg_metrics['recall'] = float(round(np.mean(recalls), 4))
            avg_metrics['recall_std'] = float(round(np.std(recalls), 4))
        if f1_scores:
            avg_metrics['f1'] = float(round(np.mean(f1_scores), 4))
            avg_metrics['f1_std'] = float(round(np.std(f1_scores), 4))
        if aucs:
            avg_metrics['auc'] = float(round(np.mean(aucs), 4))
            avg_metrics['auc_std'] = float(round(np.std(aucs), 4))
    return avg_metrics

def _resolve_cv_fold_parallelism(n_folds: int, cpu_cores: Optional[int]) -> Tuple[int, int]:
    """Split --threads budget across CV folds: (fold_workers, cores_per_fold)."""
    total = DEFAULT_N_JOBS if cpu_cores is None else max(1, int(cpu_cores))
    fold_workers = min(max(1, int(n_folds)), total)
    cores_per_fold = max(1, total // fold_workers)
    return (fold_workers, cores_per_fold)

def _set_native_thread_env(num_threads: int) -> Dict[str, Optional[str]]:
    """Set BLAS/OpenMP thread caps; return previous values for restore."""
    prev: Dict[str, Optional[str]] = {}
    value = str(max(1, int(num_threads)))
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS']:
        prev[var] = os.environ.get(var)
        os.environ[var] = value
    return prev

def _restore_native_thread_env(prev: Dict[str, Optional[str]]) -> None:
    for var, old in prev.items():
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old

def _run_one_cv_fold(fold_idx: int, n_folds: int, train_idx: np.ndarray, val_idx: np.ndarray, X_train_cv: pd.DataFrame, y_train_cv: pd.Series, model_type: str, task_type: str, best_params: Dict[str, Any], random_state: int, cores_per_fold: int) -> Tuple[int, Dict, Optional[pd.Series], Optional[np.ndarray], Optional[np.ndarray]]:
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    X_train_fold = X_train_cv.iloc[train_idx]
    X_val_fold = X_train_cv.iloc[val_idx]
    y_train_fold = y_train_cv.iloc[train_idx]
    y_val_fold = y_train_cv.iloc[val_idx]
    model_fold = init_model(model_type, task_type, random_state=random_state, cpu_cores=cores_per_fold)
    try:
        model_fold.set_params(**best_params)
    except ValueError as e:
        logger.debug(f'[Fold {fold_idx}] Failed to apply best_params to model: {e}')
    y_pred_fold = None
    y_prob_fold = None
    try:
        model_fold.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fold.predict(X_val_fold)
        if task_type == 'classification':
            try:
                if hasattr(model_fold, 'predict_proba'):
                    y_prob_fold = model_fold.predict_proba(X_val_fold)
                elif hasattr(model_fold, 'decision_function'):
                    decision_scores = model_fold.decision_function(X_val_fold)
                    from sklearn.utils.extmath import softmax
                    if len(decision_scores.shape) == 1:
                        prob_neg = 1 / (1 + np.exp(decision_scores))
                        prob_pos = 1 - prob_neg
                        y_prob_fold = np.column_stack([prob_neg, prob_pos])
                    else:
                        y_prob_fold = softmax(decision_scores)
            except Exception as e:
                logger.debug(f'[Fold {fold_idx}] Failed to obtain prediction probabilities: {str(e)}')
    except Exception as e:
        logger.error(f'[Fold {fold_idx}] Error during training/evaluation: {str(e)}', exc_info=True)
        return (fold_idx, {}, None, None, None)
    fold_metrics = evaluate_model(y_val_fold, y_pred_fold, y_prob_fold, task_type)
    if task_type == 'regression':
        corr = fold_metrics.get('pearson_correlation', 'N/A')
        pval = fold_metrics.get('pearson_p_value', 'N/A')
        logger.debug(f'[{fold_idx}/{n_folds}] Pearson correlation: {corr}, p-value: {pval}')
    else:
        acc = fold_metrics.get('accuracy', 'N/A')
        auc = fold_metrics.get('auc', 'N/A')
        logger.debug(f'[{fold_idx}/{n_folds}] CV on train — Accuracy: {acc}, AUC: {auc}')
    return (fold_idx, fold_metrics, y_val_fold, y_pred_fold, y_prob_fold)

def _run_kfold_cv_evaluation(model_type: str, task_type: str, X_train_cv: pd.DataFrame, y_train_cv: pd.Series, best_params: Dict[str, Any], n_folds: int, random_state: int, cpu_cores: Optional[int]) -> Tuple[List[Dict], Optional[Dict[str, Any]], Dict[str, float]]:
    if task_type == 'classification':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_train_cv, y_train_cv))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_train_cv, y_train_cv))
    fold_workers, cores_per_fold = _resolve_cv_fold_parallelism(n_folds, cpu_cores)
    mode = 'parallel' if fold_workers > 1 else 'serial'
    logger.debug(f'Starting {n_folds}-fold cross-validation on training set with fixed hyperparameters ({mode}: fold_workers={fold_workers}, cores/fold={cores_per_fold})')
    fold_args = [(fold_idx, train_idx, val_idx) for fold_idx, (train_idx, val_idx) in enumerate(splits, 1)]
    fold_results: List[Tuple[int, Dict, Optional[pd.Series], Optional[np.ndarray], Optional[np.ndarray]]] = []
    # Cap native libs to per-fold budget so parallel folds do not oversubscribe CPUs.
    prev_thread_env = _set_native_thread_env(cores_per_fold)
    try:
        if fold_workers == 1:
            for fold_idx, train_idx, val_idx in fold_args:
                fold_results.append(_run_one_cv_fold(fold_idx, n_folds, train_idx, val_idx, X_train_cv, y_train_cv, model_type, task_type, best_params, random_state, cores_per_fold))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=fold_workers) as executor:
                futures = {executor.submit(_run_one_cv_fold, fold_idx, n_folds, train_idx, val_idx, X_train_cv, y_train_cv, model_type, task_type, best_params, random_state, cores_per_fold): fold_idx for fold_idx, train_idx, val_idx in fold_args}
                for future in as_completed(futures):
                    fold_idx = futures[future]
                    try:
                        fold_results.append(future.result())
                    except Exception as e:
                        logger.error(f'[Fold {fold_idx}] Worker raised: {e}', exc_info=True)
                        fold_results.append((fold_idx, {}, None, None, None))
    finally:
        _restore_native_thread_env(prev_thread_env)
    fold_results.sort(key=lambda r: r[0])
    cv_fold_metrics: List[Dict] = []
    all_cv_y_true: List[pd.Series] = []
    all_cv_y_pred: List[np.ndarray] = []
    all_cv_y_prob_frames: List[pd.DataFrame] = []
    for _fold_idx, fold_metrics, y_val_fold, y_pred_fold, y_prob_fold in fold_results:
        cv_fold_metrics.append(fold_metrics)
        if y_pred_fold is not None and y_val_fold is not None:
            all_cv_y_true.append(y_val_fold)
            all_cv_y_pred.append(y_pred_fold)
            if y_prob_fold is not None:
                all_cv_y_prob_frames.append(pd.DataFrame(y_prob_fold, index=y_val_fold.index))
    avg_metrics = _aggregate_cv_fold_metrics(cv_fold_metrics, task_type)
    cv_oof_data: Optional[Dict[str, Any]] = None
    if all_cv_y_true and all_cv_y_pred:
        try:
            oof_y_true = pd.concat(all_cv_y_true, axis=0, ignore_index=False)
            if oof_y_true.index.duplicated().any():
                oof_y_true = pd.concat(all_cv_y_true, axis=0, keys=range(len(all_cv_y_true)), names=['fold', None]).droplevel(0)
            oof_y_pred = np.concatenate(all_cv_y_pred)
            oof_y_prob = None
            if all_cv_y_prob_frames:
                try:
                    oof_y_prob = _merge_oof_probabilities(all_cv_y_prob_frames, oof_y_true.index)
                except Exception as e:
                    logger.debug(f'  Failed to align CV OOF probabilities by sample ID: {e}')
            cv_oof_data = {'y_true': oof_y_true, 'y_pred': oof_y_pred, 'y_prob': oof_y_prob}
        except Exception as e:
            logger.debug(f'  Failed to assemble training-set CV OOF predictions: {e}')
    return (cv_fold_metrics, cv_oof_data, avg_metrics)

def _cv_results_for_json(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cv_results)

def _save_shap_dependence_tsv(output_prefix: Path, shap_dependence_data: Dict[str, Any], *, max_features: int=SHAP_DEPENDENCE_MAX_FEATURES) -> None:
    names = np.asarray(shap_dependence_data['feature_names'])
    shap_arr = np.asarray(shap_dependence_data['shap_values'])
    x_arr = np.asarray(shap_dependence_data['X_values'])
    keep_idx = [i for i, n in enumerate(names) if 'phenotype' not in str(n).lower()]
    if not keep_idx:
        return
    n_features = len(keep_idx)
    feature_cap = n_features
    if n_features > HIGH_DIM_SHAP_DEPENDENCE_THRESHOLD and max_features > 0:
        feature_cap = min(max_features, n_features)
        shap_kept = shap_arr[:, keep_idx]
        mean_abs = np.abs(shap_kept).mean(axis=0)
        top_local = np.argsort(mean_abs)[::-1][:feature_cap]
        keep_idx = [keep_idx[int(i)] for i in top_local]
        logger.debug(f'  SHAP dependence TSV capped to top {feature_cap} features (from {n_features:,}) to limit disk use')
    names_kept = [str(names[i]) for i in keep_idx]
    shap_kept = shap_arr[:, keep_idx]
    x_kept = x_arr[:, keep_idx]
    n_samples, n_feats = shap_kept.shape
    sample_idx = np.repeat(np.arange(n_samples), n_feats)
    feat_idx = np.tile(np.arange(n_feats), n_samples)
    dep_df = pd.DataFrame({'sample_index': sample_idx, 'feature': np.asarray(names_kept, dtype=object)[feat_idx], 'feature_value': x_kept.ravel(), 'shap_value': shap_kept.ravel()})
    dep_file = Path(f'{output_prefix}_shap_dependence.tsv')
    dep_df.to_csv(dep_file, sep='\t', index=False)
    logger.debug(f'  SHAP dependence data saved: {dep_file} ({n_samples:,} samples x {n_feats:,} features)')

def _save_test_predictions_tsv(path: Path, y_test: pd.Series, y_pred: np.ndarray, y_prob: Optional[np.ndarray], task_type: str) -> None:
    if isinstance(y_test, pd.Series):
        sample_ids = [str(x) for x in y_test.index]
        y_true = y_test.values
    else:
        y_true = np.asarray(y_test)
        sample_ids = [str(i) for i in range(len(y_true))]
    pred_df = pd.DataFrame({'sample_id': sample_ids, 'y_true': y_true, 'y_pred': np.asarray(y_pred).reshape(-1)})
    if y_prob is not None and task_type == 'classification':
        prob_arr = np.asarray(y_prob)
        if prob_arr.ndim == 1:
            pred_df['y_prob'] = prob_arr
        else:
            for class_idx in range(prob_arr.shape[1]):
                pred_df[f'y_prob_class_{class_idx}'] = prob_arr[:, class_idx]
    pred_df.to_csv(path, sep='\t', index=False)

def _save_cv_oof_tsv(path: Path, cv_oof_data: Dict[str, Any]) -> None:
    oof_y_true = cv_oof_data['y_true']
    if isinstance(oof_y_true, pd.Series):
        sample_ids = [str(x) for x in oof_y_true.index]
        y_true = oof_y_true.values
    else:
        y_true = np.asarray(oof_y_true)
        sample_ids = [str(i) for i in range(len(y_true))]
    oof_df = pd.DataFrame({'sample_id': sample_ids, 'y_true': y_true, 'y_pred': np.asarray(cv_oof_data['y_pred']).reshape(-1)})
    y_prob = cv_oof_data.get('y_prob')
    if y_prob is not None:
        prob_arr = np.asarray(y_prob)
        if prob_arr.ndim == 1:
            oof_df['y_prob'] = prob_arr
        elif prob_arr.ndim == 2:
            for class_idx in range(prob_arr.shape[1]):
                oof_df[f'y_prob_class_{class_idx}'] = prob_arr[:, class_idx]
    oof_df.to_csv(path, sep='\t', index=False)

def _shap_values_sample_by_feature(shap_values: Any, task_type: str) -> np.ndarray:
    if task_type == 'regression':
        arr = np.asarray(shap_values)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return np.mean(arr, axis=-1)
        raise ValueError(f'Unexpected regression SHAP shape: {arr.shape}')
    if isinstance(shap_values, list):
        class_arrays = [np.asarray(sv) for sv in shap_values]
        if len(class_arrays) == 2:
            return class_arrays[1]
        if len(class_arrays) == 1:
            return class_arrays[0]
        return np.mean(class_arrays, axis=0)
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        return np.mean(arr, axis=-1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f'Unexpected classification SHAP shape: {arr.shape}')

def calculate_shap_values(model: Any, model_type: str, feature_names: List[str], X_train: pd.DataFrame, y_train: pd.Series, task_type: str, shap_sample_size: Optional[int]=None) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    if not SHAP_AVAILABLE:
        logger.warning('  SHAP library not installed, cannot calculate SHAP values. Please run: pip install shap')
        return (pd.DataFrame(), None)
    try:
        shap_t0 = time.perf_counter()
        X_shap = X_train
        y_shap = y_train
        X_input = X_shap[feature_names] if feature_names else X_shap
        if shap_sample_size is not None and len(X_input) > shap_sample_size:
            X_input = X_input.sample(n=int(shap_sample_size), random_state=42)
            y_shap = y_shap.loc[X_input.index]
        if model_type in ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
        elif model_type == 'Logistic':
            explainer = shap.LinearExplainer(model, X_input)
            shap_values = explainer.shap_values(X_input)
        elif model_type == 'SVM':
            from sklearn.svm import LinearSVR, LinearSVC
            X_sur = X_input
            if len(X_sur) > 5000:
                X_sur = X_sur.sample(n=5000, random_state=42)
            y_sur = y_shap.loc[X_sur.index]
            logger.debug(f'  [SHAP][SVM] Fixed mode: linear surrogate + LinearExplainer (surrogate_samples={len(X_sur):,}, features={X_sur.shape[1]:,}, task={task_type})')
            if task_type == 'classification':
                surrogate = LinearSVC(C=1.0, dual=True, max_iter=5000)
            else:
                surrogate = LinearSVR(C=1.0, dual=True, max_iter=5000)
            surrogate.fit(X_sur, y_sur)
            explainer = shap.LinearExplainer(surrogate, X_sur, feature_perturbation='interventional')
            shap_values = explainer.shap_values(X_input)
        else:
            logger.debug(f'  {model_type} model uses KernelExplainer for SHAP values, may be slow')
            background_size = min(100, len(X_shap))
            background = X_shap[feature_names].sample(n=background_size, random_state=42) if feature_names else X_shap.sample(n=background_size, random_state=42)
            explainer = shap.KernelExplainer(model.predict if task_type == 'regression' else lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x), background)
            shap_values = explainer.shap_values(X_input)
        raw_shap = shap_values
        shap_values = _shap_values_sample_by_feature(shap_values, task_type)
        shap_values = np.asarray(shap_values)
        if task_type == 'classification':
            is_binary = isinstance(raw_shap, list) and len(raw_shap) == 2
            if not is_binary and hasattr(raw_shap, 'shape') and (len(np.asarray(raw_shap).shape) == 3):
                is_binary = np.asarray(raw_shap).shape[-1] == 2
            if is_binary:
                logger.debug('  Using positive-class SHAP values (class index 1)')
        mean_shap_abs = np.abs(shap_values).mean(axis=0)
        mean_shap_signed = shap_values.mean(axis=0)
        sign_effect = np.sign(mean_shap_signed)
        sign_effect = np.where(sign_effect == 0, 1, sign_effect)
        feature_list = list(X_input.columns)
        n_features_from_shap = len(mean_shap_abs)
        n_features_from_names = len(feature_list)
        n_features_from_sign = len(sign_effect)
        min_len = min(n_features_from_shap, n_features_from_names, n_features_from_sign)
        if not n_features_from_shap == n_features_from_names == n_features_from_sign:
            logger.debug(f' SHAP feature-length mismatch; aligning by minimum length: shap={n_features_from_shap}, names={n_features_from_names}, sign={n_features_from_sign}')
        feature_list = feature_list[:min_len]
        mean_shap_abs = mean_shap_abs[:min_len]
        sign_effect = sign_effect[:min_len]
        shap_values = shap_values[:, :min_len]
        shap_df = pd.DataFrame({'feature': feature_list, 'shap_abs': mean_shap_abs, 'effect': sign_effect.astype(int)})
        shap_df = shap_df.sort_values('shap_abs', ascending=False)
        shap_df = shap_df.reset_index(drop=True)
        dependence_bundle = {'shap_values': shap_values.astype(np.float32), 'X_values': np.asarray(X_input.iloc[:, :min_len].values, dtype=np.float32), 'feature_names': np.array(feature_list, dtype=object)}
        logger.debug(f'  SHAP values calculation completed (samples_used={len(X_input):,}, elapsed={time.perf_counter() - shap_t0:.2f}s)')
        return (shap_df, dependence_bundle)
    except Exception as e:
        logger.error(f'  SHAP values calculation failed: {str(e)}')
        import traceback
        return (pd.DataFrame(), None)

def _ensure_numpy_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.values
    else:
        return np.array(data)

def filter_phenotype_columns(columns: List[str]) -> List[str]:
    return [col for col in columns if 'phenotype' not in str(col).lower()]

def filter_phenotype_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    phenotype_cols = [col for col in df.columns if 'phenotype' in str(col).lower()]
    import re
    index_artifact_cols = [col for col in df.columns if re.match('^index(\\.\\d+)?$', str(col), flags=re.IGNORECASE)]
    drop_cols = []
    if phenotype_cols:
        drop_cols.extend(phenotype_cols)
    if index_artifact_cols:
        drop_cols.extend(index_artifact_cols)
    if drop_cols:
        seen = set()
        drop_cols_unique = []
        for c in drop_cols:
            if c not in seen:
                seen.add(c)
                drop_cols_unique.append(c)
        if index_artifact_cols:
            cols_suffix = '...' if len(index_artifact_cols) > 5 else ''
            logger.debug(f'Detected and removed index artifact columns from features: {index_artifact_cols[:5]}{cols_suffix}')
        return df.drop(columns=drop_cols_unique, errors='ignore')
    return df

def save_training_results(model: Any, metrics: Dict, selected_snps: List[str], output_dir: str, model_type: str, task_type: str, shap_df: Optional[pd.DataFrame]=None, shap_dependence_data: Optional[Dict[str, Any]]=None, y_test: Optional[pd.Series]=None, y_pred: Optional[np.ndarray]=None, y_prob: Optional[np.ndarray]=None, cv_results: Optional[Dict]=None, cv_oof_data: Optional[Dict[str, Any]]=None, peak_ram_mb: Optional[float]=None, hyperparameter_search_enabled: Optional[bool]=None, split_mode: str='cv', evaluation_results: Optional[Dict]=None, shap_dependence_max_features: Optional[int]=None) -> None:
    model_dir = Path(output_dir) / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = model_dir / model_type
    model_file = model_dir / f'{model_type}_model.pkl'
    import joblib
    joblib.dump(model, model_file)
    metrics_file = Path(f'{output_prefix}_metrics.json')
    metrics_payload = {'model_type': model_type, 'task_type': task_type, 'split_mode': split_mode, 'metrics': metrics, 'metrics_scope': 'held_out_test_set', 'training_time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if evaluation_results is not None:
        metrics_payload['train_metrics'] = evaluation_results.get('train_metrics')
        metrics_payload['val_metrics'] = evaluation_results.get('val_metrics')
        metrics_payload['test_metrics'] = evaluation_results.get('test_metrics')
        metrics_payload['split_info'] = evaluation_results.get('split_info')
    if cv_results is not None and split_mode == 'cv':
        metrics_payload['cv_metrics'] = cv_results.get('average_metrics')
        metrics_payload['cv_metrics_scope'] = cv_results.get('cv_scope', 'training_set')
        metrics_payload['test_metrics'] = cv_results.get('test_metrics')
        metrics_payload['split_info'] = cv_results.get('split_info')
        metrics_payload['cv_results_file'] = f'{model_type}_cv_results.json'
        if cv_oof_data is not None:
            metrics_payload['cv_oof_file'] = f'{model_type}_cv_oof.tsv'
    if peak_ram_mb is not None:
        metrics_payload['peak_ram_mb'] = peak_ram_mb
    if hyperparameter_search_enabled is not None:
        metrics_payload['hyperparameter_search'] = hyperparameter_search_enabled
    with open(metrics_file, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    if cv_results is not None and split_mode == 'cv':
        cv_results_file = model_dir / f'{model_type}_cv_results.json'
        try:
            with open(cv_results_file, 'w', encoding='utf-8') as f:
                json.dump(_cv_results_for_json(cv_results), f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f'  Training-set CV results saved: {cv_results_file}')
        except Exception as e:
            logger.debug(f'  Failed to save CV results JSON (ignored): {e}')
        if cv_oof_data is not None and cv_oof_data.get('y_true') is not None and cv_oof_data.get('y_pred') is not None:
            try:
                cv_oof_file = model_dir / f'{model_type}_cv_oof.tsv'
                _save_cv_oof_tsv(cv_oof_file, cv_oof_data)
                logger.debug(f'  Training-set CV OOF predictions saved: {cv_oof_file}')
            except Exception as e:
                logger.debug(f'  Failed to save CV OOF predictions (ignored): {e}')
    try:
        features_file = Path(f'{output_prefix}_training_features.json')
        feature_name_list = list(selected_snps) if selected_snps is not None else []
        with open(features_file, 'w') as f:
            json.dump({'model_type': model_type, 'task_type': task_type, 'feature_names': feature_name_list, 'feature_chr_pos_keys': build_feature_chr_pos_keys(feature_name_list), 'snp_naming_rule': 'chromosome_physicalPosition (chr_pos, e.g. 1_123456)', 'saved_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    except Exception as e:
        logger.debug(f'   Failed to save training feature names (ignored): {e}')
    if shap_df is not None and (not shap_df.empty):
        filtered_shap_df = shap_df[shap_df['feature'].astype(str).apply(lambda x: 'phenotype' not in x.lower())]
        shap_file = Path(f'{output_prefix}_shap_values.txt')
        shap_output_df = filtered_shap_df.copy()
        shap_output_df = shap_output_df.rename(columns={'shap_abs': 'importance_abs'})
        shap_output_df.to_csv(shap_file, sep='\t', index=False)
    shap_dependence_cap = _resolve_shap_dependence_max_features(shap_dependence_max_features)
    if shap_dependence_data is not None:
        try:
            _save_shap_dependence_tsv(output_prefix, shap_dependence_data, max_features=shap_dependence_cap)
        except Exception as e:
            logger.debug(f'   Failed to save SHAP dependence data (ignored): {e}')
    if y_test is not None and y_pred is not None:
        test_pred_file = Path(f'{output_prefix}_test_predictions.tsv')
        plotting_meta_file = Path(f'{output_prefix}_plotting_data.json')
        try:
            _save_test_predictions_tsv(test_pred_file, y_test, y_pred, y_prob, task_type)
        except Exception as e:
            logger.debug(f'  Failed to save test predictions TSV (ignored): {e}')
            test_pred_file = None
        plotting_payload: Dict[str, Any] = {'model_type': model_type, 'task_type': task_type, 'split_mode': split_mode, 'publication_quality': PUB_QUALITY_MODE, 'shap_dependence_top': shap_dependence_cap}
        if shap_dependence_data is not None:
            plotting_payload['shap_dependence_n_samples'] = int(len(np.asarray(shap_dependence_data['shap_values'])))
        if test_pred_file is not None:
            plotting_payload['test_predictions_file'] = test_pred_file.name
        if split_mode == 'cv' and cv_results is not None:
            plotting_payload['cv_results_file'] = f'{model_type}_cv_results.json'
        try:
            with open(plotting_meta_file, 'w', encoding='utf-8') as f:
                json.dump(plotting_payload, f, indent=2, ensure_ascii=False)
            logger.debug(f'  Plotting metadata saved: {plotting_meta_file}')
        except Exception as e:
            logger.debug(f'  Failed to save plotting metadata JSON (ignored): {e}')
    logger.debug(f'  Results saved successfully: {model_dir}')

def run_single_model(input_path: str, model_type: str, output_dir: str, task_type: Optional[str]=None, n_folds: int=5, random_state: int=42, preloaded_data: Optional[Dict[str, Any]]=None, cpu_cores: Optional[int]=None, enable_hyperparameter_search: bool=False, shap_dependence_max_features: Optional[int]=None) -> int:
    return _run_single_model_body(input_path=input_path, model_type=model_type, output_dir=output_dir, task_type=task_type, n_folds=n_folds, random_state=random_state, preloaded_data=preloaded_data, cpu_cores=cpu_cores, enable_hyperparameter_search=enable_hyperparameter_search, shap_dependence_max_features=shap_dependence_max_features)

def _run_single_model_body(input_path: str, model_type: str, output_dir: str, task_type: Optional[str]=None, n_folds: int=5, random_state: int=42, preloaded_data: Optional[Dict[str, Any]]=None, cpu_cores: Optional[int]=None, enable_hyperparameter_search: bool=False, shap_dependence_max_features: Optional[int]=None) -> int:
    if not str(input_path).endswith('_metadata.json'):
        raise ValueError(f'Training input must be the preprocess-generated metadata file (*_metadata.json).Current input: {input_path}')
    cpu_cores_value = DEFAULT_N_JOBS if cpu_cores is None else max(1, int(cpu_cores))
    if shap_dependence_max_features is None and preloaded_data is not None:
        shap_dependence_max_features = preloaded_data.get('shap_dependence_max_features')
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS']:
        os.environ[var] = str(cpu_cores_value)
    start_time = time.perf_counter()
    memory_tracker = PeakMemoryTracker()
    memory_tracker.start()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    base_output_dir = output_dir_path.parent
    tmp_root = base_output_dir / 'tmp' / 'train' / model_type
    tmp_root.mkdir(parents=True, exist_ok=True)

    def register_dir(p: Path) -> Path:
        return _temp_file_manager.register_dir(p)

    def register_file(p: Path) -> Path:
        return _temp_file_manager.register_file(p)
    run_tmp_dir = register_dir(tmp_root / f'pid_{os.getpid()}')
    preprocess_tmp_dir = None
    try:
        if preloaded_data is None:
            input_info_guard = parse_train_input_path(input_path)
            check_training_metadata_guardrails(input_info_guard.get('metadata'), model_type=model_type, task_type=task_type, enable_hyperparameter_search=enable_hyperparameter_search, n_folds=n_folds, threads=cpu_cores)
        if preloaded_data is not None:
            if preloaded_data.get('_shared_matrix'):
                X, y = _load_training_data_from_shared(preloaded_data)
            elif preloaded_data.get('_load_from_file'):
                train_file = preloaded_data['train_file']
                valid_samples = preloaded_data.get('valid_samples')
                train_file_format = (preloaded_data.get('metadata') or {}).get('train_file_format')
                X, y, snp_name_mapping = load_training_data(train_file, valid_samples, file_format=train_file_format)
            else:
                X = preloaded_data['X']
                y = preloaded_data['y']
            if preloaded_data.get('preprocess_tmp_dir'):
                preprocess_tmp_dir = Path(preloaded_data['preprocess_tmp_dir']).absolute()
                _temp_file_manager.register_preprocess_tmp_dir(preprocess_tmp_dir)

            def infer_task_type_from_y(series: pd.Series) -> str:
                values = series.values
                unique_vals = np.unique(values)
                is_all_int = np.all(values == np.round(values))
                if len(unique_vals) <= 10 and is_all_int:
                    return 'classification'
                return 'regression'
            if task_type is None:
                task_type = preloaded_data.get('task_type') or infer_task_type_from_y(y)
                logger.debug(f'Task type: {task_type}')
            validate_model_for_task(model_type, task_type)
        else:
            input_info = parse_train_input_path(input_path)
            train_file = input_info['train_file']
            valid_samples = input_info['metadata']['valid_samples'] if input_info['metadata'] else None
            if input_info['preprocess_tmp_dir']:
                preprocess_tmp_dir = Path(input_info['preprocess_tmp_dir']).absolute()
                _temp_file_manager.register_preprocess_tmp_dir(preprocess_tmp_dir)
            train_file_format = input_info['metadata'].get('train_file_format') if input_info.get('metadata') else None
            X, y, snp_name_mapping = load_training_data(train_file, valid_samples, file_format=train_file_format)

            def infer_task_type_from_y(series: pd.Series) -> str:
                values = series.values
                unique_vals = np.unique(values)
                is_all_int = np.all(values == np.round(values))
                if len(unique_vals) <= 10 and is_all_int:
                    return 'classification'
                return 'regression'
            if task_type is None:
                task_type = infer_task_type_from_y(y)
                logger.debug(f'Task type: {task_type}')
            validate_model_for_task(model_type, task_type)
        X_filtered = X
        metadata_for_split = None
        if preloaded_data:
            metadata_for_split = preloaded_data.get('metadata')
            holdout_split = preloaded_data.get('holdout_split')
        else:
            holdout_split = None
            input_info_for_split = parse_train_input_path(input_path)
            metadata_for_split = input_info_for_split.get('metadata')
        if holdout_split is None:
            holdout_split = resolve_holdout_split_from_metadata(X_filtered, y, metadata_for_split, task_type=task_type, random_state=random_state)
        split_mode = 'cv'
        evaluation_results: Optional[Dict[str, Any]] = None
        cv_results: Optional[Dict[str, Any]] = None
        cv_oof_data: Optional[Dict[str, Any]] = None
        train_idx_holdout = holdout_split['train_idx']
        test_idx_holdout = holdout_split['test_idx']
        X_train_cv = X_filtered.loc[train_idx_holdout]
        y_train_cv = y.loc[train_idx_holdout]
        X_test_holdout = X_filtered.loc[test_idx_holdout]
        y_test_holdout = y.loc[test_idx_holdout]
        X_model_train = X_train_cv
        y_model_train = y_train_cv
        logger.info(f'Dataset: {X.shape[0]:,} samples × {X.shape[1]:,} features ({task_type}); train={len(train_idx_holdout)}, test={len(test_idx_holdout)}')
        if enable_hyperparameter_search:
            logger.debug('Starting hyperparameter search on training set (hold-out tuning)...')
        _, best_params = resolve_training_hyperparameters(model_type=model_type, task_type=task_type, X_train=X_train_cv, y_train=y_train_cv, enable_hyperparameter_search=enable_hyperparameter_search, n_folds=n_folds, random_state=random_state, cpu_cores=cpu_cores)
        if enable_hyperparameter_search:
            logger.debug('Hyperparameter search completed, starting cross-validation on training set with tuned parameters')
        cv_fold_metrics, cv_oof_data, avg_metrics = _run_kfold_cv_evaluation(model_type=model_type, task_type=task_type, X_train_cv=X_train_cv, y_train_cv=y_train_cv, best_params=best_params, n_folds=n_folds, random_state=random_state, cpu_cores=cpu_cores)
        logger.info(f'{n_folds}-fold CV average: {avg_metrics}')
        final_model = _fit_model_with_params(model_type=model_type, task_type=task_type, X_train=X_train_cv, y_train=y_train_cv, best_params=best_params, random_state=random_state, cpu_cores=cpu_cores)
        y_pred_test, y_prob_test = _predict_with_optional_proba(final_model, X_test_holdout, task_type)
        test_metrics = evaluate_model(y_test_holdout, y_pred_test, y_prob_test, task_type)
        logger.info(f'Held-out test metrics: {test_metrics}')
        y_test_combined = y_test_holdout
        y_pred_combined = y_pred_test
        y_prob_combined = y_prob_test
        metrics = test_metrics
        cv_results = {'split_mode': 'train_test_cv', 'cv_scope': 'training_set', 'final_evaluation_scope': 'held_out_test_set', 'n_folds': n_folds, 'fold_metrics': cv_fold_metrics, 'average_metrics': avg_metrics, 'test_metrics': test_metrics, 'split_info': holdout_split, 'oof_summary': {'n_samples': len(cv_oof_data['y_true']) if cv_oof_data else 0}}
        feature_cols = filter_phenotype_columns(X_filtered.columns.tolist())
        shap_df, shap_dependence_data = calculate_shap_values(model=final_model, model_type=model_type, feature_names=feature_cols, X_train=X_model_train[feature_cols] if feature_cols else X_model_train, y_train=y_model_train, task_type=task_type, shap_sample_size=None)
        if shap_df.empty:
            logger.warning('SHAP values calculation failed or returned empty results')
            shap_dependence_data = None
        model_dir = Path(output_dir) / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        selected_snps = filter_phenotype_columns(X_filtered.columns.tolist())
        peak_ram_mb = memory_tracker.stop()
        save_training_results(model=final_model, metrics=metrics, selected_snps=selected_snps, output_dir=output_dir, model_type=model_type, task_type=task_type, shap_df=shap_df, shap_dependence_data=shap_dependence_data, y_test=y_test_combined, y_pred=y_pred_combined, y_prob=y_prob_combined, cv_results=cv_results, cv_oof_data=cv_oof_data, peak_ram_mb=peak_ram_mb, hyperparameter_search_enabled=enable_hyperparameter_search, split_mode=split_mode, evaluation_results=evaluation_results, shap_dependence_max_features=shap_dependence_max_features)
        total_time = round(time.perf_counter() - start_time, 2)
        logger.info(f'Training completed for {model_type} (elapsed={total_time}s, peak RAM={peak_ram_mb} MB)')
        if CLEANUP_TEMP_FILES:
            try:
                _temp_file_manager.cleanup_on_exit(cleanup_preprocess=False)
                try:
                    if tmp_root.exists() and (not any(tmp_root.iterdir())):
                        should_delete = True
                        if preprocess_tmp_dir:
                            preprocess_tmp_dir_abs = Path(preprocess_tmp_dir).absolute()
                            tmp_root_abs = tmp_root.absolute()
                            try:
                                if tmp_root_abs == preprocess_tmp_dir_abs or preprocess_tmp_dir_abs.is_relative_to(tmp_root_abs):
                                    should_delete = False
                            except AttributeError:
                                try:
                                    preprocess_tmp_dir_abs.relative_to(tmp_root_abs)
                                    should_delete = False
                                except ValueError:
                                    pass
                        if should_delete:
                            shutil.rmtree(tmp_root, ignore_errors=True)
                except Exception:
                    pass
                if preloaded_data is None:
                    if preprocess_tmp_dir:
                        try:
                            preprocess_tmp_dir_path = Path(preprocess_tmp_dir)
                            if preprocess_tmp_dir_path.exists():
                                shutil.rmtree(preprocess_tmp_dir_path, ignore_errors=True)
                                logger.debug(f'Deleted preprocess temporary directory: {preprocess_tmp_dir_path}')
                        except Exception as e:
                            logger.debug(f'Failed to delete preprocess temporary directory (ignored): {e}')
                    gwas_output_dir = output_dir_path / 'output'
                    if gwas_output_dir.exists():
                        try:
                            shutil.rmtree(gwas_output_dir, ignore_errors=True)
                        except Exception as e:
                            logger.debug(f'Failed to delete GWAS output directory (ignored): {e}')
            except Exception as cleanup_err:
                pass
        else:
            pass
        if CLEANUP_TEMP_FILES:
            try:
                import shutil
                if run_tmp_dir.exists():
                    shutil.rmtree(run_tmp_dir, ignore_errors=True)
                if preloaded_data is None:
                    try:
                        tmp_train_dir = tmp_root.parent
                        if tmp_train_dir.exists():
                            shutil.rmtree(tmp_train_dir, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        tmp_root_dir = tmp_root.parent.parent
                        if tmp_root_dir.exists():
                            shutil.rmtree(tmp_root_dir, ignore_errors=True)
                    except Exception:
                        pass
            except Exception as e:
                pass
        return 0
    except Exception as e:
        try:
            peak_ram_mb = memory_tracker.stop()
            logger.error(f'Single model training failed: {str(e)} (peak RAM: {peak_ram_mb} MB)', exc_info=True)
        except Exception:
            logger.error(f'Single model training failed: {str(e)}', exc_info=True)
        if CLEANUP_TEMP_FILES:
            try:
                _temp_file_manager.cleanup_on_exit(cleanup_preprocess=False)
                import shutil
                if run_tmp_dir.exists():
                    shutil.rmtree(run_tmp_dir, ignore_errors=True)
                if preloaded_data is None:
                    try:
                        tmp_train_dir = tmp_root.parent
                        if tmp_train_dir.exists():
                            shutil.rmtree(tmp_train_dir, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        tmp_root_dir = tmp_root.parent.parent
                        if tmp_root_dir.exists():
                            shutil.rmtree(tmp_root_dir, ignore_errors=True)
                    except Exception:
                        pass
            except Exception as cleanup_err:
                pass
        else:
            pass
        return 1

def _run_single_model_silent(*args, **kwargs) -> int:
    import logging as _logging
    root = _logging.getLogger()
    old_level = root.level
    try:
        root.setLevel(_logging.ERROR)
        return run_single_model(*args, **kwargs)
    finally:
        root.setLevel(old_level)

def _build_train_all_preloaded_data(train_file: str, valid_samples: Optional[List[str]], metadata: Optional[Dict], preprocess_tmp_dir: Optional[str], task_type: str, holdout_split: Optional[Dict[str, Any]]=None, X: Optional[pd.DataFrame]=None, y: Optional[pd.Series]=None, *, shared_matrix: Optional[Dict[str, Any]]=None, shap_dependence_max_features: Optional[int]=None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {'train_file': train_file, 'valid_samples': valid_samples, 'metadata': metadata, 'preprocess_tmp_dir': preprocess_tmp_dir, 'task_type': task_type, 'holdout_split': holdout_split, 'shap_dependence_max_features': shap_dependence_max_features}
    if shared_matrix is not None:
        payload.update(shared_matrix)
    elif X is not None and y is not None:
        payload['X'] = X
        payload['y'] = y
    return payload

def _train_all_model_task(model_type: str, input_path: str, output_dir: str, task_type: Optional[str], n_folds: int, random_state: int, preloaded_data: Dict[str, Any], cpu_cores: int, enable_hyperparameter_search: bool) -> Tuple[str, int]:
    _reset_worker_signal_handlers()
    ret_code = _run_single_model_silent(input_path, model_type, output_dir, task_type, n_folds, random_state, preloaded_data, cpu_cores=cpu_cores, enable_hyperparameter_search=enable_hyperparameter_search)
    return (model_type, ret_code)

def _train_all_model_process_entry(model_type: str, input_path: str, output_dir: str, task_type: Optional[str], n_folds: int, random_state: int, preloaded_data: Dict[str, Any], cpu_cores: int, enable_hyperparameter_search: bool, result_queue: Any) -> None:
    try:
        result_queue.put(_train_all_model_task(model_type, input_path, output_dir, task_type, n_folds, random_state, preloaded_data, cpu_cores, enable_hyperparameter_search))
    except Exception:
        result_queue.put((model_type, 1))

def _run_train_all_models_serial(supported_models: List[str], input_path: str, output_dir: str, task_type: Optional[str], n_folds: int, random_state: int, preloaded_data: Dict[str, Any], cpu_cores: int, enable_hyperparameter_search: bool) -> Dict[str, str]:
    results: Dict[str, str] = {}
    mp_ctx = mp.get_context('spawn')
    for model_type in supported_models:
        logger.debug(f'[TRAIN_ALL] Model training started: {model_type}')
        model_start = time.perf_counter()
        result_queue = mp_ctx.Queue()
        proc = mp_ctx.Process(target=_train_all_model_process_entry, args=(model_type, input_path, output_dir, task_type, n_folds, random_state, preloaded_data, cpu_cores, enable_hyperparameter_search, result_queue))
        proc.start()
        proc.join()
        ret_code = 1
        if proc.exitcode == 0 and (not result_queue.empty()):
            _, ret_code = result_queue.get()
        elif proc.exitcode not in (0, None):
            _log_worker_crash_hint(parallel_mode=False)
        status = 'success' if ret_code == 0 else 'failed'
        results[model_type] = status
        logger.info(f'[TRAIN_ALL] Model training completed: model={model_type}, status={status}, elapsed={time.perf_counter() - model_start:.1f}s')
    failed_models = [m for m, s in results.items() if s == 'failed']
    if failed_models:
        logger.debug(f'[TRAIN_ALL] {len(failed_models)} model(s) failed and were skipped: {", ".join(failed_models)}')
    return results

def run_all_models(input_path: str, output_dir: str, task_type: Optional[str]=None, n_folds: int=5, random_state: int=42, parallel_models: Optional[int]=None, threads: Optional[int]=None, enable_hyperparameter_search: bool=False, shap_dependence_max_features: Optional[int]=None) -> int:
    return _run_all_models_body(input_path=input_path, output_dir=output_dir, task_type=task_type, n_folds=n_folds, random_state=random_state, parallel_models=parallel_models, threads=threads, enable_hyperparameter_search=enable_hyperparameter_search, shap_dependence_max_features=shap_dependence_max_features)

def _run_all_models_body(input_path: str, output_dir: str, task_type: Optional[str]=None, n_folds: int=5, random_state: int=42, parallel_models: Optional[int]=None, threads: Optional[int]=None, enable_hyperparameter_search: bool=False, shap_dependence_max_features: Optional[int]=None) -> int:
    # Process-tree peak for the whole train-all job (parent + worker children when parallel).
    # Failures in sampling must never change train-all success/failure.
    job_memory_tracker: Optional[PeakMemoryTracker] = PeakMemoryTracker(include_children=True)
    job_peak_ram_mb: Optional[float] = None
    base_output_dir: Optional[Path] = None
    output_dir_path: Optional[Path] = None
    shared_matrix_bundle: Optional[Dict[str, Any]] = None
    preprocess_tmp_dir_str: Optional[str] = None
    temps_cleaned = False

    def _stop_job_memory_tracker() -> Optional[float]:
        nonlocal job_memory_tracker, job_peak_ram_mb
        if job_memory_tracker is None:
            return job_peak_ram_mb
        try:
            job_peak_ram_mb = job_memory_tracker.stop()
        except Exception:
            pass
        job_memory_tracker = None
        if job_peak_ram_mb is not None:
            logger.info(f'[TRAIN_ALL] Job peak RAM (process tree, whole train-all)={job_peak_ram_mb} MB')
        return job_peak_ram_mb

    def _cleanup_temps() -> None:
        """Idempotent temp cleanup; always run before final exit (success or failure)."""
        nonlocal temps_cleaned, shared_matrix_bundle
        if temps_cleaned:
            return
        temps_cleaned = True
        try:
            _cleanup_train_all_temp_dirs(base_output_dir, shared_matrix_bundle=shared_matrix_bundle, preprocess_tmp_dir=preprocess_tmp_dir_str, output_dir=output_dir_path)
        except Exception as e:
            logger.debug(f'[TRAIN_ALL] Temp cleanup failed (ignored): {e}')
        shared_matrix_bundle = None

    try:
        job_memory_tracker.start()
    except Exception:
        job_memory_tracker = None
    try:
        if not str(input_path).endswith('_metadata.json'):
            raise ValueError(f'Training input must be the preprocess-generated metadata file (*_metadata.json).Current input: {input_path}')
        input_info = parse_train_input_path(input_path)
        preprocess_tmp_dir_str = input_info.get('preprocess_tmp_dir')
        if task_type is None:
            task_type = input_info.get('task_type') or 'regression'
            if not input_info.get('task_type'):
                logger.debug('  task_type not specified and no task_type information in metadata, using default: regression')
        check_training_metadata_guardrails(input_info.get('metadata'), task_type=task_type, parallel_models=parallel_models, enable_hyperparameter_search=enable_hyperparameter_search, n_folds=n_folds, threads=threads)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        base_output_dir = output_dir_path.parent
        train_file = input_info['train_file']
        valid_samples = input_info['metadata']['valid_samples'] if input_info['metadata'] else None
        train_file_format = input_info['metadata'].get('train_file_format') if input_info.get('metadata') else None
        X, y, _snp_name_mapping = load_training_data(train_file, valid_samples, file_format=train_file_format)
        holdout_split = resolve_holdout_split_from_metadata(X, y, input_info.get('metadata'), task_type=task_type, random_state=random_state)
        shared_matrix_dir = base_output_dir / 'tmp' / 'train' / 'shared_matrix'
        shared_matrix_bundle = _persist_training_matrix_for_workers(X, y, shared_matrix_dir)
        del X, y
        gc.collect()
        preloaded_data = _build_train_all_preloaded_data(train_file=train_file, valid_samples=valid_samples, metadata=input_info['metadata'], preprocess_tmp_dir=preprocess_tmp_dir_str, task_type=task_type, holdout_split=holdout_split, shared_matrix=shared_matrix_bundle, shap_dependence_max_features=shap_dependence_max_features)
        supported_models = get_supported_models(task_type)
        if task_type == 'regression':
            logger.debug('Logistic model skipped (classification only)')
        results = {}
        model_metrics_summary = {}
        model_peak_ram_summary: Dict[str, float] = {}
        start_time = time.perf_counter()
        tmp_dir = base_output_dir / 'tmp' / 'train'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        parallel_model_count, cpu_cores_per_worker = _resolve_train_all_parallelism(len(supported_models), parallel_models, threads)
        use_parallel = parallel_model_count > 1
        mode_label = 'Parallel' if use_parallel else 'Serial'
        logger.info(f'[TRAIN_ALL] {mode_label} training: {len(supported_models)} models, parallel={parallel_model_count}, cores/worker={cpu_cores_per_worker}, task={task_type}')
        logger.debug('[TRAIN_ALL] Workers attach to shared training-matrix memmap (single TSV parse in parent process)')
        logger.debug('[TRAIN_ALL] Using spawn workers to isolate native libraries (LightGBM/XGBoost/SHAP/matplotlib) between models')
        model_start_times = {m: time.perf_counter() for m in supported_models}
        pool_resource_failure = False
        if use_parallel:
            from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
            mp_ctx = mp.get_context('spawn')
            try:
                with ProcessPoolExecutor(max_workers=parallel_model_count, mp_context=mp_ctx) as executor:
                    futures = {}
                    for model_type in supported_models:
                        logger.debug(f'[TRAIN_ALL] Model training started: {model_type}')
                        model_start_times[model_type] = time.perf_counter()
                        futures[executor.submit(_train_all_model_task, model_type, input_path, output_dir, task_type, n_folds, random_state, preloaded_data, cpu_cores_per_worker, enable_hyperparameter_search)] = model_type
                    for future in as_completed(futures):
                        model_type = futures[future]
                        try:
                            finished_model, ret_code = future.result()
                            model_type = finished_model
                        except BrokenExecutor as e:
                            pool_resource_failure = True
                            logger.error(f'[TRAIN_ALL] Worker process crashed: model={model_type}, err={e}')
                            ret_code = 1
                        except Exception as e:
                            logger.error(f'[TRAIN_ALL] Model training exception: model={model_type}, err={e}', exc_info=True)
                            ret_code = 1
                        results[model_type] = 'success' if ret_code == 0 else 'failed'
                        logger.info(f'[TRAIN_ALL] Model training completed: model={model_type}, status={results[model_type]}, elapsed={time.perf_counter() - model_start_times.get(model_type, start_time):.1f}s')
            except BrokenExecutor as e:
                pool_resource_failure = True
                logger.error(f'[TRAIN_ALL] Process pool broken: {e}')
            for model_type in supported_models:
                if model_type not in results:
                    results[model_type] = 'failed'
                    pool_resource_failure = True
                    logger.error(f'[TRAIN_ALL] Model marked failed (pool broken or incomplete): {model_type}')
            if pool_resource_failure:
                _log_worker_crash_hint(parallel_mode=True)
                logger.info('[TRAIN_ALL] Cleaning temporary directories before exit after worker crash')
                return 1
        else:
            results.update(_run_train_all_models_serial(supported_models, input_path, output_dir, task_type, n_folds, random_state, preloaded_data, cpu_cores_per_worker, enable_hyperparameter_search))
        logger.info(f'[TRAIN_ALL] {mode_label} training phase completed, elapsed_total={time.perf_counter() - start_time:.1f}s')
        _stop_job_memory_tracker()
        # Free shared memmap / tmp early on the success path (idempotent; finally will no-op).
        _cleanup_temps()
        best_model_type = None
        best_score = None
        best_metric_name = None
        logger.info('Model performance summary:')
        for model_type in supported_models:
            model_dir = output_dir_path / model_type
            metrics_file = model_dir / f'{model_type}_metrics.json'
            if not metrics_file.exists():
                logger.debug(f'  Metrics file not found for model {model_type}: {metrics_file}')
                continue
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            except Exception as e:
                logger.debug(f'  Failed to load metrics for model {model_type}: {e}')
                continue
            metrics = metrics_data.get('metrics', {}) or {}
            model_metrics_summary[model_type] = metrics
            raw_peak_ram = metrics_data.get('peak_ram_mb')
            if isinstance(raw_peak_ram, (int, float)):
                model_peak_ram_summary[model_type] = float(raw_peak_ram)
            if task_type == 'classification':
                raw_auc = metrics.get('auc')
                auc = raw_auc if isinstance(raw_auc, (int, float)) else None
                raw_acc = metrics.get('accuracy')
                acc = raw_acc if isinstance(raw_acc, (int, float)) else None
                peak_ram_str = f', peak_ram_mb={model_peak_ram_summary[model_type]}' if model_type in model_peak_ram_summary else ''
                logger.info(f"  {model_type} (held-out test): accuracy={metrics.get('accuracy')}, recall={metrics.get('recall')}, f1={metrics.get('f1')}, auc={metrics.get('auc')}{peak_ram_str}")
                score = None
                metric_name = None
                if auc is not None:
                    score = auc
                    metric_name = 'auc'
                elif acc is not None:
                    score = acc
                    metric_name = 'accuracy'
            else:
                raw_corr = metrics.get('pearson_correlation')
                corr = raw_corr if isinstance(raw_corr, (int, float)) else None
                peak_ram_str = f', peak_ram_mb={model_peak_ram_summary[model_type]}' if model_type in model_peak_ram_summary else ''
                pval = metrics.get('pearson_p_value')
                pval_str = f', pearson_p_value={pval}' if pval is not None and pval != 'N/A' else ''
                logger.info(f"  {model_type} (held-out test): pearson_correlation={metrics.get('pearson_correlation')}{pval_str}{peak_ram_str}")
                score = corr
                metric_name = 'pearson_correlation'
            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score
                    best_model_type = model_type
                    best_metric_name = metric_name
        if model_peak_ram_summary:
            max_mb = round(max(model_peak_ram_summary.values()), 2)
            logger.info(f'[TRAIN_ALL] Per-model peak RAM: max={max_mb} MB (see job_peak_ram_mb for whole-run pressure)')
        all_models_cv_results: Dict[str, Dict] = {}
        for model_type in supported_models:
            cv_file = output_dir_path / model_type / f'{model_type}_cv_results.json'
            if not cv_file.exists():
                continue
            try:
                with open(cv_file, 'r') as f:
                    all_models_cv_results[model_type] = json.load(f)
            except Exception as e:
                logger.debug(f'  Failed to load CV results for {model_type}: {e}')
        if all_models_cv_results:
            aggregate_cv_file = output_dir_path / 'all_models_cv_results.json'
            try:
                with open(aggregate_cv_file, 'w', encoding='utf-8') as f:
                    json.dump({'task_type': task_type, 'n_folds': n_folds, 'cv_scope': 'training_set', 'final_evaluation_scope': 'held_out_test_set', 'models': all_models_cv_results, 'generated_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2, ensure_ascii=False, default=str)
                logger.debug(f'  Aggregated CV results saved for train-all: {aggregate_cv_file} ({len(all_models_cv_results)} models)')
            except Exception as e:
                logger.debug(f'  Failed to save aggregated CV results: {e}')
        if best_model_type is not None:
            best_model_dir = output_dir_path / best_model_type
            best_model_src = best_model_dir / f'{best_model_type}_model.pkl'
            if best_model_src.exists():
                best_info_file = output_dir_path / 'best_model_info.json'
                try:
                    with open(best_info_file, 'w', encoding='utf-8') as f:
                        json.dump({'task_type': task_type, 'best_model_type': best_model_type, 'best_metric_name': best_metric_name, 'best_metric_value': best_score, 'all_model_metrics': model_metrics_summary, 'generated_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2, ensure_ascii=False, default=str)
                    logger.info(f'Best model: {best_model_type} ({best_metric_name}={best_score})')
                except Exception as e:
                    logger.debug(f'  Failed to save best_model_info.json: {e}')
            else:
                logger.debug(f'  Best model file not found for model {best_model_type}: {best_model_src}')
            for model_type in supported_models:
                if model_type == best_model_type:
                    continue
                model_dir = output_dir_path / model_type
                if model_dir.exists():
                    try:
                        shutil.rmtree(model_dir, ignore_errors=True)
                        logger.debug(f'  Removed model directory for non-best model: {model_type}')
                    except Exception as e:
                        logger.debug(f'  Failed to remove model directory {model_dir}: {e}')
        else:
            logger.debug('  No valid metrics found to select the best model.')
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        report_file = output_dir_path / 'model_comparison_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            report_payload = {'task_type': task_type, 'split_mode': 'preprocess_sample_split_cv', 'n_folds': n_folds, 'random_state': random_state, 'sample_split': (input_info.get('metadata') or {}).get('sample_split'), 'note': 'Train/test split and GWAS/LD feature selection are handled in the preprocess module', 'hyperparameter_search': enable_hyperparameter_search, 'training_results': results, 'total_training_time': round(time.perf_counter() - start_time, 2), 'generated_time': time.strftime('%Y-%m-%d %H:%M:%S')}
            if job_peak_ram_mb is not None:
                report_payload['job_peak_ram_mb'] = job_peak_ram_mb
            if model_peak_ram_summary:
                report_payload['model_peak_ram_mb'] = model_peak_ram_summary
                report_payload['model_peak_ram_mb_max'] = round(max(model_peak_ram_summary.values()), 2)
            json.dump(report_payload, f, indent=2, ensure_ascii=False)
        logger.debug('  All models training completed')
        if any(status == 'failed' for status in results.values()):
            logger.error('[TRAIN_ALL] One or more models failed; see logs above for details.')
            return 1
        return 0
    except Exception as e:
        logger.error(f'  All-models training failed: {str(e)}', exc_info=True)
        logger.info('[TRAIN_ALL] Cleaning temporary directories before exit after failure')
        return 1
    finally:
        _cleanup_temps()
        _stop_job_memory_tracker()

def _resolve_plink_genotype_prefix(input_path: str) -> Optional[str]:
    path = Path(input_path)
    if path.suffix.lower() == '.bed':
        prefix = str(path.with_suffix('').absolute())
        return prefix if Path(f'{prefix}.bim').exists() and Path(f'{prefix}.fam').exists() else None
    prefix = str(path.absolute())
    if Path(f'{prefix}.bed').exists() and Path(f'{prefix}.bim').exists() and Path(f'{prefix}.fam').exists():
        return prefix
    return None

def _is_vcf_input_path(input_path: str) -> bool:
    lowered = str(input_path).lower()
    return lowered.endswith('.vcf') or lowered.endswith('.vcf.gz')

def _is_training_matrix_input_path(input_path: str) -> bool:
    lowered = str(input_path).lower()
    return lowered.endswith('.txt') or lowered.endswith('.txt.gz')

def _genotype_to_prediction_train_file(genotype_input: str, train_chr_pos_keys: List[str], tmp_dir: Path, *, is_vcf: bool) -> str:
    from G2PInsight.bin.preprocess import genotype_to_plink, plink_to_training_data_optimized, normalize_chromosome_names, extract_snps_by_chr_pos_with_plink
    tmp_dir.mkdir(parents=True, exist_ok=True)
    plink_tmp_prefix = str(tmp_dir / 'predict_plink')
    if is_vcf:
        logger.debug('  Step 1: Converting VCF to PLINK format (temporary copy)...')
        plink_prefix = genotype_to_plink(genotype_input, plink_tmp_prefix, fmt='vcf')
    else:
        src_prefix = _resolve_plink_genotype_prefix(genotype_input)
        if src_prefix is None:
            raise FileNotFoundError(f'PLINK binary files not found for prefix: {genotype_input}')
        logger.debug(f'  Copying PLINK binary genotype to temporary prefix: {plink_tmp_prefix}')
        plink_prefix = genotype_to_plink(src_prefix, plink_tmp_prefix, fmt='plink_binary')
    normalize_chromosome_names(plink_prefix)
    plink_for_recode = plink_prefix
    if train_chr_pos_keys:
        logger.debug(f'  PLINK --extract on {len(train_chr_pos_keys):,} training chr_pos key(s) before recode...')
        extracted_prefix = str(tmp_dir / 'predict_plink_extracted')
        plink_for_recode = extract_snps_by_chr_pos_with_plink(plink_prefix, train_chr_pos_keys, extracted_prefix)
    else:
        logger.debug('  No training chr_pos keys available; recoding full genotype (may be slow and memory-intensive)')
    logger.debug('  Converting extracted PLINK subset to training-matrix format...')
    fam_file = f'{plink_for_recode}.fam'
    if not Path(fam_file).exists():
        raise FileNotFoundError(f'PLINK .fam file not found: {fam_file}')
    fam_df = pd.read_csv(fam_file, sep='\\s+', header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])
    sample_ids = fam_df['IID'].tolist()
    pheno_df = pd.DataFrame({'sample': sample_ids, 'phenotype': [0] * len(sample_ids)}).set_index('sample')
    temp_train_file = str(tmp_dir / 'predict_train_data.txt')
    actual_train_file = plink_to_training_data_optimized(plink_prefix=plink_for_recode, pheno_df=pheno_df, output_file=temp_train_file, tmp_dir=str(tmp_dir), use_cache=True).train_file
    logger.debug(f'  Genotype conversion completed: {actual_train_file}')
    return actual_train_file

def predict_with_model(input_path: str, model_path: str, output_dir: str, task_type: str) -> int:
    try:
        input_path_obj = Path(input_path)
        is_vcf_input = _is_vcf_input_path(input_path)
        plink_prefix_input = None if is_vcf_input or _is_training_matrix_input_path(input_path) else _resolve_plink_genotype_prefix(input_path)
        is_genotype_input = is_vcf_input or plink_prefix_input is not None
        tmp_dir_to_clean = None
        model_path_obj = Path(model_path).absolute()
        if not model_path_obj.is_file():
            raise FileNotFoundError(f'Model file does not exist or is not a valid file path: {model_path_obj}')
        model_file = model_path_obj
        model_dir = model_file.parent
        name = model_file.stem
        model_type = name[:-6] if name.endswith('_model') else name
        features_file = model_dir / f'{model_type}_training_features.json'
        train_features: List[str] = []
        train_chr_pos_keys: List[str] = []
        if features_file.exists():
            try:
                with open(features_file, 'r') as f:
                    feature_info = json.load(f)
                train_features = feature_info.get('feature_names') or []
                train_chr_pos_keys = feature_info.get('feature_chr_pos_keys') or []
            except Exception as e:
                logger.debug(f'  Failed to load training_features.json: {e}')
        if train_features and (not train_chr_pos_keys):
            train_chr_pos_keys = build_feature_chr_pos_keys(train_features)
            logger.debug('  feature_chr_pos_keys not in model bundle; derived from feature_names')
        if is_genotype_input:
            input_label = 'VCF' if is_vcf_input else 'PLINK binary'
            logger.debug(f'Detected {input_label} genotype input: {input_path}')
            output_dir_path = Path(output_dir)
            tmp_dir = output_dir_path / 'tmp_predict'
            tmp_dir_to_clean = tmp_dir
            genotype_source = input_path if is_vcf_input else plink_prefix_input
            input_path = _genotype_to_prediction_train_file(genotype_source, train_chr_pos_keys, tmp_dir, is_vcf=is_vcf_input)
        if not train_features:
            logger.debug('  training_features.json not found or empty; feature alignment may be unreliable.')
        chr_pos_filter = set(train_chr_pos_keys) if train_chr_pos_keys else None
        X, _, _ = load_training_data(input_path, chr_pos_filter=chr_pos_filter)
        if not train_features:
            train_features = list(X.columns)
            train_chr_pos_keys = build_feature_chr_pos_keys(train_features)
        X_aligned, matched_count, missing_count = align_features_by_chr_pos(X, train_features, train_chr_pos_keys)
        input_chr_pos_keys = {column_to_chr_pos(str(c)) or str(c) for c in X.columns}
        extra_in_predict = len(input_chr_pos_keys - set(train_chr_pos_keys))
        logger.debug(f'  Feature alignment (chr_pos): {matched_count} matched, {missing_count} missing (filled with 0), {extra_in_predict} extra chr_pos key(s) in input ignored')
        if matched_count == 0:
            logger.warning(f'  WARNING: No features matched between training and prediction data!\n  This will result in all-zero feature matrix and constant predictions.\n  Training chr_pos keys (first 20): {train_chr_pos_keys[:20]}\n  Input chr_pos keys (first 20): {[column_to_chr_pos(c) or c for c in X.columns[:20]]}')
        if missing_count > matched_count:
            logger.warning(f'  WARNING: More features missing ({missing_count}) than matched ({matched_count}).\n  This may result in poor prediction quality.')
        if X_aligned.empty:
            raise ValueError('Aligned feature matrix is empty!')
        non_zero_counts = (X_aligned != 0).sum(axis=1)
        if (non_zero_counts == 0).all():
            logger.error(f'  ERROR: All feature values are zero! This will result in constant predictions.\n  Matched features: {matched_count}\n  Missing features: {missing_count}\n  Please check if chr_pos keys match between training and prediction data.')
        elif missing_count:
            missing_examples = [k for k in train_chr_pos_keys if k not in {column_to_chr_pos(c) or c for c in X.columns}][:5]
            logger.debug(f'  {missing_count} training SNP(s) missing in predict input; filled with 0. Example chr_pos: {missing_examples}')
        if not model_file.exists():
            raise FileNotFoundError(f'Model file not found: {model_file}')
        import joblib
        model = joblib.load(model_file)
        logger.debug('  Loading pre-trained model')
        feature_variance = X_aligned.var()
        zero_variance_features = (feature_variance == 0).sum()
        if zero_variance_features > 0:
            logger.debug(f'  WARNING: {zero_variance_features} features have zero variance (constant values)')
        y_pred = model.predict(X_aligned)
        y_prob = model.predict_proba(X_aligned) if task_type == 'classification' and hasattr(model, 'predict_proba') else None
        unique_predictions = len(np.unique(y_pred))
        logger.debug(f'  Prediction statistics: {unique_predictions} unique values out of {len(y_pred)} samples')
        if unique_predictions == 1:
            logger.error(f"  ERROR: All predictions are the same value: {y_pred[0]}\n  This usually indicates one of the following issues:\n  1. chr_pos keys don't match between training and prediction data\n  2. All feature values are zero (check matched features: {matched_count})\n  3. Model is predicting a constant value for all samples\n  Please check the feature alignment logs above.")
        else:
            logger.debug(f'  Prediction range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}], mean: {np.mean(y_pred):.4f}')
        pred_output_dir = Path(output_dir)
        pred_output_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_output_dir / f'{model_type}_predictions.tsv'
        result_df = pd.DataFrame({'sample': X_aligned.index, 'prediction': y_pred})
        if task_type == 'classification' and y_prob is not None:
            for i in range(y_prob.shape[1]):
                result_df[f'prob_class_{i}'] = y_prob[:, i]
        result_df.to_csv(pred_file, sep='\t', index=False)
        logger.info(f'Prediction completed: {pred_file}')
        if is_genotype_input and tmp_dir_to_clean and tmp_dir_to_clean.exists():
            try:
                shutil.rmtree(tmp_dir_to_clean, ignore_errors=True)
            except Exception as e:
                pass
        return 0
    except Exception as e:
        logger.error(f'  Prediction failed: {str(e)}', exc_info=True)
        return 1
