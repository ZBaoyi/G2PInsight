"""Visualization for G2PInsight ``visualize`` command.

Public entry points used by ``G2PInsight.main``:

- ``EnhancedGenomeVisualizer`` — genome-wide importance / SHAP bar / dependence / summary (``-i``)
- ``resolve_shap_dependence_file`` — locate ``*_shap_dependence.tsv|.npz``
- ``plot_model_performance_from_file`` — performance / CV / confusion plots (``-I``)

Layout of this module:

1. Shared palettes & axis helpers
2. Genome-wide importance (``EnhancedGenomeVisualizer``)
3. SHAP dependence / summary helpers
4. Performance & CV plots + ``load_plotting_data``
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

warnings.filterwarnings('ignore')
try:
    from G2PInsight.bin.font_utils import setup_matplotlib_font, setup_plotly_font
    setup_matplotlib_font()
    PLOTLY_FONT = setup_plotly_font()
except ImportError:
    PLOTLY_FONT = {'family': 'Arial', 'size': 12}
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Shared palettes & axis helpers
# ---------------------------------------------------------------------------

# Cohesive palette for SHAP / genome-wide visualizations
_SHAP_PALETTE = {
    'positive': '#B5533E',
    'negative': '#3E5C76',
    'neutral': '#5C677D',
    'zero_line': '#BBBBBB',
}
# SNP genotype feature values: 0, 1, 2 (discrete colors, no gradient)
_GENOTYPE_VALUE_COLORS = ['#3E5C76', '#8DA9C4', '#C1666B']
_GENOTYPE_CMAP = ListedColormap(_GENOTYPE_VALUE_COLORS)
_GENOTYPE_NORM = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=len(_GENOTYPE_VALUE_COLORS))
_GENOTYPE_SCATTER_KW = {'cmap': _GENOTYPE_CMAP, 'norm': _GENOTYPE_NORM, 'alpha': 0.78, 'edgecolors': 'none'}

# Cohesive palette for CV / performance bar charts
_CV_PALETTE = {
    'corr': '#3E5C76',
    'accuracy': '#3E5C76',
    'recall': '#C1666B',
    'f1': '#5B8A72',
    'auc': '#7B6FA6',
    'mean_line': '#B5533E',
}
_CV_STATS_BBOX = {'boxstyle': 'round', 'facecolor': '#F5F6F8', 'alpha': 0.92, 'edgecolor': '#D0D4DA', 'linewidth': 0.8}
_CV_MEAN_LINE_KW = {'color': _CV_PALETTE['mean_line'], 'linestyle': '--', 'linewidth': 1.8, 'alpha': 0.85}
_CV_BAR_KW = {'alpha': 0.92, 'edgecolor': 'white', 'linewidth': 0.7}
_CV_GROUP_BAR_KW = {'alpha': 0.92, 'edgecolor': 'white', 'linewidth': 0.7}
_CV_BAR_WIDTH_COMPACT = 0.58

# Palette / bar style for held-out *test* metric bars (*_test_metrics.png)
_TEST_METRIC_PALETTE = {
    'test': '#4EB5FF',
}
_TEST_METRIC_BAR_WIDTH = 0.48
_TEST_METRIC_BAR_KW = {'alpha': 0.92, 'edgecolor': '#333333', 'linewidth': 1.0}
_TEST_METRIC_CLASS_YMAX = 1.06
_TEST_METRIC_REG_YMIN = -1.05
_TEST_METRIC_REG_YMAX = 1.08
_TEST_METRIC_LABEL_PAD = 0.015

# Confusion matrix heatmap
_CONFUSION_CMAP = 'Blues'

def _test_metric_score_limits(task_type: str) -> Tuple[float, float]:
    if task_type == 'classification':
        return (0.0, _TEST_METRIC_CLASS_YMAX)
    return (_TEST_METRIC_REG_YMIN, _TEST_METRIC_REG_YMAX)

def _test_metric_value_fmt(task_type: str) -> str:
    return '.3f' if task_type == 'classification' else '.4f'

def _test_metric_annotate_bar_values(ax: Axes, bar_containers, *, fmt: str='.3f') -> None:
    y_lo, y_hi = ax.get_ylim()
    pad = max(_TEST_METRIC_LABEL_PAD, (y_hi - y_lo) * 0.022)
    for container in bar_containers:
        for bar in container:
            height = bar.get_height()
            if not isinstance(height, (int, float)) or np.isnan(height):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + pad, f'{height:{fmt}}', ha='center', va='bottom', fontsize=6)

def _test_metric_finish_figure(ax: Axes, fig, bar_containers, task_type: str) -> None:
    y_min, y_max = _test_metric_score_limits(task_type)
    ax.set_ylim(y_min, y_max)
    _test_metric_annotate_bar_values(ax, bar_containers, fmt=_test_metric_value_fmt(task_type))
    ax.grid(False)
    fig.subplots_adjust(right=0.82)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False, fontsize=10, handleheight=1.0, handlelength=1.2)

def _cv_summary_stats(values: List[float], *, float_fmt: str='.4f') -> str:
    arr = np.asarray(values, dtype=float)
    return f'Mean: {np.mean(arr):{float_fmt}}\nMedian: {np.median(arr):{float_fmt}}\nStd: {np.std(arr):{float_fmt}}\nQ25: {np.percentile(arr, 25):{float_fmt}}\nQ75: {np.percentile(arr, 75):{float_fmt}}'

def _cv_brief_stats(values: List[float], *, float_fmt: str='.4f') -> str:
    arr = np.asarray(values, dtype=float)
    return f'Mean ± Std: {np.mean(arr):{float_fmt}} ± {np.std(arr):{float_fmt}}'

def _cv_place_stats(ax: Axes, text: str, *, loc: str='upper right') -> None:
    ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=8, va='top', ha='right', bbox=_CV_STATS_BBOX)

def _cv_place_mean_legend(ax: Axes) -> None:
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.97), fontsize=8, frameon=False)

def _cv_annotate_mean_line(ax: Axes, mean_val: float, n_folds: int, *, fmt: str='.4f') -> None:
    ax.text(n_folds - 0.55, mean_val, f'Mean {mean_val:{fmt}}', va='center', ha='left', fontsize=8, color=_CV_PALETTE['mean_line'])

def _cv_fold_x(n_folds: int) -> np.ndarray:
    return np.arange(n_folds)

def _cv_finish_fold_axis(ax: Axes, n_folds: int, values: List[float], *, x_labels: Optional[List[str]]=None, y_min: Optional[float]=None, y_max: Optional[float]=None, set_ylim: bool=True, y_grid: bool=False) -> None:
    x = _cv_fold_x(n_folds)
    ax.set_xticks(x)
    label_rotation = 0 if n_folds <= 5 else 45
    label_ha = 'center' if label_rotation == 0 else 'right'
    ax.set_xticklabels(x_labels if x_labels else [f'Fold {i + 1}' for i in range(n_folds)], rotation=label_rotation, ha=label_ha)
    if set_ylim:
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        elif values:
            y_min_v = min(values)
            y_max_v = max(values)
            span = y_max_v - y_min_v
            pad = span * 0.18 if span > 0 else abs(y_max_v) * 0.18 + 0.05
            lower = y_min if y_min is not None else y_min_v - pad
            ax.set_ylim(lower, y_max_v + pad)
    if y_grid:
        ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    _hide_top_right_spines(ax)

def _cv_draw_single_barplot(ax: Axes, values: List[float], color: str, *, float_fmt: str='.4f', y_min: Optional[float]=None, y_max: Optional[float]=None, x_labels: Optional[List[str]]=None, label_fn=None, stats_text: Optional[str]=None, mean_fmt: Optional[str]=None, set_ylim: bool=True, show_stats: bool=True, show_mean_legend: bool=True, highlight_outliers: bool=False, y_grid: bool=False) -> None:
    if not values:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        _hide_top_right_spines(ax)
        return
    n_folds = len(values)
    x = _cv_fold_x(n_folds)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    if highlight_outliers:
        threshold = mean_val - std_val
        bar_colors = [_CV_PALETTE['mean_line'] if val < threshold else color for val in values]
    else:
        bar_colors = [color] * n_folds
    bars = ax.bar(x, values, width=_CV_BAR_WIDTH_COMPACT, color=bar_colors, **_CV_BAR_KW)
    _cv_finish_fold_axis(ax, n_folds, values, x_labels=x_labels, y_min=y_min, y_max=y_max, set_ylim=set_ylim, y_grid=y_grid)
    y_lo, y_hi = ax.get_ylim()
    label_pad = (y_hi - y_lo) * 0.012
    for bar, val in zip(bars, values):
        height = bar.get_height()
        text = label_fn(val) if label_fn else f'{val:{float_fmt}}'
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + label_pad, text, ha='center', va='bottom', fontsize=9)
    mean_label_fmt = mean_fmt or float_fmt
    ax.axhline(y=mean_val, label=f'Mean: {mean_val:{mean_label_fmt}}' if show_mean_legend else '_nolegend_', **_CV_MEAN_LINE_KW)
    if show_stats and stats_text is not False:
        _cv_place_stats(ax, stats_text if stats_text is not None else _cv_summary_stats(values, float_fmt=float_fmt))
    if show_mean_legend:
        _cv_place_mean_legend(ax)
    elif stats_text is False or not show_stats:
        _cv_annotate_mean_line(ax, mean_val, n_folds, fmt=mean_label_fmt)

def _cv_draw_group_barplot(ax: Axes, datasets: List[List[float]], labels: List[str], colors: List[str], *, y_min: float=0.0, y_max: float=1.05, stats_text: Optional[str]=None, annotate: bool=True, legend_horizontal: bool=False) -> None:
    if not datasets:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        _hide_top_right_spines(ax)
        return
    n_folds = len(datasets[0])
    x = np.arange(n_folds)
    n_metrics = len(datasets)
    width = 0.8 / n_metrics
    for i, (data, label, color) in enumerate(zip(datasets, labels, colors)):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, data, width, label=label, color=color, **_CV_GROUP_BAR_KW)
        if annotate and n_folds <= 5:
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    label_rotation = 0 if n_folds <= 5 else 45
    label_ha = 'center' if label_rotation == 0 else 'right'
    ax.set_xticklabels([f'Fold {i + 1}' for i in range(n_folds)], rotation=label_rotation, ha=label_ha)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    _hide_top_right_spines(ax)
    if legend_horizontal:
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.06), ncol=len(labels), fontsize=9, frameon=False, borderaxespad=0)
    else:
        ax.legend(loc='upper left', fontsize=9, frameon=False)
    if stats_text:
        _cv_place_stats(ax, stats_text)

def _task_metric_specs(task_type: str) -> List[Tuple[str, str, bool]]:
    if task_type == 'classification':
        return [('accuracy', 'Accuracy', True), ('recall', 'Recall', True), ('f1', 'F1 Score', True), ('auc', 'AUC', True)]
    return [('pearson_correlation', 'Pearson r', True)]

def _regression_test_bar_specs() -> List[Tuple[str, str, bool]]:
    return [('pearson_correlation', 'Pearson r', True), ('r2', 'R²', True), ('rmse', 'RMSE', False), ('mae', 'MAE', False)]

def _collect_metric_bar_values(metrics: Dict, specs: List[Tuple[str, str, bool]]) -> Tuple[List[str], List[float]]:
    metric_names: List[str] = []
    vals: List[float] = []
    for key, label, _ in specs:
        val = _metric_float(metrics.get(key))
        if val is None:
            continue
        metric_names.append(label)
        vals.append(val)
    return metric_names, vals

def _warn_missing_regression_test_metrics(test_metrics: Dict) -> None:
    missing = [label for key, label, _ in _regression_test_bar_specs() if key != 'pearson_correlation' and _metric_float(test_metrics.get(key)) is None]
    if not missing:
        return
    logger.debug('  Held-out test metrics missing from saved training results (not inferred): %s. Re-run train to compute R²/RMSE/MAE, then visualize again.', ', '.join(missing))

def _plot_test_metric_bar_panel(ax: Axes, metric_names: List[str], vals: List[float], *, ylabel: str, ylim: Tuple[float, float], fmt: str='.4f') -> None:
    x = np.arange(len(metric_names))
    bars = ax.bar(x, vals, width=0.55, color=_TEST_METRIC_PALETTE['test'], **_TEST_METRIC_BAR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=0, ha='center')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    _test_metric_annotate_bar_values(ax, [bars], fmt=fmt)
    ax.grid(axis='y', alpha=0.25)
    _hide_top_right_spines(ax)

def _metric_float(value: Any) -> Optional[float]:
    if isinstance(value, str):
        if value.upper() == 'N/A':
            return None
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, (int, float)) and (not np.isnan(value)):
        return float(value)
    return None

def _resolve_output_prefix(output_dir: Path, output_prefix: Optional[Union[str, Path]]) -> Path:
    if output_prefix is None:
        return output_dir / 'cv'
    output_prefix_path = Path(output_prefix)
    if not output_prefix_path.parent or str(output_prefix_path.parent) == '.':
        output_prefix_path = output_dir / output_prefix_path.name
    return output_dir / output_prefix_path.name

# ---------------------------------------------------------------------------
# 2. Genome-wide importance (CLI -i)
# ---------------------------------------------------------------------------

class EnhancedGenomeVisualizer:

    def __init__(self, input_file: str, feature_col: Optional[str]=None, value_col: Optional[str]=None, max_points: Optional[int]=None, auto_sample: bool=False):
        self.input_file = Path(input_file)
        self.feature_col = feature_col
        self.value_col = value_col
        self.max_points = max_points
        self.auto_sample = auto_sample
        self.df = None
        self.threshold = None
        self.effect_col = None
        self.colors = {'Positive': _SHAP_PALETTE['positive'], 'Negative': _SHAP_PALETTE['negative']}
        try:
            self._initialize_data()
        except Exception as e:
            logger.error(f'Data initialization failed: {str(e)}')
            raise

    def _initialize_data(self):
        self.df = self._load_and_process()
        self.threshold = self._calculate_threshold()
        try:
            effect_col_display = self.effect_col or 'None'
            logger.debug(f'Visualization data loaded: file={self.input_file}, rows={len(self.df)}, feature_col={self.feature_col}, value_col={self.value_col}, effect_col={effect_col_display}')
            logger.debug(f'Significance threshold (abs value, 99th percentile): {self.threshold:.4g}')
        except Exception:
            pass


    def _detect_separator(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if '\t' in first_line:
                return '\t'
            elif ',' in first_line:
                return ','
            else:
                return '\t'

    def _detect_columns(self, columns: List[str], sep: str='\t') -> Tuple[str, str, Optional[str]]:
        sample_df = pd.read_csv(self.input_file, nrows=100, sep=sep)
        if len(columns) >= 3:
            feature_col = None
            abs_col = None
            effect_col = None
            for col in columns:
                col_lower = col.lower()
                if col_lower == 'feature':
                    feature_col = col
                elif col_lower in ['importance_abs', 'importance']:
                    abs_col = col
                elif col_lower == 'effect':
                    effect_col = col
            if feature_col and abs_col and effect_col:
                return (feature_col, abs_col, effect_col)
        if len(columns) == 3:
            feature_col = columns[0]
            abs_col = columns[1]
            effect_col = columns[2]
            return (feature_col, abs_col, effect_col)
        feature_col = None
        for col in columns:
            if 'feature' in col.lower():
                feature_col = col
                break
        if feature_col is None:
            first_col_sample = sample_df[columns[0]].dropna().astype(str).head(10)
            first_col_clean = first_col_sample.str.replace('^chr', '', case=False, regex=True)
            if first_col_clean.str.contains('^\\d+_\\d+$', na=False).any():
                feature_col = columns[0]
            else:
                feature_col = columns[0]
        abs_col = None
        for col in columns:
            if col != feature_col:
                if ('abs' in col.lower() or 'importance' in col.lower()) and pd.api.types.is_numeric_dtype(sample_df[col]):
                    abs_col = col
                    break
        if abs_col is None:
            abs_col = columns[1] if len(columns) > 1 else columns[0]
        effect_col = None
        for col in columns:
            if col != feature_col and col != abs_col:
                if 'effect' in col.lower() and pd.api.types.is_numeric_dtype(sample_df[col]):
                    unique_vals = sample_df[col].dropna().unique()
                    if len(unique_vals) <= 2 and all((v in [1, -1, 1.0, -1.0] for v in unique_vals)):
                        effect_col = col
                        break
        if effect_col is None and len(columns) >= 3:
            effect_col = columns[2]
        return (feature_col, abs_col, effect_col)

    def _load_and_process(self) -> pd.DataFrame:
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')
        sep = self._detect_separator(self.input_file)
        tab_char = '\t'
        sep_name = 'Tab' if sep == tab_char else 'Comma'
        sample_df = pd.read_csv(self.input_file, nrows=100, sep=sep)
        feature_col, abs_col, effect_col = self._detect_columns(sample_df.columns, sep=sep)
        if self.feature_col is None:
            self.feature_col = feature_col
        if self.value_col is None:
            self.value_col = abs_col
        self.effect_col = effect_col
        cols_to_load = [self.feature_col, self.value_col]
        if effect_col:
            cols_to_load.append(effect_col)
        dtype_dict = {self.feature_col: 'string', self.value_col: 'float32'}
        if effect_col:
            dtype_dict[effect_col] = 'int32'
        df = pd.read_csv(self.input_file, sep=sep, usecols=cols_to_load, dtype=dtype_dict)
        if len(df) == 0:
            raise ValueError('Input file is empty')
        if df[self.value_col].isna().sum() > 0:
            logger.debug(f'Value column {self.value_col} contains {df[self.value_col].isna().sum():,} missing values, will be automatically removed')
            df = df.dropna(subset=[self.value_col])
            if len(df) == 0:
                raise ValueError('No data remaining after removing missing values')
        value_range = df[self.value_col].max() - df[self.value_col].min()
        if value_range < 1e-09:
            logger.debug(f'Value column {self.value_col} has a very small range, which may affect visualization quality')
        logger.debug(f'Loading all data points: {len(df):,} features')
        snp_series = df[self.feature_col].astype(str)
        snp_clean = snp_series.str.replace('^chr', '', case=False, regex=True)
        chrom_pos = snp_clean.str.extract('^(?P<chrom>[0-9A-Za-z]+)[_:](?P<pos>\\d+)$')

        def _to_chrom_num(c: Any) -> Optional[int]:
            import re
            c_str = str(c).upper()
            m = re.match('^(\\d+)', c_str)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
            return 10 ** 6
        chrom_num_series = chrom_pos['chrom'].apply(_to_chrom_num)
        invalid_mask = chrom_num_series.isna() | chrom_pos['pos'].isna()
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())
            invalid_samples = df.loc[invalid_mask, self.feature_col].head(3).tolist()
            logger.debug(f"Detected invalid feature name(s) that do not match 'chrom_pos' format; will be dropped. invalid={invalid_count:,}/{len(df):,}, samples={invalid_samples}")
            df = df.loc[~invalid_mask].copy()
            chrom_pos = chrom_pos.loc[~invalid_mask]
            chrom_num_series = chrom_num_series.loc[~invalid_mask]
            if len(df) == 0:
                raise ValueError(f"Feature column must follow 'chromosome_position' format (e.g., 1_12345 or chr1_12345), but no valid rows remain after filtering.\nInvalid examples: {invalid_samples}")
        if self.effect_col and self.effect_col in df.columns:
            df = df.assign(chrom_num=chrom_num_series.astype('int32'), position=chrom_pos['pos'].astype('uint32'), chrom_label='Chr' + chrom_pos['chrom'].astype(str), value_sign=np.where(df[self.effect_col] >= 0, 'Positive', 'Negative'), value_abs=df[self.value_col], value_signed=df[self.value_col] * df[self.effect_col]).sort_values(['chrom_num', 'position'])
        else:
            raise ValueError('Input file is missing the effect column. Only 3-column format is supported: feature, importance_abs, effect (effect must be 1/-1).')
        df['x_pos'] = self._calculate_genomic_positions(df)
        return df

    def _calculate_genomic_positions(self, df: pd.DataFrame) -> np.ndarray:
        chrom_info = df[['chrom_num', 'chrom_label']].drop_duplicates()
        numeric_mask = chrom_info['chrom_num'] < 10 ** 6
        chrom_info_numeric = chrom_info[numeric_mask].sort_values(['chrom_num', 'chrom_label'])
        chrom_info_other = chrom_info[~numeric_mask].sort_values(['chrom_label'])
        chrom_info_sorted = pd.concat([chrom_info_numeric, chrom_info_other], ignore_index=True)
        label_to_base = {label: idx for idx, label in enumerate(chrom_info_sorted['chrom_label'])}
        base_pos = df['chrom_label'].map(label_to_base).astype(float)
        within_index = df.groupby('chrom_label').cumcount()
        group_sizes = df.groupby('chrom_label')['chrom_label'].transform('size')
        denom = (group_sizes - 1).replace(0, 1)
        within_norm = within_index / denom
        within_norm = np.where(group_sizes == 1, 0.5, within_norm)
        return base_pos.values + within_norm

    def _get_chromosome_boundary_xs(self) -> np.ndarray:
        """X positions at plot edges and between consecutive chromosomes."""
        chrom_stats = self.df.groupby('chrom_label')['x_pos'].agg(['min', 'max']).sort_values('min')
        if chrom_stats.empty:
            return np.array([], dtype=float)
        edge_pad = 0.5
        left_edge = float(chrom_stats['min'].iloc[0]) - edge_pad
        right_edge = float(chrom_stats['max'].iloc[-1]) + edge_pad
        if len(chrom_stats) <= 1:
            return np.array([left_edge, right_edge], dtype=float)
        inner = chrom_stats['min'].iloc[1:].to_numpy(dtype=float)
        return np.concatenate([[left_edge], inner, [right_edge]])

    def _calculate_threshold(self, percentile: float=99) -> float:
        return np.percentile(self.df['value_abs'], percentile)

    def _dynamic_style(self) -> Tuple[float, float]:
        n = len(self.df)
        log_n = math.log10(n + 1)
        size = max(4, min(52, 260 / (1 + log_n * 0.82)))
        # Tiered opacity: keep markers visible when SNP count is high (avoid alpha -> 0)
        if n <= 5000:
            alpha = 0.88
        elif n <= 50000:
            alpha = 0.78
        elif n <= 200000:
            alpha = 0.70
        elif n <= 1000000:
            alpha = 0.64
        else:
            alpha = 0.58
        return (size, alpha)

    def plot_static_scatter(self, output_file: str, dpi: int=1200, **kwargs) -> None:
        logger.debug(f'Drawing static scatter: input={self.input_file}, output={output_file}, rows={len(self.df)}, feature_col={self.feature_col}, value_col={self.value_col}')
        try:
            self._plot_static_scatter(output_file, dpi, **kwargs)
        except Exception as e:
            logger.error(f'Static scatter plot generation failed: {str(e)}')
            raise

    def _plot_static_scatter(self, output_file: str, dpi: int, **kwargs) -> None:
        sns.set(style='white')
        fig, ax = plt.subplots(figsize=(18, 8), dpi=dpi)
        self._plot_scatter_static(ax, self.colors)
        self._format_axes(ax)
        self._save_figure(fig, output_file, dpi)

    def plot_interactive_scatter(self, output_file: str, **kwargs) -> None:
        logger.debug(f'Drawing interactive scatter: input={self.input_file}, output={output_file}, rows={len(self.df)}, feature_col={self.feature_col}, value_col={self.value_col}')
        try:
            self._plot_interactive_scatter(output_file, **kwargs)
        except Exception as e:
            logger.error(f'Interactive scatter plot generation failed: {str(e)}')
            raise

    def _plot_interactive_scatter(self, output_file: str, **kwargs) -> None:
        fig = self._create_interactive_scatter()
        output_path = Path(output_file)
        if output_path.suffix == '.html':
            fig.write_html(output_path, include_plotlyjs='cdn')
        else:
            try:
                from plotly.io import write_image
            except ImportError as e:
                raise ImportError('plotly is required to export non-HTML interactive plots') from e
            try:
                write_image(fig, output_path, scale=2, engine='kaleido')
            except Exception:
                logger.error('Image writing with kaleido failed, attempting orca engine...', exc_info=True)
                try:
                    write_image(fig, output_path, scale=2, engine='orca')
                except Exception as e2:
                    logger.error(f'Image saving failed: {str(e2)}')
                    raise

    def _plot_scatter_static(self, ax: Axes, colors: Dict[str, str]):
        size, alpha = self._dynamic_style()
        n_points = len(self.df)
        draw_vlines = n_points <= 15000
        for sign, color in colors.items():
            subset = self.df[self.df['value_sign'] == sign]
            x_values = subset['x_pos'].values.flatten()
            y_values = subset['value_abs'].values.flatten()
            if draw_vlines:
                ax.vlines(x=x_values, ymin=0, ymax=y_values, colors=color, alpha=min(0.9, alpha * 0.85), linewidths=0.5, linestyles='-', zorder=1)
            ax.scatter(x=x_values, y=y_values, c=color, s=size, alpha=alpha, edgecolors='none', label=sign, zorder=2, rasterized=n_points > 50000)
        ax.legend(title='Effect Direction', loc='upper right')

    def _create_interactive_scatter(self):
        size, alpha = self._dynamic_style()
        fig = px.scatter(self.df, x=self.df['x_pos'].values.flatten(), y=self.df['value_abs'].values.flatten(), color='value_sign', color_discrete_map=self.colors, hover_data={'chrom_label': True, 'position': True, 'value_signed': ':.3f', 'value_abs': ':.3f', 'x_pos': False}, size_max=size, opacity=alpha, width=1200, height=600, title='Genome-Wide Association Plot')
        fig.update_layout(showlegend=True)
        self._format_interactive_layout(fig)
        return fig

    def _format_interactive_layout(self, fig):
        chrom_ticks = self.df.groupby('chrom_label')['x_pos'].median()
        chrom_ticks = chrom_ticks.sort_values()
        y_max = self.df['value_abs'].max()
        padding = 0.1 * y_max if y_max > 0 else 1.0
        y_min_adjusted = 0
        y_max_adjusted = y_max + padding
        x_boundaries = self._get_chromosome_boundary_xs()
        x_range = [None, None]
        if x_boundaries.size:
            x_range = [float(x_boundaries[0]), float(x_boundaries[-1])]
        fig.update_layout(font=PLOTLY_FONT, xaxis=dict(showgrid=False, title='Genomic Position', tickmode='array', tickvals=chrom_ticks.values.tolist(), ticktext=chrom_ticks.index.tolist(), tickangle=-45, range=x_range), yaxis=dict(showgrid=False, title='Feature Importance', range=[y_min_adjusted, y_max_adjusted]), title={'text': 'Genome-Wide Association Plot', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'family': PLOTLY_FONT.get('family', 'Arial')}}, legend=dict(title='Effect Direction', orientation='v', yanchor='top', y=1, xanchor='right', x=1))
        for x_boundary in x_boundaries:
            fig.add_shape(type='line', x0=x_boundary, y0=y_min_adjusted, x1=x_boundary, y1=y_max_adjusted, line=dict(color='#BBBBBB', width=1), layer='below')

    def _format_axes(self, ax: Axes):
        chrom_ticks = self.df.groupby('chrom_label')['x_pos'].median()
        chrom_ticks = chrom_ticks.sort_values()
        ax.set_xticks(chrom_ticks.values)
        ax.set_xticklabels(chrom_ticks.index, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('Genomic Position', fontsize=16)
        x_boundaries = self._get_chromosome_boundary_xs()
        for x_boundary in x_boundaries:
            ax.axvline(x_boundary, color='#BBBBBB', linestyle='-', linewidth=0.8, alpha=0.65, zorder=0)
        if x_boundaries.size:
            ax.set_xlim(float(x_boundaries[0]), float(x_boundaries[-1]))
        ax.set_ylabel('Feature Importance', fontsize=16)
        ax.set_title('Genome-Wide Association Plot', pad=20, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        y_max = self.df['value_abs'].max()
        y_range = y_max
        padding = 0.1 * y_range if y_range > 0 else 1.0
        ax.set_ylim(0, y_max + padding)
        ax.grid(False)
        _hide_top_right_spines(ax)

    def _is_shap_input(self) -> bool:
        return 'shap' in self.input_file.name.lower()

    def plot_top_snps_bar(self, output_file: str, top_n: int=20, dpi: int=300, **kwargs) -> None:
        logger.debug(f'Drawing top-SNPs bar chart: input={self.input_file}, output={output_file}, top_n={top_n}, rows={len(self.df)}')
        try:
            self._plot_top_snps_bar(output_file, top_n=top_n, dpi=dpi, **kwargs)
        except Exception as e:
            logger.error(f'Top-SNPs bar chart generation failed: {str(e)}')
            raise

    def _plot_top_snps_bar(self, output_file: str, top_n: int, dpi: int, **kwargs) -> None:
        if top_n < 1:
            raise ValueError(f'top_n must be >= 1, got {top_n}')
        n_show = min(int(top_n), len(self.df))
        if n_show == 0:
            raise ValueError('No data available for top-SNPs bar plot')
        top_df = self.df.nlargest(n_show, 'value_abs').copy()
        top_df = top_df.sort_values('value_abs', ascending=True)
        is_shap = self._is_shap_input()
        x_label = 'Mean |SHAP value|' if is_shap else 'Feature importance (absolute)'
        title_suffix = ' (SHAP)' if is_shap else ''
        title = f'Top {n_show} SNPs{title_suffix}'
        fig_height = max(4.0, 0.38 * n_show + 1.8)
        fig, ax = plt.subplots(figsize=(10, fig_height), dpi=dpi)
        bar_colors = [self.colors['Positive'] if sign == 'Positive' else self.colors['Negative'] for sign in top_df['value_sign']]
        ax.barh(top_df['feature'].astype(str), top_df['value_abs'], color=bar_colors, edgecolor='none', height=0.72)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('SNP', fontsize=14)
        ax.set_title(title, pad=14, fontsize=15)
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=12)
        ax.grid(False)
        legend_handles = [Patch(facecolor=self.colors['Positive'], label='Positive effect'), Patch(facecolor=self.colors['Negative'], label='Negative effect')]
        ax.legend(handles=legend_handles, title='Effect direction', loc='lower right')
        _hide_top_right_spines(ax)
        self._save_figure(fig, output_file, dpi)
        plt.close(fig)

    def plot_top_snps_dependence(self, dependence_npz: Union[str, Path], output_dir: Union[str, Path], output_prefix: Union[str, Path], top_n: int=20, dpi: int=300) -> List[Path]:
        top_features = self.df.nlargest(min(int(top_n), len(self.df)), 'value_abs')['feature'].astype(str).tolist()
        logger.debug(f'Drawing top-SNPs dependence plots: npz={dependence_npz}, top_n={len(top_features)}, output_prefix={output_prefix}')
        return plot_top_snps_dependence(dependence_npz=dependence_npz, top_features=top_features, output_dir=output_dir, output_prefix=output_prefix, dpi=dpi)

    def plot_shap_summary(self, dependence_npz: Union[str, Path], output_dir: Union[str, Path], output_prefix: Union[str, Path], top_n: int=20, dpi: int=300) -> Optional[Path]:
        top_features = self.df.nlargest(min(int(top_n), len(self.df)), 'value_abs')['feature'].astype(str).tolist()
        out_file = Path(output_dir) / f'{Path(output_prefix).name}_shap_summary.png'
        logger.debug(f'Drawing SHAP summary plot: npz={dependence_npz}, top_n={len(top_features)}, output={out_file}')
        return plot_shap_summary(dependence_npz=dependence_npz, top_features=top_features, output_file=out_file, dpi=dpi)

    def _save_figure(self, fig, output_path: str, dpi: int):
        path = Path(output_path)
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise PermissionError(f'Permission denied when creating directory: {path.parent}')
            except Exception as e:
                raise IOError(f'Failed to create directory: {str(e)}')
        if path.exists():
            if not os.access(path, os.W_OK):
                raise PermissionError(f'File is not writable: {path}')
        elif not os.access(path.parent, os.W_OK):
            raise PermissionError(f'Directory is not writable: {path.parent}')
        try:
            _apply_clean_spines(fig)
            fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        except Exception as e:
            logger.error(f'Save failed: {str(e)}')
            raise

# ---------------------------------------------------------------------------
# 3. SHAP dependence / summary (needs *_shap_dependence.tsv)
# ---------------------------------------------------------------------------

def _normalize_snp_name(name: str) -> str:
    import re
    return re.sub(r'^chr', '', str(name).strip(), flags=re.IGNORECASE)

def _valid_feature_values(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values >= 0)

def _genotype_codes(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 2).astype(int)

def _genotype_legend_handles() -> List[Patch]:
    return [Patch(facecolor=color, label=str(value)) for value, color in enumerate(_GENOTYPE_VALUE_COLORS)]

def _add_genotype_legend(ax: Axes, title: str='Feature value') -> None:
    ax.legend(handles=_genotype_legend_handles(), title=title, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=10, handleheight=1.0, handlelength=1.2, frameon=False)

def _draw_summary_distribution_violin(ax: Axes, shap_vals: np.ndarray, y_center: float, *, width: float=0.42) -> None:
    if len(shap_vals) < 2:
        return
    parts = ax.violinplot([shap_vals], positions=[y_center], vert=False, widths=width, showmeans=False, showmedians=True, showextrema=False)
    for body in parts.get('bodies', []):
        body.set_facecolor('#D8DEE9')
        body.set_edgecolor('#A8B4C8')
        body.set_alpha(0.5)
        body.set_zorder(1)
    median_line = parts.get('cmedians')
    if median_line is not None:
        median_line.set_color('#4C566A')
        median_line.set_linewidth(1.2)
        median_line.set_zorder(2)

def _summary_cluster_h_step(k: int, x_range: float) -> float:
    if k <= 1:
        return 0.0
    max_width = min(max(x_range * 0.22, 0.08), 0.55)
    return max_width / (k - 1)

def _summary_swarm_positions(shap_x: np.ndarray, y_center: float, *, row_width: float, seed: int=0) -> Tuple[np.ndarray, np.ndarray]:
    """X = SHAP (spread horizontally when samples share the same value); Y stays near row center."""
    n = len(shap_x)
    x_out = shap_x.astype(float).copy()
    y_out = np.full(n, float(y_center), dtype=float)
    if n <= 1:
        return x_out, y_out
    x_range = float(np.ptp(x_out))
    if x_range <= 0:
        x_range = max(abs(float(x_out[0])), 1.0)
    half_v = row_width * 0.06
    rng = np.random.default_rng(seed)
    order = np.argsort(x_out, kind='mergesort')
    tol = max(x_range * 1e-9, 1e-12)
    i = 0
    while i < n:
        j = i + 1
        base_x = float(x_out[order[i]])
        while j < n and abs(float(x_out[order[j]]) - base_x) <= tol:
            j += 1
        cluster_idx = order[i:j]
        k = len(cluster_idx)
        if k > 1:
            h_step = _summary_cluster_h_step(k, x_range)
            for c, idx in enumerate(cluster_idx):
                x_out[idx] = base_x + (c - (k - 1) / 2.0) * h_step
                y_out[idx] = y_center + (rng.random() - 0.5) * 2 * half_v
        i = j
    return x_out, y_out

def _summary_row_width(n_samples: int) -> float:
    return float(min(0.55, 0.28 + 0.0012 * max(n_samples, 1)))

def _plot_shap_summary_row(ax: Axes, shap_col: np.ndarray, feat_vals: np.ndarray, y_center: float, *, row_width: Optional[float]=None, seed: int=0) -> None:
    finite = np.isfinite(shap_col)
    if not finite.any():
        return
    shap_plot = shap_col[finite].astype(float)
    feat_plot = feat_vals[finite].astype(float)
    n_samples = len(shap_plot)
    if row_width is None:
        row_width = _summary_row_width(n_samples)
    _draw_summary_distribution_violin(ax, shap_plot, y_center, width=row_width * 0.88)
    x_pos, y_pos = _summary_swarm_positions(shap_plot, y_center, row_width=row_width, seed=seed)
    valid_geno = _valid_feature_values(feat_plot)
    point_size = max(3, min(10, int(5000 / max(n_samples, 1))))
    scatter_alpha = 0.55 if n_samples > 80 else 0.78
    scatter_kw = {**_GENOTYPE_SCATTER_KW, 'alpha': scatter_alpha}
    if valid_geno.any():
        ax.scatter(x_pos[valid_geno], y_pos[valid_geno], c=_genotype_codes(feat_plot[valid_geno]), s=point_size, **scatter_kw, linewidths=0, zorder=4)
    if (~valid_geno).any():
        ax.scatter(x_pos[~valid_geno], y_pos[~valid_geno], c=_SHAP_PALETTE['neutral'], s=max(3, point_size - 1), alpha=min(scatter_alpha, 0.45), edgecolors='none', zorder=3)

def _jitter_discrete_x(values: np.ndarray, *, jitter: float=0.08, seed: int=0) -> np.ndarray:
    x = values.astype(float).copy()
    if jitter <= 0:
        return x
    rounded = np.rint(x)
    if len(np.unique(rounded[_valid_feature_values(x)])) > 5:
        return x
    rng = np.random.default_rng(seed)
    return x + (rng.random(len(x)) - 0.5) * jitter

def _apply_genotype_dependence_x_axis(ax: Axes) -> None:
    x_lo, x_hi = -0.35, 2.35
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0', '1', '2'])
    ax.xaxis.set_major_locator(FixedLocator([0, 1, 2]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    edge_kw = {'color': _SHAP_PALETTE['zero_line'], 'linewidth': 0.8, 'linestyle': '-', 'alpha': 0.65, 'zorder': 0}
    ax.axvline(x_lo, **edge_kw)
    ax.axvline(x_hi, **edge_kw)

def _find_dependence_interaction_index(feat_idx: int, shap_values: np.ndarray, X_values: np.ndarray) -> Optional[int]:
    """SNP whose genotype correlates most with the primary SNP's SHAP values (SHAP interaction auto-select)."""
    y_shap = shap_values[:, feat_idx].astype(float)
    best_idx: Optional[int] = None
    best_corr = -1.0
    for j in range(X_values.shape[1]):
        if j == feat_idx:
            continue
        x_other = X_values[:, j].astype(float)
        mask = _valid_feature_values(x_other) & np.isfinite(y_shap)
        if int(mask.sum()) < 3:
            continue
        try:
            corr = float(np.corrcoef(x_other[mask], y_shap[mask])[0, 1])
        except Exception:
            continue
        if not np.isfinite(corr) or abs(corr) <= best_corr:
            continue
        best_corr = abs(corr)
        best_idx = j
    return best_idx

def _disperse_identical_y_within_genotype(x_raw: np.ndarray, y: np.ndarray, valid: np.ndarray, *, tol: float=1e-9) -> np.ndarray:
    """When all samples in a genotype bin share the same SHAP value, spread points vertically for visibility."""
    y_out = y.astype(float).copy()
    for x_val in (0, 1, 2):
        mask = valid & (np.rint(x_raw) == x_val)
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            continue
        y_sub = y_out[idx]
        if np.ptp(y_sub) > tol:
            continue
        n = len(idx)
        center = float(y_sub[0])
        step = max(0.05, abs(center) * 0.04 if center else 0.05)
        for k, ii in enumerate(idx):
            y_out[ii] = center + (k - (n - 1) / 2.0) * step
    return y_out

def _plot_shap_dependence_on_ax(ax: Axes, feat_idx: int, shap_values: np.ndarray, X_values: np.ndarray, feature_names: np.ndarray, *, compact: bool=False) -> None:
    names = [str(n) for n in feature_names]
    feature = names[feat_idx]
    x_raw = X_values[:, feat_idx].astype(float)
    y_shap = shap_values[:, feat_idx].astype(float)
    valid = _valid_feature_values(x_raw) & np.isfinite(y_shap)
    y_plot = _disperse_identical_y_within_genotype(x_raw, y_shap, valid)
    x_feat = _jitter_discrete_x(x_raw, seed=feat_idx)
    interaction_idx = _find_dependence_interaction_index(feat_idx, shap_values, X_values)
    interaction_name = names[interaction_idx] if interaction_idx is not None else None
    label_fs = 9 if compact else 11
    title_fs = 10 if compact else 12
    point_size = 16 if compact else 24
    if valid.any():
        if interaction_idx is not None:
            color_vals = X_values[:, interaction_idx].astype(float)
            color_mask = valid & _valid_feature_values(color_vals)
            if color_mask.any():
                ax.scatter(x_feat[color_mask], y_plot[color_mask], c=_genotype_codes(color_vals[color_mask]), s=point_size, **_GENOTYPE_SCATTER_KW, zorder=3)
            neutral_mask = valid & (~color_mask)
            if neutral_mask.any():
                ax.scatter(x_feat[neutral_mask], y_plot[neutral_mask], c=_SHAP_PALETTE['neutral'], s=point_size, alpha=0.45, edgecolors='none', zorder=2)
        else:
            ax.scatter(x_feat[valid], y_plot[valid], c=_SHAP_PALETTE['neutral'], s=point_size, alpha=0.65, edgecolors='none', zorder=3)
    ax.axhline(0.0, color=_SHAP_PALETTE['zero_line'], linewidth=0.8, linestyle='-', alpha=0.9, zorder=1)
    ax.grid(False)
    ax.set_xlabel(feature, fontsize=label_fs)
    ax.set_ylabel(f'SHAP value for {feature}', fontsize=label_fs)
    if interaction_name:
        title = f'{feature}\n(color: {interaction_name})' if compact else f'Dependence plot — {feature} (color: {interaction_name})'
    else:
        title = feature if compact else f'Dependence plot — {feature}'
    ax.set_title(title, fontsize=title_fs, pad=6 if compact else 10)
    ax.tick_params(labelsize=8 if compact else 10)
    _apply_genotype_dependence_x_axis(ax)
    _hide_top_right_spines(ax)

def _ensure_shap_matplotlib() -> bool:
    matplotlib_available, _ = _setup_matplotlib()
    if not matplotlib_available:
        logger.error('matplotlib not available, cannot draw SHAP plots')
    return matplotlib_available

def resolve_shap_dependence_file(importance_file: Union[str, Path]) -> Path:
    path = Path(importance_file)
    if path.name.endswith('_shap_values.txt'):
        tsv_path = path.parent / path.name.replace('_shap_values.txt', '_shap_dependence.tsv')
        npz_path = path.parent / path.name.replace('_shap_values.txt', '_shap_dependence.npz')
    else:
        tsv_path = path.parent / f'{path.stem}_shap_dependence.tsv'
        npz_path = path.parent / f'{path.stem}_shap_dependence.npz'
    if tsv_path.exists():
        return tsv_path
    return npz_path

def _hide_top_right_spines(ax: Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)

def _apply_clean_spines(fig: Any) -> None:
    for ax in fig.get_axes():
        if hasattr(ax, 'spines'):
            _hide_top_right_spines(ax)

def _load_shap_dependence_data(dependence_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(dependence_path)
    if not path.exists():
        raise FileNotFoundError(f'SHAP dependence data not found: {path}')
    if path.suffix == '.tsv':
        df = pd.read_csv(path, sep='\t')
        required = {'sample_index', 'feature', 'feature_value', 'shap_value'}
        if not required.issubset(df.columns):
            raise ValueError(f'SHAP dependence TSV missing columns: {sorted(required - set(df.columns))}')
        feature_names: List[str] = []
        for feature in df['feature'].astype(str):
            if feature not in feature_names:
                feature_names.append(feature)
        sample_indices = sorted(df['sample_index'].unique())
        n_samples = len(sample_indices)
        n_features = len(feature_names)
        feat_to_j = {name: j for j, name in enumerate(feature_names)}
        idx_map = {sample_idx: i for i, sample_idx in enumerate(sample_indices)}
        shap_values = np.full((n_samples, n_features), np.nan, dtype=float)
        x_values = np.full((n_samples, n_features), np.nan, dtype=float)
        for row in df.itertuples(index=False):
            i = idx_map[int(row.sample_index)]
            j = feat_to_j[str(row.feature)]
            shap_values[i, j] = float(row.shap_value)
            x_values[i, j] = float(row.feature_value)
        return (shap_values, x_values, np.array(feature_names, dtype=object))
    data = np.load(path, allow_pickle=True)
    return (np.asarray(data['shap_values']), np.asarray(data['X_values']), np.asarray(data['feature_names']))

def _resolve_top_feature_indices(top_features: List[str], feature_names: np.ndarray) -> Tuple[List[int], List[str]]:
    names_list = [str(n) for n in feature_names]
    name_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(names_list):
        name_to_idx[name] = idx
        norm = _normalize_snp_name(name)
        if norm not in name_to_idx:
            name_to_idx[norm] = idx
    indices: List[int] = []
    labels: List[str] = []
    seen: set = set()
    for feature in top_features:
        key = str(feature)
        idx = name_to_idx.get(key)
        if idx is None:
            idx = name_to_idx.get(_normalize_snp_name(key))
        if idx is None:
            logger.debug(f'  Skip SHAP plot for {feature}: Feature not found in dependence data')
            continue
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
        labels.append(names_list[idx])
    return (indices, labels)

def plot_shap_summary(dependence_npz: Union[str, Path], top_features: List[str], output_file: Union[str, Path], dpi: int=300) -> Optional[Path]:
    if not _ensure_shap_matplotlib():
        return None
    shap_values, X_values, feature_names = _load_shap_dependence_data(dependence_npz)
    n_samples = len(shap_values)
    logger.debug(f'Summary plot input: {n_samples:,} sample(s) x {len(feature_names):,} SNP(s) in saved data')
    indices, _ = _resolve_top_feature_indices(top_features, feature_names)
    if not indices:
        logger.debug('No valid features for SHAP summary plot')
        return None
    mean_abs = np.abs(shap_values[:, indices]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    sorted_indices = [indices[i] for i in order]
    sorted_names = [str(feature_names[i]) for i in sorted_indices]
    n_show = len(sorted_indices)
    row_width = _summary_row_width(n_samples)
    fig_h = max(5.0, 0.42 * n_show + 2.0)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), dpi=dpi)
    for rank, feat_idx in enumerate(sorted_indices):
        shap_col = shap_values[:, feat_idx].astype(float)
        feat_vals = X_values[:, feat_idx].astype(float)
        if not np.isfinite(shap_col).any():
            continue
        y_center = float(n_show - 1 - rank)
        _plot_shap_summary_row(ax, shap_col, feat_vals, y_center, row_width=row_width, seed=feat_idx)
    shap_subset = shap_values[:, sorted_indices].astype(float)
    finite_mask = np.isfinite(shap_subset)
    edge_kw = {'color': _SHAP_PALETTE['zero_line'], 'linewidth': 0.9, 'linestyle': '-', 'alpha': 0.95, 'zorder': 0}
    if finite_mask.any():
        data_min = float(np.min(shap_subset[finite_mask]))
        data_max = float(np.max(shap_subset[finite_mask]))
        span = data_max - data_min
        pad = max(span * 0.06, 0.02) if span > 0 else 0.1
        x_left = data_min - pad
        x_right = data_max + pad
        ax.set_xlim(x_left, x_right)
        ax.axvline(x_left, **edge_kw)
        ax.axvline(x_right, **edge_kw)
    ax.axvline(0.0, color=_SHAP_PALETTE['zero_line'], linewidth=0.9, linestyle='-', alpha=0.95, zorder=0)
    ax.set_yticks(list(range(n_show)))
    ax.set_yticklabels(list(reversed(sorted_names)), fontsize=10)
    ax.set_xlabel('SHAP value (per-sample distribution)', fontsize=13)
    ax.set_ylabel('SNP', fontsize=13)
    ax.set_title(f'Summary plot — Top {n_show} SNPs · {n_samples:,} samples', fontsize=15, pad=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(False)
    fig.subplots_adjust(right=0.82)
    _add_genotype_legend(ax, title='SNP genotype')
    _apply_clean_spines(fig)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.debug(f'SHAP summary plot saved: {out_path} ({n_show} SNPs, {n_samples:,} samples per SNP row)')
    return out_path

def plot_top_snps_dependence(dependence_npz: Union[str, Path], top_features: List[str], output_dir: Union[str, Path], output_prefix: Union[str, Path], dpi: int=300) -> List[Path]:
    if not _ensure_shap_matplotlib():
        return []
    shap_values, X_values, feature_names = _load_shap_dependence_data(dependence_npz)
    n_samples = len(shap_values)
    logger.debug(f'Dependence plot input: {n_samples:,} sample(s) x {len(feature_names):,} SNP(s) in saved data')
    prefix = Path(output_prefix)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices, plotted_features = _resolve_top_feature_indices(top_features, feature_names)
    if not indices:
        logger.debug('No valid features for SHAP dependence plot (check feature name match between shap_values.txt and shap_dependence.tsv)')
        return []
    n_plots = len(indices)
    n_cols = min(4, n_plots)
    n_rows = int(math.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.0 * n_rows + 0.6), dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()
    n_failed = 0
    for i, feat_idx in enumerate(indices):
        try:
            _plot_shap_dependence_on_ax(axes_flat[i], feat_idx, shap_values, X_values, feature_names, compact=True)
        except Exception as e:
            n_failed += 1
            logger.error(f'  Dependence plot failed for {plotted_features[i]}: {e}', exc_info=True)
            axes_flat[i].set_visible(False)
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)
    if n_failed == n_plots:
        plt.close(fig)
        logger.error('All SHAP dependence panels failed to render')
        return []
    fig.suptitle(f'Dependence plot — Top {n_plots} SNPs', fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.5, 0.965, 'X: primary SNP genotype (0/1/2) | Y: SHAP value | Color: correlated SNP genotype (0/1/2)', ha='center', va='top', fontsize=9, transform=fig.transFigure)
    fig.legend(handles=_genotype_legend_handles(), title='Correlated SNP', loc='upper right', bbox_to_anchor=(0.99, 0.93), fontsize=8, title_fontsize=8, frameon=False, handleheight=0.9, handlelength=1.0)
    fig.subplots_adjust(top=0.90, hspace=0.58, wspace=0.40)
    _apply_clean_spines(fig)
    out_file = output_dir / f'{prefix.name}_dependence.png'
    fig.savefig(out_file, dpi=dpi, bbox_inches='tight', pad_inches=0.25, facecolor='white')
    plt.close(fig)
    logger.debug(f'SHAP dependence plot saved: {out_file} ({n_plots - n_failed}/{n_plots} panels, {n_samples:,} samples per panel)')
    return [out_file]

# ---------------------------------------------------------------------------
# 4. Performance / CV plots (CLI -I)
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> Tuple[bool, Any]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            from G2PInsight.bin.font_utils import setup_matplotlib_font
            setup_matplotlib_font()
        except ImportError:
            pass
        return (True, plt)
    except ImportError:
        logger.debug('  matplotlib not installed, skipping plotting functionality')
        return (False, None)

def _ensure_numpy_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.values
    else:
        return np.array(data)

def load_cv_results_from_model_dir(model_dir: Union[str, Path], model_type: str) -> Optional[Dict]:
    model_dir = Path(model_dir)
    json_path = model_dir / f'{model_type}_cv_results.json'
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f'  Failed to load CV results from {json_path}: {e}')
        return None

def load_plotting_data(plotting_data_file: Union[str, Path]) -> Dict:
    plotting_data_file = Path(plotting_data_file)
    if not plotting_data_file.exists():
        raise FileNotFoundError(f'Plotting data file not found: {plotting_data_file}')
    if plotting_data_file.suffix == '.json' or plotting_data_file.name.endswith('_plotting_data.json'):
        meta_path = plotting_data_file if plotting_data_file.suffix == '.json' else plotting_data_file.with_suffix('.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        model_dir = meta_path.parent
        result = {'model_type': meta.get('model_type', 'Unknown'), 'task_type': meta.get('task_type', 'regression'), 'publication_quality': bool(meta.get('publication_quality', True)), 'split_mode': meta.get('split_mode', 'cv'), 'y_test': None, 'y_pred': None, 'y_prob': None, 'cv_results': None, 'evaluation_results': None}
        pred_name = meta.get('test_predictions_file')
        if pred_name:
            pred_df = pd.read_csv(model_dir / pred_name, sep='\t')
            result['y_test'] = pred_df['y_true'].values
            result['y_pred'] = pred_df['y_pred'].values
            prob_cols = [c for c in pred_df.columns if c.startswith('y_prob')]
            if len(prob_cols) == 1:
                result['y_prob'] = pred_df[prob_cols[0]].values
            elif len(prob_cols) > 1:
                result['y_prob'] = pred_df[sorted(prob_cols)].values
        cv_file = meta.get('cv_results_file')
        if cv_file:
            result['cv_results'] = load_cv_results_from_model_dir(model_dir, result['model_type'])
        eval_file = meta.get('evaluation_results_file')
        if eval_file:
            eval_path = model_dir / eval_file
            if eval_path.exists():
                with open(eval_path, 'r', encoding='utf-8') as f:
                    result['evaluation_results'] = json.load(f)
        return result
    data = np.load(plotting_data_file, allow_pickle=True)
    result = {'y_test': data['y_test'], 'y_pred': data['y_pred'], 'y_prob': data.get('y_prob', None), 'model_type': str(data['model_type'][0]) if 'model_type' in data else 'Unknown', 'task_type': str(data['task_type'][0]) if 'task_type' in data else 'regression', 'publication_quality': bool(data['publication_quality'][0]) if 'publication_quality' in data else True, 'split_mode': str(data['split_mode'][0]) if 'split_mode' in data else 'cv'}
    if 'cv_results' in data:
        cv_results_bytes = data['cv_results'][0]
        result['cv_results'] = pickle.loads(cv_results_bytes)
    else:
        result['cv_results'] = None
    if result['cv_results'] is None:
        result['cv_results'] = load_cv_results_from_model_dir(plotting_data_file.parent, result['model_type'])
    if 'evaluation_results' in data:
        result['evaluation_results'] = pickle.loads(data['evaluation_results'][0])
    else:
        result['evaluation_results'] = None
    return result

def plot_test_set_metrics(test_metrics: Dict, output_dir: Union[str, Path], model_type: str, task_type: str, output_prefix: Optional[Union[str, Path]]=None) -> Optional[Path]:
    output_dir = Path(output_dir)
    output_prefix_path = _resolve_output_prefix(output_dir, output_prefix)
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        return None
    if task_type == 'regression':
        _warn_missing_regression_test_metrics(test_metrics)
        specs = _regression_test_bar_specs()
        corr_specs = [item for item in specs if item[0] in ('pearson_correlation', 'r2')]
        err_specs = [item for item in specs if item[0] in ('rmse', 'mae')]
        corr_names, corr_vals = _collect_metric_bar_values(test_metrics, corr_specs)
        err_names, err_vals = _collect_metric_bar_values(test_metrics, err_specs)
        if not corr_names and not err_names:
            logger.debug('  Held-out test metrics plot: no plottable metrics')
            return None
        n_panels = int(bool(corr_names)) + int(bool(err_names))
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.6), squeeze=False)
        axes_flat = axes.flatten()
        panel_idx = 0
        if corr_names:
            corr_min = min(corr_vals)
            corr_max = max(corr_vals)
            if corr_min >= 0.0:
                y_lo = max(0.0, corr_min - 0.08)
                y_hi = min(1.08, corr_max + 0.06)
            else:
                y_lo = max(-1.05, corr_min - 0.08)
                y_hi = min(1.08, corr_max + 0.06)
            _plot_test_metric_bar_panel(axes_flat[panel_idx], corr_names, corr_vals, ylabel='Correlation score', ylim=(y_lo, y_hi))
            panel_idx += 1
        if err_names:
            err_max = max(err_vals)
            _plot_test_metric_bar_panel(axes_flat[panel_idx], err_names, err_vals, ylabel='Error (phenotype units)', ylim=(0.0, max(err_max * 1.18, 1e-6)))
        fig.suptitle(f'{model_type} Held-out Test Set Metrics (regression)', fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
    else:
        specs = _task_metric_specs(task_type)
        metric_names, test_vals = _collect_metric_bar_values(test_metrics, specs)
        if not metric_names:
            logger.debug('  Held-out test metrics plot: no plottable metrics')
            return None
        x = np.arange(len(metric_names))
        fig, ax = plt.subplots(figsize=(max(6, len(metric_names) * 1.6), 5))
        bars = ax.bar(x, test_vals, width=_TEST_METRIC_BAR_WIDTH, color=_TEST_METRIC_PALETTE['test'], **_TEST_METRIC_BAR_KW, label='Held-out test set')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=20, ha='right')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_type} Held-out Test Set Metrics ({task_type})', fontsize=13, fontweight='bold')
        _test_metric_finish_figure(ax, fig, [bars], task_type)
        _hide_top_right_spines(ax)
        fig.tight_layout()
    plot_file = Path(f'{output_prefix_path}_test_metrics.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.debug(f'   Held-out test metrics plot saved: {plot_file}')
    return plot_file

def plot_confusion_matrix(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray, output_dir: Union[str, Path], model_type: str, output_prefix: Optional[Union[str, Path]]=None, *, dpi: int=300) -> Optional[Path]:
    """Plot confusion matrix for classification tasks (held-out test set)."""
    output_dir = Path(output_dir)
    output_prefix_path = _resolve_output_prefix(output_dir, output_prefix)
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        return None
    y_true_values = _ensure_numpy_array(y_true)
    y_pred_values = np.asarray(y_pred)
    if len(y_true_values) == 0 or len(y_pred_values) == 0:
        logger.debug('  Confusion matrix: empty true or predicted labels')
        return None
    if len(y_true_values) != len(y_pred_values):
        logger.debug(f'  Confusion matrix: label length mismatch ({len(y_true_values)} vs {len(y_pred_values)})')
        return None
    try:
        from sklearn.metrics import confusion_matrix, accuracy_score
        labels = sorted(np.unique(np.concatenate([y_true_values, y_pred_values])), key=lambda x: (isinstance(x, str), x))
        cm = confusion_matrix(y_true_values, y_pred_values, labels=labels)
        if cm.size == 0:
            logger.debug('  Confusion matrix: no data to plot')
            return None
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
        annot = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                pct = cm_norm[i, j] * 100.0 if row_sums[i, 0] > 0 else 0.0
                annot[i, j] = f'{count}\n({pct:.1f}%)'
        n_classes = len(labels)
        fig_size = max(4.5, min(12.0, 0.85 * n_classes + 3.2))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88), dpi=dpi)
        sns.heatmap(cm, annot=annot, fmt='', cmap=_CONFUSION_CMAP, cbar=True, square=True, linewidths=0.6, linecolor='white', xticklabels=[str(l) for l in labels], yticklabels=[str(l) for l in labels], ax=ax, vmin=0)
        acc = accuracy_score(y_true_values, y_pred_values)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title(f'{model_type} Confusion Matrix (held-out test)\nAccuracy = {acc:.4f}', fontsize=13, fontweight='bold', pad=12)
        ax.tick_params(axis='both', labelsize=10)
        _hide_top_right_spines(ax)
        fig.tight_layout()
        plot_file = Path(f'{output_prefix_path}_confusion_matrix.png')
        fig.savefig(plot_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.debug(f'   Confusion matrix plot saved: {plot_file}')
        return plot_file
    except Exception as e:
        logger.debug(f'  Failed to draw confusion matrix: {e}')
        try:
            plt.close()
        except Exception:
            pass
        return None

def plot_cv_average_metrics(cv_results: Dict, output_dir: Union[str, Path], model_type: str, task_type: str, output_prefix: Optional[Union[str, Path]]=None) -> Optional[Path]:
    if task_type == 'regression':
        logger.debug('  Skipping CV mean bar chart for regression (use cv_training_curves for per-fold Pearson r)')
        return None
    avg_metrics = cv_results.get('average_metrics') or {}
    if not avg_metrics:
        logger.debug('  Training-set CV average metrics plot: no data')
        return None
    output_dir = Path(output_dir)
    output_prefix_path = _resolve_output_prefix(output_dir, output_prefix)
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        return None
    specs = _task_metric_specs(task_type)
    metric_names: List[str] = []
    vals: List[float] = []
    for key, label, _ in specs:
        v = _metric_float(avg_metrics.get(key))
        if v is None:
            continue
        metric_names.append(label)
        vals.append(v)
    if not metric_names:
        return None
    x = np.arange(len(metric_names))
    fig, ax = plt.subplots(figsize=(max(6, len(metric_names) * 1.6), 5))
    ax.bar(x, vals, width=0.55, color='#3E5C76', alpha=0.92, label='CV mean (training set)')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha='right')
    ax.set_ylabel('Score')
    if task_type == 'classification':
        ax.set_ylim(0.0, 1.05)
    elif metric_names and all(name == 'Pearson r' for name in metric_names):
        ax.set_ylim(-1.05, 1.05)
    n_folds = cv_results.get('n_folds', '?')
    ax.set_title(f'{model_type} {n_folds}-Fold CV Mean on Training Set ({task_type})', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=False)
    ax.grid(axis='y', alpha=0.3)
    _hide_top_right_spines(ax)
    fig.tight_layout()
    plot_file = Path(f'{output_prefix_path}_cv_average_metrics.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.debug(f'   Training-set CV average metrics plot saved: {plot_file}')
    return plot_file

def plot_performance_curves(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray, y_prob: Optional[np.ndarray], output_dir: Union[str, Path], model_type: str, task_type: str, publication_quality: bool=True, output_prefix: Optional[Union[str, Path]]=None) -> None:
    output_dir = Path(output_dir)
    output_prefix_path: Optional[Path] = None
    if output_prefix is not None:
        output_prefix_path = Path(output_prefix)
        if not output_prefix_path.parent or str(output_prefix_path.parent) == '.':
            output_prefix_path = output_dir / output_prefix_path.name
        output_prefix_path = output_dir / output_prefix_path.name
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        return
    if publication_quality:
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'], 'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.titlesize': 12, 'lines.linewidth': 2, 'axes.linewidth': 1.2, 'grid.linewidth': 0.8, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1})
    y_true_values = _ensure_numpy_array(y_true)
    y_pred_values = np.array(y_pred)
    if task_type == 'regression':
        if publication_quality:
            fig, axes = plt.subplots(2, 2, figsize=(7, 6))
            fig.suptitle(f'{model_type} Model Performance (Regression, held-out test)', fontsize=12, fontweight='bold')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{model_type} Performance (Regression, held-out test)', fontsize=16, fontweight='bold')
        try:
            from scipy.stats import pearsonr
            pearson_corr, pearson_p = pearsonr(y_true_values, y_pred_values)
            if np.isnan(pearson_corr):
                pearson_corr = None
                pearson_p = None
            else:
                pearson_corr = float(pearson_corr)
                pearson_p = float(pearson_p) if not np.isnan(pearson_p) else None
        except Exception as e:
            logger.debug(f'  Failed to calculate Pearson correlation coefficient: {str(e)}')
            pearson_corr = None
            pearson_p = None
        residuals = y_true_values - y_pred_values
        ax1 = axes[0, 0]
        if publication_quality:
            ax1.scatter(y_true_values, y_pred_values, alpha=0.6, s=25, color='#2E86AB', edgecolors='black', linewidths=0.3)
            min_val = min(np.min(y_true_values), np.min(y_pred_values))
            max_val = max(np.max(y_true_values), np.max(y_pred_values))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal (y=x)', dashes=(5, 3))
            try:
                z = np.polyfit(y_true_values, y_pred_values, 1)
                p = np.poly1d(z)
                ax1.plot(y_true_values, p(y_true_values), '#E63946', linewidth=2, label=f'Fit (slope={z[0]:.3f})')
            except:
                pass
            ax1.set_xlabel('True Values', fontsize=10, fontweight='normal')
            ax1.set_ylabel('Predicted Values', fontsize=10, fontweight='normal')
            ax1.set_title('(A) Predicted vs True Values', fontsize=11, fontweight='bold')
            if pearson_corr is not None:
                p_text = f'p = {pearson_p:.4g}' if pearson_p is not None else 'p = N/A'
                ax1.text(0.05, 0.95, f'r = {pearson_corr:.4f}\n{p_text}\nR² = {pearson_corr ** 2:.4f}', transform=ax1.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9, pad=0.3))
            ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9, fontsize=8, handlelength=1.5)
        else:
            ax1.scatter(y_true_values, y_pred_values, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
            min_val = min(np.min(y_true_values), np.min(y_pred_values))
            max_val = max(np.max(y_true_values), np.max(y_pred_values))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Prediction (y=x)')
            try:
                z = np.polyfit(y_true_values, y_pred_values, 1)
                p = np.poly1d(z)
                ax1.plot(y_true_values, p(y_true_values), 'b-', linewidth=1.5, alpha=0.7, label=f'Fit Line (slope={z[0]:.3f})')
            except:
                pass
            ax1.set_xlabel('True Values', fontsize=12)
            ax1.set_ylabel('Predicted Values', fontsize=12)
            ax1.set_title('Predicted vs True Values', fontsize=13)
            if pearson_corr is not None:
                p_text = f'p = {pearson_p:.4g}' if pearson_p is not None else 'p = N/A'
                ax1.text(0.05, 0.95, f'r = {pearson_corr:.4f}\n{p_text}\nR² = {pearson_corr ** 2:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        if publication_quality:
            ax2.scatter(y_pred_values, residuals, alpha=0.6, s=25, color='#2E86AB', edgecolors='black', linewidths=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, label='Zero', dashes=(5, 3))
            ax2.set_xlabel('Predicted Values', fontsize=10, fontweight='normal')
            ax2.set_ylabel('Residuals', fontsize=10, fontweight='normal')
            ax2.set_title('(B) Residual Plot', fontsize=11, fontweight='bold')
            ax2.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9, fontsize=8, handlelength=1.5)
        else:
            ax2.scatter(y_pred_values, residuals, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
            ax2.set_xlabel('Predicted Values', fontsize=12)
            ax2.set_ylabel('Residuals (True - Predicted)', fontsize=12)
            ax2.set_title('Residual Plot', fontsize=13)
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        if publication_quality:
            ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='#A23B72', linewidth=0.8)
            ax3.axvline(x=0, color='k', linestyle='--', linewidth=2, label='Zero', dashes=(5, 3))
            ax3.axvline(x=mean_residual, color='#E63946', linestyle='--', linewidth=2, label=f'Mean={mean_residual:.4f}', dashes=(5, 3))
            ax3.set_xlabel('Residuals', fontsize=10, fontweight='normal')
            ax3.set_ylabel('Frequency', fontsize=10, fontweight='normal')
            ax3.set_title('(C) Residual Distribution', fontsize=11, fontweight='bold')
            y_min, y_max = ax3.get_ylim()
            text_x = (0 + mean_residual) / 2
            ax3.text(text_x, y_min + (y_max - y_min) * 0.05, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', fontsize=8, verticalalignment='bottom', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9, pad=0.3))
            ax3.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9, fontsize=8, handlelength=1.5)
        else:
            ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
            ax3.axvline(x=mean_residual, color='g', linestyle='--', linewidth=2, label=f'Mean={mean_residual:.4f}')
            ax3.set_xlabel('Residuals', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Residual Distribution', fontsize=13)
            y_min, y_max = ax3.get_ylim()
            text_x = (0 + mean_residual) / 2
            ax3.text(text_x, y_min + (y_max - y_min) * 0.05, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', fontsize=11, verticalalignment='bottom', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax4 = axes[1, 1]
        try:
            from scipy import stats
            stats.probplot(residuals, dist='norm', plot=ax4)
            if publication_quality:
                lines = ax4.get_lines()
                if len(lines) >= 2:
                    lines[0].set_linewidth(2)
                    lines[0].set_color('#2E86AB')
                    lines[1].set_linewidth(2)
                    lines[1].set_color('#E63946')
                    lines[1].set_linestyle('--')
                ax4.set_title('(D) Q-Q Plot', fontsize=11, fontweight='bold')
                ax4.set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='normal')
                ax4.set_ylabel('Sample Quantiles', fontsize=10, fontweight='normal')
            else:
                ax4.set_title('Q-Q Plot (Residual Normality Test)', fontsize=13)
            ax4.grid(True, alpha=0.3)
        except ImportError:
            ax4.text(0.5, 0.5, 'Q-Q Plot requires scipy', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Q-Q Plot (scipy not available)', fontsize=13)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Q-Q Plot failed:\n{str(e)}', ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Q-Q Plot (Error)', fontsize=13)
        plt.tight_layout()
        _apply_clean_spines(plt.gcf())
        if publication_quality:
            if output_prefix_path is not None:
                plot_file_pdf = Path(f'{output_prefix_path}_performance_curves.pdf')
                plot_file_png = Path(f'{output_prefix_path}_performance_curves.png')
            else:
                plot_file_pdf = output_dir / 'performance_curves.pdf'
                plot_file_png = output_dir / 'performance_curves.png'
            plt.savefig(plot_file_pdf, dpi=300, bbox_inches='tight', format='pdf')
            plt.savefig(plot_file_png, dpi=300, bbox_inches='tight', format='png')
            logger.debug(f'  Publication-quality plots saved: {plot_file_pdf} and {plot_file_png}')
        else:
            if output_prefix_path is not None:
                plot_file = Path(f'{output_prefix_path}_performance_curves.png')
            else:
                plot_file = output_dir / 'performance_curves.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        if y_prob is None:
            logger.debug(' Classification task requires predicted probabilities, skip performance curves')
            return
        unique_classes = np.unique(y_true_values)
        n_classes = len(unique_classes)
        y_prob_arr = np.asarray(y_prob)
        if y_prob_arr.ndim == 1:
            y_prob_binary = y_prob_arr
        elif n_classes == 2:
            prob_col = 1 if y_prob_arr.shape[1] > 1 else 0
            y_prob_binary = y_prob_arr[:, prob_col]
        else:
            y_prob_binary = y_prob_arr[:, 0]
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, recall_score
            if publication_quality:
                fig1, axes = plt.subplots(1, 3, figsize=(14, 4))
                fig1.suptitle(f'{model_type} Performance Curves (Classification, held-out test)', fontsize=12, fontweight='bold')
            else:
                fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig1.suptitle(f'{model_type} Performance Curves (Classification, held-out test)', fontsize=16, fontweight='bold')
            if n_classes == 2:
                thresholds = np.linspace(0, 1, 100)
                accuracies = []
                recalls = []
                f1_scores = []
                fpr, tpr, roc_thresholds = roc_curve(y_true_values, y_prob_binary)
                roc_auc = auc(fpr, tpr)
                for threshold in thresholds:
                    y_pred_thresh = (y_prob_binary >= threshold).astype(int)
                    accuracies.append(accuracy_score(y_true_values, y_pred_thresh))
                    recalls.append(recall_score(y_true_values, y_pred_thresh, zero_division=0))
                    f1_scores.append(f1_score(y_true_values, y_pred_thresh, zero_division=0))
                max_acc_idx = np.argmax(accuracies)
                max_acc_threshold = thresholds[max_acc_idx]
                max_acc_value = accuracies[max_acc_idx]
                max_f1_idx = np.argmax(f1_scores)
                max_f1_threshold = thresholds[max_f1_idx]
                max_f1_value = f1_scores[max_f1_idx]
                ax1 = axes[0]
                ax1.plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess', alpha=0.5)
                if publication_quality:
                    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=10)
                    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=10)
                    ax1.set_title('ROC Curve (AUC)', fontsize=11)
                    ax1.legend(loc='lower right', fontsize=8)
                else:
                    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
                    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
                    ax1.set_title('ROC Curve (AUC)', fontsize=13)
                    ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])
                ax2 = axes[1]
                ax2.plot(thresholds, accuracies, 'r-', lw=2, label='Accuracy')
                ax2.axvline(x=max_acc_threshold, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Max Accuracy: {max_acc_value:.4f} at {max_acc_threshold:.3f}')
                if publication_quality:
                    ax2.set_xlabel('Threshold', fontsize=10)
                    ax2.set_ylabel('Accuracy', fontsize=10)
                    ax2.set_title('Accuracy Curve', fontsize=11)
                    ax2.legend(loc='best', fontsize=8)
                else:
                    ax2.set_xlabel('Threshold', fontsize=12)
                    ax2.set_ylabel('Accuracy', fontsize=12)
                    ax2.set_title('Accuracy Curve', fontsize=13)
                    ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                ax3 = axes[2]
                ax3.plot(thresholds, f1_scores, 'm-', lw=2, label='F1 Score')
                ax3.axvline(x=max_f1_threshold, color='m', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Max F1: {max_f1_value:.4f} at {max_f1_threshold:.3f}')
                if publication_quality:
                    ax3.set_xlabel('Threshold', fontsize=10)
                    ax3.set_ylabel('F1 Score', fontsize=10)
                    ax3.set_title('F1 Score Curve', fontsize=11)
                    ax3.legend(loc='best', fontsize=8)
                else:
                    ax3.set_xlabel('Threshold', fontsize=12)
                    ax3.set_ylabel('F1 Score', fontsize=12)
                    ax3.set_title('F1 Score Curve', fontsize=13)
                    ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim([0, 1])
                ax3.set_ylim([0, 1])
            else:
                from sklearn.preprocessing import label_binarize
                y_true_binarized = label_binarize(y_true_values, classes=unique_classes)
                ax1 = axes[0]
                for i, class_label in enumerate(unique_classes):
                    if y_prob.shape[1] > i:
                        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax1.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.4f})')
                ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
                if publication_quality:
                    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=10)
                    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=10)
                    ax1.set_title('ROC Curve (AUC)', fontsize=11)
                    ax1.legend(loc='lower right', fontsize=8)
                else:
                    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
                    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
                    ax1.set_title('ROC Curve (AUC)', fontsize=13)
                    ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)
                for ax in [axes[1], axes[2]]:
                    ax.axis('off')
                    if publication_quality:
                        ax.text(0.5, 0.5, 'Multi-class metrics\nnot implemented', ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Multi-class metrics\nnot implemented', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            plt.tight_layout()
            _apply_clean_spines(plt.gcf())
            if output_prefix_path is not None:
                plot_file1 = Path(f'{output_prefix_path}_performance_curves.png')
            else:
                plot_file1 = output_dir / 'performance_curves.png'
            plt.savefig(plot_file1, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.debug(f' Failed to draw performance curves: {str(e)}')

def plot_cv_training_curves(cv_results: Dict, output_dir: Union[str, Path], model_type: str, task_type: str, output_prefix: Optional[Union[str, Path]]=None) -> None:
    output_dir = Path(output_dir)
    output_prefix_path: Optional[Path] = None
    if output_prefix is not None:
        output_prefix_path = Path(output_prefix)
        if not output_prefix_path.parent or str(output_prefix_path.parent) == '.':
            output_prefix_path = output_dir / output_prefix_path.name
        output_prefix_path = output_dir / output_prefix_path.name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f'  Cannot create output directory {output_dir}: {str(e)}')
        return
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        logger.error('  matplotlib not available, cannot plot cross-validation curves')
        return
    n_folds = len(cv_results.get('fold_metrics', []))
    if n_folds == 0:
        logger.debug('  No cross-validation results, skipping training curve plotting')
        return
    if task_type == 'regression':
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5.5))
        fig.suptitle(f'{model_type} · {n_folds}-Fold CV · Pearson r (training set)', fontsize=14, fontweight='bold', y=0.98)
        pearson_corrs = []
        for i in range(n_folds):
            fold_metric = cv_results['fold_metrics'][i]
            corr_val = fold_metric.get('pearson_correlation')
            if isinstance(corr_val, (int, float)) and (not np.isnan(corr_val)):
                pearson_corrs.append(corr_val)
            else:
                logger.debug(f'  Fold {i + 1}: Invalid correlation value: {corr_val}')
        if pearson_corrs and len(pearson_corrs) > 0:
            fig.text(0.5, 0.91, _cv_brief_stats(pearson_corrs), ha='center', va='top', fontsize=10, color='#444444', transform=fig.transFigure)
            _cv_draw_single_barplot(ax1, pearson_corrs, _CV_PALETTE['corr'], y_min=0.0, y_max=1.0, stats_text=False, show_stats=False, show_mean_legend=False, highlight_outliers=True, y_grid=True)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            _hide_top_right_spines(ax1)
        ax1.set_ylabel('Pearson Correlation', fontsize=12)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_type} Model {n_folds}-Fold CV on Training Set (Classification, diagnostic)', fontsize=16, fontweight='bold', y=0.995)
        accuracies = []
        recalls = []
        f1_scores = []
        aucs = []
        for i in range(n_folds):
            fold_metric = cv_results['fold_metrics'][i]
            acc_val = fold_metric.get('accuracy')
            rec_val = fold_metric.get('recall')
            f1_val = fold_metric.get('f1')
            auc_val = fold_metric.get('auc')
            if isinstance(acc_val, (int, float)) and (not np.isnan(acc_val)):
                accuracies.append(acc_val)
            if isinstance(rec_val, (int, float)) and (not np.isnan(rec_val)):
                recalls.append(rec_val)
            if isinstance(f1_val, (int, float)) and (not np.isnan(f1_val)):
                f1_scores.append(f1_val)
            if isinstance(auc_val, (int, float)) and (not np.isnan(auc_val)) and (auc_val != 'N/A'):
                aucs.append(auc_val)
        ax1 = axes[0, 0]
        if accuracies and len(accuracies) > 0:
            _cv_draw_single_barplot(ax1, accuracies, _CV_PALETTE['accuracy'], y_min=0.0, y_max=1.05)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            _hide_top_right_spines(ax1)
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy by Fold', fontsize=13, fontweight='bold', pad=15)
        ax2 = axes[0, 1]
        recall_f1_sets: List[List[float]] = []
        recall_f1_labels: List[str] = []
        recall_f1_colors: List[str] = []
        if recalls and len(recalls) > 0:
            recall_f1_sets.append(recalls)
            recall_f1_labels.append('Recall')
            recall_f1_colors.append(_CV_PALETTE['recall'])
        if f1_scores and len(f1_scores) > 0:
            recall_f1_sets.append(f1_scores)
            recall_f1_labels.append('F1 Score')
            recall_f1_colors.append(_CV_PALETTE['f1'])
        if recall_f1_sets:
            stats_parts: List[str] = []
            if recalls and len(recalls) > 0:
                stats_parts.append('Recall:\n' + _cv_summary_stats(recalls).replace('\n', '\n  '))
            if f1_scores and len(f1_scores) > 0:
                if stats_parts:
                    stats_parts.append('')
                stats_parts.append('F1:\n' + _cv_summary_stats(f1_scores).replace('\n', '\n  '))
            _cv_draw_group_barplot(ax2, recall_f1_sets, recall_f1_labels, recall_f1_colors, y_min=0.0, y_max=1.05, stats_text='\n\n'.join(stats_parts), annotate=False, legend_horizontal=True)
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            _hide_top_right_spines(ax2)
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Recall and F1 Score by Fold', fontsize=13, fontweight='bold', pad=15)
        ax3 = axes[1, 0]
        if aucs and len(aucs) > 0:
            _cv_draw_single_barplot(ax3, aucs, _CV_PALETTE['auc'], y_min=0.0, y_max=1.05)
        else:
            ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            _hide_top_right_spines(ax3)
        ax3.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax3.set_title('AUC by Fold', fontsize=13, fontweight='bold', pad=15)
        ax4 = axes[1, 1]
        all_metrics = []
        all_labels = []
        all_colors = []
        if accuracies and len(accuracies) > 0:
            all_metrics.append(accuracies)
            all_labels.append('Accuracy')
            all_colors.append(_CV_PALETTE['accuracy'])
        if recalls and len(recalls) > 0:
            all_metrics.append(recalls)
            all_labels.append('Recall')
            all_colors.append(_CV_PALETTE['recall'])
        if f1_scores and len(f1_scores) > 0:
            all_metrics.append(f1_scores)
            all_labels.append('F1 Score')
            all_colors.append(_CV_PALETTE['f1'])
        if aucs and len(aucs) > 0:
            all_metrics.append(aucs)
            all_labels.append('AUC')
            all_colors.append(_CV_PALETTE['auc'])
        if all_metrics:
            _cv_draw_group_barplot(ax4, all_metrics, all_labels, all_colors, y_min=0.0, y_max=1.05, legend_horizontal=True)
        else:
            ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            _hide_top_right_spines(ax4)
        ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax4.set_title('All Metrics Comparison by Fold', fontsize=13, fontweight='bold', pad=15)
    try:
        if task_type == 'regression':
            plt.tight_layout(rect=[0, 0, 1, 0.88])
        else:
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.34, wspace=0.30)
        _apply_clean_spines(plt.gcf())
        if output_prefix_path is not None:
            plot_file = Path(f'{output_prefix_path}_cv_training_curves.png')
        else:
            plot_file = output_dir / 'cv_training_curves.png'
        try:
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', format='png')
            if plot_file.exists() and plot_file.stat().st_size > 0:
                logger.debug(f'   Cross-validation bar plot saved: {plot_file} (size: {plot_file.stat().st_size:,} bytes)')
            else:
                logger.error(f'   File save failed or file is empty: {plot_file}')
                if plot_file.exists():
                    plot_file.unlink()
        except Exception as save_error:
            logger.error(f'   Failed to save cross-validation curves: {str(save_error)}', exc_info=True)
            try:
                plot_file_backup = output_dir / 'cv_training_curves_backup.png'
                plt.savefig(plot_file_backup, dpi=150, bbox_inches='tight', facecolor='white', format='png')
                logger.debug(f'   Backup file saved: {plot_file_backup}')
            except Exception as backup_error:
                logger.error(f'   Backup save also failed: {str(backup_error)}')
        finally:
            plt.close()
    except Exception as e:
        logger.error(f'   Error occurred while plotting cross-validation curves: {str(e)}', exc_info=True)
        try:
            plt.close()
        except:
            pass

def plot_model_performance_from_file(plotting_data_file: Union[str, Path], output_dir: Optional[Union[str, Path]]=None, publication_quality: Optional[bool]=None, output_prefix: Optional[Union[str, Path]]=None) -> int:
    try:
        plotting_data = load_plotting_data(plotting_data_file)
        plotting_data_file = Path(plotting_data_file)
        if output_dir is None:
            output_dir = plotting_data_file.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        if output_prefix is None:
            raise ValueError('output_prefix is required for plot_model_performance_from_file')
        output_prefix_path = Path(output_prefix)
        if not output_prefix_path.parent or str(output_prefix_path.parent) == '.':
            output_prefix_path = output_dir / output_prefix_path.name
        output_prefix_path = output_dir / output_prefix_path.name
        if publication_quality is None:
            publication_quality = plotting_data['publication_quality']
        if plotting_data['y_test'] is not None and plotting_data['y_pred'] is not None:
            plot_performance_curves(plotting_data['y_test'], plotting_data['y_pred'], plotting_data['y_prob'], output_dir, plotting_data['model_type'], plotting_data['task_type'], publication_quality, output_prefix=output_prefix_path)
            logger.debug(f'Performance evaluation curves generated: {output_prefix_path}_performance_curves.png')
            if plotting_data['task_type'] == 'classification':
                plot_confusion_matrix(plotting_data['y_test'], plotting_data['y_pred'], output_dir, plotting_data['model_type'], output_prefix=output_prefix_path)
        if plotting_data['cv_results'] is not None:
            cv_results = plotting_data['cv_results']
            test_metrics = cv_results.get('test_metrics') or {}
            if plotting_data.get('evaluation_results') is not None:
                eval_payload = plotting_data['evaluation_results']
                if not test_metrics:
                    test_metrics = eval_payload.get('test_metrics') or {}
            if test_metrics:
                plot_test_set_metrics(test_metrics, output_dir, plotting_data['model_type'], plotting_data['task_type'], output_prefix=output_prefix_path)
            plot_cv_average_metrics(cv_results, output_dir, plotting_data['model_type'], plotting_data['task_type'], output_prefix=output_prefix_path)
            plot_cv_training_curves(cv_results, output_dir, plotting_data['model_type'], plotting_data['task_type'], output_prefix=output_prefix_path)
            logger.debug(f'Cross-validation curves generated (training-set CV, diagnostic): {output_prefix_path}_cv_training_curves.png')
        return 0
    except FileNotFoundError as e:
        logger.error(f'File error: {str(e)}')
        return 2
    except Exception as e:
        logger.error(f'Plotting failed: {str(e)}', exc_info=True)
        return 1
