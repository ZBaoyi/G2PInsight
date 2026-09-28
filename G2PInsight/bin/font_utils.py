from __future__ import annotations
import importlib
import logging
from functools import lru_cache
from typing import Any, Optional, Tuple, cast
logger = logging.getLogger(__name__)
_FONT_MANAGER_LOGGER = 'matplotlib.font_manager'

def _load_module(name: str) -> Any:
    return importlib.import_module(name)

def _collect_matplotlib_font_names() -> set:
    fm = _load_module('matplotlib.font_manager')
    font_manager = getattr(fm, 'fontManager', None)
    if font_manager is None:
        font_manager = cast(Any, fm.FontManager)()
    mpl_logger = logging.getLogger(_FONT_MANAGER_LOGGER)
    prev_level = mpl_logger.level
    mpl_logger.setLevel(logging.WARNING)
    try:
        return {f.name for f in font_manager.ttflist}
    finally:
        mpl_logger.setLevel(prev_level)

@lru_cache(maxsize=1)
def detect_available_fonts() -> Tuple[Optional[str], Optional[str]]:
    try:
        mpl_fonts = _collect_matplotlib_font_names()
        chinese_font_candidates = ['SimHei', 'SimSun', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong', 'STKaiti']
        english_fonts = ['Times New Roman', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'Calibri']
        chinese_font = None
        for font_name in chinese_font_candidates:
            if font_name in mpl_fonts:
                chinese_font = font_name
                break
        english_font = None
        for font_name in english_fonts:
            if font_name in mpl_fonts:
                english_font = font_name
                break
        if not english_font:
            english_font = 'DejaVu Sans'
        # Only keep a Chinese default when a real CJK face exists; otherwise prefer English.
        if not chinese_font:
            chinese_font = None
        return (chinese_font, english_font)
    except Exception:
        return (None, 'DejaVu Sans')

def setup_matplotlib_font(chinese_font: Optional[str]=None, english_font: Optional[str]=None) -> None:
    try:
        mpl = _load_module('matplotlib')
        mpl.use('Agg')
        plt = _load_module('matplotlib.pyplot')
        if chinese_font is None and english_font is None:
            chinese_font, english_font = detect_available_fonts()
        elif chinese_font is None or english_font is None:
            detected_cn, detected_en = detect_available_fonts()
            if chinese_font is None:
                chinese_font = detected_cn
            if english_font is None:
                english_font = detected_en
        # Prefer real Chinese face when present; otherwise use English (e.g. Times New Roman).
        default_font = chinese_font or english_font or 'DejaVu Sans'
        fallback_en = english_font or 'DejaVu Sans'
        serif_prefer = {'Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif', 'Georgia'}
        if default_font in serif_prefer or (not chinese_font and fallback_en in serif_prefer):
            face = default_font if default_font in serif_prefer else fallback_en
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = [face, 'Times New Roman', 'DejaVu Serif', 'Liberation Serif']
            plt.rcParams['font.sans-serif'] = [fallback_en, 'DejaVu Sans']
            # STIX math glyphs (subscripts etc.) — Times New Roman lacks Unicode ₁₀
            plt.rcParams['mathtext.fontset'] = 'stix'
        else:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [default_font, fallback_en, 'DejaVu Sans']
            if english_font:
                plt.rcParams['font.serif'] = [english_font, 'DejaVu Serif', 'Liberation Serif']
            plt.rcParams['mathtext.fontset'] = 'dejavusans'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

def setup_plotly_font(chinese_font: Optional[str]=None, english_font: Optional[str]=None) -> dict:
    try:
        if chinese_font is None and english_font is None:
            chinese_font, english_font = detect_available_fonts()
        elif chinese_font is None or english_font is None:
            detected_cn, detected_en = detect_available_fonts()
            if chinese_font is None:
                chinese_font = detected_cn
            if english_font is None:
                english_font = detected_en
        default_font = chinese_font or english_font or 'Arial'
        return {'family': default_font, 'size': 12}
    except Exception:
        return {'family': 'Arial', 'size': 12}
try:
    _cn, _en = detect_available_fonts()
    setup_matplotlib_font(_cn, _en)
except Exception:
    pass
