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
        english_fonts = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'Times New Roman', 'Calibri']
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
        if not chinese_font:
            chinese_font = 'DejaVu Sans'
        if not english_font:
            english_font = 'DejaVu Sans'
        return (chinese_font, english_font)
    except Exception:
        return ('DejaVu Sans', 'DejaVu Sans')

def setup_matplotlib_font(chinese_font: Optional[str]=None, english_font: Optional[str]=None) -> None:
    try:
        mpl = _load_module('matplotlib')
        mpl.use('Agg')
        plt = _load_module('matplotlib.pyplot')
        if chinese_font is None or english_font is None:
            chinese_font, english_font = detect_available_fonts()
        default_font = chinese_font if chinese_font else english_font
        plt.rcParams['font.sans-serif'] = [default_font, english_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

def setup_plotly_font(chinese_font: Optional[str]=None, english_font: Optional[str]=None) -> dict:
    try:
        if chinese_font is None or english_font is None:
            chinese_font, english_font = detect_available_fonts()
        default_font = chinese_font if chinese_font else english_font
        return {'family': default_font, 'size': 12}
    except Exception:
        return {'family': 'Arial', 'size': 12}
try:
    _cn, _en = detect_available_fonts()
    setup_matplotlib_font(_cn, _en)
except Exception:
    pass
