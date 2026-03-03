"""
字体检测和设置工具
自动检测系统中可用的中英文字体，并设置为matplotlib和plotly的默认字体
"""

import logging
import platform
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def detect_available_fonts() -> Tuple[Optional[str], Optional[str]]:
    """
    检测系统中可用的中英文字体
    
    使用matplotlib.font_manager.FontManager获取所有可用字体，
    然后按照指定顺序检测中文字体。
    
    Returns:
        (chinese_font, english_font): 中文字体和英文字体名称
    """
    try:
        from matplotlib.font_manager import FontManager
        
        # 获取所有可用字体列表
        mpl_fonts = set(f.name for f in FontManager().ttflist)
        
        # 中文字体候选列表（按优先级顺序）
        chinese_font_candidates = [
            'SimHei',        # 黑体
            'SimSun',        # 宋体
            'Microsoft YaHei',  # 微软雅黑
            'KaiTi',         # 楷体
            'FangSong',      # 仿宋
            'STSong',        # 华文宋体
            'STKaiti',       # 华文楷体
        ]
        
        # 英文字体候选列表（按优先级）
        english_fonts = [
            'Arial', 'DejaVu Sans', 'Liberation Sans', 
            'Helvetica', 'Times New Roman', 'Calibri'
        ]
        
        # 按照指定顺序检测中文字体
        chinese_font = None
        for font_name in chinese_font_candidates:
            if font_name in mpl_fonts:
                chinese_font = font_name
                logger.debug(f"检测到中文字体: {font_name}")
                break
        
        # 检测英文字体
        english_font = None
        for font_name in english_fonts:
            if font_name in mpl_fonts:
                english_font = font_name
                logger.debug(f"检测到英文字体: {font_name}")
                break
        
        # 如果没有找到，使用默认字体
        if not chinese_font:
            logger.debug("未检测到中文字体，使用默认字体")
            chinese_font = 'DejaVu Sans'  # matplotlib默认字体，支持基本字符
        if not english_font:
            logger.debug("未检测到英文字体，使用默认字体")
            english_font = 'DejaVu Sans'
        
        return chinese_font, english_font
        
    except Exception as e:
        logger.debug(f"字体检测失败: {str(e)}")
        return 'DejaVu Sans', 'DejaVu Sans'

def setup_matplotlib_font(chinese_font: Optional[str] = None, english_font: Optional[str] = None) -> None:
    """
    设置matplotlib的默认字体
    
    Args:
        chinese_font: 中文字体名称
        english_font: 英文字体名称
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 如果没有指定，自动检测
        if chinese_font is None or english_font is None:
            chinese_font, english_font = detect_available_fonts()
        
        # 优先使用中文字体（通常也支持英文）
        default_font = chinese_font if chinese_font else english_font
        
        # 设置matplotlib默认字体
        plt.rcParams['font.sans-serif'] = [default_font, english_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        logger.debug(f"Matplotlib字体已设置: {default_font}")
        
    except Exception as e:
        logger.debug(f"设置matplotlib字体失败: {str(e)}")

def setup_plotly_font(chinese_font: Optional[str] = None, english_font: Optional[str] = None) -> dict:
    """
    设置plotly的默认字体配置
    
    Args:
        chinese_font: 中文字体名称
        english_font: 英文字体名称
    
    Returns:
        plotly字体配置字典
    """
    try:
        # 如果没有指定，自动检测
        if chinese_font is None or english_font is None:
            chinese_font, english_font = detect_available_fonts()
        
        # 优先使用中文字体
        default_font = chinese_font if chinese_font else english_font
        
        # plotly字体配置
        font_config = {
            'family': default_font,
            'size': 12
        }
        
        logger.debug(f"Plotly字体已设置: {default_font}")
        
        return font_config
        
    except Exception as e:
        logger.debug(f"设置plotly字体失败: {str(e)}")
        return {'family': 'Arial', 'size': 12}

# 在模块导入时自动设置字体
try:
    chinese_font, english_font = detect_available_fonts()
    setup_matplotlib_font(chinese_font, english_font)
except Exception:
    pass  # 静默处理字体设置失败
