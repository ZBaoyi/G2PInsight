import argparse
import logging
import os
import sys
import warnings
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import plotly.express as px
from importlib.metadata import version
import kaleido
from plotly.io import write_image

# 忽略警告信息
warnings.filterwarnings('ignore')

# 导入字体设置工具
try:
    from assoG2P.bin.font_utils import setup_matplotlib_font, setup_plotly_font
    setup_matplotlib_font()
    PLOTLY_FONT = setup_plotly_font()
except ImportError:
    PLOTLY_FONT = {'family': 'Arial', 'size': 12}

logger = logging.getLogger(__name__)

class EnhancedGenomeVisualizer:
    """
    增强型基因组数据可视化工具
    功能：
    - 支持静态/交互式散点图可视化
    - 自动染色体位置标注
    - 动态点大小调整
    - 自动计算显著性阈值
    """

    def __init__(
        self,
        input_file: str,
        feature_col: Optional[str] = None,
        value_col: Optional[str] = None,
        max_points: Optional[int] = None,
        auto_sample: bool = False
    ):
        """
        初始化可视化工具
        参数:
            input_file: 输入文件路径
            feature_col: 特征列名(格式: chrom_pos)
            value_col: 数值列名
            max_points: 最大显示点数（已弃用，不再使用）
            auto_sample: 数据过大时自动抽样（已弃用，不再使用）
        """
        self.input_file = Path(input_file)
        self.feature_col = feature_col
        self.value_col = value_col
        self.max_points = max_points
        self.auto_sample = auto_sample
        
        # 初始化关键属性
        self.df = None
        self.threshold = None
        self.effect_col = None  # 正负效应列（三列格式的第三列）
        
        # 统一的颜色定义（静态图和交互式图保持一致）
        self.colors = {
            'Positive': '#c0392b',  # 红色
            'Negative': '#2980b9'   # 蓝色
        }
        
        try:
            self._initialize_data()
        except Exception as e:
            logger.error(f"Data initialization failed: {str(e)}")
            raise

    def _initialize_data(self):
        """数据加载和预处理"""
        self.df = self._load_and_process()
        self.threshold = self._calculate_threshold()
        # 打印基础数据信息
        try:
            logger.info(
                f"Visualization data loaded: file={self.input_file}, "
                f"rows={len(self.df)}, feature_col={self.feature_col}, "
                f"value_col={self.value_col}, effect_col={self.effect_col or 'None'}"
            )
            logger.info(f"Significance threshold (abs value, 99th percentile): {self.threshold:.4g}")
        except Exception:
            # 日志输出失败不影响后续绘图
            pass

    def _detect_feature_column(self, columns: List[str]) -> str:
        """自动检测特征列"""
        for col in columns:
            if pd.Series([col]).str.contains(r'\d+_\d+').any():
                return col
        return columns[0]

    def _detect_separator(self, file_path: Path) -> str:
        """自动检测文件分隔符"""
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # 检测tab分隔符
            if '\t' in first_line:
                return '\t'
            # 检测逗号分隔符
            elif ',' in first_line:
                return ','
            # 默认使用tab（因为我们的文件都是tab分隔）
            else:
                return '\t'
    
    def _detect_columns(self, columns: List[str], sep: str = '\t') -> Tuple[str, str, Optional[str]]:
        """
        自动检测列：适应三列格式（特征名、绝对值、正负效应）
        返回：(特征列, 绝对值列, 效应列)
        
        优先识别标准的feature_importance文件格式：
        - feature: 特征名列
        - importance_abs: 重要性绝对值列
        - effect: 正负效应列（1或-1）
        """
        sample_df = pd.read_csv(self.input_file, nrows=100, sep=sep)
        
        # 优先检查是否为标准的feature_importance格式
        if len(columns) >= 3:
            # 检查列名是否匹配标准格式
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
            
            # 如果找到了所有标准列名，直接返回
            if feature_col and abs_col and effect_col:
                return feature_col, abs_col, effect_col
        
        # 如果文件正好有三列，按顺序分配
        if len(columns) == 3:
            feature_col = columns[0]
            abs_col = columns[1]
            effect_col = columns[2]
            return feature_col, abs_col, effect_col
        
        # 否则，尝试智能检测
        # 检测特征列（通常包含染色体_位置格式或名为feature）
        feature_col = None
        for col in columns:
            if 'feature' in col.lower():
                feature_col = col
                break
        if feature_col is None:
            # 检查第一列是否包含染色体_位置格式
            first_col_sample = sample_df[columns[0]].dropna().head(10)
            if first_col_sample.str.contains(r'^\d+_\d+$', na=False).any():
                feature_col = columns[0]
            else:
                feature_col = columns[0]  # 默认使用第一列
        
        # 检测绝对值列（通常包含abs或importance）
        abs_col = None
        for col in columns:
            if col != feature_col:
                if ('abs' in col.lower() or 'importance' in col.lower()) and pd.api.types.is_numeric_dtype(sample_df[col]):
                    abs_col = col
                    break
        if abs_col is None:
            # 如果没找到，使用第二列（假设是三列格式）
            abs_col = columns[1] if len(columns) > 1 else columns[0]
        
        # 检测效应列（通常包含effect，值为1或-1）
        effect_col = None
        for col in columns:
            if col != feature_col and col != abs_col:
                if 'effect' in col.lower() and pd.api.types.is_numeric_dtype(sample_df[col]):
                    # 检查值是否为1或-1
                    unique_vals = sample_df[col].dropna().unique()
                    if len(unique_vals) <= 2 and all(v in [1, -1, 1.0, -1.0] for v in unique_vals):
                        effect_col = col
                        break
        if effect_col is None and len(columns) >= 3:
            # 如果没找到，使用第三列（假设是三列格式）
            effect_col = columns[2]
        
        return feature_col, abs_col, effect_col

    def _load_and_process(self) -> pd.DataFrame:
        """数据加载和处理流程"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")

        # 自动检测分隔符
        sep = self._detect_separator(self.input_file)
        tab_char = '\t'  # 避免在f-string中使用反斜杠
        sep_name = 'Tab' if sep == tab_char else 'Comma'

        # 列检测（适应三列格式）
        sample_df = pd.read_csv(self.input_file, nrows=100, sep=sep)
        feature_col, abs_col, effect_col = self._detect_columns(sample_df.columns, sep=sep)
        
        if self.feature_col is None:
            self.feature_col = feature_col
        if self.value_col is None:
            self.value_col = abs_col
        
        # 保存effect_col供后续使用
        self.effect_col = effect_col
        
        # 确定要加载的列
        cols_to_load = [self.feature_col, self.value_col]
        if effect_col:
            cols_to_load.append(effect_col)
        
        # 完整加载
        dtype_dict = {self.feature_col: 'string', self.value_col: 'float32'}
        if effect_col:
            dtype_dict[effect_col] = 'int32'
        
        df = pd.read_csv(
            self.input_file,
            sep=sep,
            usecols=cols_to_load,
            dtype=dtype_dict
        )

        # 数据校验
        if len(df) == 0:
            raise ValueError("输入文件为空")
        
        # 检查数值列是否包含有效值
        if df[self.value_col].isna().sum() > 0:
            logger.warning(f"Value column {self.value_col} contains {df[self.value_col].isna().sum():,} missing values, will be automatically removed")
            df = df.dropna(subset=[self.value_col])
            if len(df) == 0:
                raise ValueError("删除缺失值后数据为空")
        
        # 检查数据分布
        value_range = df[self.value_col].max() - df[self.value_col].min()
        if value_range < 1e-9:
            logger.warning(f"Value column {self.value_col} has a very small range, which may affect visualization quality")
        
        # 抽样功能已取消，使用全部数据
        logger.info(f"Loading all data points: {len(df):,} features")

        # 基因组位置解析
        chrom_pos = df[self.feature_col].str.extract(r'^(?P<chrom>\d+)_(?P<pos>\d+)$')
        if chrom_pos.isna().any().any():
            invalid_samples = df.loc[chrom_pos.isna().any(axis=1), self.feature_col].head(3).tolist()
            raise ValueError(
                f"特征列格式应为'染色体_位置'(如1_12345)\n"
                f"无效样本示例: {invalid_samples}"
            )

        # 数据增强（适应三列格式：特征名、绝对值、正负效应）
        # 如果存在effect列，使用它来确定正负效应；否则从绝对值列推断
        if self.effect_col and self.effect_col in df.columns:
            # 使用effect列（1或-1）确定正负效应
            # 第二列已经是绝对值，需要与effect列相乘得到带正负的值用于绘图
            df = df.assign(
                chrom_num=chrom_pos['chrom'].astype('uint8'),
                position=chrom_pos['pos'].astype('uint32'),
                chrom_label="Chr" + chrom_pos['chrom'],
                value_sign=np.where(df[self.effect_col] >= 0, 'Positive', 'Negative'),
                value_abs=df[self.value_col],  # 第二列已经是绝对值，不需要再计算
                value_signed=df[self.value_col] * df[self.effect_col]  # 用于绘图：绝对值 * 效应方向
            ).sort_values(['chrom_num', 'position'])
        else:
            # 兼容旧格式：从数值列推断（如果数值可能为负）
            df = df.assign(
                chrom_num=chrom_pos['chrom'].astype('uint8'),
                position=chrom_pos['pos'].astype('uint32'),
                chrom_label="Chr" + chrom_pos['chrom'],
                value_sign=np.where(df[self.value_col] >= 0, 'Positive', 'Negative'),
                value_abs=df[self.value_col].abs(),
                value_signed=df[self.value_col]  # 旧格式中value_col已经包含正负
            ).sort_values(['chrom_num', 'position'])

        # 计算基因组坐标
        df['x_pos'] = self._calculate_genomic_positions(df)
        return df

    def _calculate_genomic_positions(self, df: pd.DataFrame) -> np.ndarray:
        """计算基因组坐标"""
        chrom_sizes = df.groupby('chrom_num').size()
        offsets = chrom_sizes.cumsum().shift(1, fill_value=0)
        return df.groupby('chrom_num').cumcount() + offsets[df['chrom_num']].values

    def _calculate_threshold(self, percentile: float = 99) -> float:
        """计算显著性阈值"""
        return np.percentile(self.df['value_abs'], percentile)

    def _dynamic_style(self) -> Tuple[float, float]:
        """动态调整点样式"""
        n = len(self.df)
        # 增大基础大小系数，从50改为70使点更大
        size = max(2, 200 / (1 + np.log10(n + 1)))
        alpha = min(0.9, 0.7 / (1 + n / 5e4))
        return size, alpha

    def plot_static_scatter(
        self,
        output_file: str,
        dpi: int = 1200,
        **kwargs
    ) -> None:
        """
        绘制静态散点图
        参数:
            output_file: 输出文件路径
            dpi: 图像分辨率
        """
        # 绘图前打印基本信息
        logger.info(
            f"Drawing static scatter: input={self.input_file}, "
            f"output={output_file}, rows={len(self.df)}, "
            f"feature_col={self.feature_col}, value_col={self.value_col}"
        )
        try:
            self._plot_static_scatter(output_file, dpi, **kwargs)
        except Exception as e:
            logger.error(f"Static scatter plot generation failed: {str(e)}")
            raise

    def _plot_static_scatter(
        self,
        output_file: str,
        dpi: int,
        **kwargs
    ) -> None:
        """静态散点图绘制"""
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(18, 8), dpi=dpi)
        
        self._plot_scatter_static(ax, self.colors)
        self._format_axes(ax)
        self._save_figure(fig, output_file, dpi)

    def plot_interactive_scatter(
        self,
        output_file: str,
        **kwargs
    ) -> None:
        """
        绘制交互式散点图
        参数:
            output_file: 输出文件路径
        """
        # 绘图前打印基本信息
        logger.info(
            f"Drawing interactive scatter: input={self.input_file}, "
            f"output={output_file}, rows={len(self.df)}, "
            f"feature_col={self.feature_col}, value_col={self.value_col}"
        )
        try:
            self._plot_interactive_scatter(output_file, **kwargs)
        except Exception as e:
            logger.error(f"Interactive scatter plot generation failed: {str(e)}")
            raise

    def _plot_interactive_scatter(
        self,
        output_file: str,
        **kwargs
    ) -> None:
        """交互式散点图绘制"""
        fig = self._create_interactive_scatter()
        
        output_path = Path(output_file)
        if output_path.suffix == '.html':
            fig.write_html(output_path, include_plotlyjs='cdn')
        else:
            try:
                write_image(fig, output_path, scale=2, engine='kaleido')
            except Exception as e:
                logger.error(f"Image writing failed, attempting to use orca engine...", exc_info=True)
                try:
                    write_image(fig, output_path, scale=2, engine='orca')
                except Exception as e2:
                    logger.error(f"Image saving failed: {str(e2)}")
                    raise

    def _plot_scatter_static(self, ax: Axes, colors: Dict[str, str]):
        """静态散点图绘制实现"""
        size, alpha = self._dynamic_style()
        
        for sign, color in colors.items():
            subset = self.df[self.df['value_sign'] == sign]
            x_values = subset['x_pos'].values.flatten()
            # 使用value_abs列（绝对值）用于绘图，所有点都绘制在横轴上方
            y_values = subset['value_abs'].values.flatten()
            
            # 绘制点到X轴的垂直连线
            # 使用vlines绘制垂直线，从y=0到点的y值
            ax.vlines(
                x=x_values,
                ymin=0,  # 从X轴（y=0）开始
                ymax=y_values,
                colors=color,
                alpha=alpha * 0.5,  # 连线透明度稍低，避免过于突出
                linewidths=0.5,  # 细线
                linestyles='-',
                zorder=1  # 在散点下方
            )
            
            # 绘制散点（在连线上方）
            ax.scatter(
                x=x_values,
                y=y_values,
                c=color,
                s=size,
                alpha=alpha,
                edgecolors='none',
                label=sign,
                zorder=2  # 在连线上方
            )
        
        # 确保显示图例
        ax.legend(title='Effect Direction', loc='upper right')

    def _create_interactive_scatter(self):
        """交互式散点图创建"""
        size, alpha = self._dynamic_style()
        
        # 创建散点图（使用统一的颜色定义，所有点都绘制在横轴上方）
        fig = px.scatter(
            self.df,
            x=self.df['x_pos'].values.flatten(),
            y=self.df['value_abs'].values.flatten(),  # 使用value_abs列（绝对值），所有点都在横轴上方
            color='value_sign',
            color_discrete_map=self.colors,  # 使用统一的颜色定义
            hover_data={
                'chrom_label': True,
                'position': True,
                'value_signed': ':.3f',
                'value_abs': ':.3f',
                'x_pos': False
            },
            size_max=size,
            opacity=alpha,
            width=1200,
            height=600,
            title="Genome-Wide Association Plot"  # 与静态图标题一致
        )

        # 确保显示图例
        fig.update_layout(showlegend=True)
        
        # 应用与静态图一致的格式化
        self._format_interactive_layout(fig)
        return fig

    def _format_interactive_layout(self, fig):
        """格式化交互图表布局（与静态图保持一致）"""
        # 计算染色体刻度位置（与静态图一致）
        chrom_ticks = self.df.groupby('chrom_label')['x_pos'].median()
        
        # 计算Y轴范围：从0开始，所有点都在横轴上方
        y_max = self.df['value_abs'].max()
        # 稍微加一点padding，避免点贴边
        padding = 0.1 * y_max if y_max > 0 else 1.0
        y_min_adjusted = 0  # Y轴从0开始
        y_max_adjusted = y_max + padding
        
        # 更新布局（与静态图格式保持一致）并应用字体设置
        fig.update_layout(
            font=PLOTLY_FONT,
            xaxis=dict(
                showgrid=False,
                title="Genomic Position",
                tickmode='array',
                tickvals=chrom_ticks.values.tolist(),
                ticktext=chrom_ticks.index.tolist(),
                tickangle=-45
            ),
            yaxis=dict(
                showgrid=False,
                title="Feature Importance",
                range=[y_min_adjusted, y_max_adjusted]  # 与静态图Y轴范围一致
            ),
            title={
                'text': "Genome-Wide Association Plot",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': PLOTLY_FONT.get('family', 'Arial')}
            },
            legend=dict(
                title="Effect Direction",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1
            )
        )
        
        # 添加染色体分隔线（与静态图一致）
        for chrom, group in self.df.groupby('chrom_label'):
            x_sep = group['x_pos'].min() - 0.5
            fig.add_shape(
                type="line",
                x0=x_sep,
                y0=y_min_adjusted,  # 从0开始
                x1=x_sep,
                y1=y_max_adjusted,
                line=dict(
                    color='gray',
                    width=1,
                    dash='dot'
                ),
                layer='below'
            )

    def _format_axes(self, ax: Axes):
        """格式化坐标轴"""
        # 染色体刻度
        chrom_ticks = self.df.groupby('chrom_label')['x_pos'].median()
        ax.set_xticks(chrom_ticks.values)
        ax.set_xticklabels(chrom_ticks.index, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel("Genomic Position", fontsize=16)
        
        # 染色体分隔线
        for chrom, group in self.df.groupby('chrom_label'):
            ax.axvline(group['x_pos'].min() - 0.5, color='gray', linestyle=':', alpha=0.3)

        # 标签和标题
        ax.set_ylabel('Feature Importance', fontsize=16)
        ax.set_title(
            "Genome-Wide Association Plot",
            pad=20,
            fontsize=16
        )
        
        # 设置y轴刻度字体大小
        ax.tick_params(axis='y', labelsize=16)
        
        # 动态调整Y轴范围：从0开始，所有点都在横轴上方
        y_max = self.df['value_abs'].max()
        y_range = y_max
        # 扩展Y轴范围10%，从0开始
        padding = 0.1 * y_range if y_range > 0 else 1.0
        ax.set_ylim(0, y_max + padding)
        
        # 散点图不显示网格线
        ax.grid(False)

    def _save_figure(self, fig, output_path: str, dpi: int):
        """保存图像"""
        path = Path(output_path)
        # 验证输出路径
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise PermissionError(f"没有权限创建目录: {path.parent}")
            except Exception as e:
                raise IOError(f"创建目录失败: {str(e)}")
        
        # 检查路径可写性
        if path.exists():
            if not os.access(path, os.W_OK):
                raise PermissionError(f"文件不可写: {path}")
        else:
            if not os.access(path.parent, os.W_OK):
                raise PermissionError(f"目录不可写: {path.parent}")
        
        try:
            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
            )
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            raise

def run_visualization(
    input_file: str,
    output_prefix: str,
    feature_col: Optional[str] = None,
    value_col: Optional[str] = None,
    dpi: int = 1200,
    static_only: bool = False,
    interactive_only: bool = False,
    **kwargs
) -> int:
    """
    可视化入口函数
    
    支持三种模式：
    1) 默认（不传 static_only/interactive_only）：同时生成静态和交互式散点图
    2) static_only=True：只生成静态散点图
    3) interactive_only=True：只生成交互式散点图
    
    返回:
        0: 成功
        1: 常规错误
        2: 文件错误
        3: 数据格式错误
        4: 依赖错误
    """
    try:
        visualizer = EnhancedGenomeVisualizer(
            input_file=input_file,
            feature_col=feature_col,
            value_col=value_col
        )

        # 根据参数决定生成哪种图
        generate_static = True
        generate_interactive = True
        if static_only and not interactive_only:
            generate_interactive = False
        elif interactive_only and not static_only:
            generate_static = False

        static_output = None
        interactive_output = None

        if generate_static:
            static_output = f"{output_prefix}_static.png"
            visualizer.plot_static_scatter(
                output_file=static_output,
                dpi=dpi,
                **kwargs
            )

        if generate_interactive:
            interactive_output = f"{output_prefix}_interactive.html"
            visualizer.plot_interactive_scatter(
                output_file=interactive_output,
                **kwargs
            )

        logger.info("Visualization completed")
        return 0

    except FileNotFoundError:
        logger.error(f"File error: Cannot find input file '{input_file}'\nPlease check if the file path is correct")
        return 2
    except ValueError as e:
        logger.error(f"Data format error: {str(e)}\nInput file: {input_file}\nFeature column: {feature_col}, Value column: {value_col}")
        return 3
    except ImportError as e:
        logger.error(f"Dependency error: {str(e)}\nPlease run 'pip install plotly kaleido' to install required dependencies")
        return 4
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return 1

# ======================== 模型性能可视化函数 ========================

def _setup_matplotlib() -> Tuple[bool, Any]:
    """
    设置matplotlib环境（提取公共代码）
    
    Returns:
        (is_available, plt_module): matplotlib是否可用和plt模块（如果可用）
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        # 设置字体
        try:
            from assoG2P.bin.font_utils import setup_matplotlib_font
            setup_matplotlib_font()
        except ImportError:
            pass
        return True, plt
    except ImportError:
        logger.warning("  matplotlib not installed, skipping plotting functionality")
        return False, None  # type: ignore

def _ensure_numpy_array(data: Any) -> np.ndarray:
    """
    确保数据是numpy数组（提取公共代码）
    
    :param data: 输入数据（可能是pd.Series、list或np.ndarray）
    :return: numpy数组
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.values
    else:
        return np.array(data)

def load_plotting_data(plotting_data_file: Union[str, Path]) -> Dict:
    """
    从plotting_data.npz文件加载所有绘图数据
    
    参数:
        plotting_data_file: plotting_data.npz文件路径
        
    返回:
        包含所有绘图数据的字典：
        {
            'y_test': np.ndarray,
            'y_pred': np.ndarray,
            'y_prob': np.ndarray or None,
            'cv_results': Dict or None,
            'model_type': str,
            'task_type': str,
            'publication_quality': bool
        }
    """
    plotting_data_file = Path(plotting_data_file)
    if not plotting_data_file.exists():
        raise FileNotFoundError(f"绘图数据文件不存在: {plotting_data_file}")
    
    # 加载NPZ文件
    data = np.load(plotting_data_file, allow_pickle=True)
    
    # 提取数据
    result = {
        'y_test': data['y_test'],
        'y_pred': data['y_pred'],
        'y_prob': data.get('y_prob', None),
        'model_type': str(data['model_type'][0]) if 'model_type' in data else 'Unknown',
        'task_type': str(data['task_type'][0]) if 'task_type' in data else 'regression',
        'publication_quality': bool(data['publication_quality'][0]) if 'publication_quality' in data else True
    }
    
    # 反序列化cv_results（如果存在）
    if 'cv_results' in data:
        cv_results_bytes = data['cv_results'][0]
        result['cv_results'] = pickle.loads(cv_results_bytes)
    else:
        result['cv_results'] = None
    
    return result

def plot_performance_curves(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray], 
    output_dir: Union[str, Path], 
    model_type: str, 
    task_type: str, 
    publication_quality: bool = True
) -> None:
    """
    绘制性能评估指标变化曲线（分类/回归）
    
    :param y_true: 真实值（pd.Series或np.ndarray）
    :param y_pred: 预测值（np.ndarray）
    :param y_prob: 预测概率（分类任务需要，回归任务为None）
    :param output_dir: 输出目录
    :param model_type: 模型类型
    :param task_type: 任务类型（classification/regression）
    :param publication_quality: 是否生成期刊发表质量图表（高分辨率、矢量格式、专业配色）
    """
    output_dir = Path(output_dir)
    
    # 设置matplotlib环境
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        return
    
    # 期刊发表质量设置
    if publication_quality:
        # 设置期刊标准字体和样式
        plt.rcParams.update({
            'font.family': 'serif',  # 使用serif字体（如Times New Roman）
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
            'font.size': 9,  # 基础字体大小（减小）
            'axes.labelsize': 10,  # 轴标签字体大小（减小）
            'axes.titlesize': 11,  # 标题字体大小（减小）
            'xtick.labelsize': 8,  # x轴刻度字体大小（减小）
            'ytick.labelsize': 8,  # y轴刻度字体大小（减小）
            'legend.fontsize': 8,  # 图例字体大小（减小）
            'figure.titlesize': 12,  # 图形标题字体大小（减小）
            'lines.linewidth': 2,  # 线条宽度
            'axes.linewidth': 1.2,  # 坐标轴线宽
            'grid.linewidth': 0.8,  # 网格线宽
            'axes.grid': True,  # 默认显示网格
            'grid.alpha': 0.3,  # 网格透明度
            'figure.dpi': 300,  # 高分辨率
            'savefig.dpi': 300,  # 保存时高分辨率
            'savefig.bbox': 'tight',  # 紧密边界
            'savefig.pad_inches': 0.1,  # 边距
        })
    
    # 确保y_true是numpy数组
    y_true_values = _ensure_numpy_array(y_true)
    y_pred_values = np.array(y_pred)
    
    if task_type == "regression":
        # ========== Regression performance curves ==========
        # 绘制多个回归评估图：散点图、残差图、残差分布、Q-Q图
        if publication_quality:
            # 期刊标准尺寸：单栏宽度约3.5英寸，双栏约7英寸
            fig, axes = plt.subplots(2, 2, figsize=(7, 6))  # 适合双栏布局
            fig.suptitle(f'{model_type} Model Performance (Regression)', fontsize=12, fontweight='bold')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{model_type} Performance (Regression)', fontsize=16, fontweight='bold')
        
        # 计算皮尔逊相关系数和P值
        try:
            from scipy.stats import pearsonr
            pearson_corr, pearson_pvalue = pearsonr(y_true_values, y_pred_values)
        except ImportError:
            logger.warning("  scipy not installed, cannot calculate Pearson correlation coefficient and P-value")
            pearson_corr = None
            pearson_pvalue = None
        except Exception as e:
            logger.warning(f"  Failed to calculate Pearson correlation coefficient: {str(e)}")
            pearson_corr = None
            pearson_pvalue = None
        
        # 计算残差
        residuals = y_true_values - y_pred_values
        
        # 1. 预测值 vs 真实值散点图
        ax1 = axes[0, 0]
        if publication_quality:
            # 期刊标准：使用灰度或专业配色，更大的点，清晰的边缘
            ax1.scatter(y_true_values, y_pred_values, alpha=0.6, s=25, 
                       color='#2E86AB', edgecolors='black', linewidths=0.3)
            # 理想预测线：黑色虚线
            min_val = min(np.min(y_true_values), np.min(y_pred_values))
            max_val = max(np.max(y_true_values), np.max(y_pred_values))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
                    label='Ideal (y=x)', dashes=(5, 3))
            # 回归拟合线：深蓝色实线
            try:
                z = np.polyfit(y_true_values, y_pred_values, 1)
                p = np.poly1d(z)
                ax1.plot(y_true_values, p(y_true_values), "#E63946", linewidth=2, 
                        label=f'Fit (slope={z[0]:.3f})')
            except:
                pass
            ax1.set_xlabel('True Values', fontsize=10, fontweight='normal')
            ax1.set_ylabel('Predicted Values', fontsize=10, fontweight='normal')
            ax1.set_title('(A) Predicted vs True Values', fontsize=11, fontweight='bold')
            if pearson_corr is not None:
                ax1.text(0.05, 0.95, f'r = {pearson_corr:.4f}\nR² = {pearson_corr**2:.4f}', 
                        transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                                linewidth=0.8, alpha=0.9, pad=0.3))
            ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', 
                      framealpha=0.9, fontsize=8, handlelength=1.5)
        else:
            ax1.scatter(y_true_values, y_pred_values, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
            min_val = min(np.min(y_true_values), np.min(y_pred_values))
            max_val = max(np.max(y_true_values), np.max(y_pred_values))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Prediction (y=x)')
            try:
                z = np.polyfit(y_true_values, y_pred_values, 1)
                p = np.poly1d(z)
                ax1.plot(y_true_values, p(y_true_values), "b-", linewidth=1.5, alpha=0.7, label=f'Fit Line (slope={z[0]:.3f})')
            except:
                pass
            ax1.set_xlabel('True Values', fontsize=12)
            ax1.set_ylabel('Predicted Values', fontsize=12)
            ax1.set_title('Predicted vs True Values', fontsize=13)
            if pearson_corr is not None:
                ax1.text(0.05, 0.95, f'r = {pearson_corr:.4f}\nR² = {pearson_corr**2:.4f}', 
                        transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差图（残差 vs 预测值）
        ax2 = axes[0, 1]
        if publication_quality:
            ax2.scatter(y_pred_values, residuals, alpha=0.6, s=25, 
                       color='#2E86AB', edgecolors='black', linewidths=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, 
                       label='Zero', dashes=(5, 3))
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
        
        # 3. 残差分布直方图
        ax3 = axes[1, 0]
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        if publication_quality:
            ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, 
                    color='#A23B72', linewidth=0.8)
            ax3.axvline(x=0, color='k', linestyle='--', linewidth=2, 
                       label='Zero', dashes=(5, 3))
            ax3.axvline(x=mean_residual, color='#E63946', linestyle='--', linewidth=2, 
                       label=f'Mean={mean_residual:.4f}', dashes=(5, 3))
            ax3.set_xlabel('Residuals', fontsize=10, fontweight='normal')
            ax3.set_ylabel('Frequency', fontsize=10, fontweight='normal')
            ax3.set_title('(C) Residual Distribution', fontsize=11, fontweight='bold')
            # 获取y轴范围，将文本放在zero线和Mean线之间的正下方（y轴底部）
            y_min, y_max = ax3.get_ylim()
            # 计算zero线和Mean线的中点位置
            text_x = (0 + mean_residual) / 2
            # 使用数据坐标，将文本放在两条线之间的正下方
            ax3.text(text_x, y_min + (y_max - y_min) * 0.05, 
                    f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                    fontsize=8, verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                            linewidth=0.8, alpha=0.9, pad=0.3))
            ax3.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9, fontsize=8, handlelength=1.5)
        else:
            ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
            ax3.axvline(x=mean_residual, color='g', linestyle='--', linewidth=2, 
                       label=f'Mean={mean_residual:.4f}')
            ax3.set_xlabel('Residuals', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Residual Distribution', fontsize=13)
            # 获取y轴范围，将文本放在zero线和Mean线之间的正下方（y轴底部）
            y_min, y_max = ax3.get_ylim()
            # 计算zero线和Mean线的中点位置
            text_x = (0 + mean_residual) / 2
            # 使用数据坐标，将文本放在两条线之间的正下方
            ax3.text(text_x, y_min + (y_max - y_min) * 0.05, 
                    f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                    fontsize=11, verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q图（残差正态性检验）
        ax4 = axes[1, 1]
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            if publication_quality:
                # 优化Q-Q图的线条样式
                lines = ax4.get_lines()
                if len(lines) >= 2:
                    lines[0].set_linewidth(2)  # 数据点线
                    lines[0].set_color('#2E86AB')
                    lines[1].set_linewidth(2)  # 理论线
                    lines[1].set_color('#E63946')
                    lines[1].set_linestyle('--')
                ax4.set_title('(D) Q-Q Plot', fontsize=11, fontweight='bold')
                ax4.set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='normal')
                ax4.set_ylabel('Sample Quantiles', fontsize=10, fontweight='normal')
            else:
                ax4.set_title('Q-Q Plot (Residual Normality Test)', fontsize=13)
            ax4.grid(True, alpha=0.3)
        except ImportError:
            ax4.text(0.5, 0.5, 'Q-Q Plot requires scipy', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Q-Q Plot (scipy not available)', fontsize=13)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Q-Q Plot failed:\n{str(e)}', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Q-Q Plot (Error)', fontsize=13)
        
        plt.tight_layout()
        
        # 保存图形
        if publication_quality:
            # 期刊标准：保存为PDF（矢量格式）和PNG（高分辨率）
            plot_file_pdf = output_dir / "performance_curves.pdf"
            plot_file_png = output_dir / "performance_curves.png"
            plt.savefig(plot_file_pdf, dpi=300, bbox_inches='tight', format='pdf')
            plt.savefig(plot_file_png, dpi=300, bbox_inches='tight', format='png')
            logger.info(f"  Publication-quality plots saved: {plot_file_pdf} and {plot_file_png}")
        else:
            plot_file = output_dir / "performance_curves.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    else:
        # ========== Classification performance curves ==========
        if y_prob is None:
            logger.warning(" Classification task requires predicted probabilities, skip performance curves")
            return
        
        # 获取类别信息
        unique_classes = np.unique(y_true_values)
        n_classes = len(unique_classes)
        
        # 确定二分类时使用的概率列
        if n_classes == 2:
            prob_col = 1 if y_prob.shape[1] > 1 else 0
            y_prob_binary = y_prob[:, prob_col]
        else:
            # 多分类：使用第一个类别的概率
            prob_col = 0
            y_prob_binary = y_prob[:, prob_col]
        
        # ========== 图1：多个性能曲线合并在一张图中（1x3子图布局，删除召回率曲线）==========
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, recall_score
            
            if publication_quality:
                fig1, axes = plt.subplots(1, 3, figsize=(14, 4))
                fig1.suptitle(f'{model_type} Performance Curves (Classification)', fontsize=12, fontweight='bold')
            else:
                fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig1.suptitle(f'{model_type} Performance Curves (Classification)', fontsize=16, fontweight='bold')
            
            if n_classes == 2:
                # 二分类：计算不同阈值下的指标
                thresholds = np.linspace(0, 1, 100)
                accuracies = []
                recalls = []
                f1_scores = []
                
                # 计算ROC曲线
                fpr, tpr, roc_thresholds = roc_curve(y_true_values, y_prob_binary)
                roc_auc = auc(fpr, tpr)
                
                # 计算不同阈值下的准确率、召回率和F1得分
                for threshold in thresholds:
                    y_pred_thresh = (y_prob_binary >= threshold).astype(int)
                    accuracies.append(accuracy_score(y_true_values, y_pred_thresh))
                    recalls.append(recall_score(y_true_values, y_pred_thresh, zero_division=0))
                    f1_scores.append(f1_score(y_true_values, y_pred_thresh, zero_division=0))
                
                # 找到准确率和F1得分的最高值点
                max_acc_idx = np.argmax(accuracies)
                max_acc_threshold = thresholds[max_acc_idx]
                max_acc_value = accuracies[max_acc_idx]
                
                max_f1_idx = np.argmax(f1_scores)
                max_f1_threshold = thresholds[max_f1_idx]
                max_f1_value = f1_scores[max_f1_idx]
                
                # 子图1：ROC曲线（AUC曲线）
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
                
                # 子图2：准确率曲线（添加最高值点的垂直虚线）
                ax2 = axes[1]
                ax2.plot(thresholds, accuracies, 'r-', lw=2, label='Accuracy')
                # 在最高值点处添加垂直虚线
                ax2.axvline(x=max_acc_threshold, color='r', linestyle='--', linewidth=1.5, alpha=0.7, 
                           label=f'Max Accuracy: {max_acc_value:.4f} at {max_acc_threshold:.3f}')
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
                
                # 子图3：F1得分曲线（添加最高值点的垂直虚线）
                ax3 = axes[2]
                ax3.plot(thresholds, f1_scores, 'm-', lw=2, label='F1 Score')
                # 在最高值点处添加垂直虚线
                ax3.axvline(x=max_f1_threshold, color='m', linestyle='--', linewidth=1.5, alpha=0.7,
                           label=f'Max F1: {max_f1_value:.4f} at {max_f1_threshold:.3f}')
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
                # 多分类：绘制每个类别的ROC曲线
                from sklearn.preprocessing import label_binarize
                y_true_binarized = label_binarize(y_true_values, classes=unique_classes)
                
                # 子图1：ROC曲线
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
                
                # 其他子图留空或显示提示
                for ax in [axes[1], axes[2]]:
                    ax.axis('off')
                    if publication_quality:
                        ax.text(0.5, 0.5, 'Multi-class metrics\nnot implemented', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Multi-class metrics\nnot implemented', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            plt.tight_layout()
            plot_file1 = output_dir / "performance_curves.png"
            plt.savefig(plot_file1, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f" Failed to draw performance curves: {str(e)}")

def plot_cv_training_curves(
    cv_results: Dict, 
    output_dir: Union[str, Path], 
    model_type: str, 
    task_type: str
) -> None:
    """
    绘制交叉验证训练过程柱形图
    
    :param cv_results: 交叉验证结果字典，包含每折的指标
    :param output_dir: 输出目录
    :param model_type: 模型类型
    :param task_type: 任务类型
    """
    output_dir = Path(output_dir)
    
    # 确保输出目录存在
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"  Cannot create output directory {output_dir}: {str(e)}")
        return
    
    # 设置matplotlib环境
    matplotlib_available, plt = _setup_matplotlib()
    if not matplotlib_available:
        logger.error("  matplotlib not available, cannot plot cross-validation curves")
        return
    
    n_folds = len(cv_results.get('fold_metrics', []))
    if n_folds == 0:
        logger.warning("  No cross-validation results, skipping training curve plotting")
        return
    
    # 创建图形
    if task_type == "regression":
        # 回归任务：绘制皮尔逊相关系数和P值柱形图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{model_type} Model {n_folds}-Fold Cross-Validation (Regression)', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # 提取指标（使用字典保存每个折的索引，以便正确对应）
        pearson_corrs = []
        pearson_pvalues = []
        pvalue_fold_indices = []  # 记录每个P值对应的折索引
        
        for i in range(n_folds):
            fold_metric = cv_results['fold_metrics'][i]
            corr_val = fold_metric.get('pearson_correlation')
            pval_val = fold_metric.get('pearson_pvalue')
            
            # 提取相关系数
            if isinstance(corr_val, (int, float)) and not np.isnan(corr_val):
                pearson_corrs.append(corr_val)
            else:
                logger.warning(f"  Fold {i+1}: Invalid correlation value: {corr_val}")
            
            # 提取P值：P值应该 >= 0，但可能非常小（接近0），所以只检查是否为有效数值
            if pval_val is not None:
                # 处理字符串类型的P值（如"N/A"）
                if isinstance(pval_val, str):
                    if pval_val.upper() != 'N/A':
                        try:
                            pval_val = float(pval_val)
                        except (ValueError, TypeError):
                            logger.warning(f"  Fold {i+1}: Cannot convert P-value string: {pval_val}")
                            pval_val = None
                    else:
                        pval_val = None
                
                # 检查是否为有效数值
                if isinstance(pval_val, (int, float)) and not np.isnan(pval_val):
                    # P值应该 >= 0，但允许等于0（虽然罕见）
                    if pval_val >= 0:
                        pearson_pvalues.append(pval_val)
                        pvalue_fold_indices.append(i)  # 记录对应的折索引
                    else:
                        logger.warning(f"  Fold {i+1}: P-value is negative, skipping: {pval_val}")
                elif pval_val is not None:
                    logger.warning(f"  Fold {i+1}: Invalid P-value: {pval_val} (type: {type(pval_val)})")
            else:
                logger.warning(f"  Fold {i+1}: P-value is None")
        
        # 诊断信息：检查提取到的数据
        logger.debug(f"  Extracted {len(pearson_corrs)} correlation values, {len(pearson_pvalues)} P-values")
        if len(pearson_pvalues) != n_folds:
            logger.warning(f"  P-value count ({len(pearson_pvalues)}) does not match number of folds ({n_folds})!")
            if len(pearson_pvalues) > 0:
                logger.warning(f"  Valid P-values from folds: {[idx+1 for idx in pvalue_fold_indices]}")
            logger.warning(f"  This may cause the P-value plot to display data from only some folds")
        
        # 1. 皮尔逊相关系数柱形图
        if pearson_corrs and len(pearson_corrs) > 0:
            x_positions = np.arange(len(pearson_corrs))
            bars1 = ax1.bar(x_positions, pearson_corrs, color='#4A90E2', alpha=0.7, 
                           edgecolor='black', linewidth=1.5, width=0.6)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars1, pearson_corrs)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 添加统计信息文本
            mean_corr = np.mean(pearson_corrs)
            median_corr = np.median(pearson_corrs)
            std_corr = np.std(pearson_corrs)
            q25 = np.percentile(pearson_corrs, 25)
            q75 = np.percentile(pearson_corrs, 75)
            
            # 添加均值线
            ax1.axhline(y=mean_corr, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_corr:.4f}', alpha=0.7)
            
            stats_text = f'Mean: {mean_corr:.4f}\nMedian: {median_corr:.4f}\nStd: {std_corr:.4f}\nQ25: {q25:.4f}\nQ75: {q75:.4f}'
            ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))
            
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels([f'Fold {i+1}' for i in range(len(pearson_corrs))], rotation=45, ha='right')
            ax1.legend(loc='upper left', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
        
        ax1.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
        ax1.set_title('Pearson Correlation by Fold', fontsize=13, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        if pearson_corrs:
            y_min = min(pearson_corrs) - 0.1 * abs(min(pearson_corrs))
            y_max = max(pearson_corrs) + 0.1 * abs(max(pearson_corrs))
            ax1.set_ylim([y_min, y_max])
        
        # 2. P值柱形图（对数尺度）
        if pearson_pvalues and len(pearson_pvalues) > 0:
            # 使用记录的折索引来正确标记X轴
            if len(pvalue_fold_indices) == len(pearson_pvalues):
                # 使用实际对应的折索引
                x_positions = np.arange(len(pearson_pvalues))
                x_labels = [f'Fold {idx+1}' for idx in pvalue_fold_indices]
            else:
                # 如果索引记录不完整，使用顺序索引
                x_positions = np.arange(len(pearson_pvalues))
                x_labels = [f'Fold {i+1}' for i in range(len(pearson_pvalues))]
                if len(pearson_pvalues) < n_folds:
                    logger.warning(f"  P-value count ({len(pearson_pvalues)}) is less than number of folds ({n_folds}), some folds may have missing P-values")
            
            bars2 = ax2.bar(x_positions, pearson_pvalues, color='#E74C3C', alpha=0.7, 
                           edgecolor='black', linewidth=1.5, width=0.6)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars2, pearson_pvalues)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2e}', ha='center', va='bottom', fontsize=8)
            
            # 添加统计信息文本
            mean_pvalue = np.mean(pearson_pvalues)
            median_pvalue = np.median(pearson_pvalues)
            std_pvalue = np.std(pearson_pvalues)
            q25 = np.percentile(pearson_pvalues, 25)
            q75 = np.percentile(pearson_pvalues, 75)
            
            # 添加均值线
            ax2.axhline(y=mean_pvalue, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_pvalue:.2e}', alpha=0.7)
            
            stats_text = f'Mean: {mean_pvalue:.6f}\nMedian: {median_pvalue:.6f}\nStd: {std_pvalue:.6f}\nQ25: {q25:.6f}\nQ75: {q75:.6f}'
            if len(pearson_pvalues) < n_folds:
                stats_text += f'\n\n注意: 仅显示 {len(pearson_pvalues)}/{n_folds} 折的数据'
            ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))
            
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(x_labels, rotation=45, ha='right')
            ax2.legend(loc='upper left', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        ax2.set_ylabel('P-value (log scale)', fontsize=12, fontweight='bold')
        ax2.set_title('P-value by Fold', fontsize=13, fontweight='bold', pad=15)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
    else:
        # 分类任务：绘制所有指标的柱形图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_type} Model {n_folds}-Fold Cross-Validation (Classification)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 提取指标
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
            
            if isinstance(acc_val, (int, float)) and not np.isnan(acc_val):
                accuracies.append(acc_val)
            if isinstance(rec_val, (int, float)) and not np.isnan(rec_val):
                recalls.append(rec_val)
            if isinstance(f1_val, (int, float)) and not np.isnan(f1_val):
                f1_scores.append(f1_val)
            if isinstance(auc_val, (int, float)) and not np.isnan(auc_val) and auc_val != 'N/A':
                aucs.append(auc_val)
        
        # 1. 准确率柱形图
        ax1 = axes[0, 0]
        if accuracies and len(accuracies) > 0:
            x_positions = np.arange(len(accuracies))
            bars1 = ax1.bar(x_positions, accuracies, color='#3498DB', alpha=0.7, 
                           edgecolor='black', linewidth=1.5, width=0.6)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars1, accuracies)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 添加统计信息
            mean_acc = np.mean(accuracies)
            median_acc = np.median(accuracies)
            std_acc = np.std(accuracies)
            q25 = np.percentile(accuracies, 25)
            q75 = np.percentile(accuracies, 75)
            
            # 添加均值线
            ax1.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_acc:.4f}', alpha=0.7)
            
            stats_text = f'Mean: {mean_acc:.4f}\nMedian: {median_acc:.4f}\nStd: {std_acc:.4f}\nQ25: {q25:.4f}\nQ75: {q75:.4f}'
            ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))
            
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels([f'Fold {i+1}' for i in range(len(accuracies))], rotation=45, ha='right')
            ax1.legend(loc='upper left', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
        
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy by Fold', fontsize=13, fontweight='bold', pad=15)
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 2. 召回率和F1得分柱形图（分组柱形图）
        ax2 = axes[0, 1]
        if (recalls and len(recalls) > 0) or (f1_scores and len(f1_scores) > 0):
            x = np.arange(len(recalls) if recalls else len(f1_scores))
            width = 0.35
            
            if recalls and len(recalls) > 0:
                bars_recall = ax2.bar(x - width/2, recalls, width, label='Recall', 
                                      color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=1.5)
                # 添加数值标签
                for bar, val in zip(bars_recall, recalls):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            
            if f1_scores and len(f1_scores) > 0:
                bars_f1 = ax2.bar(x + width/2, f1_scores, width, label='F1 Score', 
                                  color='#27AE60', alpha=0.7, edgecolor='black', linewidth=1.5)
                # 添加数值标签
                for bar, val in zip(bars_f1, f1_scores):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            
            # 添加统计信息
            stats_text = ""
            if recalls and len(recalls) > 0:
                mean_rec = np.mean(recalls)
                median_rec = np.median(recalls)
                stats_text += f'Recall:\n  Mean: {mean_rec:.4f}\n  Median: {median_rec:.4f}\n\n'
            if f1_scores and len(f1_scores) > 0:
                mean_f1 = np.mean(f1_scores)
                median_f1 = np.median(f1_scores)
                stats_text += f'F1 Score:\n  Mean: {mean_f1:.4f}\n  Median: {median_f1:.4f}'
            
            ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))
            
            ax2.set_xticks(x)
            n_folds_display = len(recalls) if recalls else len(f1_scores)
            ax2.set_xticklabels([f'Fold {i+1}' for i in range(n_folds_display)], rotation=45, ha='right')
            ax2.legend(loc='upper left', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Recall and F1 Score by Fold', fontsize=13, fontweight='bold', pad=15)
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 3. AUC柱形图
        ax3 = axes[1, 0]
        if aucs and len(aucs) > 0:
            x_positions = np.arange(len(aucs))
            bars3 = ax3.bar(x_positions, aucs, color='#9B59B6', alpha=0.7, 
                           edgecolor='black', linewidth=1.5, width=0.6)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars3, aucs)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 添加统计信息
            mean_auc = np.mean(aucs)
            median_auc = np.median(aucs)
            std_auc = np.std(aucs)
            q25 = np.percentile(aucs, 25)
            q75 = np.percentile(aucs, 75)
            
            # 添加均值线
            ax3.axhline(y=mean_auc, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_auc:.4f}', alpha=0.7)
            
            stats_text = f'Mean: {mean_auc:.4f}\nMedian: {median_auc:.4f}\nStd: {std_auc:.4f}\nQ25: {q25:.4f}\nQ75: {q75:.4f}'
            ax3.text(0.98, 0.02, stats_text, transform=ax3.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))
            
            ax3.set_xticks(x_positions)
            ax3.set_xticklabels([f'Fold {i+1}' for i in range(len(aucs))], rotation=45, ha='right')
            ax3.legend(loc='upper left', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
        
        ax3.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax3.set_title('AUC by Fold', fontsize=13, fontweight='bold', pad=15)
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 4. 所有指标综合柱形图（分组柱形图）
        ax4 = axes[1, 1]
        all_metrics = []
        all_labels = []
        all_colors = []
        
        if accuracies and len(accuracies) > 0:
            all_metrics.append(accuracies)
            all_labels.append('Accuracy')
            all_colors.append('#3498DB')
        if recalls and len(recalls) > 0:
            all_metrics.append(recalls)
            all_labels.append('Recall')
            all_colors.append('#E74C3C')
        if f1_scores and len(f1_scores) > 0:
            all_metrics.append(f1_scores)
            all_labels.append('F1 Score')
            all_colors.append('#27AE60')
        if aucs and len(aucs) > 0:
            all_metrics.append(aucs)
            all_labels.append('AUC')
            all_colors.append('#9B59B6')
        
        if all_metrics:
            n_metrics = len(all_metrics)
            n_folds_display = len(all_metrics[0])
            x = np.arange(n_folds_display)
            width = 0.8 / n_metrics  # 根据指标数量调整宽度
            
            for i, (data, label, color) in enumerate(zip(all_metrics, all_labels, all_colors)):
                offset = (i - n_metrics/2 + 0.5) * width
                bars = ax4.bar(x + offset, data, width, label=label, 
                              color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
                # 添加数值标签（仅显示前几个，避免过于拥挤）
                if n_folds_display <= 5:
                    for bar, val in zip(bars, data):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
            
            ax4.set_xticks(x)
            ax4.set_xticklabels([f'Fold {i+1}' for i in range(n_folds_display)], rotation=45, ha='right')
            ax4.legend(loc='upper left', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
        
        ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax4.set_title('All Metrics Comparison by Fold', fontsize=13, fontweight='bold', pad=15)
        ax4.set_ylim([0, 1.1])
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    try:
        plt.tight_layout()
        
        # 保存图形
        plot_file = output_dir / "cv_training_curves.png"
        
        # 确保文件路径有效
        try:
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', format='png')
            
            # 验证文件是否成功创建且不为空
            if plot_file.exists() and plot_file.stat().st_size > 0:
                logger.info(f"   Cross-validation bar plot saved: {plot_file} (size: {plot_file.stat().st_size:,} bytes)")
            else:
                logger.error(f"   File save failed or file is empty: {plot_file}")
                if plot_file.exists():
                    plot_file.unlink()  # 删除损坏的文件
        except Exception as save_error:
            logger.error(f"   Failed to save cross-validation curves: {str(save_error)}", exc_info=True)
            # 尝试保存为备用格式
            try:
                plot_file_backup = output_dir / "cv_training_curves_backup.png"
                plt.savefig(plot_file_backup, dpi=150, bbox_inches='tight', facecolor='white', format='png')
                logger.warning(f"   Backup file saved: {plot_file_backup}")
            except Exception as backup_error:
                logger.error(f"   Backup save also failed: {str(backup_error)}")
        finally:
            plt.close()
    except Exception as e:
        logger.error(f"   Error occurred while plotting cross-validation curves: {str(e)}", exc_info=True)
        try:
            plt.close()
        except:
            pass

def plot_model_performance_from_file(
    plotting_data_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    publication_quality: Optional[bool] = None
) -> int:
    """
    从plotting_data.npz文件读取数据并绘制模型性能曲线
    
    参数:
        plotting_data_file: plotting_data.npz文件路径
        output_dir: 输出目录（默认与plotting_data_file同目录）
        publication_quality: 是否生成期刊质量图表（默认从文件读取）
        
    返回:
        0: 成功
        1: 失败
    """
    try:
        # 加载绘图数据
        plotting_data = load_plotting_data(plotting_data_file)
        
        # 确定输出目录
        plotting_data_file = Path(plotting_data_file)
        if output_dir is None:
            output_dir = plotting_data_file.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定publication_quality
        if publication_quality is None:
            publication_quality = plotting_data['publication_quality']
        
        # 绘制性能曲线
        if plotting_data['y_test'] is not None and plotting_data['y_pred'] is not None:
            plot_performance_curves(
                plotting_data['y_test'],
                plotting_data['y_pred'],
                plotting_data['y_prob'],
                output_dir,
                plotting_data['model_type'],
                plotting_data['task_type'],
                publication_quality
            )
            logger.info(f"Performance evaluation curves generated: {output_dir / 'performance_curves.png'}")
        
        # 绘制交叉验证曲线
        if plotting_data['cv_results'] is not None:
            plot_cv_training_curves(
                plotting_data['cv_results'],
                output_dir,
                plotting_data['model_type'],
                plotting_data['task_type']
            )
            logger.info(f"Cross-validation curves generated: {output_dir / 'cv_training_curves.png'}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        return 2
    except Exception as e:
        logger.error(f"Plotting failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="基因组数据散点图可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="输入文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出文件前缀")
    parser.add_argument("--feature-col", help="特征列名")
    parser.add_argument("--value-col", help="数值列名")
    parser.add_argument("--dpi", type=int, default=1200, help="静态图像分辨率")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--static-only", action="store_true", help="仅生成静态散点图（PNG）")
    group.add_argument("--interactive-only", action="store_true", help="仅生成交互式散点图（HTML）")
    
    args = parser.parse_args()
    sys.exit(run_visualization(**vars(args)))