import argparse
import logging
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List
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
        max_points: int = 500000,
        auto_sample: bool = True
    ):
        """
        初始化可视化工具
        参数:
            input_file: 输入文件路径
            feature_col: 特征列名(格式: chrom_pos)
            value_col: 数值列名
            max_points: 最大显示点数
            auto_sample: 数据过大时自动抽样
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
            'Positive': '#2980b9',  # 蓝色
            'Negative': '#c0392b'   # 红色
        }
        
        try:
            self._initialize_data()
        except Exception as e:
            logger.error(f"数据初始化失败: {str(e)}")
            raise

    def _initialize_data(self):
        """数据加载和预处理"""
        self.df = self._load_and_process()
        self.threshold = self._calculate_threshold()
        # 打印基础数据信息
        try:
            logger.info(
                f"📥 Visualization data loaded: file={self.input_file}, "
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
            logger.warning(f"数值列 {self.value_col} 包含 {df[self.value_col].isna().sum()} 个缺失值，将被自动删除")
            df = df.dropna(subset=[self.value_col])
            if len(df) == 0:
                raise ValueError("删除缺失值后数据为空")
        
        # 检查数据分布
        value_range = df[self.value_col].max() - df[self.value_col].min()
        if value_range < 1e-9:
            logger.warning(f"数值列 {self.value_col} 取值范围过小，可能影响可视化效果")
        
        if len(df) > self.max_points and self.auto_sample:
            before_n = len(df)
            df = df.sample(self.max_points, random_state=42)
            logger.info(
                f"Auto-sampling enabled: down-sampled from {before_n} to {len(df)} points "
                f"(max_points={self.max_points})"
            )

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
        dpi: int = 300,
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
            logger.error(f"静态散点图绘制失败: {str(e)}")
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
            logger.error(f"交互式散点图绘制失败: {str(e)}")
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
                logger.error(f"图像写入失败，尝试使用orca引擎...", exc_info=True)
                try:
                    write_image(fig, output_path, scale=2, engine='orca')
                except Exception as e2:
                    logger.error(f"图像保存失败: {str(e2)}")
                    raise

    def _plot_scatter_static(self, ax: Axes, colors: Dict[str, str]):
        """静态散点图绘制实现"""
        size, alpha = self._dynamic_style()
        
        for sign, color in colors.items():
            subset = self.df[self.df['value_sign'] == sign]
            x_values = subset['x_pos'].values.flatten()
            # 使用value_signed列（包含正负的值）用于绘图
            y_values = subset['value_signed'].values.flatten()
            
            # 绘制散点
            ax.scatter(
                x=x_values,
                y=y_values,
                c=color,
                s=size,
                alpha=alpha,
                edgecolors='none',
                label=sign
            )
            
            # 添加从每个点到Y=0的垂直线
            for x, y in zip(x_values, y_values):
                ax.plot([x, x], [0, y], color=color, linestyle='-', alpha=0.3, linewidth=2)
        
        # 确保显示图例
        ax.legend(title='Effect Direction', loc='upper right')

    def _create_interactive_scatter(self):
        """交互式散点图创建"""
        size, alpha = self._dynamic_style()
        
        # 创建散点图（使用统一的颜色定义，仅绘制点本身，交互式查看细节）
        fig = px.scatter(
            self.df,
            x=self.df['x_pos'].values.flatten(),
            y=self.df['value_signed'].values.flatten(),  # 使用value_signed列（包含正负的值）
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
            title="Genome-Wide Association Scatter Plot"  # 与静态图标题一致
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
        
        # 计算Y轴范围：围绕0对称，便于上下居中观察正负效应
        y_min = self.df['value_signed'].min()
        y_max = self.df['value_signed'].max()
        y_max_abs = max(abs(y_min), abs(y_max))
        # 稍微加一点padding，避免点贴边
        padding = 0.1 * y_max_abs if y_max_abs > 0 else 1.0
        y_min_adjusted = - (y_max_abs + padding)
        y_max_adjusted = (y_max_abs + padding)
        
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
                'text': "Genome-Wide Association Scatter Plot",
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
                y0=y_min_adjusted,
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
            "Genome-Wide Association Scatter Plot",
            pad=20,
            fontsize=16
        )
        
        # 设置y轴刻度字体大小
        ax.tick_params(axis='y', labelsize=16)
        
        # 动态调整Y轴范围以增强可视性
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        # 扩展Y轴范围10%并确保包含原点
        ax.set_ylim(min(y_min - 0.1*y_range, 0), max(y_max + 0.1*y_range, 0))
        
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
            logger.error(f"保存失败: {str(e)}")
            raise

def run_visualization(
    input_file: str,
    output_prefix: str,
    feature_col: Optional[str] = None,
    value_col: Optional[str] = None,
    dpi: int = 300,
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

        logger.info(f"可视化完成")
        return 0

    except FileNotFoundError:
        logger.error(f"文件错误: 无法找到输入文件 '{input_file}'\n请检查文件路径是否正确")
        return 2
    except ValueError as e:
        logger.error(f"数据格式错误: {str(e)}\n输入文件: {input_file}\n特征列: {feature_col}, 数值列: {value_col}")
        return 3
    except ImportError as e:
        logger.error(f"依赖错误: {str(e)}\n请运行 'pip install plotly kaleido' 安装必要依赖")
        return 4
    except Exception as e:
        logger.error(f"可视化失败: {str(e)}")
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
    parser.add_argument("--dpi", type=int, default=300, help="静态图像分辨率")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--static-only", action="store_true", help="仅生成静态散点图（PNG）")
    group.add_argument("--interactive-only", action="store_true", help="仅生成交互式散点图（HTML）")
    
    args = parser.parse_args()
    sys.exit(run_visualization(**vars(args)))