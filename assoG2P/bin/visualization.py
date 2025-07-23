import argparse
import logging
import os
import sys
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
logger = logging.getLogger(__name__)

class EnhancedGenomeVisualizer:
    """
    增强型基因组数据可视化工具
    功能：
    - 支持静态/交互式可视化
    - 自动染色体位置标注
    - 动态点大小调整
    - 自动计算显著性阈值
    - 多图表类型支持
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
        
        try:
            self._initialize_data()
            logger.info(f"成功加载 {len(self.df)} 个数据点")
            logger.debug(f"数据列信息: {self.df.columns.tolist()}")
            logger.debug(f"数值列统计: {self.df[self.value_col].describe()}")
        except Exception as e:
            logger.error(f"数据初始化失败: {str(e)}", exc_info=True)
            raise

    def _initialize_data(self):
        """数据加载和预处理"""
        self.df = self._load_and_process()
        self.threshold = self._calculate_threshold()

    def _detect_feature_column(self, columns: List[str]) -> str:
        """自动检测特征列"""
        for col in columns:
            if pd.Series([col]).str.contains(r'\d+_\d+').any():
                return col
        return columns[0]

    def _detect_value_column(self, columns: List[str]) -> str:
        """自动检测数值列"""
        sample_df = pd.read_csv(self.input_file, nrows=100)
        for col in columns:
            if col != self.feature_col and pd.api.types.is_numeric_dtype(sample_df[col]):
                return col
        return columns[1] if len(columns) > 1 else columns[0]

    def _load_and_process(self) -> pd.DataFrame:
        """数据加载和处理流程"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")

        # 列检测
        sample_df = pd.read_csv(self.input_file, nrows=100)
        if self.feature_col is None:
            self.feature_col = self._detect_feature_column(sample_df.columns)
        if self.value_col is None:
            self.value_col = self._detect_value_column(sample_df.columns)

        # 完整加载
        df = pd.read_csv(
            self.input_file,
            usecols=[self.feature_col, self.value_col],
            dtype={self.feature_col: 'string', self.value_col: 'float32'}
        )

        # 数据校验
        if len(df) == 0:
            raise ValueError("输入文件为空")
        logger.debug(f"原始数据形状: {df.shape}, 列: {df.columns.tolist()}")
        logger.debug(f"数值列统计摘要: {df[self.value_col].describe()}")
        
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
            df = df.sample(self.max_points, random_state=42)
            logger.info(f"数据抽样至 {self.max_points} 个点")
            logger.debug(f"抽样后数据形状: {df.shape}")

        # 基因组位置解析
        chrom_pos = df[self.feature_col].str.extract(r'^(?P<chrom>\d+)_(?P<pos>\d+)$')
        if chrom_pos.isna().any().any():
            invalid_samples = df.loc[chrom_pos.isna().any(axis=1), self.feature_col].head(3).tolist()
            raise ValueError(
                f"特征列格式应为'染色体_位置'(如1_12345)\n"
                f"无效样本示例: {invalid_samples}"
            )

        # 数据增强
        df = df.assign(
            chrom_num=chrom_pos['chrom'].astype('uint8'),
            position=chrom_pos['pos'].astype('uint32'),
            chrom_label="Chr" + chrom_pos['chrom'],
            value_sign=np.where(df[self.value_col] >= 0, 'Positive', 'Negative'),
            value_abs=df[self.value_col].abs()
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
        size = max(1, 50 / (1 + np.log10(n + 1)))
        alpha = min(0.9, 0.7 / (1 + n / 5e4))
        return size, alpha

    def plot(
        self,
        output_file: str,
        plot_type: str = "scatter",
        interactive: bool = False,
        dpi: int = 150,
        **kwargs
    ) -> None:
        """
        主绘图方法
        参数:
            output_file: 输出文件路径
            plot_type: 图表类型(scatter/bar/line/hist)
            interactive: 是否使用交互模式
            dpi: 图像分辨率(静态图)
        """
        plot_types = ["scatter", "bar", "line", "hist"]
        if plot_type not in plot_types:
            raise ValueError(f"不支持的图表类型: {plot_type} (可选: {plot_types})")

        # 绘图前综合调试摘要
        logger.debug(
            "===== 绘图参数调试摘要 =====\n"\
            f"图表类型: {plot_type}, 交互模式: {interactive}\n"\
            f"输入文件: {self.input_file}, 输出文件: {output_file}\n"\
            f"特征列: {self.feature_col}, 数值列: {self.value_col}\n"\
            f"数据点数量: {len(self.df)}, 染色体数量: {self.df['chrom_num'].nunique()}\n"\
            f"数值范围: [{self.df[self.value_col].min():.4f}, {self.df[self.value_col].max():.4f}]\n"\
            f"显著性阈值: {self.threshold:.4f}\n"\
            "=============================="
        )
        
        try:
            if interactive:
                self._plot_interactive(output_file, plot_type, **kwargs)
            else:
                self._plot_static(output_file, plot_type, dpi,** kwargs)
            logger.info(f"图表成功保存到 {output_file}")
        except Exception as e:
            logger.error(f"绘图失败: {str(e)}", exc_info=True)
            logger.error(f"绘图参数: 类型={plot_type}, 交互={interactive}, 输出={output_file}")
            raise

    def _plot_static(
        self,
        output_file: str,
        plot_type: str,
        dpi: int,
        **kwargs
    ) -> None:
        """静态图绘制"""
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(18, 8), dpi=dpi)
        
        # 颜色定义
        colors = {'Positive': '#2980b9', 'Negative': '#c0392b'}
        
        if plot_type == "scatter":
            self._plot_scatter_static(ax, colors)
        elif plot_type == "bar":
            self._plot_bar_static(ax, colors)
        elif plot_type == "line":
            self._plot_line_static(ax, colors)
        elif plot_type == "hist":
            self._plot_hist_static(ax, colors)

        if plot_type == 'scatter':
            self._add_threshold_lines(ax)
        self._format_axes(ax, plot_type)
        self._save_figure(fig, output_file, dpi)

    def _plot_scatter_static(self, ax: Axes, colors: Dict[str, str]):
        """静态散点图"""
        size, alpha = self._dynamic_style()
        
        for sign, color in colors.items():
            subset = self.df[self.df['value_sign'] == sign]
            ax.scatter(
                x=subset['x_pos'].values.flatten(),
                y=subset[self.value_col].values.flatten(),
                c=color,
                s=size,
                alpha=alpha,
                edgecolors='none',
                label=sign
            )

    def _plot_bar_static(self, ax: Axes, colors: Dict[str, str]):
        """静态柱状图"""
        chrom_stats = self.df.groupby('chrom_label', as_index=False).agg(
            mean_value=(self.value_col, 'mean'),
            mean_abs=('value_abs', 'mean'),
            median_x=('x_pos', 'median')  # 添加中位数位置
        )
        # 按染色体编号排序
        chrom_stats = chrom_stats.sort_values(
            by='chrom_label',
            key=lambda x: x.str.extract('(\d+)', expand=False).astype(int)
        )
        # 确保数据为1维数组
        chrom_stats['mean_abs'] = chrom_stats['mean_abs'].values.flatten()
        chrom_stats['mean_value'] = chrom_stats['mean_value'].values.flatten()
        chrom_stats['median_x'] = chrom_stats['median_x'].values.flatten()
        
        # 计算相邻染色体位置的最小间距
        sorted_x = chrom_stats['median_x']
        min_spacing = sorted_x.diff().dropna().min() if len(sorted_x) > 1 else 1
        # 使用最小间距的80%作为柱子宽度
        dynamic_width = min_spacing * 0.8
        
        # 创建颜色列表
        colors_list = [colors['Positive'] if val >= 0 else colors['Negative'] for val in chrom_stats['mean_value']]
        
        # 绘制柱状图
        ax.bar(
            x=chrom_stats['median_x'],
            height=chrom_stats['mean_abs'],
            color=colors_list,
            edgecolor='black',
            width=dynamic_width
        )
        
        # 调试颜色分配和宽度计算
        for _, row in chrom_stats.iterrows():
            logger.debug(f"染色体 {row['chrom_label']} 均值: {row['mean_value']}, 颜色: {colors['Positive'] if row['mean_value'] >= 0 else colors['Negative']}")
        logger.debug(f"动态宽度计算: 最小间距={min_spacing}, 宽度={dynamic_width}")

    def _plot_line_static(self, ax: Axes, colors: Dict[str, str]):
        """静态折线图"""
        stats = self.df.groupby('chrom_label').agg(
              mean=('value_abs', 'mean'),
              std=('value_abs', 'std')
          ).sort_values(
              by='chrom_label',
              key=lambda x: x.str.extract('(\d+)', expand=False).astype(int)
          )
        logger.debug(f"折线图统计数据形状: {stats.shape}")
        logger.debug(f"折线图统计数据类型: {stats.dtypes}")
        logger.debug(f"前5行统计数据: {stats.head().to_dict()}")
        # 按染色体编号排序
        stats = stats.reindex(sorted(stats.index, key=lambda x: int(x.replace('Chr', ''))))
        
        x = np.arange(len(stats))
        # Flatten arrays to ensure 1-dimensional input
        mean_values = stats['mean'].values.flatten()
        std_values = stats['std'].values.flatten()
        
        # 验证数据维度
        logger.debug(f"静态折线图x形状: {x.shape}, y形状: {mean_values.shape}")
        if x.ndim != 1 or mean_values.ndim != 1 or std_values.ndim != 1:
            logger.error(f"静态折线图数据维度错误: x={x.ndim}D, y={mean_values.ndim}D, std={std_values.ndim}D")
            raise ValueError("折线图数据必须是1维数组")
        
        try:
            ax.plot(x, mean_values, color=colors['Positive'], marker='o')
            ax.fill_between(
                x,
                mean_values - std_values,
                mean_values + std_values,
                color=colors['Positive'],
                alpha=0.2
            )
            logger.debug(f"静态折线图绘制成功，数据点数量: {len(x)}")
        except Exception as e:
            logger.error(f"静态折线图绘制失败: {str(e)}", exc_info=True)
            raise
        ax.set_xticks(x)
        ax.set_xticklabels(stats.index)

    def _plot_hist_static(self, ax: Axes, colors: Dict[str, str]):
        """静态直方图"""
        bins = min(50, int(len(self.df) ** 0.5))  # 自适应bin数量
        pos = self.df[self.df['value_sign'] == 'Positive'][self.value_col].values.flatten()
        neg = self.df[self.df['value_sign'] == 'Negative'][self.value_col].values.flatten()
        
        ax.hist(
            [pos, neg],
            bins=bins,
            color=[colors['Positive'], colors['Negative']],
            stacked=True,
            label=['Positive', 'Negative']
        )

    def _plot_interactive(
        self,
        output_file: str,
        plot_type: str,
        **kwargs
    ) -> None:
        """交互式绘图"""

        if plot_type == "scatter":
            fig = self._create_interactive_scatter()
        elif plot_type == "bar":
            fig = self._create_interactive_bar()
        elif plot_type == "line":
            fig = self._create_interactive_line()
        elif plot_type == "hist":
            fig = self._create_interactive_hist()
        else:
            raise ValueError(f"交互模式暂不支持 {plot_type} 图表类型")

        output_path = Path(output_file)
        if output_path.suffix == '.html':
            fig.write_html(output_path, include_plotlyjs='cdn')
            logger.debug(f"成功写入交互式HTML图表: {output_path}")
        else:
            try:
                write_image(fig, output_path, scale=2, engine='kaleido')
                logger.debug(f"成功写入静态图像: {output_path}")
            except Exception as e:
                logger.error(f"图像写入失败，尝试使用orca引擎...", exc_info=True)
                try:
                    write_image(fig, output_path, scale=2, engine='orca')
                    logger.debug(f"成功使用orca引擎写入静态图像: {output_path}")
                except Exception as e2:
                    logger.error(f"使用orca引擎写入图像也失败: {str(e2)}", exc_info=True)
                    raise

    def _create_interactive_line(self):
        """交互式折线图"""
        stats = self.df.groupby('chrom_label').agg(
            mean=('value_abs', 'mean'),
            std=('value_abs', 'std')
        ).reset_index()
        logger.debug(f"交互式折线图统计数据形状: {stats.shape}")
        logger.debug(f"交互式统计数据列: {stats.columns.tolist()}")
        logger.debug(f"交互式统计数据前5行: {stats.head().to_dict()}")
        # 确保数据为1维数组
        stats['mean'] = stats['mean'].values.flatten()
        stats['std'] = stats['std'].values.flatten()
        # 按染色体编号排序
        stats = stats.sort_values(
            by='chrom_label',
            key=lambda x: x.str.extract('(\d+)', expand=False).astype(int)
        )
        
        # 验证数据维度
        logger.debug(f"交互式折线图x形状: {stats['chrom_label'].shape}, y形状: {stats['mean'].shape}")
        if stats['chrom_label'].ndim != 1 or stats['mean'].ndim != 1 or stats['std'].ndim != 1:
            logger.error(f"交互式折线图数据维度错误: x={stats['chrom_label'].ndim}D, y={stats['mean'].ndim}D, std={stats['std'].ndim}D")
            raise ValueError("折线图数据必须是1维数组")
        
        try:
            fig = px.line(
                stats,
                x='chrom_label',
                y='mean',
                error_y='std',
                markers=True,
                title=f'染色体区域{self.value_col}绝对值平均值',
                labels={'mean': f'{self.value_col}绝对值平均值', 'chrom_label': '染色体'}
            )
            logger.debug(f"交互式折线图创建成功，数据点数量: {len(stats)}")
        except Exception as e:
            logger.error(f"交互式折线图创建失败: {str(e)}", exc_info=True)
            raise
        
        fig.update_traces(line=dict(color='#3498db'))
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        return fig

    def _create_interactive_hist(self):
        """交互式直方图"""
        pos = self.df[self.df['value_sign'] == 'Positive'][self.value_col].values.flatten()
        neg = self.df[self.df['value_sign'] == 'Negative'][self.value_col].values.flatten()
        
        logger.debug(f"直方图数据: 正样本={len(pos)}, 负样本={len(neg)}")
        if len(pos) == 0 and len(neg) == 0:
            raise ValueError("没有可绘制的数据点，请检查输入数据")
        
        # 根据数据存在情况创建直方图轨迹
        if len(pos) > 0:
            fig = px.histogram(
                x=pos,
                color_discrete_sequence=['#3498db'],
                name='Positive',
                title=f'{self.value_col} 分布直方图',
                labels={"x": self.value_col, "count": "频率"}
            )
        else:
            fig = px.histogram(
                x=neg,
                color_discrete_sequence=['#e74c3c'],
                name='Negative',
                title=f'{self.value_col} 分布直方图',
                labels={"x": self.value_col, "count": "频率"}
            )
        
        # 添加另一组数据（如果存在）
        if len(pos) > 0 and len(neg) > 0:
            fig.add_histogram(
                x=neg,
                color_discrete_sequence=['#e74c3c'],
                name='Negative'
            )
        elif len(neg) > 0 and len(pos) == 0:
            pass  # 已在初始fig中创建
        elif len(pos) > 0 and len(neg) == 0:
            pass  # 已在初始fig中创建
        
        fig.update_layout(
            barmode='overlay',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            legend=dict(
                title='效应方向',
                itemsizing='constant',
                traceorder='normal'
            )
        )
        
        fig.update_traces(opacity=0.75)
        return fig

    def _create_interactive_scatter(self):
        """交互式散点图"""
        size, alpha = self._dynamic_style()
        
        fig = px.scatter(
            self.df,
            x=self.df['x_pos'].values.flatten(),
            y=self.df[self.value_col].values.flatten(),
            color='value_sign',
            color_discrete_map={
                'Positive': '#3498db',
                'Negative': '#e74c3c'
            },
            hover_data={
                'chrom_label': True,
                'position': True,
                self.value_col: ':.3f',
                'x_pos': False
            },
            size_max=size,
            opacity=alpha,
            width=1200,
            height=500,
            title=f"Genome-Wide Association (Top 1% threshold: ±{self.threshold:.2f})"
        )
        
        self._format_interactive_layout(fig, 'scatter')
        return fig

    def _create_interactive_bar(self):
        """交互式柱状图"""
        chrom_stats = self.df.groupby('chrom_label', as_index=False).agg(
            mean_value=(self.value_col, 'mean'),
            mean_abs=('value_abs', 'mean')
        )
        
        # Flatten arrays to ensure 1-dimensional input
        chrom_stats['mean_abs'] = chrom_stats['mean_abs'].values.flatten()
        chrom_stats['mean_value'] = chrom_stats['mean_value'].values.flatten()
        
        fig = px.bar(
            chrom_stats,
            x='chrom_label',
            y='mean_abs',
            color=np.where(chrom_stats['mean_value'] >= 0, 'Positive', 'Negative'),
            color_discrete_map={
                'Positive': '#3498db',
                'Negative': '#e74c3c'
            },
            hover_data={
                'mean_value': ':.3f',
                'mean_abs': ':.3f'
            },
            width=1200,
            height=500,
            title=f"Chromosome-wise Mean Values (Threshold: ±{self.threshold:.2f})"
        )
        
        self._format_interactive_layout(fig, 'bar')
        return fig

    def _format_interactive_layout(self, fig, plot_type: str):
        """格式化交互图表布局"""
        if plot_type in ['scatter', 'bar', 'hist']:
            fig.update_layout(
                xaxis_title=self.value_col if plot_type == 'hist' else "Chromosome",
                yaxis_title="Frequency" if plot_type == 'hist' else ("Value" if plot_type == 'scatter' else "Mean Absolute Value"),
                hovermode="x unified",
                showlegend=True
            )
            
            # 添加阈值线
            if plot_type == 'scatter':
                fig.add_hline(
                    y=self.threshold,
                    line_dash="dot",
                    line_color="grey",
                    annotation_text=f"Top 1%: {self.threshold:.2f}"
                )
                fig.add_hline(
                    y=-self.threshold,
                    line_dash="dot",
                    line_color="grey"
                )

    def _add_threshold_lines(self, ax: Axes):
        """添加阈值线"""
        ax.axhline(self.threshold, color='#2c3e50', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(-self.threshold, color='#2c3e50', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(
            0.01, 0.95,
            f"Top 1% Threshold: ±{self.threshold:.2f}",
            transform=ax.transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    def _format_axes(self, ax: Axes, plot_type: str):
        """格式化坐标轴"""
        # 染色体刻度
        if plot_type in ['scatter', 'bar']:
            chrom_ticks = self.df.groupby('chrom_label')['x_pos'].median()
            ax.set_xticks(chrom_ticks.values)
            ax.set_xticklabels(chrom_ticks.index, rotation=45, ha='right')
            ax.set_xlabel("Genomic Position")
            
            # 染色体分隔线
            for chrom, group in self.df.groupby('chrom_label'):
                ax.axvline(group['x_pos'].min() - 0.5, color='gray', linestyle=':', alpha=0.3)

        # 标签和标题
        ylabel = {
            'scatter': 'Association Value',
            'bar': 'Mean Absolute Value',
            'line': 'Value ± SD',
            'hist': 'Frequency'
        }.get(plot_type, 'Value')
        
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{plot_type.capitalize()} Plot "
            f"(Total {len(self.df):,} variants, {self.df['chrom_num'].nunique()} chromosomes)",
            pad=20
        )
        # 动态调整Y轴范围以增强可视性
        if plot_type == 'bar':
            # 柱状图从0开始并添加适当边距
            y_max = ax.get_ylim()[1]
            padding = max(0.1 * y_max, 0.1)  # 10% padding或最小0.1
            ax.set_ylim(0, y_max + padding)
        else:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            # 扩展Y轴范围10%并确保包含原点
            ax.set_ylim(min(y_min - 0.1*y_range, 0), max(y_max + 0.1*y_range, 0))
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        # 增大字体大小提高可读性
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_size(14)
        
        # 图例
        if plot_type in ['scatter', 'hist']:
            ax.legend(title='Effect Direction', loc='upper right')

    def _save_figure(self, fig, output_path: str, dpi: int):
        """保存图像"""
        path = Path(output_path)
        # 验证输出路径
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"创建输出目录: {path.parent}")
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
            logger.debug(f"尝试保存静态图像: {path}, DPI: {dpi}")
            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
            )
            logger.debug(f"静态图像保存成功，文件大小: {os.path.getsize(path)} bytes")
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            raise

def run_visualization(
    input_file: str,
    output_file: str,
    feature_col: Optional[str] = None,
    value_col: Optional[str] = None,
    plot_type: str = "scatter",
    interactive: bool = False,
    dpi: int = 300,
    **kwargs
) -> int:
    """
    可视化入口函数
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
        
        visualizer.plot(
            output_file=output_file,
            plot_type=plot_type,
            interactive=interactive,
            dpi=dpi,
            **kwargs
        )
        logger.info("可视化任务成功完成")
        return 0
        
    except FileNotFoundError as e:
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
        description="基因组数据可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="输入文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出文件路径")
    parser.add_argument("--feature-col", help="特征列名")
    parser.add_argument("--value-col", help="数值列名")
    parser.add_argument("-t", "--type", default="scatter", 
                       choices=["scatter", "bar", "line", "hist"], help="图表类型")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--dpi", type=int, default=300, help="图像分辨率(静态图)")
    
    args = parser.parse_args()
    sys.exit(run_visualization(**vars(args)))