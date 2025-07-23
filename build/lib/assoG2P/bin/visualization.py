
import argparse
import logging
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
# Initialize module logger
logger = logging.getLogger(__name__)

class DataVisualizer:
    """Main class for data visualization operations"""
    
    def __init__(
        self, 
        data_path: str,
        feature_col: Optional[str] = None,
        shap_col: Optional[str] = None
    ):
        """Initialize visualizer with data source
        
        Args:
            data_path: Path to input data file
            feature_col: Name of feature column (default: first column)
            shap_col: Name of SHAP value column (default: second column)
        """
        self.data_path = data_path
        self.feature_col = feature_col
        self.shap_col = shap_col
        self.df = self._load_data()
        self._validate_data()
        self._preprocess_data()
        self._setup_chromosome_ticks()

    def _load_data(self) -> pd.DataFrame:
        """Load input data with automatic delimiter detection
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If data format is invalid
        """
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {self.data_path}")
        
        try:
            # Auto-detect delimiter
            with open(path, 'r') as f:
                sample = f.readline()
                
            if '\t' in sample:
                df = pd.read_csv(path, sep='\t')
            elif ',' in sample:
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, sep='\s+')  # Fallback to whitespace
                
            logger.info(f"Loaded data from {path}: {df.shape[0]} rows")
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _validate_data(self) -> None:
        """Validate data columns and types"""
        if self.feature_col is None:
            self.feature_col = self.df.columns[0]
        if self.shap_col is None:
            self.shap_col = self.df.columns[1] if len(self.df.columns) > 1 else None
            
        if self.feature_col not in self.df.columns:
            raise ValueError(f"Feature column '{self.feature_col}' not found")
        if self.shap_col is None:
            raise ValueError("Data must contain at least 2 columns (features and SHAP values)")
        if not pd.api.types.is_numeric_dtype(self.df[self.shap_col]):
            raise ValueError(f"SHAP column '{self.shap_col}' must be numeric")

    def _preprocess_data(self) -> None:
        """Preprocess data for visualization"""
        sample_features = self.df[self.feature_col].head()
        if not all(re.match(r'\d+_\d+', str(x)) for x in sample_features):
            raise ValueError("The trait name must be 'chromosome_location' format (such as 1_1973872)")
        # Extract chromosome info from feature names
        chrom_data = self._extract_chromosome_info(self.df[self.feature_col])
        
        # Add metadata columns
        self.df = self.df.assign(
            chromosome=chrom_data['chromosome'],
            position=chrom_data['position'],
            chrom_num=chrom_data['chromosome'].str.extract(r'(\d+)').astype(int),
            shap_abs=self.df[self.shap_col].abs(),
            shap_sign=np.where(self.df[self.shap_col] >= 0, 'Positive', 'Negative')
        )
        
        # Sort by chromosome and position
        self.df = self.df.sort_values(['chrom_num', 'position'])
        logger.info(f"Processed {len(self.df)} variants across {self.df['chrom_num'].nunique()} chromosomes")

    def _extract_chromosome_info(self, features: pd.Series) -> pd.DataFrame:
        chrom_pos = []
        pattern = re.compile(r'^(\d+)_(\d+)$') 
    
        for feat in features:
            match = pattern.match(str(feat))
            if match:
                chrom, pos = match.groups()
                chrom_pos.append((feat, f"Chr{chrom}", int(pos)))
            else:
                raise ValueError(f"Invalid feature name format: {feat}")
    
        return pd.DataFrame(chrom_pos, columns=['feature', 'chromosome', 'position'])

    def _setup_chromosome_ticks(self) -> None:
        """Calculate chromosome positions for plot ticks"""
        chrom_groups = self.df.groupby('chrom_num', sort=True)
        self.chrom_centers = []
        self.chrom_ticks = {}
        
        start_pos = 0
        for chrom, group in chrom_groups:
            end_pos = start_pos + len(group)
            center = (start_pos + end_pos) / 2
            self.chrom_centers.append((chrom, center))
            
            # Create ticks (max 10 per chromosome)
            step = max(1, len(group) // 10)
            self.chrom_ticks[chrom] = np.arange(start_pos, end_pos, step)
            start_pos = end_pos

    def generate_plot(
        self,
        plot_type: str = "scatter",
        output_path: Optional[str] = None,
        dpi: int = 300,
        interactive: bool = False,
        **style_kwargs
    ) -> Optional[plt.Figure]:
        """Generate and save visualization
        
        Args:
            plot_type: Type of plot (scatter/bar/line/hist)
            output_path: Path to save output image
            dpi: Image resolution for static plots
            interactive: Whether to use interactive mode
            **style_kwargs: Additional styling parameters
            
        Returns:
            matplotlib Figure if static mode, None otherwise
        """
        plot_methods = {
            'scatter': self._plot_scatter_static,
            'bar': self._plot_bar_static,
            'line': self._plot_line_static,
            'hist': self._plot_hist_static
        }
        
        if plot_type not in plot_methods:
            raise ValueError(f"Invalid plot type: {plot_type}")
            
        logger.info(f"Generating {plot_type} plot")
        
        if interactive:
            self._plot_interactive(plot_type, output_path, **style_kwargs)
            return None
        else:
            fig = self._plot_static(plot_type, **style_kwargs)
            if output_path:
                self._save_static_plot(fig, output_path, dpi)
            return fig

    def _plot_static(
        self, 
        plot_type: str,
        **style_kwargs
    ) -> plt.Figure:
        """Generate static matplotlib plot"""
        plt.style.use('seaborn-v0_8') if 'seaborn-v0_8' in plt.style.available else plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=style_kwargs.get('figsize', (12, 6)))
        
        # Select and execute plot method
        plot_method = {
            'scatter': self._plot_scatter_static,
            'bar': self._plot_bar_static,
            'line': self._plot_line_static,
            'hist': self._plot_hist_static
        }[plot_type]
        
        plot_method(ax, **style_kwargs)
        self._format_axes(ax, plot_type)
        plt.tight_layout()
        return fig

    def _plot_scatter_static(self, ax: Axes):
        # 计算x轴坐标（按染色体和位置排序）
        self.df['x_pos'] = np.arange(len(self.df))
        
        # 绘制
        sns.scatterplot(
            x='x_pos',
            y='Importance',
            hue='shap_sign',
            data=self.df,
            palette={'Positive':'#3498db', 'Negative':'#e74c3c'},
            alpha=0.6,
            s=15
        )
        
        # 设置染色体分隔线
        for chrom in self.df['chrom_num'].unique():
            chrom_df = self.df[self.df['chrom_num'] == chrom]
            if len(chrom_df) > 0:
                x_min = chrom_df['x_pos'].min()
                x_max = chrom_df['x_pos'].max()
                ax.axvline(x=x_min, color='gray', linestyle='--', alpha=0.5)
        
        # 设置刻度标签
        ax.set_xticks([pos for _, pos in self.chrom_centers])
        ax.set_xticklabels([f"Chr{c}" for c, _ in self.chrom_centers])

    def _plot_bar_static(self, ax: Axes, **kwargs) -> None:
        """Static bar plot by chromosome"""
        chrom_stats = self.df.groupby('chrom_num')['shap_abs'].mean()
        chrom_stats.plot.bar(
            ax=ax,
            color=kwargs.get('color', 'skyblue'),
            width=0.8
        )

    def _plot_line_static(self, ax: Axes, **kwargs) -> None:
        """Static line plot with confidence intervals"""
        chrom_stats = self.df.groupby('chrom_num').agg({
            'shap_abs': ['mean', 'std']
        })
        chrom_stats.columns = ['mean', 'std']
        
        x = chrom_stats.index
        y = chrom_stats['mean']
        y_err = chrom_stats['std']
        
        ax.plot(x, y, color='royalblue', marker='o')
        ax.fill_between(
            x, 
            y - y_err, 
            y + y_err, 
            color='lightblue', 
            alpha=0.3
        )

    def _plot_hist_static(self, ax: Axes, **kwargs) -> None:
        """Static histogram of SHAP values"""
        sns.histplot(
            data=self.df,
            x='shap_abs',
            hue='chrom_num',
            multiple='stack',
            bins=kwargs.get('bins', 30),
            ax=ax
        )

    def _plot_interactive(
        self,
        plot_type: str,
        output_path: Optional[str],
        **style_kwargs
    ) -> None:
        """Generate interactive plot using Plotly"""
        try:
            import plotly.express as px
            from plotly.io import write_image
        except ImportError:
            logger.error("Interactive mode requires plotly. Install with: pip install plotly kaleido")
            raise
            
        if plot_type == 'scatter':
            fig = px.scatter(
                self.df,
                x=np.arange(len(self.df)),
                y='shap_abs',
                color='shap_sign',
                color_discrete_map={'Positive': 'blue', 'Negative': 'red'},
                hover_data=['chromosome', 'position']
            )
        elif plot_type == 'bar':
            chrom_stats = self.df.groupby('chrom_num', as_index=False)['shap_abs'].mean()
            fig = px.bar(
                chrom_stats,
                x='chrom_num',
                y='shap_abs',
                labels={'shap_abs': 'Mean |SHAP|'}
            )
        
        if output_path:
            write_image(fig, output_path, scale=2)
            logger.info(f"Saved interactive plot to {output_path}")

    def _format_axes(self, ax: Axes, plot_type: str) -> None:
        """Format plot axes and titles"""
        ax.set_title(f"SHAP Values - {plot_type.capitalize()} Plot")
        if plot_type != 'hist':
            ax.set_xlabel('Genomic Position')
            ax.set_ylabel('|SHAP Value|')
            
            # Set chromosome ticks
            chrom_labels = [f"Chr{c}" for c, _ in self.chrom_centers]
            ax.set_xticks([pos for _, pos in self.chrom_centers])
            ax.set_xticklabels(chrom_labels, rotation=45)
        else:
            ax.set_xlabel('SHAP Value')
            ax.set_ylabel('Count')

    def _save_static_plot(
        self,
        fig: plt.Figure,
        output_path: str,
        dpi: int = 300
    ) -> None:
        """Save static plot to file
        
        Args:
            fig: matplotlib Figure object
            output_path: Destination file path
            dpi: Image resolution
        """
        path = Path(output_path)
        path.parent.mkdir(exist_ok=True)
        
        try:
            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white'
            )
            logger.info(f"Saved plot to {path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to save plot: {str(e)}")
            raise

def run_visualization(
    input_path: str,
    output_path: str,
    plot_type: str = "scatter",
    feature_col: Optional[str] = None,
    shap_col: Optional[str] = None,
    interactive: bool = False,
    dpi: int = 300
) -> int:
    """Main visualization workflow
    
    Args:
        input_path: Input data file path
        output_path: Output image path
        plot_type: Type of visualization
        feature_col: Feature column name
        shap_col: SHAP value column name
        interactive: Use interactive plotting
        dpi: Image resolution (static only)
        
    Returns:
        0 on success, 1 on failure
    """
    try:
        visualizer = DataVisualizer(
            input_path,
            feature_col=feature_col,
            shap_col=shap_col
        )
        
        visualizer.generate_plot(
            plot_type=plot_type,
            output_path=output_path,
            dpi=dpi,
            interactive=interactive
        )
        return 0
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return 1

def cli_main() -> None:
    """Command line interface entry point"""
    parser = argparse.ArgumentParser(
        description="Genomic data visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Input data file")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument("-t", "--type", default="scatter",
                       choices=["scatter", "bar", "line", "hist"], help="Plot type")
    parser.add_argument("--feature-col", help="Feature column name")
    parser.add_argument("--shap-col", help="SHAP value column name")
    parser.add_argument("--interactive", action="store_true", help="Use interactive mode")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution (static)")
    
    args = parser.parse_args()
    sys.exit(run_visualization(
        input_path=args.input,
        output_path=args.output,
        plot_type=args.type,
        feature_col=args.feature_col,
        shap_col=args.shap_col,
        interactive=args.interactive,
        dpi=args.dpi
    ))

if __name__ == "__main__":
    cli_main()