#!/usr/bin/env python3
"""
Genotype-phenotype association analysis tool

Usage:
    association [command] [options]

Commands:
    preprocess    Data preprocessing (with optional GWAS/LD filtering)
    train         Model training (GWAS/LD filtering should be done in preprocess stage)
    train-all     Train all models (GWAS/LD filtering should be done in preprocess stage)
    predict       Prediction with trained model
    visualize     Result visualization

Examples:
    association preprocess -h
    association train -h
    association train-all -h
    association predict -h
    association visualize -h
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
import time
from typing import Optional

# 忽略警告信息
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ======================== 关键修复：确保 assoG2P 包在直接运行 main.py 时可导入 ========================
# 常见场景：用户在 /path/to/assoG2P 目录内直接运行 `python3 main.py`
# 此时 sys.path 默认包含当前目录（即 assoG2P 包目录本身），但不包含其父目录，
# 导致 `import assoG2P` 失败。这里将父目录加入 sys.path，保证 CLI 能正常导入子模块。
_pkg_dir = Path(__file__).resolve().parent
_pkg_parent = _pkg_dir.parent
if (_pkg_dir / "__init__.py").exists() and str(_pkg_parent) not in sys.path:
    sys.path.insert(0, str(_pkg_parent))

def init_logging(log_file: Optional[Path] = None) -> None:
    """Unified log configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True 
    )
    logging.captureWarnings(True)

def run_preprocess(args) -> int:
    """Perform data preprocessing"""
    from assoG2P.bin.preprocess import run_preprocess as preprocess_func
    try:
        logger.info("Starting preprocessing")
        # 解析SNP过滤参数
        filter_snps_value = not args.no_filter_snps
        # 获取GWAS/LD参数（如果未指定则使用默认值）
        feature_selection_mode = getattr(args, 'feature_selection_mode', 1)
        gwas_pvalue = getattr(args, 'gwas_pvalue', 0.01)
        ld_window_kb = getattr(args, 'ld_window_kb', 50)
        ld_window = getattr(args, 'ld_window', 5)
        ld_window_r2 = getattr(args, 'ld_window_r2', 0.2)
        use_cache = not getattr(args, 'no_cache', False)
        keep_temp_files = getattr(args, 'keep_temp_files', False)
        
        return preprocess_func(
            genotype_file=args.genotype,
            phenotype_file=args.phenotype,
            output_file=args.output,
            pheno_col=args.pheno_col,
            filter_snps=filter_snps_value,
            use_cache=use_cache,
            feature_selection_mode=feature_selection_mode,
            gwas_pvalue=gwas_pvalue,
            ld_window_kb=ld_window_kb,
            ld_window=ld_window,
            ld_window_r2=ld_window_r2,
            keep_temp_files=keep_temp_files,
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return 1

def run_training(args) -> int:
    """Perform model training (GWAS/LD filtering should be done in preprocess stage)"""
    try:
        from assoG2P.bin.modeltraining import run_single_model
        
        logger.info(f"训练: {args.model}")
        
        # GWAS/LD特征筛选已在preprocess模块完成，训练阶段不再执行筛选
        return run_single_model(
            input_path=args.json,
            model_type=args.model,
            output_dir=args.output_dir,
            task_type=args.task_type,
            n_folds=args.n_folds,
            random_state=args.random_state,
            calculate_feature_importance=getattr(args, 'feature_importance', False)
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

def run_train_all(args) -> int:
    """Train all models and compare results (GWAS/LD filtering should be done in preprocess stage)"""
    try:
        from assoG2P.bin.modeltraining import run_all_models
        
        logger.info(f"训练所有模型")
        
        # GWAS/LD特征筛选已在preprocess模块完成，训练阶段不再执行筛选
        return run_all_models(
            input_path=args.json,
            output_dir=args.output_dir,
            task_type=args.task_type,
            n_folds=args.n_folds,
            random_state=args.random_state,
            calculate_feature_importance=getattr(args, 'feature_importance', False)
        )
        
    except Exception as e:
        logger.error(f"Training all models failed: {str(e)}")
        return 1

def run_predict(args) -> int:
    """Perform prediction using trained models"""
    try:
        from assoG2P.bin.modeltraining import predict_with_model
        
        logger.info(f"预测")
        
        return predict_with_model(
            input_path=args.input,
            model_path=args.model,
            output_dir=args.output_dir,
            task_type=args.task_type
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return 1

def run_performance_visualization(args) -> int:
    """Perform model performance curves visualization"""
    try:
        from assoG2P.bin.visualization import plot_model_performance_from_file
        from pathlib import Path
        
        # 确定plotting_data.npz文件路径
        plotting_data_file = None
        if args.file:
            plotting_data_file = Path(args.file)
        elif args.model_dir:
            plotting_data_file = Path(args.model_dir) / "plotting_data.npz"
        else:
            logger.error("必须指定 --file 或 --model-dir 参数")
            return 1
        
        if not plotting_data_file.exists():
            logger.error(f"绘图数据文件不存在: {plotting_data_file}")
            return 2
        
        # 确定输出目录
        output_dir = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif args.model_dir:
            output_dir = Path(args.model_dir)
        elif args.file:
            output_dir = Path(args.file).parent
        
        # 确定publication_quality
        publication_quality = args.publication_quality if hasattr(args, 'publication_quality') else None
        
        logger.info(f"从文件读取绘图数据: {plotting_data_file}")
        return plot_model_performance_from_file(
            plotting_data_file=plotting_data_file,
            output_dir=output_dir,
            publication_quality=publication_quality
        )
        
    except Exception as e:
        logger.error(f"性能可视化失败: {str(e)}")
        return 1

def run_unified_visualization(args) -> int:
    """
    统一的可视化处理函数，支持同时处理特征重要性可视化（-i）和模型性能可视化（-f/-d）
    
    可以同时指定-i和-f/-d，同时生成两种图表
    执行顺序：先执行性能可视化（-f/-d），再执行特征重要性可视化（-i）
    """
    result_code = 0
    
    # 检查是否至少指定了一个参数
    has_importance = args.input is not None
    has_performance = args.file is not None or args.model_dir is not None
    
    if not has_importance and not has_performance:
        logger.error("必须至少指定以下参数之一：")
        logger.error("  -i/--input: 特征重要性数据文件（用于重要性可视化）")
        logger.error("  -f/--file 或 -d/--model-dir: 绘图数据文件或模型目录（用于性能可视化）")
        return 1
    
    # 先处理模型性能可视化（-f/-d）
    if has_performance:
        try:
            logger.info("开始生成模型性能可视化...")
            from assoG2P.bin.visualization import plot_model_performance_from_file
            
            # 确定plotting_data.npz文件路径
            plotting_data_file = None
            if args.file:
                plotting_data_file = Path(args.file)
            elif args.model_dir:
                plotting_data_file = Path(args.model_dir) / "plotting_data.npz"
            
            if not plotting_data_file.exists():
                logger.error(f"绘图数据文件不存在: {plotting_data_file}")
                result_code = 1
            else:
                # 确定输出目录
                output_dir = None
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                elif args.model_dir:
                    output_dir = Path(args.model_dir)
                elif args.file:
                    output_dir = Path(args.file).parent
                
                # 确定publication_quality
                publication_quality = args.publication_quality if hasattr(args, 'publication_quality') else None
                
                logger.info(f"从文件读取绘图数据: {plotting_data_file}")
                perf_result = plot_model_performance_from_file(
                    plotting_data_file=plotting_data_file,
                    output_dir=output_dir,
                    publication_quality=publication_quality
                )
                if perf_result != 0:
                    result_code = perf_result
                else:
                    logger.info("模型性能可视化完成")
                    
        except Exception as e:
            logger.error(f"模型性能可视化失败: {str(e)}")
            result_code = 1
    
    # 再处理特征重要性可视化（-i）
    if has_importance:
        if not args.output:
            logger.error("使用 -i/--input 时必须指定 --output 参数")
            result_code = 1
        else:
            try:
                logger.info("开始生成特征重要性可视化...")
                from assoG2P.bin.visualization import EnhancedGenomeVisualizer
                
                if not Path(args.input).exists():
                    raise FileNotFoundError(f"输入文件不存在: {args.input}")
                
                # 创建可视化器实例
                visualizer = EnhancedGenomeVisualizer(
                    input_file=args.input,
                    feature_col=args.feature_col,
                    value_col=getattr(args, 'value_col', None) or getattr(args, 'shap_col', None)
                )
                
                # 根据参数决定生成哪种图
                generate_static = True
                generate_interactive = True
                if getattr(args, "static_only", False) and not getattr(args, "interactive_only", False):
                    generate_interactive = False
                elif getattr(args, "interactive_only", False) and not getattr(args, "static_only", False):
                    generate_static = False
                
                if generate_static:
                    static_output = f"{args.output}_static.png"
                    visualizer.plot_static_scatter(
                        output_file=static_output,
                        dpi=args.dpi
                    )
                    logger.info(f"静态图表已生成: {static_output}")
                
                if generate_interactive:
                    interactive_output = f"{args.output}_interactive.html"
                    visualizer.plot_interactive_scatter(
                        output_file=interactive_output
                    )
                    logger.info(f"交互式图表已生成: {interactive_output}")
                
                logger.info("特征重要性可视化完成")
                
            except ImportError as e:
                logger.error(f"模块导入错误: {str(e)}")
                logger.error("对于交互式图表，请安装依赖: pip install plotly kaleido")
                result_code = 1
            except Exception as e:
                logger.error(f"特征重要性可视化失败: {str(e)}")
                result_code = 1
    
    return result_code

def run_visualization(args) -> int:
    """Perform scatter plot visualization - static / interactive can be generated separately"""
    try:
        from assoG2P.bin.visualization import EnhancedGenomeVisualizer
        
        if not Path(args.input).exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
            
        # 创建可视化器实例
        visualizer = EnhancedGenomeVisualizer(
            input_file=args.input,
            feature_col=args.feature_col,
            value_col=getattr(args, 'value_col', None) or getattr(args, 'shap_col', None)  # 兼容旧参数名
        )
        
        # 根据参数决定生成哪种图
        generate_static = True
        generate_interactive = True
        if getattr(args, "static_only", False) and not getattr(args, "interactive_only", False):
            generate_interactive = False
        elif getattr(args, "interactive_only", False) and not getattr(args, "static_only", False):
            generate_static = False

        static_output = None
        interactive_output = None

        if generate_static:
            static_output = f"{args.output}_static.png"
            visualizer.plot_static_scatter(
                output_file=static_output,
                dpi=args.dpi
            )
        if generate_interactive:
            interactive_output = f"{args.output}_interactive.html"
            visualizer.plot_interactive_scatter(
                output_file=interactive_output
            )

        return 0
        
    except ImportError as e:
        logger.error(f"Module import error: {str(e)}")
        logger.error("For interactive plots, install required dependencies: pip install plotly kaleido")
        return 1
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return 1

def print_banner() -> None:
    print("=" * 50)
    print("assocG2P Genomic analysis platform v1.0.0")
    print("=" * 50)

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genotype-phenotype machine learning association analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  association preprocess -g genotype.vcf -p phenotype.csv -o preprocessed_data
  association train -i preprocessed_data/train_data.txt -m LightGBM -f 1 -o results
  association predict -i new_data.csv -m LightGBM -o predictions
  association visualize -i feature_importance.csv -o plot
  association visualize -f results/LightGBM/plotting_data.npz
  association visualize -i feature_importance.csv -o plot -d results/LightGBM

For more details, use: association [command] -h
        """
    )
    
    subparsers = parser.add_subparsers(
        title="Available commands", 
        dest="command",
        metavar=""
    )
    
    # 数据预处理命令（新增GWAS/LD过滤参数）
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Data preprocessing (with optional GWAS/LD filtering)"
    )
    preprocess_parser.add_argument("-g", "--genotype", required=True, help="Genotypic data in VCF format")
    preprocess_parser.add_argument("-p", "--phenotype", required=True, help="Phenotypic data")
    preprocess_parser.add_argument("-o", "--output", required=True, help="Output file path")
    preprocess_parser.add_argument("--pheno-col", help="Custom phenotype column name (for non-'phenotype' header)")
    preprocess_parser.add_argument("--no-filter-snps", action="store_true", help="不进行SNP质量过滤")
    preprocess_parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    preprocess_parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="保留预处理阶段产生的GWAS/LD临时文件（默认自动清理，仅保留必要结果文件）"
    )
    
    # GWAS/LD特征筛选参数（在preprocess阶段执行）
    preprocess_parser.add_argument("-f", "--feature_selection_mode", type=int, default=1, choices=[1, 2, 3, 4],
                                 help="Feature selection mode: 1=空白对照(不使用GWAS和LD, 默认), 2=GWAS筛选(仅使用GWAS), 3=LD过滤(仅使用LD), 4=GWAS和LD综合过滤(先GWAS后LD)")
    preprocess_parser.add_argument("--gwas_pvalue", type=float, default=0.01, help="P-value threshold for GWAS SNP selection (default: 0.01). Used when -f is 2 or 4")
    preprocess_parser.add_argument("--ld_window_kb", type=int, default=50, help="LD window size in KB (default: 50). Used when -f is 3 or 4")
    preprocess_parser.add_argument("--ld_window", type=int, default=5, help="LD window variant count (default: 5). Used when -f is 3 or 4")
    preprocess_parser.add_argument("--ld_window_r2", type=float, default=0.2, help="LD r² threshold (default: 0.2). Used when -f is 3 or 4")
    
    preprocess_parser.set_defaults(func=run_preprocess)
    
    # 单个模型训练命令 - 注意：GWAS/LD特征筛选已前移到preprocess阶段
    train_parser = subparsers.add_parser(
        "train",
        help="Single model training (GWAS/LD filtering should be done in preprocess stage)"
    )
    train_parser.add_argument(
        "-j",
        "--json",
        required=True,
        help="Preprocess metadata JSON file (*_metadata.json). Training is metadata-driven; direct .txt input is not supported."
    )
    train_parser.add_argument("-m", "--model", required=True, 
                             choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"],help="Select a model")
    train_parser.add_argument("--task_type", required=False,
                             choices=["classification", "regression"],
                             help="Task type (classification/regression). If not specified, will be automatically read from metadata or default to 'regression'")
    train_parser.add_argument("-o", "--output_dir", required=True, 
                             help="Output directory (will create model-specific subdirectories)")
    train_parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation (default: 5)")
    train_parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # 特征重要性计算参数（可选）
    train_parser.add_argument("--feature_importance", action="store_true", help="Calculate and save feature importance (SHAP values are calculated by default)")
    
    train_parser.set_defaults(func=run_training)
    
    # 全模型训练命令 - 注意：GWAS/LD特征筛选已前移到preprocess阶段
    train_all_parser = subparsers.add_parser(
        "train-all",
        help="Train all models and compare performance (GWAS/LD filtering should be done in preprocess stage)"
    )
    train_all_parser.add_argument(
        "-j",
        "--json",
        required=True,
        help="Preprocess metadata JSON file (*_metadata.json). Training is metadata-driven; direct .txt input is not supported."
    )
    train_all_parser.add_argument("--task_type", required=False,
                                 choices=["classification", "regression"],
                                 help="Task type (classification/regression). If not specified, will be automatically read from metadata or default to 'regression'")
    train_all_parser.add_argument("-o", "--output_dir", required=True, 
                                 help="Output directory")
    train_all_parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation (default: 5)")
    train_all_parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # 特征重要性计算参数（可选）
    train_all_parser.add_argument("--feature_importance", action="store_true", help="Calculate and save feature importance (SHAP values are calculated by default)")
    
    train_all_parser.set_defaults(func=run_train_all)
    
    # 预测命令（无修改）
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict using trained model"
    )
    predict_parser.add_argument("-i", "--input", required=True, help="Input data for prediction")
    # 将 -m 参数改为模型文件路径
    predict_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to trained model file (.pkl), e.g. /path/to/LightGBM_model.pkl"
    )
    predict_parser.add_argument("-o", "--output_dir", required=True, 
                               help="Output directory for predictions")
    predict_parser.add_argument("--task_type",
                               choices=["classification", "regression"],
                               help="Task type (classification/regression, optional)")
    predict_parser.set_defaults(func=run_predict)
    
    # 可视化命令（统一接口，支持同时使用-i和-f）
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Visualization tools (feature importance scatter plot and/or model performance curves)"
    )
    
    # 特征重要性可视化参数（-i）
    viz_parser.add_argument("-i", "--input", 
                           help="Feature importance data file (for importance visualization)")
    viz_parser.add_argument("--output", 
                           help="Output file prefix for importance plots (will add *_static.png and/or *_interactive.html)")
    viz_parser.add_argument("--feature-col", 
                           help="Feature column name (default: first column)")
    viz_parser.add_argument("--value-col", 
                           help="Feature importance value column name (default: auto-detect)")
    viz_parser.add_argument("--shap-col", 
                           dest='value_col',
                           help="(Deprecated) Use --value-col instead. Feature importance value column name (default: auto-detect)")
    viz_parser.add_argument("--dpi", 
                           type=int, 
                           default=1200,
                           help="Image resolution for static plots (default: 1200)")
    importance_group = viz_parser.add_mutually_exclusive_group()
    importance_group.add_argument("--static-only",
                                  action="store_true",
                                  help="Only generate static scatter plot (PNG) for importance visualization")
    importance_group.add_argument("--interactive-only",
                                   action="store_true",
                                   help="Only generate interactive scatter plot (HTML) for importance visualization")
    
    # 模型性能可视化参数（-f或-d）
    viz_parser.add_argument("-f", "--file", 
                           help="plotting_data.npz file path (for performance visualization)")
    viz_parser.add_argument("-d", "--model-dir", 
                           help="Model output directory (will look for plotting_data.npz in this directory)")
    viz_parser.add_argument("--output-dir", 
                           help="Output directory for performance plots (default: same as model directory or plotting_data.npz directory)")
    viz_parser.add_argument("--publication-quality", 
                            action="store_true",
                            help="Generate publication-quality plots (high resolution, PDF format) for performance visualization")
    
    viz_parser.set_defaults(func=run_unified_visualization)
    
    return parser

def format_runtime(start_time: float) -> str:
    seconds = time.time() - start_time
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h{minutes}m{seconds}s"

def main() -> None:
    start_time = time.time()
    init_logging(Path("association.log"))
    print_banner()
    
    parser = setup_argparse()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    try:
        ret_code = args.func(args)
        if ret_code != 0:
            logger.error(f"{args.command} failed with error code {ret_code}")
            sys.exit(ret_code)
            
        logger.info(f"Operation completed successfully! Total time: {format_runtime(start_time)}")
    
    except FileNotFoundError as e:
        logger.critical(f"File not found: {str(e)}")
        sys.exit(2)
    except ValueError as e:
        logger.critical(f"Invalid data format: {str(e)}")
        sys.exit(3)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()