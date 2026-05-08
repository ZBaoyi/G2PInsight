#!/usr/bin/env python3
"""
Genotype-phenotype G2PInsight analysis tool

Usage:
    G2PInsight [command] [options]

Commands:
    preprocess    Data preprocessing (with optional GWAS/LD filtering)
    train         Model training (GWAS/LD filtering should be done in preprocess stage)
    train-all     Train all models (GWAS/LD filtering should be done in preprocess stage)
    predict       Prediction with trained model
    visualize     Result visualization

Examples:
    G2PInsight preprocess -h
    G2PInsight train -h
    G2PInsight train-all -h
    G2PInsight predict -h
    G2PInsight visualize -h
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

_pkg_dir = Path(__file__).resolve().parent
_pkg_parent = _pkg_dir.parent
if (_pkg_dir / "__init__.py").exists() and str(_pkg_parent) not in sys.path:
    sys.path.insert(0, str(_pkg_parent))

def _prepare_stage_output_dir(output_dir: str, stage: str) -> str:
    """Prepare standardized stage output directory (e.g. {output_dir}/train)."""
    base_output_dir = Path(output_dir).absolute()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    stage_output_dir = base_output_dir / stage
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    return str(stage_output_dir)

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
    from G2PInsight.bin.preprocess import run_preprocess as preprocess_func
    try:
        logger.info("Starting preprocessing")
        # 解析SNP过滤参数
        filter_snps_value = not args.no_filter_snps
        # 解析GWAS/LD特征筛选配置（LD使用三合一配置参数）
        feature_selection_mode = getattr(args, 'feature_selection_mode', 1)
        gwas_pvalue = getattr(args, 'gwas_pvalue', 0.01)
        # LD三合一配置，格式: \"<window_kb>,<window_variants>,<r2_threshold>\"
        ld_config = getattr(args, 'ld_config', None) or "50,5,0.2"
        # `--no-cache` 参数已移除，缓存逻辑使用默认行为（启用缓存）
        use_cache = True
        
        return preprocess_func(
            genotype_file=args.genotype,
            phenotype_file=args.phenotype,
            output_file=args.output,
            filter_snps=filter_snps_value,
            use_cache=use_cache,
            feature_selection_mode=feature_selection_mode,
            gwas_pvalue=gwas_pvalue,
            ld_config=ld_config,
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return 1

def run_training(args) -> int:
    """Perform model training (GWAS/LD filtering should be done in preprocess stage)"""
    try:
        from G2PInsight.bin.modeltraining import run_single_model
        
        logger.info(f"Training model: {args.model}")
        
        # GWAS/LD特征筛选已在preprocess模块完成，训练阶段不再执行筛选
        return run_single_model(
            input_path=args.json,
            model_type=args.model,
            output_dir=_prepare_stage_output_dir(args.output_dir, "train"),
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
        from G2PInsight.bin.modeltraining import run_all_models
        
        logger.info("Training all supported models")
        
        # GWAS/LD特征筛选已在preprocess模块完成，训练阶段不再执行筛选
        return run_all_models(
            input_path=args.json,
            output_dir=_prepare_stage_output_dir(args.output_dir, "train"),
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
        from G2PInsight.bin.modeltraining import predict_with_model
        
        logger.info("Running prediction")
        
        return predict_with_model(
            input_path=args.input,
            model_path=args.model,
            output_dir=_prepare_stage_output_dir(args.output_dir, "predict"),
            task_type=args.task_type
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return 1

def run_unified_visualization(args) -> int:
    """
    统一的可视化处理函数
    - 使用 -i/--importance 输入特征重要性文件，生成全基因组散点图
    - 使用 -I/--indicator 输入模型评估结果文件（plotting_data.npz），生成模型性能与CV曲线
    - 使用 -o/--output 指定输出前缀（standardized），用于生成相关图形文件
    """
    result_code = 0
    
    # 检查是否至少指定了一个输入
    has_importance = getattr(args, "importance", None) is not None
    has_indicator = getattr(args, "indicator", None) is not None
    
    if not has_importance and not has_indicator:
        logger.error("At least one input must be specified:")
        logger.error("  -i/--importance: feature-importance file for importance visualization")
        logger.error("  -I/--indicator: model plotting data file (plotting_data.npz) for performance/CV visualization")
        return 1
    
    if not args.output:
        logger.error("When using -i/--importance or -I/--indicator, -o/--output must be provided")
        return 1
    
    # ======================== 输出目录结构规范化（standardized） ========================
    # 在 -o 指定的目录下创建 visualize 子目录
    output_prefix = Path(args.output)
    # 若用户只给了文件名而无目录，则默认当前工作目录
    if not output_prefix.parent or str(output_prefix.parent) == ".":
        output_prefix = Path.cwd() / output_prefix.name
    
    base_output_dir = output_prefix.parent
    base_output_dir.mkdir(parents=True, exist_ok=True)
    stage_output_dir = base_output_dir / "visualize"
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出前缀调整为在 visualize 目录下
    output_prefix = stage_output_dir / output_prefix.name
    output_dir = stage_output_dir
    
    # 先处理模型性能/评估指标可视化（-I）
    if has_indicator:
        try:
            logger.info("Generating model performance visualizations...")
            from G2PInsight.bin.visualization import plot_model_performance_from_file
            
            plotting_data_file = Path(args.indicator)
            if not plotting_data_file.exists():
                logger.error(f"Model plotting data file does not exist: {plotting_data_file}")
                result_code = 1
            else:
                logger.info(f"Loading model plotting data from file: {plotting_data_file}")
                perf_result = plot_model_performance_from_file(
                    plotting_data_file=plotting_data_file,
                    output_dir=output_dir,
                    publication_quality=None,
                    output_prefix=output_prefix
                )
                if perf_result != 0:
                    result_code = perf_result
                else:
                    logger.info("Model performance visualization completed")
        except Exception as e:
            logger.error(f"Model performance visualization failed: {str(e)}")
            result_code = 1
    
    # 再处理特征重要性可视化（-i）
    if has_importance:
        try:
            logger.info("Generating feature-importance visualizations...")
            from G2PInsight.bin.visualization import EnhancedGenomeVisualizer
            
            importance_file = Path(args.importance)
            if not importance_file.exists():
                raise FileNotFoundError(f"Feature-importance file not found: {importance_file}")
            
            # 创建可视化器实例，列名自动检测
            visualizer = EnhancedGenomeVisualizer(
                input_file=str(importance_file),
                feature_col=None,
                value_col=None
            )
            
            # 统一使用输出前缀，生成静态和交互式图
            static_output = f"{output_prefix}_importance_static.png"
            interactive_output = f"{output_prefix}_importance_interactive.html"
            
            visualizer.plot_static_scatter(output_file=str(static_output))
            logger.info(f"Static feature-importance plot generated: {static_output}")
            
            visualizer.plot_interactive_scatter(output_file=str(interactive_output))
            logger.info(f"Interactive feature-importance plot generated: {interactive_output}")
            
            logger.info("Feature-importance visualization completed")
            
        except ImportError as e:
            logger.error(f"Module import error: {str(e)}")
            logger.error("For interactive plots, install dependencies: pip install plotly kaleido")
            result_code = 1
        except Exception as e:
            logger.error(f"Feature-importance visualization failed: {str(e)}")
            result_code = 1
    
    return result_code

def print_banner() -> None:
    print("=" * 50)
    print("assocG2P Genomic analysis platform v1.0.0")
    print("=" * 50)

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genotype-phenotype machine learning G2PInsight analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  G2PInsight preprocess -g genotype.vcf -p phenotype.csv -o preprocessed_data
  G2PInsight train -j preprocessed_data/*_metadata.json -m LightGBM -o results
  G2PInsight predict -i new_data.txt -m results/train/LightGBM/LightGBM_model.pkl -o predictions
  G2PInsight visualize -i feature_importance.txt -o results/plot
  G2PInsight visualize -I results/train/LightGBM/LightGBM_plotting_data.npz -o results/plot

For more details, use: G2PInsight [command] -h
        """
    )
    
    subparsers = parser.add_subparsers(
        title="Available commands", 
        dest="command",
        metavar=""
    )
    
    # 数据预处理命令（新增GWAS/LD过滤参数, LD参数使用三合一配置）
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Data preprocessing (with optional GWAS/LD filtering)"
    )
    preprocess_parser.add_argument("-g", "--genotype", required=True, help="Genotypic data in VCF format")
    preprocess_parser.add_argument("-p", "--phenotype", required=True, help="Phenotypic data")
    preprocess_parser.add_argument("-o", "--output", required=True, help="Output file path")
    preprocess_parser.add_argument("--no-filter-snps", action="store_true", help="Disable SNP quality filtering")
    
    # GWAS/LD特征筛选参数（在preprocess阶段执行, LD参数已三合一）
    preprocess_parser.add_argument("-f", "--feature_selection_mode", type=int, default=1, choices=[1, 2, 3, 4],
                                 help="Feature selection mode: 1=no selection (default), 2=GWAS only, 3=LD only, 4=GWAS+LD (GWAS first, then LD)")
    preprocess_parser.add_argument("--gwas_pvalue", type=float, default=0.01, help="P-value threshold for GWAS SNP selection (default: 0.01). Used when -f is 2 or 4")
    preprocess_parser.add_argument(
        "--ld-config",
        type=str,
        default="50,5,0.2",
        help=(
            "LD three-in-one configuration (standardized): "
            "\"<window_kb>,<window_variants>,<r2_threshold>\", "
            "例如 \"50,5,0.2\" 表示窗口50KB、窗口内5个变体、r²阈值0.2；"
            "used only when -f is 3 or 4"
        ),
    )
    
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
        help="Preprocess metadata JSON file (*_metadata.json)."
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
    
    # 可视化命令（统一接口，使用-i/-I 输入，-o 指定输出前缀）
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Visualization tools (feature importance scatter plot and/or model performance curves)"
    )
    
    # 特征重要性可视化输入（-i）
    viz_parser.add_argument(
        "-i",
        "--importance",
        help="Feature-importance file path (for genome-wide importance scatter visualization)"
    )
    
    # 模型评估指标可视化输入（-I）
    viz_parser.add_argument(
        "-I",
        "--indicator",
        help="Model plotting data file (plotting_data.npz) for performance and CV-curve visualization"
    )
    
    # 统一输出前缀
    viz_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output filename prefix (standardized) for feature-importance and/or model-performance plots"
    )
    
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
    init_logging(Path("G2PInsight.log"))
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