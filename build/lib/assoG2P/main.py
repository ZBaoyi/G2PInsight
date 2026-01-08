#!/usr/bin/env python3
"""
Genotype-phenotype association analysis tool

Usage:
    association [command] [options]

Commands:
    preprocess    Data preprocessing
    train         Model training (integrated LD/GWAS)
    train-all     Train all models (integrated LD/GWAS)
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
        logger.info(f"预处理")
        # 如果threads为None，表示用户未指定，使用0表示自动分配
        threads_value = args.threads if args.threads is not None else 0
        return preprocess_func(
            genotype_file=args.genotype,
            phenotype_file=args.phenotype,
            output_file=args.output,
            threads=threads_value,
            pheno_col=args.pheno_col
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return 1

def run_training(args) -> int:
    """Perform model training with integrated LD/GWAS"""
    try:
        from assoG2P.bin.modeltraining import run_single_model
        
        logger.info(f"训练: {args.model}")
        
        # 核心修改：使用feature_selection_mode统一控制GWAS和LD
        return run_single_model(
            input_path=args.input,
            model_type=args.model,
            output_dir=args.output_dir,
            feature_selection_mode=args.feature_selection_mode,
            task_type=args.task_type,
            n_folds=args.n_folds,
            random_state=args.random_state,
            gwas_genotype=args.gwas_genotype,  # 同时用于LD/GWAS的基因型路径
            gwas_pvalue=args.gwas_pvalue,
            ld_window_kb=args.ld_window_kb,
            ld_window=args.ld_window,          # 新增LD窗口变体数参数
            ld_window_r2=args.ld_window_r2,    # 统一参数名（原ld_r2）
            ld_threads=args.ld_threads
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

def run_train_all(args) -> int:
    """Train all models and compare results with integrated LD/GWAS"""
    try:
        from assoG2P.bin.modeltraining import run_all_models
        
        logger.info(f"训练所有模型")
        
        # 核心修改：使用feature_selection_mode统一控制GWAS和LD
        return run_all_models(
            input_path=args.input,
            output_dir=args.output_dir,
            feature_selection_mode=args.feature_selection_mode,
            task_type=args.task_type,
            n_folds=args.n_folds,
            random_state=args.random_state,
            gwas_genotype=args.gwas_genotype,
            gwas_pvalue=args.gwas_pvalue,
            ld_window_kb=args.ld_window_kb,
            ld_window=args.ld_window,
            ld_window_r2=args.ld_window_r2,
            ld_threads=args.ld_threads
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
            model_type=args.model,
            output_dir=args.output_dir,
            task_type=args.task_type
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return 1

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
  association train -i preprocessed_data.csv -m LightGBM -f 1 -o results
  association predict -i new_data.csv -m LightGBM -o predictions
  association visualize -i feature_importance.csv -o plot

For more details, use: association [command] -h
        """
    )
    
    subparsers = parser.add_subparsers(
        title="Available commands", 
        dest="command",
        metavar=""
    )
    
    # 数据预处理命令（无修改）
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Data preprocessing"
    )
    preprocess_parser.add_argument("-g", "--genotype", required=True, help="Genotypic data in VCF format")
    preprocess_parser.add_argument("-p", "--phenotype", required=True, help="Phenotypic data")
    preprocess_parser.add_argument("-o", "--output", required=True, help="Output file path")
    preprocess_parser.add_argument("--threads", type=int, default=None, help="并行进程数上限（可选，默认自动根据染色体数和CPU核心数智能分配，约每个进程处理2条染色体）")
    preprocess_parser.add_argument("--pheno-col", help="Custom phenotype column name (for non-'phenotype' header)")
    preprocess_parser.add_argument("--serial", action="store_true", 
                                   help="强制使用串行模式（完全禁用多进程，逐染色体处理，绕过OOM问题）。适用于内存受限环境（如SLURM内存限制≤32GB）")
    # preprocess_parser.add_argument("--mem-limit", type=int, default=PLINK_MEM_LIMIT, help="PLINK内存限制（MB）")
    preprocess_parser.set_defaults(func=run_preprocess)
    
    # 单个模型训练命令 - 核心修改：集成LD/GWAS，统一参数名
    train_parser = subparsers.add_parser(
        "train",
        help="Single model training (integrated LD filtering & GWAS feature selection)"
    )
    train_parser.add_argument("-i", "--input", required=True, help="Training data file (e.g., preprocess output directory/train_data.txt) or preprocess metadata file (*_metadata.json)")
    train_parser.add_argument("-m", "--model", required=True, 
                             choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"],
                             help="Select a model")
    train_parser.add_argument("-f", "--feature_selection_mode", type=int, required=True, choices=[1, 2, 3, 4],
                             help="Feature selection mode (required): 1=空白对照(不使用GWAS和LD), 2=GWAS筛选(仅使用GWAS), 3=LD过滤(仅使用LD), 4=GWAS和LD综合过滤(先GWAS后LD)")
    train_parser.add_argument("--task_type", required=False,
                             choices=["classification", "regression"],
                             help="Task type (classification/regression). If not specified, will be automatically read from metadata or default to 'regression'")
    train_parser.add_argument("-o", "--output_dir", required=True, 
                             help="Output directory (will create model-specific subdirectories)")
    train_parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation (default: 5)")
    train_parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # GWAS相关参数（当-f为2或4时需要）
    train_parser.add_argument("--gwas_genotype", help="Genotype file/prefix for GWAS/LD (VCF/PLINK binary/text). If not specified, will be automatically read from metadata when -f is 2 or 4")
    train_parser.add_argument("--gwas_pvalue", type=float, default=0.01, help="P-value threshold to select SNP features via GWAS (default: 5e-8). Used when -f is 2 or 4")
    
    # LD过滤相关参数（当-f为3或4时需要）
    train_parser.add_argument("--ld_window_kb", type=int, default=50, help="LD window size in KB (default: 50). Used when -f is 3 or 4")
    train_parser.add_argument("--ld_window", type=int, default=5, help="LD window variant count (default: 5). Used when -f is 3 or 4")
    train_parser.add_argument("--ld_window_r2", type=float, default=0.2, help="LD r² threshold (default: 0.2). Used when -f is 3 or 4")
    train_parser.add_argument("--ld_threads", type=int, default=1, help="Threads for LD pruning (default: 1). Used when -f is 3 or 4")
    
    train_parser.set_defaults(func=run_training)
    
    # 全模型训练命令 - 核心修改：同步LD/GWAS参数
    train_all_parser = subparsers.add_parser(
        "train-all",
        help="Train all models and compare performance (integrated LD/GWAS)"
    )
    train_all_parser.add_argument("-i", "--input", required=True, help="Training data file (e.g., preprocess output directory/train_data.txt) or preprocess metadata file (*_metadata.json)")
    train_all_parser.add_argument("-f", "--feature_selection_mode", type=int, required=True, choices=[1, 2, 3, 4],
                                 help="Feature selection mode (required): 1=空白对照(不使用GWAS和LD), 2=GWAS筛选(仅使用GWAS), 3=LD过滤(仅使用LD), 4=GWAS和LD综合过滤(先GWAS后LD)")
    train_all_parser.add_argument("--task_type", required=False,
                                 choices=["classification", "regression"],
                                 help="Task type (classification/regression). If not specified, will be automatically read from metadata or default to 'regression'")
    train_all_parser.add_argument("-o", "--output_dir", required=True, 
                                 help="Output directory")
    train_all_parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation (default: 5)")
    train_all_parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # GWAS参数（当-f为2或4时需要）
    train_all_parser.add_argument("--gwas_genotype", help="Genotype file/prefix for GWAS/LD (VCF/PLINK binary/text). If not specified, will be automatically read from metadata when -f is 2 or 4")
    train_all_parser.add_argument("--gwas_pvalue", type=float, default=0.01, help="P-value threshold to select SNP features via GWAS (default: 5e-8). Used when -f is 2 or 4")
    
    # LD参数（当-f为3或4时需要）
    train_all_parser.add_argument("--ld_window_kb", type=int, default=50, help="LD window size in KB (default: 50). Used when -f is 3 or 4")
    train_all_parser.add_argument("--ld_window", type=int, default=5, help="LD window variant count (default: 5). Used when -f is 3 or 4")
    train_all_parser.add_argument("--ld_window_r2", type=float, default=0.2, help="LD r² threshold (default: 0.2). Used when -f is 3 or 4")
    train_all_parser.add_argument("--ld_threads", type=int, default=1, help="Threads for LD pruning (default: 1). Used when -f is 3 or 4")
    
    train_all_parser.set_defaults(func=run_train_all)
    
    # 预测命令（无修改）
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict using trained model"
    )
    predict_parser.add_argument("-i", "--input", required=True, help="Input data for prediction")
    predict_parser.add_argument("-m", "--model", required=True,
                               choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"],
                               help="Select a trained model")
    predict_parser.add_argument("-o", "--output_dir", required=True, 
                               help="Output directory for predictions")
    predict_parser.add_argument("--task_type",
                               choices=["classification", "regression"],
                               help="Task type (classification/regression, optional)")
    predict_parser.set_defaults(func=run_predict)
    
    # 可视化命令
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Feature importance scatter plot visualization (static PNG and/or interactive HTML)"
    )
    viz_parser.add_argument("-i", "--input", required=True, help="Feature importance data file")
    viz_parser.add_argument("-o", "--output", required=True, 
                           help="Output file prefix (will add *_static.png and/or *_interactive.html)")
    viz_parser.add_argument("--feature-col", 
                           help="Feature column name (default: first column)")
    viz_parser.add_argument("--value-col", 
                           help="Feature importance value column name (default: auto-detect)")
    viz_parser.add_argument("--shap-col", 
                           dest='value_col',
                           help="(Deprecated) Use --value-col instead. Feature importance value column name (default: auto-detect)")
    viz_parser.add_argument("--dpi", 
                           type=int, 
                           default=300,
                           help="Image resolution for static plots (default: 300)")
    group = viz_parser.add_mutually_exclusive_group()
    group.add_argument("--static-only",
                       action="store_true",
                       help="Only generate static scatter plot (PNG)")
    group.add_argument("--interactive-only",
                       action="store_true",
                       help="Only generate interactive scatter plot (HTML)")
    viz_parser.set_defaults(func=run_visualization)
    
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
    
    # 参数合法性校验（LD/GWAS依赖）
    # 注意：如果用户未指定--gwas_genotype，系统会尝试从元数据自动获取
    # 因此这里只检查模式是否有效，实际的基因型文件检查在modeltraining模块中进行
    if args.command in ["train", "train-all"]:
        if args.feature_selection_mode not in [1, 2, 3, 4]:
            logger.critical(f"Error: feature_selection_mode must be 1, 2, 3, or 4, got {args.feature_selection_mode}")
            sys.exit(2)
    
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