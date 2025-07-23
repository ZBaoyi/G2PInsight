#!/usr/bin/env python3
"""
Genotype-phenotype association analysis tool

Usage:
    association [command] [options]

Commands:
    preprocess    Data preprocessing
    train        Model training
    visualize    Result visualization

Examples:
    association preprocess -h
    association train -h
    association visualize -h
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from typing import Optional

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
        logger.info(f"Start preprocessing: genotype={args.genotype}, phenotype={args.phenotype}")
        return preprocess_func(
            genotype_file=args.genotype,
            phenotype_file=args.phenotype,
            output_file=args.output
        )
    except ImportError as e:
        logger.error(f"Failed to import preprocessing module: {str(e)}")
        return 1

def run_training(args) -> int:
    """Perform model training"""
    from assoG2P.bin.modeltraining import ModelTrainer
    try:
        logger.info(f"Start training: model={args.model}, input={args.input}")
        trainer = ModelTrainer(random_state=args.random_state)
        
        X, y = trainer.load_data(args.input)
        model, _ = trainer.train_model(
            model_type=args.model,
            X=X,
            y=y,
            test_size=args.test_size,
            task_type=args.task_type,
            random_state=args.random_state
        )
        
        importance = trainer.calculate_feature_importance(model, X, args.model)
        importance.to_csv(args.output, index=False)
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

def run_visualization(args) -> int:
    """Perform visualization with enhanced features"""
    try:
        # 修改1：导入优化后的可视化类
        from assoG2P.bin.visualization import EnhancedGenomeVisualizer
        
        # 修改2：增加输入文件验证
        if not Path(args.input).exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
            
        # 修改3：初始化优化后的可视化工具
        visualizer = EnhancedGenomeVisualizer(
            input_file=args.input,
            feature_col=args.feature_col,
            value_col=args.shap_col
        )
        
        # 修改4：使用新的绘图接口
        visualizer.plot(
            output_file=args.output,
            plot_type=args.type,
            interactive=args.interactive,
            dpi=args.dpi
        )
        return 0
        
    except ImportError as e:
        if "plotly" in str(e) and args.interactive:
            logger.error("Interactive mode requires plotly. Install with: pip install plotly kaleido")
        else:
            logger.error(f"Module import error: {str(e)}")
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
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(
        title="Available commands", 
        dest="command",
        metavar=""
    )
    
    # Preprocessing commands (保持不变)
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Data preprocessing"
    )
    preprocess_parser.add_argument("-g", "--genotype", required=True, help="Genotypic data in VCF format")
    preprocess_parser.add_argument("-p", "--phenotype", required=True, help="Phenotypic data")
    preprocess_parser.add_argument("-o", "--output", required=True, help="Output file path")
    preprocess_parser.set_defaults(func=run_preprocess)
    
    # Training commands (保持不变)
    train_parser = subparsers.add_parser(
        "train",
        help="Model training"
    )
    train_parser.add_argument("-i", "--input", required=True, help="Training data")
    train_parser.add_argument("-m", "--model", required=True, 
                             choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"],
                             help="Select a model")
    train_parser.add_argument("--task_type", required=True,
                             choices=["classification", "regression"],
                             help="Task type (classification/regression)")
    train_parser.add_argument("-o", "--output", required=True, help="Feature importance output file")
    train_parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    train_parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    train_parser.set_defaults(func=run_training)
    
    # Visualization commands (保持不变，参数已兼容)
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Feature importance visualization"
    )
    viz_parser.add_argument("-i", "--input", required=True, help="Feature importance data file")
    viz_parser.add_argument("-o", "--output", required=True, help="Output image path")
    viz_parser.add_argument("-t", "--type", default="scatter",
                           choices=["scatter", "bar", "line", "hist"], 
                           help="Chart type")
    viz_parser.add_argument("--feature-col", 
                           help="Feature column name (default: first column)")
    viz_parser.add_argument("--shap-col", 
                           help="SHAP value column name (default: second column)")
    viz_parser.add_argument("--interactive", 
                           action="store_true",
                           help="Enable interactive visualization (requires plotly)")
    viz_parser.add_argument("--dpi", 
                           type=int, 
                           default=300,
                           help="Image resolution for static plots")
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
    
    try:
        # 修改5：简化预处理检查（已在各函数内部处理）
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