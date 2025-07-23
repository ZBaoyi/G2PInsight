#!/usr/bin/env python3
"""
Usage:
    association [command] [options]
Examples:
    preprocess -h    data preprocessing help
    train -h        Model Training help
    visualize -h     Visualization of results help 
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from typing import Optional
logger = logging.getLogger(__name__)
# Initialize the log configuration
def init_logging(log_file: Optional[Path] = None) -> None:
    """Unified log configuration
    
    Args:
        log_file: Optional log file path
    """
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

# Functions for modular imports
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
        logger.error(f"Failed to import the preprocessing module: {str(e)}")
        return 1
from assoG2P.bin.modeltraining import ModelTrainer
def run_training(args) -> int:
    """Perform model training"""
    try:
        logger.info(f"Start the training: model={args.model}, input={args.input}")
        trainer = ModelTrainer(random_state=args.random_state)
        
        X, y = trainer.load_data(args.input)

        model, _ = trainer.train_model(
            model_type=args.model,
            X=X,
            y=y,
            test_size=args.test_size
        )
        
        importance = trainer.calculate_feature_importance(model, X, args.model)
        importance.to_csv(args.output, index=False)
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

from assoG2P.bin.visualization import run_visualization as viz_func
def run_visualization(args) -> int:
    """Perform visualization"""
    try:
        logger.info(f"Generate visualizations: type={args.type}, input={args.input}")
        return viz_func(
            args.input,
            args.output,
            plot_type=args.type, 
            interactive=args.interactive
        )
    except ImportError as e:
        logger.error(f"Failed to import the visualization module: {str(e)}")
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
    
    # Preprocessing commands
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="data preprocessing"
    )
    preprocess_parser.add_argument("-g", "--genotype", required=True, help="Genotypic data in VCF format")
    preprocess_parser.add_argument("-p", "--phenotype", required=True, help="Phenotypic data")
    preprocess_parser.add_argument("-o", "--output", required=True, help="Output file path")
    preprocess_parser.set_defaults(func=run_preprocess)
    
    # Training commands
    train_parser = subparsers.add_parser(
        "train",
        help="Model training"
    )
    train_parser.add_argument("-i", "--input", required=True, help="Training data")
    train_parser.add_argument("-m", "--model", required=True, 
                             choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"],
                             help="Select a model")
    train_parser.add_argument("-o", "--output", required=True, help="Feature importance output file")
    # train_parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    # train_parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    train_parser.set_defaults(func=run_training)
    
    # Visualization commands
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Feature importance analysis visualization"
    )
    viz_parser.add_argument("-i", "--input", required=True, help="Feature importance data files")
    viz_parser.add_argument("-o", "--output", required=True, help="Output image path")
    viz_parser.add_argument("-t", "--type", default="scatter",
                           choices=["scatter", "bar", "line", "hist"], help="Chart type")
    viz_parser.add_argument("--interactive", action="store_true", help="Enable interactive charts")
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
    logger = logging.getLogger(__name__)
    print_banner()
    
    parser = setup_argparse()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    try:
        ret_code = args.func(args)
        if ret_code != 0:
            logger.error("The process execution failed")
            sys.exit(ret_code)
            
        logger.info(f"The operation completed successfully! Total time spent: {format_runtime(start_time)}")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()