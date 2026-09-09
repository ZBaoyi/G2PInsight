#!/usr/bin/env python3
import argparse
import logging
import sys
import warnings
from pathlib import Path
import time
from typing import Optional
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
_pkg_dir = Path(__file__).resolve().parent
_pkg_parent = _pkg_dir.parent
if (_pkg_dir / '__init__.py').exists() and str(_pkg_parent) not in sys.path:
    sys.path.insert(0, str(_pkg_parent))

def _prepare_stage_output_dir(output_dir: str, stage: str) -> str:
    base_output_dir = Path(output_dir).absolute()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    stage_output_dir = base_output_dir / stage
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    return str(stage_output_dir)

def init_logging(log_file: Optional[Path]=None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    root.addHandler(console)
    if log_file:
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        root.addHandler(file_handler)
    logging.captureWarnings(True)

def run_preprocess(args) -> int:
    from G2PInsight.bin.preprocess import run_preprocess as preprocess_func, resolve_genotype_input
    try:
        logger.info('Starting preprocessing')
        filter_snps_value = not args.no_filter_snps
        feature_selection_mode = getattr(args, 'feature_selection_mode', 1)
        gwas_pvalue = getattr(args, 'gwas_pvalue', 0.01)
        ld_config = getattr(args, 'ld_config', None) or '50,5,0.2'
        genotype_path, genotype_format = resolve_genotype_input(args)
        return preprocess_func(genotype_path=genotype_path, genotype_format=genotype_format, phenotype_file=args.phenotype, output_file=args.output, filter_snps=filter_snps_value, feature_selection_mode=feature_selection_mode, gwas_pvalue=gwas_pvalue, ld_config=ld_config, train_ids_file=getattr(args, 'train_ids_file', None), test_ids_file=getattr(args, 'test_ids_file', None), random_state=getattr(args, 'random_state', 42))
    except Exception as e:
        logger.error(f'Preprocessing failed: {str(e)}')
        return 1

def run_training(args) -> int:
    try:
        from G2PInsight.bin.modeltraining import run_single_model
        logger.info(f'Training model: {args.model}')
        return run_single_model(input_path=args.json, model_type=args.model, output_dir=_prepare_stage_output_dir(args.output_dir, 'train'), task_type=args.task_type, n_folds=5, random_state=args.random_state, enable_hyperparameter_search=args.hyperparameter_search, cpu_cores=getattr(args, 'threads', None), shap_dependence_max_features=getattr(args, 'shap_dependence_top', None))
    except Exception as e:
        logger.error(f'Training failed: {str(e)}')
        return 1

def run_train_all(args) -> int:
    try:
        from G2PInsight.bin.modeltraining import run_all_models
        logger.info('Training all supported models')
        return run_all_models(input_path=args.json, output_dir=_prepare_stage_output_dir(args.output_dir, 'train'), task_type=args.task_type, n_folds=5, random_state=args.random_state, parallel_models=getattr(args, 'parallel_models', None), enable_hyperparameter_search=args.hyperparameter_search, threads=getattr(args, 'threads', None), shap_dependence_max_features=getattr(args, 'shap_dependence_top', None))
    except Exception as e:
        logger.error(f'Training all models failed: {str(e)}')
        return 1

def run_predict(args) -> int:
    try:
        from G2PInsight.bin.modeltraining import predict_with_model
        logger.info('Running prediction')
        return predict_with_model(input_path=args.input, model_path=args.model, output_dir=_prepare_stage_output_dir(args.output_dir, 'predict'), task_type=args.task_type)
    except Exception as e:
        logger.error(f'Prediction failed: {str(e)}')
        return 1

def run_unified_visualization(args) -> int:
    result_code = 0
    has_importance = getattr(args, 'importance', None) is not None
    has_indicator = getattr(args, 'indicator', None) is not None
    if not has_importance and (not has_indicator):
        logger.error('At least one input must be specified:')
        logger.error('  -i/--importance: feature-importance file for importance visualization')
        logger.error('  -I/--indicator: model plotting metadata (plotting_data.json) for performance/CV visualization')
        return 1
    if not args.output:
        logger.error('When using -i/--importance or -I/--indicator, -o/--output must be provided')
        return 1
    output_prefix = Path(args.output)
    if not output_prefix.parent or str(output_prefix.parent) == '.':
        output_prefix = Path.cwd() / output_prefix.name
    base_output_dir = output_prefix.parent
    base_output_dir.mkdir(parents=True, exist_ok=True)
    stage_output_dir = base_output_dir / 'visualize'
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = stage_output_dir / output_prefix.name
    output_dir = stage_output_dir
    if has_indicator:
        try:
            logger.info('Generating model performance visualizations...')
            from G2PInsight.bin.visualization import plot_model_performance_from_file
            plotting_data_file = Path(args.indicator)
            if not plotting_data_file.exists():
                logger.error(f'Model plotting data file does not exist: {plotting_data_file}')
                result_code = 1
            else:
                logger.debug(f'Loading model plotting data from file: {plotting_data_file}')
                perf_result = plot_model_performance_from_file(plotting_data_file=plotting_data_file, output_dir=output_dir, publication_quality=None, output_prefix=output_prefix)
                if perf_result != 0:
                    result_code = perf_result
                else:
                    logger.info('Model performance visualization completed')
        except Exception as e:
            logger.error(f'Model performance visualization failed: {str(e)}')
            result_code = 1
    if has_importance:
        try:
            logger.info('Generating feature-importance visualizations...')
            from G2PInsight.bin.visualization import EnhancedGenomeVisualizer
            importance_file = Path(args.importance)
            if not importance_file.exists():
                raise FileNotFoundError(f'Feature-importance file not found: {importance_file}')
            visualizer = EnhancedGenomeVisualizer(input_file=str(importance_file), feature_col=None, value_col=None)
            static_output = f'{output_prefix}_importance_static.png'
            interactive_output = f'{output_prefix}_importance_interactive.html'
            is_shap_input = 'shap' in importance_file.name.lower()
            bar_output = f'{output_prefix}_shap_top_snps_bar.png' if is_shap_input else f'{output_prefix}_top_snps_bar.png'
            top_snps_n = getattr(args, 'top_snps', 20)
            if top_snps_n is None:
                top_snps_n = 20
            visualizer.plot_static_scatter(output_file=str(static_output))
            logger.debug(f'Static feature-importance plot generated: {static_output}')
            visualizer.plot_interactive_scatter(output_file=str(interactive_output))
            logger.debug(f'Interactive feature-importance plot generated: {interactive_output}')
            visualizer.plot_top_snps_bar(output_file=str(bar_output), top_n=int(top_snps_n))
            logger.debug(f'Top-SNPs bar chart generated: {bar_output}')
            if is_shap_input:
                from G2PInsight.bin.visualization import resolve_shap_dependence_file
                dep_file = resolve_shap_dependence_file(importance_file)
                if dep_file.exists():
                    plotting_meta_file = importance_file.parent / importance_file.name.replace('_shap_values.txt', '_plotting_data.json')
                    if plotting_meta_file.exists():
                        try:
                            import json
                            with open(plotting_meta_file, 'r', encoding='utf-8') as meta_f:
                                plotting_meta = json.load(meta_f)
                            dep_cap = plotting_meta.get('shap_dependence_top')
                            if dep_cap and int(top_snps_n) > int(dep_cap):
                                logger.debug(f'--top_snps {int(top_snps_n)} exceeds training --shap_dependence_top {int(dep_cap)}; dependence/summary plots cover at most {int(dep_cap)} SNPs')
                        except Exception:
                            pass
                    dep_saved = visualizer.plot_top_snps_dependence(dependence_npz=str(dep_file), output_dir=str(output_dir), output_prefix=str(output_prefix), top_n=int(top_snps_n))
                    if dep_saved:
                        logger.debug(f'SHAP dependence plot generated: {dep_saved[0]} ({len(dep_saved)} panel file, {int(top_snps_n)} SNPs)')
                    summary_path = visualizer.plot_shap_summary(dependence_npz=str(dep_file), output_dir=str(output_dir), output_prefix=str(output_prefix), top_n=int(top_snps_n))
                    if summary_path:
                        logger.debug(f'SHAP summary plot generated: {summary_path}')
                else:
                    logger.debug(f'SHAP dependence data not found: {dep_file}. Re-run model training to produce *_shap_dependence.tsv, then visualize again.')
            logger.info('Feature-importance visualization completed')
        except ImportError as e:
            logger.error(f'Module import error: {str(e)}')
            logger.error('For interactive plots, install dependencies: pip install plotly kaleido')
            result_code = 1
        except Exception as e:
            logger.error(f'Feature-importance visualization failed: {str(e)}')
            result_code = 1
    return result_code

def print_banner() -> None:
    from G2PInsight import __version__
    print('=' * 50)
    print(f'G2PInsight Genomic analysis platform v{__version__}')
    print('=' * 50)

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Compact, aligned help: short usage metasyntax + readable option columns."""

    def __init__(self, prog: str, indent_increment: int=2, max_help_position: int=32, width: Optional[int]=None):
        if width is None:
            try:
                import shutil
                width = min(100, max(80, shutil.get_terminal_size().columns))
            except Exception:
                width = 90
        super().__init__(prog, indent_increment=indent_increment, max_help_position=max_help_position, width=width)

    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):
            rows = []
            name_width = max((len(c.dest) for c in action._choices_actions), default=8)
            for choice in action._choices_actions:
                help_text = (choice.help or '').strip()
                rows.append(f'  {choice.dest:<{name_width}}  {help_text}')
            return '\n'.join(rows) + ('\n' if rows else '')
        return super()._format_action(action)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings and isinstance(action, argparse._SubParsersAction):
            return ''
        return super()._format_action_invocation(action)

def add_genotype_input_arguments(parser: argparse.ArgumentParser, *, required: bool=True) -> argparse._MutuallyExclusiveGroup:
    """Lightweight CLI helper (kept in main to avoid importing preprocess for -h)."""
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--bfile', metavar='PREFIX', dest='bfile', help='PLINK binary prefix (.bed/.bim/.fam)')
    group.add_argument('--file', metavar='PREFIX', dest='file', help='PLINK text prefix (.map/.ped)')
    group.add_argument('--vcf', metavar='PATH', dest='vcf', help='VCF file (.vcf / .vcf.gz)')
    return group

def _add_subparser(subparsers: argparse._SubParsersAction, name: str, *, help: str, description: str, epilog: Optional[str]=None) -> argparse.ArgumentParser:
    return subparsers.add_parser(name, help=help, description=description, epilog=epilog, formatter_class=_HelpFormatter)

def setup_argparse() -> argparse.ArgumentParser:
    from G2PInsight import __version__
    top_epilog = """\
examples:
  G2PInsight preprocess --bfile geno -p pheno.txt -o results/ -f 4
  G2PInsight train -j results/preprocess/results_metadata.json -m LightGBM -o results/
  G2PInsight train-all -j results/preprocess/results_metadata.json -o results/ --threads 4
  G2PInsight predict -i new_geno -m results/train/LightGBM/LightGBM_model.pkl -o results/
  G2PInsight visualize -I results/train/LightGBM/LightGBM_plotting_data.json -o results/plot/perf

Run "G2PInsight <command> -h" for command-specific options.
"""
    parser = argparse.ArgumentParser(prog='G2PInsight', description='G2PInsight — genotype–phenotype ML toolkit', formatter_class=_HelpFormatter, epilog=top_epilog)
    parser.add_argument('--version', action='version', version=f'G2PInsight {__version__}')
    subparsers = parser.add_subparsers(title='commands', dest='command', metavar='COMMAND')
    # Longer one-line helps avoid awkward wraps under the COMMAND column
    _cmd_help = {
        'preprocess': 'QC + optional GWAS/LD → matrix & metadata',
        'train': 'Train one model (fixed 5-fold CV)',
        'train-all': 'Train all models; keep best by test',
        'predict': 'Predict with a saved .pkl model',
        'visualize': 'Plots from SHAP / plotting JSON',
    }

    # --- preprocess ---
    preprocess_parser = _add_subparser(subparsers, 'preprocess', help=_cmd_help['preprocess'], description='Build the sample×SNP training matrix and metadata JSON.\nAfter QC, fixes a reproducible train/test split; GWAS/LD use train only.', epilog='Example:\n  G2PInsight preprocess --bfile geno -p pheno.txt -o results/ -f 4')
    geno_g = preprocess_parser.add_argument_group('genotype (choose one)')
    geno_mx = geno_g.add_mutually_exclusive_group(required=True)
    geno_mx.add_argument('--bfile', metavar='PREFIX', dest='bfile', help='PLINK binary prefix (.bed/.bim/.fam)')
    geno_mx.add_argument('--file', metavar='PREFIX', dest='file', help='PLINK text prefix (.map/.ped)')
    geno_mx.add_argument('--vcf', metavar='PATH', dest='vcf', help='VCF file (.vcf / .vcf.gz)')
    input_g = preprocess_parser.add_argument_group('phenotype / output')
    input_g.add_argument('-p', '--phenotype', required=True, metavar='FILE', help='Phenotype file (2 columns, no header: sample_id value)')
    input_g.add_argument('-o', '--output', required=True, metavar='PATH', help='Output directory or file prefix')
    feat_g = preprocess_parser.add_argument_group('feature selection')
    feat_g.add_argument('--no-filter-snps', action='store_true', help='Skip MAF/missingness QC (not for whole-genome)')
    feat_g.add_argument('-f', '--feature_selection_mode', type=int, default=1, choices=[1, 2, 3, 4], metavar='MODE', help='1=none (default), 2=GWAS, 3=LD, 4=GWAS→LD')
    feat_g.add_argument('--gwas_pvalue', type=float, default=0.01, metavar='P', help='GWAS P threshold when MODE is 2 or 4 (default: 0.01)')
    feat_g.add_argument('--ld-config', type=str, default='50,5,0.2', metavar='KB,N,R2', help='LD prune settings when MODE is 3 or 4 (default: 50,5,0.2)')
    split_g = preprocess_parser.add_argument_group('sample split')
    split_g.add_argument('--train-ids-file', dest='train_ids_file', default=None, metavar='FILE', help='Train sample IDs (one per line; use with --test-ids-file)')
    split_g.add_argument('--test-ids-file', dest='test_ids_file', default=None, metavar='FILE', help='Test sample IDs (one per line; use with --train-ids-file)')
    split_g.add_argument('--random_state', type=int, default=42, metavar='N', help='Seed for default sorted-ID 80/20 split (default: 42)')
    preprocess_parser.set_defaults(func=run_preprocess)

    # --- train ---
    train_parser = _add_subparser(subparsers, 'train', help=_cmd_help['train'], description='Train one model using preprocess metadata.\nUses fixed 5-fold CV on the train split; test set is evaluation-only.', epilog='Example:\n  G2PInsight train -j results/preprocess/results_metadata.json -m LightGBM -o results/')
    train_req = train_parser.add_argument_group('required')
    train_req.add_argument('-j', '--json', required=True, metavar='FILE', help='Preprocess *_metadata.json')
    train_req.add_argument('-m', '--model', required=True, metavar='MODEL', choices=['LightGBM', 'RandomForest', 'XGBoost', 'SVM', 'CatBoost', 'Logistic'], help='LightGBM | RandomForest | XGBoost | SVM | CatBoost | Logistic')
    train_req.add_argument('-o', '--output_dir', required=True, metavar='DIR', help='Output directory (creates <DIR>/train/<MODEL>/)')
    train_opt = train_parser.add_argument_group('optional')
    train_opt.add_argument('--task_type', required=False, metavar='TYPE', choices=['classification', 'regression'], help='classification | regression (default: from metadata)')
    train_opt.add_argument('--random_state', type=int, default=42, metavar='N', help='Seed for CV / model fitting (default: 42)')
    train_opt.add_argument('--threads', type=int, default=None, metavar='N', help='CPU threads per model (default: 1)')
    train_opt.add_argument('--shap_dependence_top', type=int, default=50, metavar='N', help='Cap SNPs in *_shap_dependence.tsv when features > 1000 (default: 50; 0=all)')
    train_opt.add_argument('--hyperparameter_search', dest='hyperparameter_search', action='store_true', help='Enable RandomizedSearchCV (off by default)')
    train_parser.set_defaults(hyperparameter_search=False, func=run_training)

    # --- train-all ---
    train_all_parser = _add_subparser(subparsers, 'train-all', help=_cmd_help['train-all'], description='Train every supported model on the same preprocess split.\nKeeps the best model by held-out test metrics.', epilog='Example:\n  G2PInsight train-all -j results/preprocess/results_metadata.json -o results/ --threads 4')
    ta_req = train_all_parser.add_argument_group('required')
    ta_req.add_argument('-j', '--json', required=True, metavar='FILE', help='Preprocess *_metadata.json')
    ta_req.add_argument('-o', '--output_dir', required=True, metavar='DIR', help='Output directory (creates <DIR>/train/)')
    ta_opt = train_all_parser.add_argument_group('optional')
    ta_opt.add_argument('--task_type', required=False, metavar='TYPE', choices=['classification', 'regression'], help='classification | regression (default: from metadata)')
    ta_opt.add_argument('--random_state', type=int, default=42, metavar='N', help='Seed for CV / model fitting (default: 42)')
    ta_opt.add_argument('--parallel_models', type=int, default=1, metavar='N', help='Models to train in parallel (default: 1)')
    ta_opt.add_argument('--threads', type=int, default=None, metavar='N', help='CPU threads per model process (default: 1)')
    ta_opt.add_argument('--shap_dependence_top', type=int, default=50, metavar='N', help='Cap SNPs in *_shap_dependence.tsv when features > 1000 (default: 50; 0=all)')
    ta_opt.add_argument('--hyperparameter_search', dest='hyperparameter_search', action='store_true', help='Enable RandomizedSearchCV (off by default)')
    train_all_parser.set_defaults(hyperparameter_search=False, func=run_train_all)

    # --- predict ---
    predict_parser = _add_subparser(subparsers, 'predict', help=_cmd_help['predict'], description='Apply a saved .pkl model to new samples.\nWrites <output_dir>/predict/{Model}_predictions.tsv.', epilog='Example:\n  G2PInsight predict -i new_geno -m results/train/LightGBM/LightGBM_model.pkl -o results/')
    pred_req = predict_parser.add_argument_group('required')
    pred_req.add_argument('-i', '--input', required=True, metavar='PATH', help='Input matrix (.txt/.gz) or genotype prefix/VCF')
    pred_req.add_argument('-m', '--model', required=True, metavar='FILE', help='Trained model .pkl')
    pred_req.add_argument('-o', '--output_dir', required=True, metavar='DIR', help='Output directory')
    pred_opt = predict_parser.add_argument_group('optional')
    pred_opt.add_argument('--task_type', metavar='TYPE', choices=['classification', 'regression'], help='classification | regression (optional)')
    predict_parser.set_defaults(func=run_predict)

    # --- visualize ---
    viz_parser = _add_subparser(subparsers, 'visualize', help=_cmd_help['visualize'], description='Render importance and/or performance plots.\nProvide at least one of -i/--importance or -I/--indicator.', epilog='Examples:\n  G2PInsight visualize -I results/train/LightGBM/LightGBM_plotting_data.json -o results/plot/perf\n  G2PInsight visualize -i results/train/LightGBM/LightGBM_shap_values.txt -o results/plot/shap')
    viz_in = viz_parser.add_argument_group('input (at least one)')
    viz_in.add_argument('-i', '--importance', metavar='FILE', help='Feature-importance / SHAP values file')
    viz_in.add_argument('-I', '--indicator', metavar='FILE', help='Model plotting_data.json')
    viz_out = viz_parser.add_argument_group('output')
    viz_out.add_argument('-o', '--output', required=True, metavar='PREFIX', help='Output filename prefix')
    viz_out.add_argument('--top_snps', type=int, default=20, metavar='N', help='Top SNPs in bar/dependence plots (default: 20)')
    viz_parser.set_defaults(func=run_unified_visualization)
    return parser

def format_runtime(start_time: float) -> str:
    seconds = time.perf_counter() - start_time
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    seconds = int(seconds % 60)
    return f'{hours}h{minutes}m{seconds}s'

def _is_help_or_version_request(argv: Optional[list]=None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    return any((a in ('-h', '--help', '--version') for a in args))

def main() -> None:
    start_time = time.perf_counter()
    help_or_version = _is_help_or_version_request()
    if not help_or_version:
        init_logging(Path('G2PInsight.log'))
        print_banner()
    else:
        # Minimal console logging for -h/--version (avoid slow path side effects)
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = setup_argparse()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    try:
        ret_code = args.func(args)
        if ret_code != 0:
            logger.error(f'{args.command} failed with error code {ret_code}')
            sys.exit(ret_code)
        if not help_or_version:
            logger.info(f'Operation completed successfully! Total time: {format_runtime(start_time)}')
    except FileNotFoundError as e:
        logger.critical(f'File not found: {str(e)}')
        sys.exit(2)
    except ValueError as e:
        logger.critical(f'Invalid data format: {str(e)}')
        sys.exit(3)
    except Exception as e:
        logger.critical(f'Unexpected error: {str(e)}', exc_info=True)
        sys.exit(1)
if __name__ == '__main__':
    main()
