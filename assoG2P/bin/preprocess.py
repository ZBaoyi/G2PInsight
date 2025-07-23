
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict
import pandas as pd
# Initialize module-level logger
logger = logging.getLogger(__name__)

def load_phenotype_data(phenotype_file: str) -> Dict[str, float]:
    """Load phenotype data and return sample ID to phenotype mapping
    
    Args:
        phenotype_file: Path to phenotype data file
        
    Returns:
        Dictionary mapping sample IDs to phenotype values
        
    Raises:
        ValueError: If file format is invalid
    """
    try:
        phenotype_df = pd.read_csv(
            phenotype_file, 
            sep='\t', 
            header=None, 
            na_values=["nan", "NA", " "],
            names=["ID", "Phenotype"]
        )
        return dict(zip(phenotype_df['ID'], phenotype_df['Phenotype']))
    except Exception as e:
        logger.error(f"Failed to load phenotype data: {str(e)}")
        raise

def process_genotype_file(
    genotype_file: str,
    output_file: str,
    phenotype_dict: Dict[str, float]
) -> str:
    """Process genotype file and write merged output
    
    Args:
        genotype_file: Input VCF file path
        output_file: Output file path
        phenotype_dict: Sample to phenotype mapping
        
    Returns:
        Path to generated output file
        
    Raises:
        ValueError: If genotype format is invalid
    """
    # Genotype encoding mapping
    genotype_map = {
        "0|0": "0",
        "0|1": "1",
        "1|0": "1",
        "1|1": "2"
    }
    
    try:
        with open(genotype_file, 'r') as infile:
            # First pass: Extract sample IDs and SNP positions
            sample_ids = []
            chrom_pos_list = []
            
            for line in infile:
                if line.startswith('#CHROM'):
                    sample_ids = line.strip().split('\t')[9:]
                elif not line.startswith('#'):
                    chrom_pos_list.append(line.split('\t')[2])
        
        # Write output with header
        with open(output_file, "w") as outfile:
            outfile.write("sample\t" + "\t".join(chrom_pos_list) + "\tphenotype\n")
            
            # Second pass: Process and write genotypes
            with open(genotype_file, 'r') as infile:
                sample_genotypes = {sample: [] for sample in sample_ids}
                
                for line in infile:
                    if line.startswith('#'):
                        continue
                        
                    parts = line.strip().split('\t')
                    genotypes = [genotype_map.get(g, g) for g in parts[9:]]
                    
                    for i, sample in enumerate(sample_ids):
                        sample_genotypes[sample].append(genotypes[i])
                
                # Write sample data
                for sample in sample_ids:
                    if sample in phenotype_dict:
                        outfile.write(f"{sample}\t" + "\t".join(sample_genotypes[sample]) + 
                                    f"\t{phenotype_dict[sample]}\n")
        
        logger.info(f"Processed data written to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Genotype processing failed: {str(e)}")
        raise

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments
    
    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Genotype-phenotype data preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-g", "--genotype",
        required=True,
        help="Input genotype file (VCF format)"
    )
    parser.add_argument(
        "-p", "--phenotype",
        required=True,
        help="Input phenotype file (tab-delimited)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path"
    )
    return parser.parse_args()

def run_preprocess(
    genotype_file: str,
    phenotype_file: str,
    output_file: str
) -> int:
    """Main preprocessing workflow
    
    Args:
        genotype_file: Input VCF path
        phenotype_file: Phenotype data path
        output_file: Output file path
        
    Returns:
        0 on success, 1 on failure
    """
    try:
        logger.info("Starting data preprocessing")
        phenotype_data = load_phenotype_data(phenotype_file)
        process_genotype_file(genotype_file, output_file, phenotype_data)
        logger.info("Preprocessing completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return 1

def cli_main() -> None:
    """Command line entry point"""
    args = parse_arguments()
    sys.exit(run_preprocess(
        genotype_file=args.genotype,
        phenotype_file=args.phenotype,
        output_file=args.output
    ))

if __name__ == "__main__":
    cli_main()