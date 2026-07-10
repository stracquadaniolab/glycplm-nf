#!/usr/bin/env python

import argparse
import json
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_glycosylation_binary( 
        sequence,   
        glycosylation_data  
        ):
    
    '''
    Convert glycosylation data to a binary string representation for a given protein sequence.

    Parameters:
    sequence: str
        The protein sequence.
    glycosylation_data: str
        The glycosylation data string from the TSV file.

    Returns:
    str
        A binary string where '1' indicates a glycosylation site and 
        '0' indicates no glycosylation at that position.
    '''
    
    if not sequence or pd.isna(sequence):
        return ''
    
    if not glycosylation_data or pd.isna(glycosylation_data):
        return '0' * len(sequence)
    
    # Extract positions
    positions = re.findall(r'CARBOHYD\s+(\d+)', str(glycosylation_data))
    positions = [int(p) for p in positions]
    
    # Create glycosylation binary
    binary = [0] * len(sequence)

    for pos in positions:
        if 1 <= pos <= len(sequence):
            binary[pos - 1] = 1

    return binary

def load_data(
        file_path,
        drop_non_glycosylated=True,
        length_filter=None
        ): 
    
    '''
    Load raw data from a TSV file into a pandas DataFrame and process glycosylation information.
    
    Parameters:
    file_path: str
        The path to the TSV file containing protein sequences and glycosylation data.
    drop_non_glycosylated: bool
        If True, rows without glycosylation annotations will be dropped from the DataFrame.
    length_filter: int
        If provided, proteins longer than this length will be filtered out.
    

    Returns:
    pd.DataFrame
        A DataFrame containing the original data along with a new column for glycosylation binary representation.
    '''
    
    df = pd.read_csv(file_path, sep='\t')
    logger.info(f"Loaded {len(df)} raw rows from {file_path}")

    # Filter out non glycosylated proteins
    if drop_non_glycosylated:
        df = df.loc[df['Glycosylation'].notnull(),] 
        logger.info(f"{len(df)} rows remain after filtering for glycosylation annotations")

    # Filter out long proteins
    if length_filter:
        df = df.loc[df['Sequence'].str.len() <= length_filter,]
        logger.info(f"{len(df)} rows remain after filtering for sequence length <= {length_filter}")

    # Create a new column for glycosylation binary representation
    df['Glycosylation_binary'] = df.apply(
        lambda row: get_glycosylation_binary(row['Sequence'], row['Glycosylation']), 
        axis=1
        )

    n_pos_sites = df['Glycosylation_binary'].apply(lambda s: s.count('1')).sum()
    logger.info(f"Total annotated glycosylation sites: {n_pos_sites}")

    df = df[['Entry', 'Entry Name', 'Sequence', 'Glycosylation_binary']].reset_index(drop=True)

    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess glycosylation TSV data")
    parser.add_argument("--input", required=True, help="Path to raw input TSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file (processed DataFrame)")
    parser.add_argument("--drop_non_glycosylated", action='store_true', help="Drop sequences without glycosylation annotations")
    parser.add_argument("--length_filter", type=int, default=None, help="Filter out sequences longer than this length")
    args = parser.parse_args()
 
    df = load_data(
        file_path=args.input,
        drop_non_glycosylated=args.drop_non_glycosylated,
        length_filter=args.length_filter
    )

    df.to_json(args.output, orient='records', indent=2)

    logger.info(f"Loaded and processed {len(df)} proteins.")
    logger.info(f"Saved processed data to {args.output}")
 
 
if __name__ == "__main__":
    main()