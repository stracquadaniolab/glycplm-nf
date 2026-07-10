#!/usr/bin/env python

import argparse
import json
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def split_train_val_test(
        df,
        test_size,
        random_state
        ):
    '''
    Split a protein-level DataFrame into a train+val set and a held-out test set.
 
    Parameters:
    df: pd.DataFrame
        Preprocessed protein-level data (Entry, Entry Name, Sequence, Glycosylation_binary).
    test_size: float
        Fraction of proteins to hold out as the test set. If 0/None, no split
        is performed and the test set is empty (useful when you just want a
        single processed file, e.g. for standalone prediction).
    random_state: int
        Random seed for reproducibility.
 
    Returns:
    train_val_df: pd.DataFrame
    test_df: pd.DataFrame
    '''
    if test_size and test_size > 0:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
    else:
        train_val_df, test_df = df, df.iloc[0:0]
        logger.warning("No test split performed: test_size is 0 or None, returning empty test set.")
 
    return train_val_df.reset_index(drop=True), test_df.reset_index(drop=True)
 
 
def main():
    parser = argparse.ArgumentParser(description="Split preprocessed protein data into train+val / test JSON files")
    parser.add_argument("--input", required=True, help="Path to processed.json (from preprocess.py)")
    parser.add_argument("--train_val_out", required=True, help="Path to output train+val JSON file")
    parser.add_argument("--test_out", required=True, help="Path to output test JSON file")
    parser.add_argument("--test_size", type=float, help="Fraction of proteins held out as the test set")
    parser.add_argument("--random_state", type=int, help="Random seed for the split")
    args = parser.parse_args()
 
    df = pd.read_json(args.input, orient='records')
    logger.info(f"Loaded {len(df)} proteins from {args.input}")
 
    train_val_df, test_df = split_train_val_test(
        df, test_size=args.test_size, random_state=args.random_state
    )
    logger.info(f"Split: {len(train_val_df)} train+val / {len(test_df)} test proteins")
    
    train_val_df.to_json(args.train_val_out, orient='records', indent=2)
    test_df.to_json(args.test_out, orient='records', indent=2)
 
    logger.info(f"Saved train+val proteins to {args.train_val_out}")
    logger.info(f"Saved test proteins to {args.test_out}")
 
 
if __name__ == "__main__":
    main()