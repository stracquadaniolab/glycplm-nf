#!/usr/bin/env python

import argparse
import torch
import pandas as pd
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "/opt/conda/models/ESMC-300M"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    device_map="auto",
).eval()

def get_embeddings(
        entries,
        entry_names,
        sequences, 
        labels,
        batch_size,
        model=model, 
        tokenizer=tokenizer
        ):

    '''
    Get embeddings for a list of protein sequences using the ESMC model.
    
    Parameters:
    sequences: list[str]
        List of protein sequences.
    labels: list[str]
        List of glycosylation binary strings corresponding to the sequences.
    entries: list[str]
        List of entry IDs corresponding to the sequences.
    entry_names: list[str]
        List of entry names corresponding to the sequences.
    batch_size: int
        Number of sequences to process in a batch.
    model: AutoModelForMaskedLM
        The ESMC model for generating embeddings.
    tokenizer: AutoTokenizer
        The tokenizer for the ESMC model.

    Returns:
    list[dict]
        A list of dictionaries, each containing 'entry', 'entry_name', 
        'sequence', 'embedding' and 'label' for a protein.
    '''

    logger.info(f"Generating embeddings for {len(sequences)} sequences, batch_size={batch_size}")
    embedding_data = []
    n_skipped = 0

    for i in tqdm(range(0, len(sequences), batch_size), mininterval=30, maxinterval=60):
        batch_seqs = sequences[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_entries = entries[i:i+batch_size]
        batch_entry_names = entry_names[i:i+batch_size]
        
        # Tokenise
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding='longest')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            # Get hidden states from the encoder
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, padded_seq_len, 960]
        
        # Align and save each sequence in the batch
        for j in range(len(batch_seqs)):
            # 1. Get the valid token mask (ignore padding)
            valid_positions = inputs['attention_mask'][j].nonzero().squeeze()
            
            # 2. Remove the [CLS] (first) and [EOS] (last) tokens
            if len(valid_positions) >= 2:
                valid_positions = valid_positions[1:-1]
            
            # 3. Extract the matching embeddings
            seq_embedding = hidden_states[j, valid_positions, :].cpu()  # Shape: [protein_length, 960]
            
            # 4. Extract the matching label string and convert to tensor of integers
            label = batch_labels[j]
            # Ensure the label string length matches the number of valid positions
            try:
                assert len(label) == seq_embedding.shape[0], "Length mismatch! Check labels."
            except AssertionError as e:
                n_skipped += 1
                logger.warning(f"Skipping sequence {i+j}: {e}")
                continue
            seq_label = torch.tensor(label, dtype=torch.long)

            embedding_data.append({
                'entry': batch_entries[j],
                'entry_name': batch_entry_names[j],
                'sequence': batch_seqs[j],
                'embedding': seq_embedding,
                'label': seq_label
            })
    
    logger.info(f"Embedded {len(embedding_data)} proteins, skipped {n_skipped}")

    return embedding_data

def main():
    parser = argparse.ArgumentParser(description="Generate ESMC embeddings for protein sequences")
    parser.add_argument("--input", required=True, help="Path to processed/split JSON file (from preprocess.py)")
    parser.add_argument("--output", required=True, help="Path to output .pt file with embedding_data list")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding generation")
    args = parser.parse_args()
 
    df = pd.read_json(args.input, orient='records')
 
    entries = df["Entry"].tolist()
    entry_names = df["Entry Name"].tolist()
    sequences = df["Sequence"].tolist()
    labels = df["Glycosylation_binary"].tolist()
 
    embedding_data = get_embeddings(entries, entry_names, sequences, labels, batch_size=args.batch_size)
 
    torch.save(embedding_data, args.output)
    logger.info(f"Saved embedding data to {args.output}")
 
 
if __name__ == "__main__":
    main()