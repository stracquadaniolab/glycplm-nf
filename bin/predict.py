#!/usr/bin/env python

import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from classifier import ResidueClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict(classifier, embedding_data, device, threshold):
    '''
    Run the classifier on each protein's embedding and produce per-residue predictions.

    Parameters:
    classifier: ResidueClassifier
        The trained model.
    embedding_data: list[dict]
        List of {'entry', 'entry_name', 'sequence', 'embedding', 'label'} dicts (label may be None).
    device: str
        'cuda' or 'cpu'
    threshold: float
        Decision threshold for predicted_label (should match the threshold optimised during training).

    Returns:
    predictions_df: pd.DataFrame
        One row per residue: protein_index, position, amino_acid, predicted_prob,
        predicted_label, true_label (true_label is empty if no ground truth).
    all_labels: list[int]
        Flattened true labels, only for residues where ground truth was available.
    all_probs: list[float]
        Flattened predicted probabilities aligned with all_labels.
    '''
    records = []
    all_labels = []
    all_probs = []

    classifier.eval()
    with torch.no_grad():
        for idx, entry in enumerate(embedding_data):
            embed = entry['embedding'].to(device)
            logits = classifier(embed)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)

            ent = entry['entry']
            ent_name = entry['entry_name']
            seq = entry['sequence']
            label = entry['label']
            label_arr = label.numpy() if label is not None else [None] * len(seq)

            for pos, (aa, p, pr, lab) in enumerate(zip(seq, probs, preds, label_arr), start=1):
                records.append({
                    'entry': ent,
                    'entry_name': ent_name,
                    'protein_index': idx,
                    'position': pos,
                    'amino_acid': aa,
                    'predicted_prob': float(p),
                    'predicted_label': int(pr),
                    'true_label': int(lab) if lab is not None else None
                })

            if label is not None:
                all_labels.extend(label_arr.tolist())
                all_probs.extend(probs.tolist())

    return pd.DataFrame.from_records(records), all_labels, all_probs

def compute_metrics(labels, probs, threshold):
    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    preds_arr = (probs_arr >= threshold).astype(int)
 
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_arr, preds_arr, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(labels_arr, probs_arr)
        pr_auc = average_precision_score(labels_arr, probs_arr)
    except ValueError:
        roc_auc, pr_auc = float('nan'), float('nan')
        logger.warning("AUC metrics undefined: only one class present among labeled residues")
 
    return {
        "n_residues": int(len(labels_arr)),
        "threshold": float(threshold),  # CHANGED: record which threshold was used
        "precision": float(precision),  # CHANGED: explicit float() cast for JSON serialisation safety
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }
 

def main():
    parser = argparse.ArgumentParser(
        description="Run a trained residue classifier on new embedding data and output per-residue predictions"
    )
    parser.add_argument("--embedding_input", required=True, help="Path to embedding_data .pt file (from get_embeddings.py)")
    parser.add_argument("--model_ckpt", required=True, help="Path to trained model state_dict (.pt, from train.py)")
    parser.add_argument("--hidden_size", type=int, help="Embedding hidden size (must match training)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for glycosylation prediction (use the optimised threshold from train.py)")
    parser.add_argument("--predictions_out", required=True, help="Path to save per-residue predictions (.json)")
    parser.add_argument("--metrics_out", default=None, help="Path to save aggregate metrics (.json); only written if ground-truth labels are present")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Using decision threshold: {args.threshold:.3f}")

    embedding_data = torch.load(args.embedding_input, weights_only=True)
    logger.info(f"Loaded embeddings for {len(embedding_data)} proteins")

    classifier = ResidueClassifier(hidden_size=args.hidden_size).to(device)
    state_dict = torch.load(args.model_ckpt, map_location=device, weights_only=True)
    classifier.load_state_dict(state_dict)
    logger.info(f"Loaded model checkpoint from {args.model_ckpt}")

    predictions_df, all_labels, all_probs = predict(classifier, embedding_data, device, args.threshold)
    predictions_df.to_json(args.predictions_out, orient='records', indent=2)
    logger.info(f"Saved {len(predictions_df)} residue-level predictions to {args.predictions_out}")

    if all_labels and args.metrics_out:
        metrics = compute_metrics(all_labels, all_probs, args.threshold)
        logger.info(
            f"Test set: n={metrics['n_residues']} | Precision = {metrics['precision']:.3f} "
            f"| Recall = {metrics['recall']:.3f} | F1 = {metrics['f1']:.3f} "
            f"| ROC-AUC = {metrics['roc_auc']:.3f} | PR-AUC = {metrics['pr_auc']:.3f}"
        )
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {args.metrics_out}")
    elif args.metrics_out:
        logger.warning("Metrics unavailable: no ground-truth labels found in embedding data")


if __name__ == "__main__":
    main()