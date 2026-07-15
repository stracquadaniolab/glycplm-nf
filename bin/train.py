#!/usr/bin/env python

import argparse
import json
import torch
import torch.nn as nn
import logging
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, precision_recall_fscore_support
from classifiers import ResidueClassifier, ResidueClassifierMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def split_embedding_data(
        embedding_data,
        val_size,
        test_size,
        random_state
        ):
    
    '''
    Split a list of {'embedding':..., 'label':...} dicts into training and
    validation sets.

    Parameters:
    embedding_data: list[dict]
        List of dictionaries containing embeddings and labels for each protein.
    val_size: float
        Fraction of the data (from full dataset) to be used for validation.
    test_size: float
        Fraction of the dataset held out as the test set.
    random_state: int
        Random seed for reproducibility.   

    Returns:
    train_data: list[dict]
    val_data: list[dict]
    '''
    
    # Split the remainder into train/val
    relative_val_size = val_size / (1 - test_size)
    
    train_data, val_data = train_test_split(
        embedding_data,
        test_size=relative_val_size,
        random_state=random_state
    )
    
    return train_data, val_data

def flatten_embeddings(
        data
        ):
    
    '''
    Flatten a list of {'embedding': [seq_len, hidden], 'label': [seq_len]} dicts
    into single tensors over all residues across all proteins.

    Parameters:
    data: list[dict]
        List of dictionaries containing embeddings and labels for each protein.

    Returns:
    all_embeds: torch.Tensor
        Concatenated embeddings of shape [total_residues, hidden].
    all_labels: torch.Tensor
        Concatenated labels of shape [total_residues].
    '''

    all_embeds = torch.cat([entry['embedding'] for entry in data], dim=0)  # [total_residues, hidden]
    all_labels = torch.cat([entry['label'] for entry in data], dim=0)      # [total_residues]
    
    return all_embeds, all_labels

def compute_class_weight(
        labels
        ):
    '''
    Compute class weight for binary classification based on the training labels.
    
    Parameters:
    labels: torch.Tensor
        Tensor of shape [num_samples], with values 0 or 1

    Returns:
    class_weight: Tensor of shape [2], where class_weight[0] is the weight for class 0 
    and class_weight[1] is the weight for class 1.
    '''
    num_pos = labels.sum().item()
    num_neg = len(labels) - num_pos
    class_weight = torch.tensor([1.0, num_neg / num_pos], device=labels.device)

    logger.info(f"Class balance -> neg: {num_neg}, pos: {num_pos}, weight: {class_weight[1]}")

    return class_weight

def find_optimal_threshold(
        val_probs, 
        val_labels, 
        metric='f1'
        ):
    '''
    Find threshold that maximises the chosen metric on validation set.

    Parameters:
    val_probs: torch.Tensor
        Probabilities of residues being glycosylated.
    val_labels: torch.Tensor
        True labels.
    metric: 'f1', 'precision', 'recall', or 'youden' (maximizes TPR - FPR)
        Metric to optimise.

    Returns:
    best_threshold: float
        Best threshold for glycosylated/nonglycosylated prediction.
    best_metric:float
        Best value of the chosen metric.

    '''
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
    
    if metric == 'f1':
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # exclude last threshold (0)
        best_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]
    elif metric == 'youden':
        # Youden's J statistic
        fpr = 1 - precisions  # approximate, or compute from confusion matrix
        youden = recalls - (1 - precisions)  # TPR - FPR
        best_idx = np.argmax(youden[:-1])
        best_threshold = thresholds[best_idx]
        best_score = youden[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return best_threshold, best_score

def evaluate(
        classifier, 
        embeds, 
        labels, 
        loss_fn,
        threshold):
    '''
    Evaluate the classifier on a held-out set.

    Parameters:
    classifier: nn.Module
        The trained classifier model.
    embeds: torch.Tensor
        Embeddings of shape [num_samples, hidden_size].
    labels: torch.Tensor
        True labels of shape [num_samples].
    loss_fn: nn.Module
        Loss function used for evaluation.
    threshold: float
        Threshold to make decisions whether resie is glycosylated or not.

    Returns:
    dict with loss, precision, recall, f1, ROC AUC, PR AUC
    '''
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeds)
        loss = loss_fn(logits, labels).item()
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= threshold).long()

        labels_np = labels.cpu().numpy()
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average="binary", zero_division=0
        )
        
        try:
            roc_auc = roc_auc_score(labels_np, probs_np)
            pr_auc = average_precision_score(labels_np, probs_np)
        except ValueError:
            roc_auc, pr_auc = float('nan'), float('nan')
            logger.warning("AUC undefined: Only one class present in the split")

    return {
        "loss": loss, "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }

def train_classifier(
        classifier,
        train_embeds, 
        train_labels, 
        val_embeds, 
        val_labels,
        num_epochs, 
        batch_size, 
        lr, 
        weight,
        metric):
    """
    Train the residue-level classifier using a class-weighted loss to handle imbalance.

    Parameters:
    train_embeds: torch.Tensor
        [num_train_residues, hidden_size]
    train_labels: torch.Tensor
        [num_train_residues], long dtype, values in {0, 1}
    val_embeds: torch.Tensor
        [num_val_residues, hidden_size]
    val_labels: torch.Tensor
        [num_val_residues], long dtype, values in {0, 1}
    num_epochs: int
        Number of training epochs
    batch_size: int
        Batch size for training
    lr: float 
        Learning rate
    weight: torch.Tensor or None
        Class weight for the loss function
    metric: 
        Metric used to optimise threshold.

    Returns:
    dict
        History of per-epoch train/val metrics, evaluated at each epoch's
        freshly-optimised threshold.
    """
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    weight = compute_class_weight(train_labels) if weight is None else weight
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    train_dataset = TensorDataset(train_embeds, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logger.info(f'Using {metric} to optimise threshold.')

    history = {
        "train_loss": [],
        "val_loss": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "best_threshold": []
        }

    for epoch in range(num_epochs):
        # Training
        classifier.train()
        total_loss = 0

        for batch_embeds, batch_labels in train_loader:
            logits = classifier(batch_embeds)
            loss = loss_fn(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation 
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_embeds)
            val_loss = loss_fn(val_logits, val_labels).item()
            val_probs = torch.softmax(val_logits, dim=-1)[:, 1]

        val_probs_np = val_probs.cpu().numpy()
        val_labels_np = val_labels.cpu().numpy()

        # Find this epoch's optimal threshold from the current probability distribution
        best_thresh, _ = find_optimal_threshold(val_probs_np, val_labels_np, metric)

        # Evaluate at that threshold
        threshold_metrics = evaluate(classifier, val_embeds, val_labels, loss_fn, threshold=best_thresh)

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(threshold_metrics["loss"]))
        history["precision"].append(float(threshold_metrics["precision"]))
        history["recall"].append(float(threshold_metrics["recall"]))
        history["f1"].append(float(threshold_metrics["f1"]))
        history["roc_auc"].append(float(threshold_metrics["roc_auc"]))
        history["pr_auc"].append(float(threshold_metrics["pr_auc"]))
        history["best_threshold"].append(float(best_thresh))

        logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {threshold_metrics['loss']:.4f} "
            f"| Precision = {threshold_metrics['precision']:.3f} | Recall = {threshold_metrics['recall']:.3f} "
            f"| F1 = {threshold_metrics['f1']:.3f} | ROC-AUC = {threshold_metrics['roc_auc']:.3f} "
            f"| PR-AUC = {threshold_metrics['pr_auc']:.3f} | Best Thresh = {best_thresh:.3f}")

    return history

def main():
    parser = argparse.ArgumentParser(description="Train residue-level glycosylation classifier")
    parser.add_argument("--input", required=True, help="Path to embedding_data .pt file (from get_embeddings.py)")
    parser.add_argument("--model_out", required=True, help="Path to save trained model state_dict (.pt)")
    parser.add_argument("--history_out", required=True, help="Path to save training history (.json)")
    parser.add_argument("--threshold_out", required=True, help="Path to save the final optimised decision threshold (.json)")
    parser.add_argument("--val_size", type=float, help="Fraction of proteins held out for validation")
    parser.add_argument("--test_size", type=float, help="Fraction of proteins held out for testing")
    parser.add_argument("--random_state", type=int, help="Random seed for the train/val/test split")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size (residues per batch)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, help="Embedding hidden size (ESMC-300M = 960)")
    parser.add_argument("--dropout", type=float, help="Dropout rate for classifier training")
    parser.add_argument("--optimise_metric", help="Metric used to optimise threshold for glycosylation prediction")
    parser.add_argument("--classifier", help="Whether to use single linear layer classifier ('LL') or MLP ('MLP')")
    parser.add_argument("--mlp_hidden_size", type=int, help="If using MLP, which hidden dimensions to use")
    args = parser.parse_args()

    embedding_data = torch.load(args.input, weights_only=True)

    train_data, val_data = split_embedding_data(
        embedding_data,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    logger.info(f"Split: {len(train_data)} train / {len(val_data)} val proteins")

    train_embeds, train_labels = flatten_embeddings(train_data)
    val_embeds, val_labels = flatten_embeddings(val_data)
    logger.info(f"Train residues: {train_embeds.shape[0]}, Val residues: {val_embeds.shape[0]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training using device: {device}")
    train_embeds, train_labels = train_embeds.to(device), train_labels.to(device)
    val_embeds, val_labels = val_embeds.to(device), val_labels.to(device)

    if args.classifier == 'LL':
        classifier = ResidueClassifier(dropout=args.dropout, input_dim=args.hidden_size).to(device)
    elif args.classifier == 'MLP':
        classifier = ResidueClassifierMLP(dropout=args.dropout, input_dim=args.hidden_size, hidden_dim=args.mlp_hidden_size).to(device)
    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")

    weight = compute_class_weight(train_labels)

    history = train_classifier(
        classifier,
        train_embeds,
        train_labels,
        val_embeds,
        val_labels,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight=weight,
        metric=args.optimise_metric
    )

    torch.save(classifier.state_dict(), args.model_out)
    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)

    final_threshold = history["best_threshold"][-1]
    with open(args.threshold_out, "w") as f:
        json.dump({"threshold": final_threshold, "metric": args.optimise_metric}, f, indent=2)
    logger.info(f"Saved final threshold ({final_threshold:.3f}, optimised for {args.optimise_metric}) to {args.threshold_out}")
 
    logger.info(f"Saved trained model to {args.model_out}")
    logger.info(f"Saved training history to {args.history_out}")


if __name__ == "__main__":
    main()