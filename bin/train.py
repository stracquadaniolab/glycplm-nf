#!/usr/bin/env python

import argparse
import json
import torch
import torch.nn as nn
import logging
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from classifier import ResidueClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def split_embedding_data(
        embedding_data,
        val_size=0.2,
        random_state=42
        ):
    
    '''
    Split a list of {'embedding':..., 'label':...} dicts into training and validation sets.

    Parameters:
    embedding_data: list[dict]
        List of dictionaries containing embeddings and labels for each protein.
    val_size: float
        Fraction of the dataset to be used for validation.
    random_state: int
        Random seed for reproducibility.   

    Returns:
    train_data: list[dict]
        List of dictionaries for the training set.
    val_data: list[dict]
        List of dictionaries for the validation set.
    '''
    
    train_data, val_data = train_test_split(
        embedding_data,
        test_size=val_size,
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

def compute_class_weight(labels):
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

def train_classifier(
        classifier,
        train_embeds, 
        train_labels, 
        val_embeds, 
        val_labels,
        num_epochs=5, 
        batch_size=256, 
        lr=1e-4, 
        weight=None):
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

    Returns:
    dict
        History of per-epoch train/val metrics
    """
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    weight = compute_class_weight(train_labels) if weight is None else weight
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    train_dataset = TensorDataset(train_embeds, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "precision": [], "recall": [], "f1": []}

    for epoch in range(num_epochs):
        # Training
        classifier.train()
        total_loss = 0

        for batch_embeds, batch_labels in train_loader:
            logits = classifier(batch_embeds)  # [batch_size, 2]
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
            preds = torch.argmax(val_logits, dim=-1)

            tp = ((preds == 1) & (val_labels == 1)).sum().item()
            fp = ((preds == 1) & (val_labels == 0)).sum().item()
            fn = ((preds == 0) & (val_labels == 1)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["f1"].append(f1)

        logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {val_loss:.4f} "
                    f"| Precision = {precision:.3f} | Recall = {recall:.3f} | F1 = {f1:.3f}")

    return history

def main():
    parser = argparse.ArgumentParser(description="Train residue-level glycosylation classifier")
    parser.add_argument("--input", required=True, help="Path to embedding_data .pt file (from get_embeddings.py)")
    parser.add_argument("--model_out", required=True, help="Path to save trained model state_dict (.pt)")
    parser.add_argument("--history_out", required=True, help="Path to save training history (.json)")
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction of proteins held out for validation")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for the train/val split")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (residues per batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=960, help="Embedding hidden size (ESMC-300M = 960)")
    args = parser.parse_args()
 
    embedding_data = torch.load(args.input)
 
    train_data, val_data = split_embedding_data(
        embedding_data,
        val_size=args.val_size,
        random_state=args.random_state,
    )
 
    train_embeds, train_labels = flatten_embeddings(train_data)
    val_embeds, val_labels = flatten_embeddings(val_data)
    logger.info(f"Train proteins: {len(train_data)}, Val proteins: {len(val_data)}")
    logger.info(f"Train residues: {train_embeds.shape[0]}, Val residues: {val_embeds.shape[0]}")
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_embeds, train_labels = train_embeds.to(device), train_labels.to(device)
    val_embeds, val_labels = val_embeds.to(device), val_labels.to(device)
 
    classifier = ResidueClassifier(hidden_size=args.hidden_size).to(device)
 
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
    )
 
    torch.save(classifier.state_dict(), args.model_out)
    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)
 
    print(f"Saved trained model to {args.model_out}")
    print(f"Saved training history to {args.history_out}")
 
if __name__ == "__main__":
    main()
 