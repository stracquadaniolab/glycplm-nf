#!/usr/bin/env python

import torch.nn as nn

class ResidueClassifier(nn.Module):
    """
    Simple linear classifier for per-residue binary classification.
    A dropout layer is applied to the input embedding before the linear
    layer for regularisation.
    """
    
    def __init__(
        self, 
        dropout,
        input_dim=960):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class ResidueClassifierMLP(nn.Module):

    def __init__(
        self, 
        dropout,
        input_dim=960, 
        hidden_dim=256
        ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.classifier(x)