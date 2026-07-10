#!/usr/bin/env python

import torch.nn as nn

class ResidueClassifier(nn.Module):
    """
    Simple linear classifier for per-residue binary classification.
    A dropout layer is applied to the input embedding before the linear
    layer for regularisation.
    """
    
    def __init__(self, hidden_size=960, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)