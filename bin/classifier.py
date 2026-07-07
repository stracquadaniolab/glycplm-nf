#!/usr/bin/env python

import torch.nn as nn

class ResidueClassifier(nn.Module):
    """
    Simple linear baseline for per-residue binary classification.
    Just a single linear layer - no hidden layers, no activation.
    """
    
    def __init__(self, hidden_size=960):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        return self.linear(x)