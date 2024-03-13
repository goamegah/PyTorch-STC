from typing import List

import torch
import torch.nn as nn

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, input_dim):
        super(ClusteringLayer, self).__init__()
        self.clusters = nn.Parameter(torch.randn(n_clusters, input_dim))

    def forward(self, x):
        q = 1.0 / (1.0 + (torch.sum((x.unsqueeze(1) - self.clusters) ** 2, dim=2) / 1.0))
        q = q ** 2 / q.sum(0)
        q = (q.T / q.sum(1)).T
        return q