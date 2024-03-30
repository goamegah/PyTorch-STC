from typing import List

import torch
import torch.nn as nn

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, input_dim):
        super(ClusteringLayer, self).__init__()
        self.clusters = nn.Parameter(torch.randn(n_clusters, input_dim))

    def init_clusters(self, x: torch.Tensor):
        self.clusters.data = x
    
    def solf_clustering(self, x: torch.Tensor):
        assert len(x.shape) == 2, "Input tensor must be 2D." 

        q = 1.0 / (1.0 + (torch.sum((x.unsqueeze(1) - self.clusters) ** 2, dim=2) / 1.0))
        q = q ** 2 / q.sum(0)
        q = (q.T / q.sum(1)).T
        return q

    def target_distribution(self, q : torch.Tensor):
        p = q ** 2 / q.sum(0)
        p = (p.T / p.sum(1)).T
        return p

    def forward(self, x : torch.Tensor):
        q = self.solf_clustering(x)
        p = self.target_distribution(q)
        return q, p