import pytest

import torch

from torchSTC.modules import ClusteringLayer

class TestClusteringLayer:
    def _create_clustering_layer(self, 
                                 n_clusters: int, 
                                 input_dim: int):
        layer = ClusteringLayer(n_clusters, input_dim)
        return layer

    def test_model(self):
        n_clusters = 10
        input_dim = 32
        layer = self._create_clustering_layer(n_clusters, input_dim)
        # create random input and compare input and output shapes: laters should be equal
        x = torch.rand(32, input_dim)
        # check if forward method works
        q, p = layer.forward(x)
        assert q.shape == (32, n_clusters)
        assert p.shape == (32, n_clusters)
        # check if init_clusters method works
        x = torch.rand(n_clusters, input_dim)
        layer.init_clusters(x)
        assert torch.all(layer.clusters == x)
        # check if solf_clustering method works
        x = torch.rand(32, input_dim)
        q = layer.solf_clustering(x)
        assert q.shape == (32, n_clusters)
        # check if target_distribution method works
        p = layer.target_distribution(q)
        assert p.shape == (32, n_clusters)
        

