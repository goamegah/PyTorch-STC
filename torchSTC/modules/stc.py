from typing import List, Union

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchSTC.models import AutoEncoder
from torchSTC.modules import ClusteringLayer

from torchSTC.metrics import metrics



class STC(nn.Module):
    def __init__(self, 
                 hidden_dims: List[int], 
                 n_clusters: int,
                 pretrained_autoencoder_path: Union[str, None] = None, 
                 alpha: float = 1.0) -> None:
        
        super(STC, self).__init__()

        self.hidden_dims = hidden_dims
        self.n_clusters = n_clusters
        self.alpha = alpha

        # Autoencoder
        self.autoencoder = AutoEncoder(hidden_dims)

        # Clustering layer
        self.clustering_layer = ClusteringLayer(n_clusters, input_dim=hidden_dims[-1])

    def from_pretrained(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
    
    def partial_forward(self, x: torch.Tensor):
        encoded = self.autoencoder.encoder(x)
        return encoded

    def forward(self, x: torch.Tensor):
        #q, p = self.model(x)
        encoded = self.autoencoder.encoder(x)
        #encoded = encoded.view(encoded.size(0), -1)
        q, p = self.clustering_layer(encoded)
        return q, p
    
    # def parameters(self, **kwargs):
    #     """
    #     Renvoie les paramètres nécessitant un gradient pour l'optimisation.
    #     Ici, seuls les paramètres de l'encodeur et de la couche de clustering nécessitent un gradient.
    #     """
    #     return list(self.autoencoder.encoder.parameters()) + list(self.clustering_layer.parameters())