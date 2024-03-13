from typing import List, Union

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchclust.models import AutoEncoder
from torchclust.modules import ClusteringLayer

from torchclust.metrics import metrics



class STC(nn.Module):
    def __init__(self, 
                 hidden_dims: List[int], 
                 n_clusters: int, 
                 autoencoder: Union[AutoEncoder, None] = None,
                 pretrained_autoencoder_path: Union[str, None] = None, 
                 alpha: float = 1.0) -> None:
        
        super(STC, self).__init__()

        self.hidden_dims = hidden_dims
        self.n_clusters = n_clusters
        self.alpha = alpha

        # Initialize autoencoder
        if autoencoder is None:
            self.autoencoder = AutoEncoder(hidden_dims)
        else:
            self.autoencoder = autoencoder

        # Load pretrained autoencoder weights if provided
        if pretrained_autoencoder_path is not None:
            self.autoencoder.load_state_dict(torch.load(pretrained_autoencoder_path))

        # Clustering layer
        self.clustering_layer = ClusteringLayer(n_clusters, input_dim=hidden_dims[-1])

    def forward(self, x: torch.Tensor):
        encoded = self.autoencoder.encoder(x)
        decoded = self.autoencoder.decoder(encoded)
        return encoded, decoded

    def pretrain_autoencoder(self, 
                             train_loader: torch.utils.data.DataLoader, 
                             optimizer: optim.Optimizer, 
                             epochs: int, 
                             batch_size: Union[int, None] = None,
                             save_path: Union[str, None] = None) -> None:
        
        self.autoencoder.train()

        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for data, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
                optimizer.zero_grad()
                encoded, decoded = self(data)
                loss = criterion(decoded, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

        torch.save(self.autoencoder.state_dict(), save_path)
        print(f"Pretrained autoencoder weights saved to: {save_path}")

    def soft_clustering(self, x, n_clusters):
        encoded, _ = self(x)
        self.clusters = nn.Parameter(torch.randn(n_clusters, encoded.size(1)))
        self.clusters.data = self.clusters.data * 0.01

    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def update_encoder_weights(self, 
                               x: torch.Tensor, 
                               y: torch.Tensor, 
                               optimizer: optim.Optimizer, 
                               maxiter: int, 
                               batch_size: int, 
                               tol: float) -> torch.Tensor:
        
        self.clustering_layer.train()

        # Initialize KMeans
        z = self.autoencoder.encoder(x)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z.detach().numpy())
        self.clustering_layer.clusters.data = nn.Parameter(torch.tensor(kmeans.cluster_centers_))

        criterion = nn.KLDivLoss(reduction='batchmean')
        prev_y_pred = None

        for ite in tqdm(range(maxiter), desc="Clustering Iterations"):
            optimizer.zero_grad()
            z = self.autoencoder.encoder(x)
            q = self.clustering_layer(z)
            p = self.target_distribution(q)
            loss = criterion(torch.log(p), q)

            # Check convergence
            if prev_y_pred is not None:
                delta_label = torch.sum(torch.tensor(prev_y_pred) != y).float() / y.size(0)
                if delta_label < tol:
                    print(f"Delta label: {delta_label:.6f} < Tol: {tol:.6f}. Stopping training.")
                    break

            loss.backward()
            optimizer.step()

            # Update y_pred
            prev_y_pred = torch.argmax(q, dim=1)

            # Print progress
            if (ite + 1) % 10 == 0:
                acc = metrics.acc(y, prev_y_pred.numpy())
                nmi = metrics.nmi(y, prev_y_pred.numpy())
                print(f"Iteration {ite + 1}/{maxiter}, Loss: {loss.item():.6f}, Acc: {acc:.6f}, NMI: {nmi:.6f}")

        return prev_y_pred
