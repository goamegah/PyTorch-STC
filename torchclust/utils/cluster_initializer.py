from sklearn.cluster import KMeans

import torch


def get_clusters(x: torch.Tensor,
                 n_clusters: int,
                 kind: str = "kmeans") -> torch.Tensor:
        if kind == "kmeans":
            kmeans = KMeans(n_clusters=n_clusters, n_init=100)
            _ = kmeans.fit_predict(x.detach().numpy())
            return torch.Tensor(kmeans.cluster_centers_), torch.Tensor(kmeans.labels_)