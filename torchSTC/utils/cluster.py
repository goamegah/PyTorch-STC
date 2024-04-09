from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from spherecluster import VonMisesFisherMixture
#from soyclustering import SphericalKMeans
#from coclust.clustering.spherical_kmeans import SphericalKmeans
from torchSTC.utils.spherical_kmeans import SphericalKmeans
from torchSTC.utils.sphericalKmeans_optim import SphericalKMeans
import torch


def get_clusters(x: torch.Tensor,
                 n_clusters: int,
                 kind: str = "Kmeans") -> torch.Tensor:
        if kind == "Kmeans":
            kmeans = KMeans(n_clusters=n_clusters, n_init=100)
            _ = kmeans.fit_predict(x.detach().numpy())
            return torch.Tensor(kmeans.cluster_centers_), torch.Tensor(kmeans.labels_)


        elif kind == "movMF-soft":
            # spharical kmeans
            # spherical-k-means
            # k-means++
            vmf_soft = VonMisesFisherMixture(n_clusters=n_clusters, 
                                             posterior_type='soft')
            vmf_soft.fit(x.detach().numpy())
            return torch.Tensor(vmf_soft.cluster_centers_), torch.Tensor(vmf_soft.labels_)

        elif kind == "SphericalKmeans":
            skmeans = SphericalKmeans(n_clusters=n_clusters,
                                      max_iter=100,
                                      n_init=50)
            skmeans.fit(x.detach().numpy())
            return torch.Tensor(skmeans.centers), torch.Tensor(skmeans.labels_)
        
        elif kind == "SphericalKmeans++":
            skmeanspp = SphericalKMeans(n_clusters=n_clusters)
            x = csr_matrix(x.detach().numpy())
            skmeanspp.fit(x)
            return torch.Tensor(skmeanspp.cluster_centers_), torch.Tensor(skmeanspp.labels_)