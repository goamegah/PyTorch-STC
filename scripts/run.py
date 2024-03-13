import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import des modules STC et AutoEncoder
from torchclust.modules import STC
from torchclust.models import AutoEncoder
from torchclust.metrics import metrics
from torchclust.data import load_data

def main(args):
    # Chargement des données
    # Assurez-vous que les données sont chargées sous forme de tenseurs
    x, y = load_data(args.dataset)
    n_clusters = len(torch.unique(y))

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    # conversion des données en tenseurs
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Création du modèle STC
    stc = STC(hidden_dims=[X_train.shape[-1], 500, 500, 2000, 20], n_clusters=n_clusters)

    # Préentraînement de l'autoencodeur si les poids ne sont pas déjà préentraînés
    if not os.path.exists(args.ae_weights):
        print("Préentraînement de l'autoencodeur...")
        autoencoder_optimizer = optim.Adam(stc.autoencoder.parameters())
        train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        stc.pretrain_autoencoder(train_loader, autoencoder_optimizer, args.pretrain_epochs, args.batch_size, args.ae_weights)
    else:
        print("Chargement des poids préentraînés de l'autoencodeur...")
        stc.autoencoder.load_state_dict(torch.load(args.ae_weights))

    # Initialisation du clustering
    print("Initialisation du clustering...")
    stc.soft_clustering(X_train, n_clusters)

    # Optimisation de l'encodeur avec le clustering
    print("Optimisation de l'encodeur avec le clustering...")
    clustering_optimizer = optim.Adam(stc.clustering_layer.parameters())
    y_pred_train = stc.update_encoder_weights(X_train, y_train, clustering_optimizer, args.maxiter, args.batch_size, args.tol)

    # Évaluation du modèle sur l'ensemble de test
    _, decoded_test = stc(X_test)
    q_test = stc.clustering_layer(X_test)
    y_pred_test = torch.argmax(q_test, dim=1)
    acc_test = metrics.acc(y_test, y_pred_test.numpy())
    nmi_test = metrics.nmi(y_test, y_pred_test.numpy())
    print(f"Accuracy on test set: {acc_test:.4f}")
    print(f"NMI on test set: {nmi_test:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='stackoverflow',
                        choices=['stackoverflow', 'biomedical', 'search_snippets'])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--maxiter', default=1000, type=int)
    parser.add_argument('--pretrain_epochs', default=15, type=int)
    parser.add_argument('--update_interval', default=30, type=int)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--ae_weights', default='/data/search_snippets/results/ae_weights.pth')
    parser.add_argument('--save_dir', default='/data/search_snippets/results/')
    args = parser.parse_args()

    # Exécution du programme principal
    main(args)