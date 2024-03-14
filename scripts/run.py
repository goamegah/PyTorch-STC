import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchinfo import summary

# Import des modules STC et AutoEncoder
from torchclust.modules import STC
from torchclust.metrics import metrics
from torchclust.data import load_data
from torchclust.utils import pretrain_autoencoder, self_train

def main(args):
    # Chargement des données
    # Assurez-vous que les données sont chargées sous forme de tenseurs
    x, y = load_data(args.dataset)
    n_clusters = len(torch.unique(torch.tensor(y)))

    print("nclusters = ", n_clusters)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    # conversion des données en tenseurs
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), 
                                  batch_size=args.batch_size, 
                                  shuffle=True)
    
    # Création du modèle STC
    #print(torch.Tensor(X_train).shape)
    #print(X_train.shape[-1])
    #test_ae = AutoEncoder([X_train.shape[-1], 500, 500, 2000, 20])
    #print(test_ae.encoder(torch.Tensor(X_train)).shape)
    #summary(test_ae.encoder, torch.Tensor(X_train).shape)
    stc = STC(hidden_dims=[torch.Tensor(X_train).shape[-1], 500, 500, 2000, 20], n_clusters=n_clusters)
    #print(stc.autoencoder.encoder(torch.Tensor(X_train)).shape)
    #print(len(stc.clustering_layer(stc.autoencoder.encoder(torch.Tensor(X_train)))))
    

    # Préentraînement de l'autoencodeur si les poids ne sont pas déjà préentraînés
    if not os.path.exists(args.ae_weights):
        os.makedirs(os.path.dirname(args.ae_weights), exist_ok=True)
        print("Préentraînement de l'autoencodeur...")
        ae_optimizer = optim.Adam(stc.autoencoder.parameters())
        ae_criterion = nn.MSELoss()
        pretrain_autoencoder(stc.autoencoder,
                             train_loader, 
                             ae_optimizer, 
                             ae_criterion,
                             args.pretrain_epochs, 
                             args.ae_weights)
    else:
        print("Chargement des poids préentraînés de l'autoencodeur...")
        stc.autoencoder.from_pretrained(args.ae_weights)

   
    summary(stc, input_size=torch.Tensor(X_train).shape)

    # Freeze parameters of the decoder
    for param in stc.autoencoder.decoder.parameters():
        param.requires_grad = False

    # Optimisation de l'encodeur avec le clustering
    print("Optimisation de l'encodeur avec le clustering...")
    st_optimizer = optim.Adam(stc.parameters())
    st_criterion = nn.KLDivLoss(reduction='batchmean')

    y_pred_train = self_train(stc,
                              st_criterion,
                              st_optimizer,
                              X_train, 
                              y_train, 
                              args.maxiter, 
                              args.batch_size, 
                              args.tol,
                              args.update_interval,
                              args.save_dir,
                              rand_seed=0)

    # # Évaluation du modèle sur l'ensemble de test
    # _, decoded_test = stc(X_test)
    # q_test = stc.clustering_layer(X_test)
    # y_pred_test = torch.argmax(q_test, dim=1)
    # acc_test = metrics.acc(y_test, y_pred_test.numpy())
    # nmi_test = metrics.nmi(y_test, y_pred_test.numpy())
    # print(f"Accuracy on test set: {acc_test:.4f}")
    # print(f"NMI on test set: {nmi_test:.4f}")

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
    parser.add_argument('--ae_weights', default='data/stackoverflow/artefacts/ae_weights.pth')
    parser.add_argument('--save_dir', default='data/stackoverflow/artefacts/')
    args = parser.parse_args()

    # Exécution du programme principal
    main(args)