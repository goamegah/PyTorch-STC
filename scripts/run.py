import argparse
import os

import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchinfo import summary

# Import des modules STC et AutoEncoder
from torchstc.modules import STC
#from torchclust.metrics import metrics
from torchstc.data import load_data
from torchstc.utils import pretrain_autoencoder, self_train
#from torchclust.utils.cluster import SphericalKmeans
#from torchclust.metrics import Evaluate

def main(args):
    print('\n*********************************** Step1 - loading data, Embedding *******************************\n')
    # Chargement des données
    # Assurez-vous que les données sont chargées sous forme de tenseurs
    #x, y = load_data(args.dataset, emb_type="W2V-SIF", normalize_type='MinMax')
    x, y = load_data(args.dataset, word_emb=args.word_emb, transform=args.transform_emb, scaler=args.scaler, norm=args.norm)
    #x, y = load_data(args.dataset, emb_type="JOSE", normalize_type='MinMax')
    #x, y = load_data(args.dataset, emb_type="JOSE", normalize_type='Spherical')
    #x, y = load_data(args.dataset, emb_type="JOSE-SIF", normalize_type='MinMax')
    #x, y = load_data(args.dataset, emb_type="JOSE-SIF", normalize_type='Spherical')
    n_clusters = len(torch.unique(torch.tensor(y)))

    print(f"Data shape: {x.shape}, Labels shape: {y.shape}")

    # print settings
    print(f"n_clusters: {n_clusters}")

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

    hidden_dims = [torch.Tensor(X_train).shape[-1], 500, 500, 2000, 20]
    stc = STC(hidden_dims=hidden_dims, n_clusters=n_clusters)
    #print(stc.autoencoder.encoder(torch.Tensor(X_train)).shape)
    #print(len(stc.clustering_layer(stc.autoencoder.encoder(torch.Tensor(X_train)))))
    
    # build string indicating the autoencoder hidden dim format like this: d:d2:...:dn
    hidden_units = [str(d) for d in hidden_dims]
    hidden_units = ":".join(hidden_units)
    print(f"Autoencoder hidden units: {hidden_units}")

    # Préentraînement de l'autoencodeur si les poids ne sont pas déjà préentraînés
    print('\n ****************************** Step 2 - Pretraining Auto encoder ***************************\n')
    data = args.dataset.split("/")[-1]
    art_dir = f'{args.save_dir}STC-d{hidden_units}-epoch{args.pretrain_epochs}-dat{data}-wde{args.word_emb}-sca{args.scaler}-tfe{args.transform_emb}-norm{args.norm}-init{args.init}/'
    ae_file = f'{art_dir}ae-d{hidden_units}-epoch{args.pretrain_epochs}.pth'

    if not os.path.exists(art_dir):
        os.makedirs(os.path.dirname(art_dir), exist_ok=True)
        args.save_dir = art_dir
        print("Préentraînement de l'autoencodeur...")
        ae_optimizer = optim.Adam(stc.autoencoder.parameters())
        ae_criterion = nn.MSELoss()
        pretrain_autoencoder(stc.autoencoder,
                             train_loader, 
                             ae_optimizer, 
                             ae_criterion,
                             args=args)
    else:
        args.save_dir = art_dir
        print("Chargement des poids préentraînés de l'autoencodeur...")
        stc.autoencoder.from_pretrained(ae_file)

   
    summary(stc, input_size=torch.Tensor(X_train).shape)

    # Freeze parameters of the decoder
    for param in stc.autoencoder.decoder.parameters():
        param.requires_grad = False

    print('\n*********************************** Step 3 - Self-traning **********************************\n')


    # Optimisation de l'encodeur avec le clustering
    print("Optimisation de l'encodeur avec le clustering...")
    st_optimizer = optim.Adam(stc.parameters())
    # st_optimizer = optim.SGD(stc.parameters(), lr=0.1, momentum=0.9)
    st_criterion = nn.KLDivLoss(reduction='batchmean')

    y_pred_train = self_train(stc, st_criterion, st_optimizer, args, x=X_train, y=y_train)

    # eval = Evaluate()

    # # # Évaluation du modèle sur l'ensemble de test
    # z = stc.autoencoder.encoder(X_test)

    # # comprehension list with 5 runs of Sherical kmeans, get average and std of metrics
    # avg_hgf_mmx_iskm = []
    # tmp = []
    # for _ in range(5):
    #     skmeans = SphericalKmeans(n_clusters=n_clusters, n_init=50)
    #     skmeans.fit(z.detach().numpy())
    #     y_skm_pred = skmeans.labels_
    #     tmp.append(eval.allMetrics(y_test.detach().numpy(), y_skm_pred))

    # avg_hgf_mmx_iskm = np.array(tmp)
    # np.round(avg_hgf_mmx_iskm.mean(axis=0), 3) * 100, avg_hgf_mmx_iskm.std(axis=0)

    # avg_hgf_mmx_ikm = []
    # tmp = []
    # for _ in range(5):
    #     kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    #     kmeans.fit(z.detach().numpy())
    #     y_km_pred = kmeans.labels_
    #     tmp.append(eval.allMetrics(y_test.detach().numpy(), y_km_pred))

    # avg_hgf_mmx_ikm = np.array(tmp)
    # np.round(avg_hgf_mmx_ikm.mean(axis=0), 3) * 100, avg_hgf_mmx_ikm.std(axis=0)



    # print(f"SKmeans on X_test: {np.round(avg_hgf_mmx_iskm.mean(axis=0), 3) * 100}, {avg_hgf_mmx_iskm.std(axis=0)}")
    # print(f"Kmeans on X_test: {np.round(avg_hgf_mmx_ikm.mean(axis=0), 3) * 100}, {avg_hgf_mmx_ikm.std(axis=0)}")


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
    
    parser.add_argument('--word_emb', default='Word2Vec', 
                        choices= ['Word2Vec', 'HuggingFace', 'Jose'])
    
    parser.add_argument('--transform_emb', default='SIF', 
                        choices= [None, 'SIF'])
    
    parser.add_argument('--scaler', default='MinMax', 
                        choices=[None, 'MinMax', 'Standard'])
    
    parser.add_argument('--norm', default=None,
                        choices=['l2', 'l1', 'max'])
    
    parser.add_argument('--init', default='Kmeans', 
                        choices=['Kmeans', 'movMF-soft', 
                                 'SphericalKmeans', 'SphericalKmeans++'])
    
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--save_dir', default='datasets/Biomedical/artefacts/')
    args = parser.parse_args()

    print(args)

    if args.dataset == 'search_snippets':
        args.update_interval = 30
        args.maxiter = 100
        args.pretrain_epochs = 30

        args.dataset = 'datasets/SearchSnippets'
        args.save_dir = 'datasets/SearchSnippets/artefacts/'

    elif args.dataset == 'stackoverflow':
        args.update_interval = 500
        args.maxiter = 1500
        args.pretrain_epochs = 15

        args.dataset = 'datasets/stackoverflow'
        args.save_dir = 'datasets/stackoverflow/artefacts/'

    elif args.dataset == 'biomedical':
        args.update_interval = 500
        args.pretrain_epochs = 15
        args.maxiter = 1500

        args.dataset = 'datasets/Biomedical'
        args.save_dir = 'datasets/Biomedical/artefacts/'

    else:
        raise Exception("Dataset not found!")


    # Exécution du programme principal
    main(args)