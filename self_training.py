from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchclust.models import AutoEncoder
from torchclust.modules import STC
from torchclust.metrics import metrics
from torchclust.utils.cluster_initializer import get_clusters

def self_train(model: STC, 
               criterion: nn.Module, 
               optimizer: optim.Optimizer, 
               x: torch.Tensor,  
               y: Union[torch.Tensor, None]=None,
               maxiter: int=2e4, 
               batch_size: int=256, 
               tol: float=1e-3,
               update_interval: int=140, 
               save_dir='data/stackoverflow/artefacts', 
               rand_seed: Union[int, None]=None):
        
    print('Update interval', update_interval)
    save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
    print('Save interval', save_interval)

    # Step 1: initialize cluster centers using k-means
    print('Initializing cluster centers with k-means.')
    z = model.partial_forward(x)
    clusters, y_pred = get_clusters(z, 
                                    n_clusters=model.n_clusters, 
                                    kind="kmeans")
    model.clustering_layer.init_clusters(torch.Tensor(clusters))
    y_pred_last = np.copy(y_pred.detach().numpy())

    loss = 0
    index = 0
    index_array = np.arange(x.shape[0])
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q, p = model(x)

            _, y_pred = q.max(1)
            y_pred = y_pred.detach().numpy()
            if y is not None:
                acc = np.round(metrics.acc(y, y_pred), 5)
                nmi = np.round(metrics.nmi(y, y_pred), 5)
                print('Iter %d: acc = %.5f, nmi = %.5f' % (ite, acc, nmi))

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break

        idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
        x_batch = x[idx].clone().detach().float()
        p_batch = p[idx].clone().detach().float()

        # Step 3: Forward pass
        q_batch, _ = model(x_batch)

        # Step 4: Compute the KL divergence loss
        loss = criterion(q_batch.log(), p_batch)

        # Step 5: Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    # save the trained model
    print('saving model to:', save_dir + 'STC_model_final.pth')
    torch.save(model.state_dict(), save_dir + 'STC_model_final.pth')

    return y_pred


def pretrain_autoencoder(model: AutoEncoder, 
                         train_loader: torch.utils.data.DataLoader,
                         optimizer: optim.Optimizer, 
                         criterion: nn.Module,
                         epochs: int = 200,
                         save_path: Union[str, None] = None) -> None:
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for data, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
                optimizer.zero_grad()
                decoded = model(data)
                loss = criterion(decoded, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

        torch.save(model.state_dict(), save_path)
        print(f"Pretrained autoencoder weights saved to: {save_path}")