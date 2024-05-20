from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchSTC.models import AutoEncoder
from torchSTC.modules import STC
from torchSTC.utils.cluster import get_clusters
from torchSTC.metrics import Evaluate

def self_train(model: STC, 
               criterion: nn.Module, 
               optimizer: optim.Optimizer,
               args,
               x: torch.Tensor,  
               y: Union[torch.Tensor, None]=None) -> np.ndarray:
        
    print('Update interval', args.update_interval)
    save_interval = int(x.shape[0] / args.batch_size) * 5  # 5 epochs
    print('Save interval', save_interval)

    # instantiate evaluate class
    eval = Evaluate()

    # Stage 1: initialize cluster centers using k-means-like algorithm
    print(f'Initializing cluster centers with {args.init}.')
    z = model.partial_forward(x)

    clusters, y_pred = get_clusters(z, 
                                    n_clusters=model.n_clusters, 
                                    kind=args.init)
    
    y_pred = y_pred.detach().numpy()
    model.clustering_layer.init_clusters(torch.Tensor(clusters))

    loss = 0
    index = 0
    index_array = np.arange(x.shape[0])

    # Stage 2: jointly optimize the cluster centers and the network parameters
    for ite in range(int(args.maxiter)):
        if ite % args.update_interval == 0:
            # compute all embeddings points 
            # compute soft assignment probabilities q 
            # update the target distribution p using the soft assignment probabilities q 
            q, p = model(x)

            # save the last label assignment y_pred_last 
            y_pred_last = np.copy(y_pred)

            # compute new label assignment y_pred
            _, y_pred = q.max(1)
            y_pred = y_pred.detach().numpy()
            if y is not None:
                all_metrics = eval.allMetrics(y, y_pred, prec=5)
                print(f"Iter {ite}: acc = {all_metrics[0]}, nmi = {all_metrics[1]}, ari = {all_metrics[2]}, loss = {loss:.5f}")

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]

            if ite > 0 and delta_label < args.tol:
                print('delta_label ', delta_label, '< tol ', args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        
        # Choose a batch of samples S = {x_i} from x and their corresponding targets P = {p_i} from p 
        idx = index_array[index * args.batch_size: min((index + 1) * args.batch_size, x.shape[0])]
        x_batch = x[idx].clone().detach().float()
        p_batch = p[idx].clone().detach().float()

        # update the cluster centers and the network parameters using the batch of samples S and their corresponding targets P 

            # Step 3: Forward pass
        q_batch, _ = model(x_batch)

            # Step 4: Compute the KL divergence loss
        loss = criterion(q_batch.log(), p_batch)

            # Step 5: Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        index = index + 1 if (index + 1) * args.batch_size <= x.shape[0] else 0

    data = args.dataset.split("/")[-1]
    # save the trained model
    print('saving model to:', args.save_dir + f'STC-dat{data}-wde{args.word_emb}-sca{args.scaler}-tfe{args.transform_emb}-norm{args.norm}-init{args.init}.pth')
    torch.save(model.state_dict(), args.save_dir + f'STC-dat{data}-wde{args.word_emb}-sca{args.scaler}-tfe{args.transform_emb}-norm{args.norm}-init{args.init}.pth')
    # save optimal metrics into a file csv with config

    return y_pred


def pretrain_autoencoder(model: AutoEncoder, 
                         train_loader: torch.utils.data.DataLoader,
                         optimizer: optim.Optimizer, 
                         criterion: nn.Module,
                         args) -> None:
        
        model.train()
        for epoch in range(args.pretrain_epochs):
            total_loss = 0.0
            for data, _ in tqdm(train_loader, 
                                desc=f'Epoch {epoch + 1}/{args.pretrain_epochs}', 
                                leave=False):
                optimizer.zero_grad()
                decoded = model(data)
                loss = criterion(decoded, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{args.pretrain_epochs}, Loss: {total_loss / len(train_loader):.6f}")

        # buid string to indicate the hidden units in the format 'd1:d2:...:dn'
        hidden_units = [str(d) for d in model.hidden_units]
        hidden_units = ":".join(hidden_units)

        torch.save(model.state_dict(), f'{args.save_dir}ae-d{hidden_units}-epoch{args.pretrain_epochs}.pth')
        print(f"Pretrained autoencoder weights saved to: {args.save_dir}ae-d{hidden_units}-epoch{args.pretrain_epochs}.pth'")