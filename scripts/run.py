# -*- coding: utf-8 -*-

import os
from time import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from torchclust.metrics import metrics
from torchclust.data.data_loader import load_data


if __name__ == "__main__":
    # args
    ####################################################################################
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='stackoverflow',
                        choices=['stackoverflow', 'biomedical', 'search_snippets'])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--maxiter', default=1000, type=int)
    parser.add_argument('--pretrain_epochs', default=15, type=int)
    parser.add_argument('--update_interval', default=30, type=int)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--ae_weights', default='/data/stackoverflow/results/ae_weights.h5')
    parser.add_argument('--save_dir', default='/data/stackoverflow/results/')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.dataset == 'search_snippets':
        args.update_interval = 100
        args.maxiter = 100
    elif args.dataset == 'stackoverflow':
        args.update_interval = 500
        args.maxiter = 1500
        args.pretrain_epochs = 12
    elif args.dataset == 'biomedical':
        args.update_interval = 300
    else:
        raise Exception("Dataset not found!")

    print(args)

    # load dataset
    ####################################################################################
    x, y = load_data(args.dataset)
    n_clusters = len(np.unique(y))

    X_test, X_dev, y_test, y_dev = train_test_split(x, y, test_size=0.1, random_state=0)
    x, y = shuffle(X_test, y_test)
