import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import sys
from os.path import abspath
sys.path.insert(0, abspath('..'))

from os.path import join

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
import numpy as np


import torch
from torchSTC.data import load_data
from torchSTC.modules import STC
from torchSTC.metrics import SpacePlot, Evaluate
from torchSTC.utils.cluster import SphericalKmeans

plot = SpacePlot()
eval = Evaluate()

cur = abspath("")
dataset = 'stackoverflow'
data_in_dir=join(cur, "..", "datasets")