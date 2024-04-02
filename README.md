# PyTorch-Short-Text-Clustering
PyTorch version of Self-training approch for short text clustering

### Original Paper: [A Self-Training Approach for Short Text Clustering](https://aclanthology.org/W19-4322/)

![Image of STC Arch](assets/cover.png)

## Installation

- Using **venv**

Create environement like below.

```
$ python -m venv torchSTC
$ source torchSTC/bin/activate
```

Clone repository and run installation step

```
$ git clone git@github.com:goamegah/torchSTC.git
$ cd torchSTC
$ pip install .
$ python scripts/run.py
```

 In such case you might want to make visualisation or use PyTorch libs like *torchinfo*, you have to lunch instead command below  

```
$ python -m venv torchSTC
$ source torchSTC/bin/activate
$ git clone git@github.com:goamegah/torchSTC.git
$ cd torchSTC
$ pip install ".[dev, vis]"
$ python scripts/run.py
```

- Using **conda** 

```
$ conda env create --name torchSTC --file env.yaml
$ conda activate torchSTC
$ python scripts/run.py
```





## Config file

Before running torchSTC, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py --dataset stackoverflow --word_emb Word2Vec --init Kmeans --max_iter 1500

```

## Feature Evaluation

Feature evaluation is done by running Kmeans 5 times and keep average results. 

First, we learned features using **autoencoder** on the ```Stackoverflow``` set. Then, we finetune both autoencoder and cluster centers by frozing decoder part and using Adam as optimizer with default params. Objective function use **KL-divergence** on two distribution: **Q** make by ***soft-assignment*** and an target distribution **P**. After convergence, use run 5 runs clustering algorithm like **Kmeans**.

Check the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/goamegah/torchSTC/blob/main/demos/stackoverlow/stc_final_assignment_hgf_sof.ipynb) notebook for reproducibility.


## Acknowledgments
The authors would like to thank the anonymous reviewers for their constructive feedback.