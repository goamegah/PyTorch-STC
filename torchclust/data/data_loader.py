# -*- coding: utf-8 -*-

from typing import Literal, Union
import os
from pathlib import Path
from collections import Counter

import ast

import nltk
nltk.download('punkt')  # adding new version nltk download
import pandas as pd
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler

from torchclust.data.utils import get_emb, read_label, get_wv_emb, get_avg_emb_from_word_vectors

#HERE = Path().cwd().resolve()
#print(">>>>> data_loader.py HERE: ", HERE)
cwd = os.getcwd()
print(">>>>> data_loader.py cwd: ", cwd)

def get_word_vectors(data_path="datasets/stackoverflow", 
                     word_emb='Word2Vec'):
    
    # data_path = cwd + 'datasets' / f'{dataset}'

    #data_path = cwd + '/datasets' + f'/{dataset}'
    if data_path.endswith('stackoverflow'):
        #data_path = 'datasets/stackoverflow'
        if word_emb == 'Word2Vec':
            # ids_file = data_path / 'vocab_withIdx.dic'
            # vec_dict_file = data_path / 'vocab_emb_Word2vec_48_index.dic'
            # vec_file = data_path / 'vocab_emb_Word2vec_48.vec'

            ids_file = data_path + '/vocab_withIdx.dic'
            vec_dict_file = data_path + '/vocab_emb_Word2vec_48_index.dic'
            vec_file = data_path + '/vocab_emb_Word2vec_48.vec'

            # load SO embedding
            with open(ids_file, 'r') as inp_indx, \
                    open(vec_dict_file, 'r') as inp_dic, \
                    open(vec_file) as inp_vec:
                pair_dic = inp_indx.readlines()
                word_index = {}
                for pair in pair_dic:
                    word, index = pair.replace('\n', '').split('\t')
                    word_index[word] = index

                index_word = {v: k for k, v in word_index.items()}

                del pair_dic

                emb_index = inp_dic.readlines()
                emb_vec = inp_vec.readlines()
                word_vectors = {}
                for index, vec in zip(emb_index, emb_vec):
                    word = index_word[index.replace('\n', '')]
                    word_vectors[word] = np.array(list((map(float, vec.split()))))

                del emb_index
                del emb_vec

                # message indicating the type words embedding process
                print('Word2Vec words embedding loaded...')

            return word_vectors
        
        elif word_emb == 'Jose':
            vec_file = data_path + '/jose/jose_sof_wv.txt'
            word_vectors, _, _ = get_wv_emb(vec_file=vec_file)
            
            # message indicating the type words process
            print('Jose words embedding loaded...')
            return word_vectors
        

    if data_path.endswith('Biomedical'):
        if word_emb == 'Word2Vec':
            mat = scipy.io.loadmat(data_path + '/Biomedical-STC2.mat')

            emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
            emb_vec = mat['vocab_emb_Word2vec_48']


            # load SO embedding
            with open(data_path + '/Biomedical_vocab2idx.dic', 'r') as inp_indx:
                # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
                # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
                pair_dic = inp_indx.readlines()
                word_index = {}
                for pair in pair_dic:
                    word, index = pair.replace('\n', '').split('\t')
                    word_index[word] = index

                index_word = {v: k for k, v in word_index.items()}

                del pair_dic

                word_vectors = {}
                for index, vec in zip(emb_index, emb_vec.T):
                    word = index_word[str(index)]
                    word_vectors[word] = vec

                del emb_index
                del emb_vec

            # message indicating the type words embedding process
            print('Word2Vec words embedding loaded...')
            return word_vectors


def embed_docs(data_path="datasets/stackoverflow", 
               transform: Union[Literal['SIF'], None] = None,
               word_emb: Literal['Word2Vec', 'Jose', 'Glove']='Word2Vec',
               decompose_type: Union[Literal['PCA', 'SVD'], None]='PCA'):
    # start embedding
    print('### Embedding started...')
    
    # data_path = cwd / 'datasets' / f'{dataset}'
    #data_path = cwd + 'datasets' + f'/{dataset}'
    word_vectors = get_word_vectors(data_path=data_path, word_emb=word_emb)

    if transform == 'SIF':
        if data_path.endswith('stackoverflow'):
            text_file = data_path + '/title_StackOverflow.txt'
            # text_file = data_path / 'title_StackOverflow.txt'
            #text_file = 'datasets/stackoverflow/title_StackOverflow.txt'
            XX = sif_emb(text_path=text_file, 
                         word_vectors=word_vectors,
                         decompose_type=decompose_type)
            # end embedding
            print('### Embedding completed...')

        elif data_path.endswith('Biomedical'):
            text_file = data_path + '/Biomedical.txt'
            # text_file = data_path / 'Biomedical.txt'
            #text_file = 'datasets/Biomedical/Biomedical.txt'
            XX = sif_emb(text_path=text_file, 
                         word_vectors=word_vectors,
                         decompose_type=decompose_type)
            # end embedding
            print('### Embedding completed...')
        return XX
    else:
        XX = get_avg_emb_from_word_vectors(dataset=data_path, 
                                           word_vectors=word_vectors)
        # end embedding
        print('### Embedding completed...')

        return XX
            


def sif_emb(text_path, 
            word_vectors: dict[str, np.array], 
            decompose_type: Literal['PCA', 'SVD']='PCA') -> np.ndarray:
    
    message = '#### SIF embedding started...'
    print(message)

    with open(text_path, 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        d = len(list(word_vectors.values())[0])

        # message indicating the embedding process
        print(f'SIF-Embedding {len(all_lines)} documents with {d}-dimensional word vectors...') 
         
        all_vector_representation = np.zeros(shape=(20000, d))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[d, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    if decompose_type == 'PCA':
        decompose = PCA(n_components=1)
    elif decompose_type == 'SVD':
        decompose = TruncatedSVD(n_components=1, n_iter=20)
    else:
        raise Exception('decompose type not found...')
    
    decompose.fit(all_vector_representation)
    component = decompose.components_

    XX1 = all_vector_representation - all_vector_representation.dot(component.transpose()) * component

    XX = XX1

    message = '### SIF embedding completed...'
    print(message)
    
    return XX

def load_stackoverflow(data_path='datasets/stackoverflow',
                       transform: Union[Literal['SIF'], None]=None,
                       scaler: Union[Literal['MinMax', 'Standard'], None]='MinMax',
                       norm: Union[Literal['l2', 'l1', 'max'], None]=None):

    doc_embedded = embed_docs(data_path=data_path, 
                              transform=transform, 
                              word_emb='Word2Vec',
                              decompose_type='PCA')
    y_true = read_label(dataset=data_path)

    XX = doc_embedded
    # print check if doc_embedded contains NaN
    if np.isnan(doc_embedded).any():
        print('doc_embedded contains NaN values...')

    # normalize data 
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)

            # message indicating the type normalization process
            print('Spherical normalization completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)

            # message indicating the type normalization process
            print('Spherical normalization completed...')

    return XX, y_true

def load_stackoverflow_jose(data_path='datasets/stackoverflow',
                            transform: Union[Literal['SIF'], None]=None,
                            scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                            norm: Union[Literal['l2', 'l1', 'max'], None]=None):

    doc_embedded = embed_docs(dataset=data_path+ '/jose',
                              transform=transform, 
                              word_emb='Jose',
                              decompose_type='PCA')
    y_true = read_label(data_dir=data_path)
    
    XX = doc_embedded
    # normalize data 
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)

    return XX, y_true


def load_search_snippet2(data_path='datasets/SearchSnippets/'):
    mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r') as inp_indx:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'SearchSnippets.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        all_lines = [line for line in all_lines]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(12340, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_

    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_biomedical(data_path='datasets/Biomedical',
                    transform: Union[Literal['SIF'], None]=None,
                    scaler: Union[Literal['MinMax', 'Standard'], None]='MinMax',
                    norm: Union[Literal['l2', 'l1', 'max'], None]=None):

    doc_embedded = embed_docs(data_path=data_path, 
                              transform=transform, 
                              word_emb='Word2Vec',
                              decompose_type='SVD')

    XX = doc_embedded
    mat = scipy.io.loadmat(data_path + '/Biomedical-STC2.mat')
    y = np.squeeze(mat['labels_All'])

    # print check if doc_embedded contains NaN
    if np.isnan(doc_embedded).any():
        print('doc_embedded contains NaN values...')

    # normalize data 
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)

            # message indicating the type normalization process
            print(f'{norm} normalization completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)

            # message indicating the type normalization process
            print(f'{norm} normalization completed...')

    return XX, y


def load_biomedical_jose(data_path='datasets/Biomedical',
                         transform: Union[Literal['SIF'], None]=None,
                         scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                         norm: Union[Literal['l2', 'l1', 'max'], None]=None):

    doc_embedded = embed_docs(dataset=data_path+ '/jose',
                              transform=transform, 
                              word_emb='Jose',
                              decompose_type='SVD')
    y_true = read_label(data_dir=data_path)
    
    XX = doc_embedded
    # normalize data 
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        if norm is not None:
            XX = normalize(XX, norm=norm)

    return XX, y_true


def load_biomedical_hf(data_path='datasets/Biomedical',
                       transform: Union[Literal['SIF'], None]=None,
                       scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                       norm: Union[Literal['l2', 'l1', 'max'], None]=None):
    
    output = pd.read_csv(data_path+'/HuggingFace/HF_doc_emb.csv')
    embeddings2 = output['embeddings2']
    embeddings_array = np.array(embeddings2)
    processed_data = [ast.literal_eval(embedding_str) for embedding_str in embeddings_array]

    if norm is not None:
        processed_data = normalize(processed_data, norm=norm)

    XX = np.array(processed_data)
    y_true = pd.read_excel(data_path+'/HuggingFace/y.xlsx', header=None)
    y_true = y_true.values.flatten()
    return XX, y_true



def load_data(dataset='datasets/stackoverflow', 
              word_emb: Union[Literal['Word2Vec', 'Jose', 'Glove'], None]=None,
              transform: Union[Literal['SIF'], None]=None,
              scaler: Union[Literal['MinMax', 'Standard'], None]=None,
              norm: Union[Literal['l2', 'l1', 'max'], None]=None):
    
    if dataset.endswith('stackoverflow'):
        if word_emb == "Word2Vec":
            return load_stackoverflow(data_path=dataset,
                                      transform=transform, 
                                      scaler=scaler, 
                                      norm=norm)
        elif word_emb == "Jose":
            return load_stackoverflow_jose(transform=transform, 
                                           scaler=scaler, 
                                           norm=norm)
        else:
            raise Exception('embedding type not found...')
        

    elif dataset.endswith('Biomedical'):
        if word_emb == "Word2Vec":
            return load_biomedical(data_path=dataset,
                                   transform=transform, 
                                   scaler=scaler, 
                                   norm=norm)
        elif word_emb == "Jose":
            return load_biomedical_jose(data_path=dataset,
                                        transform=transform, 
                                        scaler=scaler, 
                                        norm=norm)
        elif word_emb == "HuggingFace":
            return load_biomedical_hf(data_path=dataset,
                                      transform=transform, 
                                      scaler=scaler, 
                                      norm=norm)
        else:
            raise Exception('embedding type not found...')
        
    elif dataset == 'search_snippets':
        return load_search_snippet2()
    else:
        raise Exception('dataset not found...')
