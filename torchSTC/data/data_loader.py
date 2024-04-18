# -*- coding: utf-8 -*-

from typing import Literal, Union
import os

from collections import Counter

import ast

import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD, NMF, KernelPCA
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler

from .utils import read_label, get_wv_emb, get_avg_emb_from_word_vectors

cwd = os.getcwd()
print(">>>>> data_loader.py cwd: ", cwd)

def get_word_vectors(data_path="datasets/stackoverflow", 
                     word_emb='Word2Vec'):
    
    if "stackoverflow" in data_path.split("/"):
        # print("Test sof w2v bis ==> ==> ==> ")
        # if word_emb == 'Word2Vec':
        #     vec_file = data_path + '/w2v_sof_d48.txt'
        #     word_vectors, _, _ = get_wv_emb(vec_file=vec_file)
        if word_emb == 'Word2Vec':

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

                print('Word2Vec words embedding loaded...')

            return word_vectors
        
        elif word_emb == 'Jose':
            if data_path.endswith('jose'):
                data_path = "/".join(data_path.split("/")[:-1])
            vec_file = data_path + '/jose/jose_sof_wv_d48.txt'
            word_vectors, _, _ = get_wv_emb(vec_file=vec_file)
            
            print('Jose words embedding loaded...')
            return word_vectors
        

    elif "Biomedical" in data_path.split("/"):
        print("Test bio w2v bis ==> ==> ==> ")
        if word_emb == 'Word2Vec':
            vec_file = data_path + '/w2v_bio_d48.txt'
            word_vectors, _, _ = get_wv_emb(vec_file=vec_file)
        # if word_emb == 'Word2Vec':
        #     mat = scipy.io.loadmat(data_path + '/Biomedical-STC2.mat')

        #     emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
        #     emb_vec = mat['vocab_emb_Word2vec_48']


        #     # load SO embedding
        #     with open(data_path + '/Biomedical_vocab2idx.dic', 'r') as inp_indx:
        #         # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
        #         # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        #         pair_dic = inp_indx.readlines()
        #         word_index = {}
        #         for pair in pair_dic:
        #             word, index = pair.replace('\n', '').split('\t')
        #             word_index[word] = index

        #         index_word = {v: k for k, v in word_index.items()}

        #         del pair_dic

        #         word_vectors = {}
        #         for index, vec in zip(emb_index, emb_vec.T):
        #             word = index_word[str(index)]
        #             word_vectors[word] = vec

        #         del emb_index
        #         del emb_vec

            # message indicating the type words embedding process
            print('Word2Vec words embedding loaded...')
            return word_vectors
        elif word_emb == 'Jose':
            if data_path.endswith('jose'):
                data_path = "/".join(data_path.split("/")[:-1])
            vec_file = data_path + '/jose/jose_bio_wv_d48.txt'
            word_vectors, _, _ = get_wv_emb(vec_file=vec_file)
            
            print('Jose words embedding loaded...')
            return word_vectors
    
    elif "SearchSnippets" in data_path.split("/"):
        # print("Test Sst w2v bis ==> ==> ==> ")
        # if word_emb == 'Word2Vec':
        #     vec_file = data_path + '/w2v_sst_d300.txt'
        #     word_vectors, _, _ = get_wv_emb(vec_file=vec_file)
        if word_emb == 'Word2Vec':
            mat = scipy.io.loadmat(data_path + '/SearchSnippets-STC2.mat')

            emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
            emb_vec = mat['vocab_emb_Word2vec_48']

            # load SO embedding
            with open(data_path + '/SearchSnippets_vocab2idx.dic', 'r') as inp_indx:
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

            print('Word2Vec words embedding loaded...')
            return word_vectors
        

def embed_docs(data_path="datasets/stackoverflow", 
               transform: Union[Literal['SIF'], None] = None,
               word_emb: Literal['Word2Vec', 'Jose', 'Glove']='Word2Vec',
               decompose_type: Literal['PCA', 'SVD', 'NMF', 'KPCA']='PCA'):
    
    print('### Embedding started...')
    word_vectors = get_word_vectors(data_path=data_path, word_emb=word_emb)

    XX = None
    if transform == 'SIF':
        if "stackoverflow" in data_path.split("/"): 
            if data_path.endswith('jose'):
                data_path = "/".join(data_path.split("/")[:-1])
            text_file = data_path + '/title_StackOverflow.txt'
            XX = sif_emb(text_path=text_file, 
                         word_vectors=word_vectors,
                         decompose_type=decompose_type)
            print('### Embedding completed...')

        elif "Biomedical" in data_path.split("/"):
            if data_path.endswith('jose'):
                data_path = "/".join(data_path.split("/")[:-1])
            text_file = data_path + '/Biomedical.txt'
            XX = sif_emb(text_path=text_file, 
                         word_vectors=word_vectors,
                         decompose_type=decompose_type)
            print('### Embedding completed...')

        elif "SearchSnippets" in data_path.split("/"):
            if data_path.endswith('jose'):
                data_path = "/".join(data_path.split("/")[:-1])
            text_file = data_path + '/SearchSnippets.txt'
            XX = sif_emb(text_path=text_file, 
                         word_vectors=word_vectors,
                         decompose_type=decompose_type)
            print('### Embedding completed...')

        print('[embed_docs] XX shape: ', XX.shape)

        return XX
    
    else:
        XX = get_avg_emb_from_word_vectors(dataset=data_path, 
                                           word_vectors=word_vectors)
        print('### Embedding completed...')

        return XX
            


def sif_emb(text_path, 
            word_vectors: dict, 
            decompose_type: Literal['PCA', 'SVD', 'NMF', 'KPCA']='PCA') -> np.ndarray:
    
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

        n_samples = len(all_lines) + 1
        dim_emb = len(word_vectors[list(word_vectors.keys())[0]])

        print(f'SIF-Embedding {len(all_lines)} documents with {dim_emb}-dimensional word vectors...') 
        
        all_vector_representation = np.zeros(shape=(n_samples, dim_emb))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[dim_emb, ])
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
        print('PCA decomposition...')
    elif decompose_type == 'SVD':
        decompose = TruncatedSVD(n_components=1, n_iter=20)
        print('SVD decomposition...')
    elif decompose_type == 'NMF':
        decompose = NMF(n_components=1)
        print('NMF decomposition...')
    elif decompose_type == 'KPCA':
        decompose = KernelPCA(n_components=1, kernel='cosine')
        print('KPCA decomposition...')
    else:
        raise Exception('decompose type not found...')
    
    decompose.fit(all_vector_representation)
    if decompose_type == 'KPCA':
        component = decompose.components_
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

    if np.isnan(doc_embedded).any():
        print('doc_embedded contains NaN values...')

    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')

    if norm is not None:
        XX = normalize(XX, norm=norm)
        print(f'{norm} normalization completed...')
    else:
        print('No normalization applied...')

    return XX, y_true


def load_stackoverflow_jose(data_path='datasets/stackoverflow',
                            transform: Union[Literal['SIF'], None]=None,
                            scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                            norm: Union[Literal['l2', 'l1', 'max'], None]=None):

    doc_embedded = embed_docs(data_path=data_path+ '/jose',
                              transform=transform, 
                              word_emb='Jose',
                              decompose_type='PCA')
    print(">>>", doc_embedded.shape)
    y_true = read_label(dataset=data_path)
    
    XX = doc_embedded
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')
        
    if norm is not None:
        XX = normalize(XX, norm=norm)
        
        print(f'{norm} normalization completed...')
    else:
        print('No normalization applied...')

    return XX, y_true

def load_stackoverflow_hf(data_path='datasets/stackoverflow',
                          transform: Union[Literal['SIF'], None]=None,
                          scaler: Union[Literal['MinMax', 'Standard'], None]='MinMax',
                          norm: Union[Literal['l2', 'l1', 'max'], None]=None):
    
    
    file = pd.read_excel(data_path+'/HuggingFace/output_stack.xlsx')
    embeddings2 = file['embeddings2']
    embeddings_array = np.array(embeddings2)
    processed_data = [ast.literal_eval(embedding_str) for embedding_str in embeddings_array]

    if norm is not None:
        processed_data = normalize(processed_data, norm=norm)
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        processed_data = scaler.fit_transform(processed_data)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(processed_data)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')
    

    x = np.array(processed_data)
    
    file = pd.read_excel(data_path+'/HuggingFace/y.xlsx', header=None)
    y = file.values.flatten()

    return x, y
    
def load_search_snippet2(data_path='datasets/SearchSnippets',
                         transform: Union[Literal['SIF'], None]=None,
                         scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                         norm: Union[Literal['l2', 'l1', 'max'], None]=None):
    
    doc_embedded = embed_docs(data_path=data_path, 
                              transform=transform, 
                              word_emb='Word2Vec',
                              decompose_type='SVD')
    
    print(">>> | ", doc_embedded.shape)

    XX = doc_embedded
    mat = scipy.io.loadmat(data_path + '/SearchSnippets-STC2.mat')
    y = np.squeeze(mat['labels_All'])
    del mat
    
    # check if doc_embedded contains NaN
    if np.isnan(doc_embedded).any():
        print('doc_embedded contains NaN values...')

    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')
        
    if norm is not None:
        XX = normalize(XX, norm=norm)
        
        print(f'{norm} normalization completed...')
    else:
        print('No normalization applied...')

    return XX, y

def load_search_snippet2_hf(data_path='datasets/SearchSnippets',
                            transform: Union[Literal['SIF', None]]=None,
                            scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                            norm: Union[Literal['l2', 'l1', 'max'], None]=None):
    
    output = pd.read_excel(data_path+'/HuggingFace/output_snippet.xlsx')
    embeddings2 = output['embeddings2']
    embeddings_array = np.array(embeddings2)
    processed_data = [ast.literal_eval(embedding_str) for embedding_str in embeddings_array]

    if norm is not None:
        processed_data = normalize(processed_data, norm=norm)
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        processed_data = scaler.fit_transform(processed_data)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(processed_data)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')
    
    x = np.array(processed_data)

    y = pd.read_excel(data_path+'/HuggingFace/y.xlsx', header = None)
    y = y.values.flatten()

    return x, y

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

    if np.isnan(doc_embedded).any():
        print('doc_embedded contains NaN values...')

    # normalize data 
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')
        
    if norm is not None:
        XX = normalize(XX, norm=norm)
        
        print(f'{norm} normalization completed...')
    else:
        print('No normalization applied...')

    return XX, y


def load_biomedical_jose(data_path='datasets/Biomedical',
                         transform: Union[Literal['SIF'], None]=None,
                         scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                         norm: Union[Literal['l2', 'l1', 'max'], None]=None):

    doc_embedded = embed_docs(data_path=data_path+ '/jose',
                              transform=transform, 
                              word_emb='Jose',
                              decompose_type='NMF')
    print(">>>>>>", data_path)
    y_true = read_label(dataset=data_path)
    
    XX = doc_embedded
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        XX = scaler.fit_transform(doc_embedded)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')
        
    if norm is not None:
        XX = normalize(XX, norm=norm)
      
        print(f'{norm} normalization completed...')
    else:
        print('No normalization applied...')

    return XX, y_true


def load_biomedical_hf(data_path='datasets/Biomedical',
                       transform: Union[Literal['SIF'], None]=None,
                       scaler: Union[Literal['MinMax', 'Standard'], None]=None,
                       norm: Union[Literal['l2', 'l1', 'max'], None]=None):
    
    
    output = pd.read_excel(data_path+'/HuggingFace/output_biomed.xlsx')
    embeddings2 = output['embeddings2']
    embeddings_array = np.array(embeddings2)
    processed_data = [ast.literal_eval(embedding_str) for embedding_str in embeddings_array]

    if norm is not None:
        processed_data = normalize(processed_data, norm=norm)
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        processed_data = scaler.fit_transform(processed_data)
        print('MinMax scaling completed...')
    elif scaler == 'Standard':
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(processed_data)
        print('Standard scaling completed...')
    else:
        print('No scaling applied...')
    

    x = np.array(processed_data)
    
    y = pd.read_excel(data_path+'/HuggingFace/y.xlsx', header=None)
    y = y.values.flatten()
    return x, y



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
            return load_stackoverflow_jose(data_path=dataset,
                                           transform=transform, 
                                           scaler=scaler, 
                                           norm=norm)
        elif word_emb == "HuggingFace":
            return load_stackoverflow_hf(data_path=dataset,
                                        transform=transform, 
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
        
    elif dataset.endswith('SearchSnippets'):
        if word_emb == "Word2Vec":
            return load_search_snippet2(data_path=dataset,
                                        transform=transform, 
                                        scaler=scaler, 
                                        norm=norm)
        elif word_emb == "HuggingFace":
            return load_search_snippet2_hf(data_path=dataset,
                                          transform=transform, 
                                          scaler=scaler, 
                                          norm=norm)
    else:
        raise Exception('dataset not found...')
