import os 
from pathlib import Path
import numpy as np
import scipy.io

import nltk
nltk.download('punkt')  # adding new version nltk download

HERE = Path().cwd()
print(">>>>>", HERE)


def get_wv_emb(vec_file):
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word
    return word_emb, vocabulary, vocabulary_inv


def get_emb(vec_file):
    f = open(vec_file, 'r')
    tmp = f.readlines()
    contents = tmp[1:]
    doc_emb = np.zeros([int(x) for x in tmp[0].split(' ')])
    
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        doc_emb[i] = np.array(vec)

    return doc_emb

def get_avg_emb_from_word_vectors(dataset='datasets/stackoverflow',
                                  word_vectors: dict[str, np.array]=None): 

    #data_path = HERE / 'datasets' / f'{dataset}'
    data_path = dataset
    if "stackoverflow" in dataset.split("/"): 
        if dataset.endswith('jose'):
            # back to link without jose
            # and turn it to path with join
            data_path = "/".join(dataset.split("/")[:-1])
        text_file = data_path+"/title_StackOverflow.txt"
    elif "Biomedical" in dataset.split("/"): 
        if dataset.endswith('jose'):
            # back to link without jose
            # and turn it to path with join
            data_path = "/".join(dataset.split("/")[:-1])
        print(">>>>> agv_bio", data_path)
        text_file = data_path+"/Biomedical.txt"
    
    elif 'SearchSnippets' in dataset.split("/"):
        if dataset.endswith('jose'):
            # back to link without jose
            # and turn it to path with join
            data_path = "/".join(dataset.split("/")[:-1])
        text_file = data_path+"/SearchSnippets.txt"
    elif dataset == '20news':
        pass

    #print(">>>>>", text_file)

    # read the text file and tokenize it using nltk
    with open(text_file, 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])

        n_samples = len(all_lines) + 1
        dim_emb = len(word_vectors[list(word_vectors.keys())[0]])
        all_vector_representation = np.zeros((n_samples, dim_emb))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[dim_emb, ])
            #print(n_samples, dim_emb)
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                sent_rep = sent_rep + wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    message = 'Average embedding completed...'
    print(message)

    return all_vector_representation



def get_avg_emb_from_vec(vec_file, text):
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word

    f = open(text, 'r')
    contents = f.readlines()
    doc_emb = np.zeros((len(contents), len(word_emb[vocabulary_inv[0]])))
    for i, content in enumerate(contents):
        content = content.strip()
        doc = content.split(" ")
        emb = np.array([word_emb[w] for w in doc if w in word_emb])
        doc_emb[i] = np.average(emb, axis=0)

    return doc_emb

def read_label(dataset='datasets/stackoverflow'):
    #data_path = HERE / 'datasets' / f'{dataset}'
    if dataset.endswith('stackoverflow'):
        label_file = dataset + '/label_StackOverflow.txt'
        f = open(label_file, 'r')
        docs = f.readlines()
        y_true = np.array([int(doc.strip())-1 for doc in docs])

    elif dataset.endswith('Biomedical'):
        mat_file = dataset + '/Biomedical-STC2.mat'
        mat = scipy.io.loadmat(mat_file)
        y_true = np.squeeze(mat['labels_All'])
        del mat
    elif dataset == '20news':
        pass

    return y_true