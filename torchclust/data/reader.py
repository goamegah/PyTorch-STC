import numpy as np
import os 


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

def get_avg_emb(vec_file, text):
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

def read_label(data_dir):
    f = open(os.path.join(data_dir, 'label_StackOverflow.txt'))
    docs = f.readlines()
    y_true = np.array([int(doc.strip())-1 for doc in docs])
    return y_true