
import os, sys, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import config
import numpy as np
import pandas as pd
from collections import defaultdict
from time import time
from copy import deepcopy
from gensim.models import Word2Vec
from ABAE import ABAE

import data_abae as data_utils
import config_abae as conf

from Logging import Logging
def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()

def review2ids(opt, review):
    review = clean_str(review)
    review_ids = []
    for w in review.strip().split():
        if len(w) > 2 and w in opt.vocab:
            review_ids.append(opt.word2index[w])


    if len(review_ids) > opt.r_max_len:
        review_ids = review_ids[:opt.r_max_len]
    else:
        review_ids += [0] * (opt.r_max_len - len(review_ids))
    return review_ids

if __name__ == '__main__':
    
    dataset = sys.argv[2]
    opt = getattr(config, dataset + '_Config')()
    opt.parse({})
    n_aspect = 15

    model_path = f'aspect/checkpoints/model_{dataset}_{n_aspect}'
    aspect_params = torch.load(model_path)

    c = aspect_params['transform_T.weight'].transpose(0, 1) # (aspect_dimesion, word_dimension)
    x = aspect_params['word_embedding.weight'] # (num_words, word_dimension)

    x_i = F.normalize(x[:, None, :], p=2, dim=2) # (num_words, 1, word_dimension)
    c_j = F.normalize(c[None, :, :], p=2, dim=2) # (1, aspect_dimesion, word_dimension)
    
    D_ij = torch.transpose((x_i * c_j).sum(-1), 0, 1) # (aspect_dimesion, num_words)

    print('aspect, vocab matrix shape : ', D_ij.size())

    K = 100
    values, indices = torch.topk(D_ij, K) # (aspect_dimesion, K)

    from gensim.models import Word2Vec
    w2v = Word2Vec.load(f'aspect/data/word2vec_{dataset}.model')

    aspect_indices = []
    aspect_words = []
    for idx, word_idx_list in enumerate(indices):
        aspect_word_list = 'aspect_%d: ' % (idx+1)
        words = []
        for word_idx in word_idx_list:
            word = w2v.wv.index_to_key[word_idx.item()]
            word = str(word).lower()
            aspect_word_list += f'{word}, '  
            words.append(word)
        aspect_indices.append(idx)
        aspect_words.append(words)
        print(aspect_word_list)

    df = pd.DataFrame({
        'aspect_index':aspect_indices, 
        'aspect_words':aspect_words, 
        })
    df.to_csv(f'aspect/data/{dataset}_{n_aspect}.csv', index=None)
    print(f'{dataset}_{n_aspect}')
