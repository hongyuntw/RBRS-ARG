import torch

import numpy as np 
from collections import defaultdict
import re
import torch.utils.data as data
import config_abae as conf
import pandas as pd
from copy import deepcopy
# from utils import review2ids, clean_str
from utils import get_stopwords

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

PAD = 0


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



def train_abae_w2v(opt):
    dataset = opt.dataset
    all_df = pd.read_csv(f'./dataset/{dataset}/data.csv')
    all_df = all_df[all_df['reviews'].notna()]
    all_df['clean_reviews'] = all_df['reviews'].apply(clean_str)
    all_df = all_df[all_df['clean_reviews'].notna()]
    reviews = all_df['clean_reviews'].tolist()

    print(len(reviews))

    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    print('nltk stop ' ,len(stop))

    print('----remove stop words ----')
    corpus = []
    for review in reviews:
        review = review.split()
        tmp = []
        for token in review:
            if len(token) > 2 and token not in stop:
                tmp.append(token)
        if len(tmp) >= 10:
            corpus.append(tmp)
    print(len(corpus))

    from gensim.models import Word2Vec
    w2v_model = Word2Vec(min_count = 10,
                         window=10,
                         vector_size=300,
                         negative=5,
                         workers=4)
    print('---- build vocab ----')
    w2v_model.build_vocab(corpus, progress_per=10000)
    print('--- training word2vec ----')
    w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=20, report_delay=1)
    w2v_model.save(f'aspect/data/word2vec_{dataset}.model')
    return w2v_model
    


def review2ids(opt, w2v, review):
    review = clean_str(review)
    review_ids = []
    for w in review.strip().split():
        if w in w2v.wv.key_to_index:
            review_ids.append(w2v.wv.key_to_index[w])

    if len(review_ids) > opt.r_max_len:
        review_ids = review_ids[:opt.r_max_len]
    else:
        review_ids += [0] * (opt.r_max_len - len(review_ids))
    return review_ids




def load_all(opt, w2v):
    dataset = opt.dataset
    data_train = pd.read_csv(f'./dataset/{dataset}/data_train.csv')
    data_val = pd.read_csv(f'./dataset/{dataset}/data_val.csv')
    data_test = pd.read_csv(f'./dataset/{dataset}/data_test.csv')

    train_reviews = data_train['reviews'].tolist()
    val_reviews = data_val['reviews'].tolist()
    test_reviews = data_test['reviews'].tolist()

    sent_id = 0
    train_data = {}
    for review in train_reviews:
        try:
            ids = review2ids(opt, w2v, clean_str(review))
        except:
            print(f'error review : {review}')
            continue
        train_data[sent_id] = ids
        sent_id += 1

    sent_id = 0
    val_data = {}
    for review in val_reviews:
        try:
            ids = review2ids(opt, w2v, clean_str(review))
        except:
            print(f'error review : {review}')
            continue
        val_data[sent_id] = ids
        sent_id += 1


    sent_id = 0
    test_data = {}
    for review in test_reviews:
        try:
            ids = review2ids(opt, w2v, clean_str(review))
        except:
            print(f'error review : {review}')
            continue
        test_data[sent_id] = ids
        sent_id += 1
    return train_data, val_data, test_data

