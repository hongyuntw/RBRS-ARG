import re
import time
import numpy as np
import torch


def get_stopwords():
    fp = open('./stopwords-en.txt', 'r')
    stopwords = []
    for line in fp.readlines():
        line = line.strip()
        line = line.replace('\n', '')
        stopwords.append(line)
    fp.close()
    return stopwords
    

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
        if w in opt.vocab:
            review_ids.append(opt.word2index[w])


    if len(review_ids) > opt.r_max_len:
        review_ids = review_ids[:opt.r_max_len]
    else:
        review_ids += [0] * (opt.r_max_len - len(review_ids))
    return review_ids


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def unpack_input(opt, x):
    uids_batch , iids_batch = x
    uids_batch = list(uids_batch)
    iids_batch = list(iids_batch)


    user_reviews_batch = []
    user_item2id_batch = []
    user_doc_batch = []
    item_reviews_batch = []
    item_user2id_batch = []
    item_doc_batch = []

    for i in range(len(uids_batch)):
        uids = uids_batch[0]
        iids = iids_batch[0]
        

        # num_items * #reviews, review_max_length
        user_reviews = opt.users_review_list[uids] 
        # num_items * #reviews
        user_item2id = opt.user2itemid_list[uids]  
        # num_items * doc_max_length
        user_doc = opt.user_doc[uids]

        item_reviews = opt.items_review_list[iids]
        item_user2id = opt.item2userid_list[iids]  
        item_doc = opt.item_doc[iids]

        user_reviews_batch.append(user_reviews)
        user_item2id_batch.append(user_item2id)
        user_doc_batch.append(user_doc)
        item_reviews_batch.append(item_reviews)
        item_user2id_batch.append(item_user2id)
        item_doc_batch.append(item_doc)


    data = [user_reviews_batch, item_reviews_batch, uids_batch, iids_batch, user_item2id_batch, item_user2id_batch, user_doc_batch, item_doc_batch]
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
    return data


