import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans

import os
import sys
import inspect
from ABAE import ABAE
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import config

import data_abae as data_utils




def get_centroids(w2v, aspects_count):
    """
        Clustering all word vectors with K-means and returning L2-normalizes
        cluster centroids; used for ABAE aspects matrix initialization
    """

    km = MiniBatchKMeans(n_clusters=aspects_count, verbose=0, n_init=100)

    print('run kmeans')
    vectors = np.asarray(w2v.wv.vectors)
    print(vectors.shape)
    m = np.matrix(vectors)
    print('matrix', m.shape)
    km.fit(m)
    clusters = km.cluster_centers_

    # L2 normalization
    norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)

    return norm_aspect_matrix

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == '__main__':

    ## GET config
    import sys
    dataset = sys.argv[2]
    opt = getattr(config, dataset + '_Config')()
    opt.parse({})
    # opt.load_w2v_model()

    train_w2v = True
    if train_w2v:
        w2v = data_utils.train_abae_w2v(opt)
    else:
        from gensim.models import Word2Vec
        w2v = Word2Vec.load(f'aspect/data/word2vec_{dataset}.model')

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all(opt, w2v)
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## CREATE MODEL ##############################
    # n_aspect = opt.abae_n_aspect
    n_aspect = 15
    vocab_size = len(w2v.wv.index_to_key)
    print('vocab size : ', vocab_size)
    re_kmean = True
    # vocab_size = len(opt.word2vec)
    model = ABAE(opt, vocab_size=vocab_size, n_aspect=n_aspect)

    
    model_params = model.state_dict()

    ### using trained embeddings on custom dataset 
    for index, word in enumerate(w2v.wv.index_to_key):
        model_params['word_embedding.weight'][index] = torch.FloatTensor(w2v.wv[word])


    k_means_path = f'aspect/data/{dataset}_kmeans_{n_aspect}.npy'
    if re_kmean:
        k_means_weight = get_centroids(w2v, n_aspect)
        np.save(k_means_path, k_means_weight)
    else:
        try:
            k_means_weight = np.load(k_means_path) # (aspect_dimesion, word_dimension)
        except:
            k_means_weight = get_centroids(w2v, n_aspect)
            np.save(k_means_path, k_means_weight)
    # gensim                             3.8.3 -> 4.0.0

    model_params['transform_T.weight'] = torch.FloatTensor(k_means_weight.transpose()) # (word_dim,  asp_dim)
    
    model.load_state_dict(model_params)
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.abae_lr)

    ########################### FIRST TRAINING #####################################
  
    train_model_path = f'aspect/checkpoints/model_{dataset}_{n_aspect}'
    print(train_model_path)
    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data)
    val_dataset = data_utils.TrainData(val_data)
    test_dataset = data_utils.TrainData(test_data)

    train_batch_sampler = data.BatchSampler(data.RandomSampler(range(train_dataset.length)), batch_size=opt.abae_batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.RandomSampler(range(val_dataset.length)), batch_size=opt.abae_batch_size, drop_last=False)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(range(test_dataset.length)), batch_size=opt.abae_batch_size, drop_last=False)

    # Start Training !!!
    min_loss = 0
    for epoch in range(1, opt.abae_train_epochs+1):
        t0 = time()
        model.train()

        train_loss = []
        for batch_idx_list in train_batch_sampler:
            pos_sent, neg_sent = train_dataset.get_batch(batch_idx_list)
            c1, c2, out_loss, obj = model(pos_sent, neg_sent)
            '''
            print(torch.mean(c1))
            print(torch.mean(c2))
            '''
            #print(torch.mean(out_loss))
            train_loss.extend(tensorToScalar(out_loss))
            model.zero_grad(); obj.backward(); optimizer.step()
        t1 = time()

        # evaluate the performance of the model with following code
        model.eval()
        
        val_loss = []
        for batch_idx_list in val_batch_sampler:
            pos_sent, neg_sent = val_dataset.get_batch(batch_idx_list)
            _, _, out_loss, obj = model(pos_sent, neg_sent)
            val_loss.extend(tensorToScalar(out_loss))
        t2 = time()

        test_loss = []
        for batch_idx_list in test_batch_sampler:
            pos_sent, neg_sent = test_dataset.get_batch(batch_idx_list)
            _, _, out_loss, obj = model(pos_sent, neg_sent) 
            test_loss.extend(tensorToScalar(out_loss))
        t3 = time()

        train_loss, val_loss, test_loss = np.mean(train_loss), np.mean(val_loss), np.mean(test_loss)

        if epoch == 1:
            min_loss = val_loss
        if val_loss <= min_loss:
            torch.save(model.state_dict(), train_model_path)
        min_loss = min(val_loss, min_loss)

        # log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        # log.record('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_loss, val_loss, test_loss))

        print('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        print('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_loss, val_loss, test_loss))