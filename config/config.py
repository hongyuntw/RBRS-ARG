# -*- coding: utf-8 -*-

import numpy as np
import pickle
import gensim

"""
important data:
- userNum
- itemNum
- r_max_len

"""

PRE_W2V_BIN_PATH = "./GoogleNews-vectors-negative300.bin"  # the pre-trained word2vec files

class DefaultConfig:

    model = 'DeepCoNN'
    dataset = 'Digital_Music_data'

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []

    seed = 2019
    num_epochs = 40
    num_workers = 0

    optimizer = 'Adam'
    weight_decay = 1e-3  # optimizer rameteri
    lr = 2e-3
    loss_method = 'mse'
    drop_out = 0.5

    use_word_embedding = True

    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 32

    doc_len = 1000
    filters_num = 100
    kernel_size = 3

    num_fea = 1  # id feature, review feature, doc feature
    use_review = True
    use_doc = True
    self_att = False

    r_id_merge = 'cat'  # review and ID feature
    ui_merge = 'cat'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    fine_step = False  # save mode in step level, defualt in epoch
    pth_path = ""  # the saved pth path for test
    print_opt = 'default'

    train_negN = 4
    val_negN = 49
    topk = 10

    data_mode = "train"
    exp_name = ''
    used_epoch = ''


    ### for ABAE ###
    abae_lr = 1e-3
    abae_n_aspect = 15
    abae_num_neg_sent = 20
    abae_lr_lambda = 1
    abae_batch_size = 50
    abae_train_epochs = 20

    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.w2v_path = f'{prefix}/w2v.npy'

        self.word2index_path = f'{self.data_root}/word2index'
        self.index2word_path = f'{self.data_root}/index2word'
        self.vocab_path = f'{self.data_root}/vocab'

        self.index2user_path = f'{self.data_root}/index2user'
        self.index2item_path = f'{self.data_root}/index2item'
        self.item2index_path = f'{self.data_root}/item2index'
        self.user2index_path = f'{self.data_root}/user2index'


    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        with open(self.word2index_path, 'rb') as handle:
            self.word2index = pickle.load(handle)

        with open(self.index2word_path, 'rb') as handle:
            self.index2word = pickle.load(handle)

        with open(self.vocab_path, 'rb') as handle:
            self.vocab = pickle.load(handle)
            self.vocab['<unk>'] = 0


        with open(self.user2index_path, 'rb') as handle:
            self.user2index = pickle.load(handle)

        with open(self.item2index_path, 'rb') as handle:
            self.item2index = pickle.load(handle)

        with open(self.index2item_path, 'rb') as handle:
            self.index2item = pickle.load(handle)

        with open(self.index2user_path, 'rb') as handle:
            self.index2user = pickle.load(handle)

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')

    def load_w2v_model(self):
        binary = 'bin' in PRE_W2V_BIN_PATH
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH, binary=binary)



    

class Digital_Music_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Digital_Music_data')

    dataset = 'Digital_Music_data'
    vocab_size = 50002
    word_dim = 300

    r_max_len = 202

    u_max_r = 13
    i_max_r = 24

    train_data_size = 51764
    test_data_size = 6471
    val_data_size = 6471

    user_num = 5541 + 2
    item_num = 3568 + 2

    batch_size = 4
    print_step = 100

    

class Video_Games_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Video_Games_data')


    dataset = 'Video_Games_data'
    vocab_size = 50002
    word_dim = 300

    r_max_len = 169

    u_max_r = 9
    i_max_r = 34

    train_data_size = 379264
    val_data_size = 47348
    test_data_size = 47348

    user_num = 55217 + 2
    item_num = 17408 + 2

    batch_size = 128
    print_step = 100


    train_negN = 4
    val_negN = 99



class Musical_Instruments_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Musical_Instruments_data')

    dataset = 'Musical_Instruments_data'
    vocab_size = 50002
    word_dim = 300

    r_max_len = 76

    u_max_r = 9
    i_max_r = 23

    train_data_size = 177618
    val_data_size = 22175
    test_data_size = 22175

    user_num = 27528 + 2
    item_num = 10620 + 2

    batch_size = 4
    print_step = 100


    train_negN = 4
    val_negN = 99


class Office_Products_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Office_Products_data')

    dataset = 'Office_Products_data'
    vocab_size = 50002
    word_dim = 300

    r_max_len = 63

    u_max_r = 245
    i_max_r = 1783

    train_data_size = 599646
    val_data_size = 74816
    test_data_size = 74815

    user_num = 101498 + 2
    item_num = 27965 + 2

    batch_size = 4
    print_step = 100


    train_negN = 4
    val_negN = 99