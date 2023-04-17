# -*- coding: utf-8 -*-

import os
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from collections import OrderedDict
from random import shuffle
import pandas as pd
from tqdm import tqdm
import pickle


class ReviewData(Dataset):

    def __init__(self, root_path, mode):
        if mode == 'Train':
            path = os.path.join(root_path, 'train/')
            print('loading train data')
            self.data = np.load(path + 'Train.npy', encoding='bytes')
            self.scores = np.load(path + 'Train_Score.npy')
        elif mode == 'Val':
            path = os.path.join(root_path, 'val/')
            print('loading val data')
            self.data = np.load(path + 'Val.npy', encoding='bytes')
            self.scores = np.load(path + 'Val_Score.npy')
        else:
            path = os.path.join(root_path, 'test/')
            print('loading test data')
            self.data = np.load(path + 'Test.npy', encoding='bytes')
            self.scores = np.load(path + 'Test_Score.npy')
        self.x = list(zip(self.data, self.scores))

    
    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


class RLDataset(Dataset):
    def __init__(self, root_path, opt, data_mode='test'):
        self.data_mode = data_mode
        path = os.path.join(root_path, f'{data_mode}/')
        data = np.load(path + f'{data_mode.capitalize()}.npy', encoding='bytes')
        scores = np.load(path + f'{data_mode.capitalize()}_Score.npy')
        df_path = f'./dataset/{opt.dataset}/data.csv'
        df = pd.read_csv(df_path)

        self.item_dict_path = f'./dataset/{opt.dataset}/rl_{self.data_mode}_item_dict'
        if os.path.exists(self.item_dict_path):
            with open(self.item_dict_path, 'rb') as handle:
                self.item_dict = pickle.load(handle)
        else:
            self.item_dict = OrderedDict()
            self.build_instances(data, scores, df, opt)
        self.iids = list(self.item_dict.keys())

        self.max_users_per_batch = 10
        print(f'total {len(self.iids)} items')

    def build_instances(self, data, scores, df, opt):
        for i, d in enumerate(tqdm(data)):
            u_idx, i_idx = d
            score = scores[i]
            i_id = opt.index2item[i_idx]
            u_id = opt.index2user[u_idx]

            target_row = df.query(f'user_id=="{u_id}" and item_id=="{i_id}"')
            review = target_row.iloc[0]['reviews']
            rating = target_row.iloc[0]['ratings']
            if i_id in self.item_dict:
                self.item_dict[i_id].append([u_id, str(score), review])
            else:
                self.item_dict[i_id] = [[u_id, str(score), review]]

        with open(self.item_dict_path, 'wb') as handle:
            pickle.dump(self.item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        i_id = self.iids[idx]
        tmp = self.item_dict[i_id]

        if len(tmp) >= self.max_users_per_batch:
            tmp = tmp[:self.max_users_per_batch]
        else:
            for _ in range(self.max_users_per_batch - len(self.item_dict[i_id])):
                tmp.append(['PAD_USER', '0.0', 'PAD_REVIEW'])
        
        ratings = [x[1] for x in tmp]
        reviews = [x[2] for x in tmp]

        return i_id, ratings, reviews

    def __len__(self):
        return len(self.iids)



class AttackDataset(Dataset):

    def __init__(self, root_path, opt, attack_item_id=None, data_mode='test', data=None, scores=None):
        
        path = os.path.join(root_path, f'{data_mode}/')

        if data is None:
            data = np.load(path + f'{data_mode.capitalize()}.npy', encoding='bytes')

        if scores is None:
            scores = np.load(path + f'{data_mode.capitalize()}_Score.npy')

        self.attack_data = []
        self.attack_score = []
        for i, d in enumerate(data):
            uid, iid = d

            if int(iid) == attack_item_id:
                self.attack_data.append(data[i])
                self.attack_score.append(scores[i]) 

        self.x = list(zip(self.attack_data, self.attack_score))

    
    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


