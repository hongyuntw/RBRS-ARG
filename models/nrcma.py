# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
torch.set_printoptions(precision=4)


class NRCMA(nn.Module):
    '''
    NRCMA: CIKM 2021
    '''
    def __init__(self, opt):
        super(NRCMA, self).__init__()
        self.opt = opt
        self.num_fea = 2  # ID + Review
        attention_dim = 80

        seq_of_review = opt.r_max_len


        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.item_id_embedding = nn.Embedding(self.opt.item_num, self.opt.id_emb_size)
        self.user_id_embedding = nn.Embedding(self.opt.user_num, self.opt.id_emb_size)
        self.Bk_linear_user = nn.Linear(self.opt.id_emb_size, attention_dim)
        self.Bu_linear_user = nn.Linear(self.opt.id_emb_size, attention_dim)
        self.transform_B1_user = nn.Linear(attention_dim, opt.filters_num, bias=False) # weight: word_dimension * word_dimension
        self.transform_B2_user = nn.Linear(attention_dim, seq_of_review, bias=False) # weight: word_dimension * word_dimension


        self.Bk_linear_item = nn.Linear(self.opt.id_emb_size, attention_dim)
        self.Bu_linear_item = nn.Linear(self.opt.id_emb_size, attention_dim)
        self.transform_B1_item = nn.Linear(attention_dim, opt.filters_num, bias=False) # weight: word_dimension * word_dimension
        self.transform_B2_item = nn.Linear(attention_dim, seq_of_review, bias=False) # weight: word_dimension * word_dimension


        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        
        self.user_dropout = nn.Dropout(0.0)
        self.item_dropout = nn.Dropout(0.0)

        self.reset_para()

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, _, _, user_doc, item_doc = datas

        ## For User reviews
        user_reviews =  self.word_embs(user_reviews)
        bs, r_num, r_len, wd = user_reviews.size()
        user_reviews = user_reviews.view(-1, r_len, wd)
        user_reviews = F.relu(self.user_cnn(user_reviews.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        user_reviews = self.user_dropout(user_reviews)
        user_reviews = user_reviews.view(-1, r_num, user_reviews.size(1), user_reviews.size(2))
        user_reviews = user_reviews.permute(0, 1, 3, 2) # B_size, num_of_review, seq_len, num_filter


        item_embedding = self.item_id_embedding(iids)
        Bk = F.relu(self.Bk_linear_user(item_embedding)) # batch_size, attention_dim
        bc = self.transform_B1_user(Bk) # batch_size, num_filter
        bc = bc.unsqueeze(1).unsqueeze(1)
        bc = torch.matmul(bc, user_reviews.permute(0, 1, 3, 2))


        ac = F.softmax(bc, -1)
        duk = ac.permute(0, 1, 3, 2) * user_reviews
        duk = duk.sum(-1)


        Bu = F.relu(self.Bu_linear_user(item_embedding)) # batch_size, attention_dim
        bk = self.transform_B2_user(Bu)
        bk = bk.unsqueeze(1)
        bk = torch.matmul(bk, duk.permute(0, 2, 1))


        ak = F.softmax(bk, -1)
        du = ak.permute(0, 2, 1) * duk
        u_fea = du.sum(-1)

        ## For item reviews
        item_reviews =  self.word_embs(item_reviews)
        bs, r_num, r_len, wd = item_reviews.size()
        item_reviews = item_reviews.view(-1, r_len, wd)
        item_reviews = F.relu(self.item_cnn(item_reviews.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        item_reviews = self.item_dropout(item_reviews)
        item_reviews = item_reviews.view(-1, r_num, item_reviews.size(1), item_reviews.size(2))
        item_reviews = item_reviews.permute(0, 1, 3, 2) # B_size, num_of_review, seq_len, num_filter


        user_embedding = self.user_id_embedding(uids)
        Bk_item = F.relu(self.Bk_linear_item(user_embedding)) # batch_size, attention_dim
        bc_item = self.transform_B1_item(Bk_item) # batch_size, num_filter
        bc_item = bc_item.unsqueeze(1).unsqueeze(1)
        bc_item = torch.matmul(bc_item, item_reviews.permute(0, 1, 3, 2))


        ac_item = F.softmax(bc_item, -1)
        duk_item = ac_item.permute(0, 1, 3, 2) * item_reviews
        duk_item = duk_item.sum(-1)


        Bu_item = F.relu(self.Bu_linear_item(user_embedding)) # batch_size, attention_dim
        bk_item = self.transform_B2_item(Bu_item)
        bk_item = bk_item.unsqueeze(1)
        bk_item = torch.matmul(bk_item, duk_item.permute(0, 2, 1))


        ak_item = F.softmax(bk_item, -1)
        du_item = ak_item.permute(0, 2, 1) * duk_item
        i_fea = du_item.sum(-1)

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)


    def reset_para(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)

