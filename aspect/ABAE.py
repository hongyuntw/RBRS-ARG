import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from gensim.models import Word2Vec

# import config_abae as conf 

PAD = 0; SOS = 1; EOS = 2

margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='none')
mse_loss = nn.MSELoss(reduction='sum')

class ABAE(nn.Module):
    def __init__(self, opt, vocab_size, n_aspect=15):
        super(ABAE, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, opt.word_dim) 
        self.word_embedding.weight.requires_grad = False
        
        self.transform_M = nn.Linear(opt.word_dim, opt.word_dim, bias=False) # weight: word_dimension * word_dimension
        self.transform_W = nn.Linear(opt.word_dim, n_aspect) # weight: aspect_dimension * word_diension
        self.transform_T = nn.Linear(n_aspect, opt.word_dim, bias=False) # weight: word_dimension * aspect_dimension



        self.num_neg_sent = opt.abae_num_neg_sent
        self.seq_len = opt.r_max_len
        self.word_dim = opt.word_dim
        self.n_aspect = n_aspect
        self.lr_lambda = opt.abae_lr_lambda
        self.reinit()
    
    def reinit(self):
        nn.init.zeros_(self.transform_W.bias)

        nn.init.xavier_uniform_(self.transform_M.weight)
        nn.init.xavier_uniform_(self.transform_W.weight)
        nn.init.xavier_uniform_(self.transform_T.weight)

    # e_w: (batch_size, sequence_length, word_dimension)
    # Y_s: (batch_size, word_dimension)
    # pos_rev: (batch_size, sequence_length)
    def _attention(self, pos_rev, e_w, y_s):
        mask = (pos_rev > 0).long() # (batch_size, seq_len)

        # self.transform_M(e_w): (batch_size, sequence_length, word_dimension)
        dx = torch.matmul(self.transform_M(e_w), y_s).view(-1, self.seq_len) # (batch_size, sequence_length)
        dx_mask = (dx < 80).long()
        dx = dx_mask * dx

        ax_1 = torch.exp(dx) # (batch_size, seq_len)
        ax_2 = ax_1 * mask # (batch_size, seq_len)
        ax_3 = torch.sum(ax_2, dim=1, keepdim=True) + 1e-6 # (batch_size, 1)
        ax_4 = (ax_2 / ax_3).view(-1, self.seq_len, 1) # (batch_size, seq_len, 1)
        
        # e_w.transpose(1, 2): (batch_size, word_dimension, sequence_length)
        # torch.matmul(e_w, a): (batch_size, word_dimension, 1)
        z_s = torch.sum(e_w*ax_4, dim=1).view(-1, self.word_dim) # (batch_size, word_dimension)

        #import  pdb; pdb.set_trace()
        return z_s

    # w: (batch_size, sequence_length)
    # y_s: (batch_size, word_dimension)
    # z_n: (batch_size * num_negative_reviews, word_dimension)
    def forward(self, pos_rev, neg_rev):
        #positive_review: (batch_size, sequence_length)
        #negative_review: (batch_size*num_negative_reviews, sequence_length)

        pos_rev_emb = self.word_embedding(pos_rev) # (batch_size, sequence_length, word_dimension)
        neg_rev_emb = self.word_embedding(neg_rev) # (batch_size*num_negative_reviews, sequence_length, word_dimension)

        y_s = torch.sum(pos_rev_emb, 1) # (batch_size, word_dimension, 1)
        z_n = torch.sum(neg_rev_emb, 1) # (batch_size * num_negative_reviews, word_dimension)
        
        pos_rev_mask = (pos_rev > 0).long()
        neg_rev_mask = (neg_rev > 0).long()

        pos_rev_mask = torch.sum(pos_rev_mask, dim=1, keepdim=True) + 1e-6
        neg_rev_mask = torch.sum(neg_rev_mask, dim=1, keepdim=True) + 1e-6

        y_s = (y_s / pos_rev_mask).view(-1, self.word_dim, 1)
        z_n = (z_n / neg_rev_mask).view(-1, self.word_dim)

        #import pdb; pdb.set_trace()

        z_s = self._attention(pos_rev, pos_rev_emb, y_s) # (batch_size, word_dimension)
        
        #p_t = self.transform_W(z_s)
        p_t = F.softmax(self.transform_W(z_s), dim=1) # (batch_size, aspect_dimension)
        r_s = self.transform_T(p_t) # (batch_size, word_dimension)

        # cosine similarity betwee r_s and z_s
        c1 = (F.normalize(r_s, p=2, dim=1) * F.normalize(z_s, p=2, dim=1)).sum(-1, keepdim=True) # (batch_size, 1)
        c1 = c1.repeat(1, self.num_neg_sent).view(-1) # (batch_size * num_negative)

        # z_n.view(conf.batch_size, conf.num_negative_reviews, -1): (batch_size, num_negative_reviews, word_dimension)
        # r_s.view(conf.batch_size, 1, -1): (batch_size, 1, word_dimension)
        # z_n * r_s: (batch_size, num_negative_reviews, word_dimension)
        # (z_n * r_s).sum(-1): (batch_size, num_negative)
        # (z_n * r_s).sum(-1).view(-1): (batch_size)
        c2 = (F.normalize(z_n.view(y_s.shape[0], self.num_neg_sent, -1), p=2, dim=2) \
             * F.normalize(r_s.view(y_s.shape[0], 1, -1), p=2, dim=2)).sum(-1).view(-1) # (batch_size * num_negative)
        
        out_loss = margin_ranking_loss(c1, c2, torch.FloatTensor([1.0]).cuda())
        
        J_loss = torch.mean(out_loss)

        transform_T_weight = F.normalize(self.transform_T.weight, p=2, dim=0) # word_dimension * aspect_dimension
        U_loss = mse_loss(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(self.n_aspect).cuda())
        return c1, c2, out_loss, self.lr_lambda * U_loss + J_loss


    def get_aspect_emb(self, review_ids):

        reviews_emb = self.word_embedding(review_ids)
        y_s = torch.sum(reviews_emb, 1)
        review_mask = (review_ids > 0).long()
        review_mask = torch.sum(review_mask, dim=1, keepdim=True) + 1e-6
        y_s = (y_s / review_mask).view(-1, self.word_dim, 1)
        z_s = self._attention(review_ids, reviews_emb, y_s) # (batch_size, word_dimension)
        p_t = F.softmax(self.transform_W(z_s), dim=1) # (batch_size, aspect_dimension)

        return p_t