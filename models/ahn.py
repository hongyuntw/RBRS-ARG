# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Gated_Attention(nn.Module):
    def __init__(self):
        super(Gated_Attention, self).__init__()
        self.proj_matrix = nn.Linear(300, 300, bias=False)
        self.gated_matrix = nn.Linear(300, 300, bias=False)
        self.item_att_v = nn.Parameter(torch.randn(300, 1), requires_grad=True)

    #     config.coatt_hidden_size
    def forward(self, item_input, i_mask):
        item_att_v = self.item_att_v.unsqueeze(0).expand(item_input.size(0), -1, -1)
        # bsz * L_u
        weighted_ir = torch.bmm(torch.tanh(self.proj_matrix(item_input)) * torch.sigmoid(self.gated_matrix(item_input)),
                                item_att_v).squeeze(2)  # bsz * 1 * L_u
        weighted_ir = weighted_ir / torch.sqrt(torch.tensor(300.0).cuda())

        # i_mask = torch.ones(weighted_ir.size()).cuda()
        # i_att_weight = F.softmax(weighted_ir, -1).unsqueeze(1)

        i_att_weight = masked_softmax(weighted_ir, i_mask).unsqueeze(1)  # bsz * 1* L_i
        overall_w = masked_softmax(weighted_ir.view(-1), i_mask.view(-1)).view(weighted_ir.size(0), -1)

        item_rep = i_att_weight.transpose(1, 2).expand_as(item_input) * item_input
        return item_rep, i_att_weight, overall_w


class AHN(nn.Module):
    '''
    AHN AAAI 2020
    '''
    def __init__(self, opt):
        super(AHN, self).__init__()
        self.opt = opt
        self.num_fea = 2  # ID + Review

        # self.user_net = Net(opt, 'user')
        # self.item_net = Net(opt, 'item')


        self.user_id_embedding = nn.Embedding(self.opt.user_num, self.opt.id_emb_size)
        self.item_id_embedding = nn.Embedding(self.opt.item_num, self.opt.id_emb_size)
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)

        self.seq_encoder = nn.LSTM(input_size=300, hidden_size=150,
                                   num_layers=1, batch_first=True,
                                   bidirectional=True)
        self.item_review_att = Gated_Attention()
        self.coatt_rev = Co_Attention_wATT()
        self.dropout = nn.Dropout(0.5)

        self.u_linear = nn.Linear(300, 300)
        self.i_linear = nn.Linear(300, 300)
        # lstm 150
        self.reset_para()


    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, _, _, _, _ = datas

        check = torch.sum(user_reviews, -1)
        user_reviews_mask = check > 0
        user_reviews_mask = user_reviews_mask.cuda()

        check = torch.sum(item_reviews, -1)
        item_reviews_mask = check > 0
        item_reviews_mask = item_reviews_mask.cuda()


        # user_reviews_mask = torch.ones((user_reviews.size(0),user_reviews.size(1))).cuda()
        # item_reviews_mask = torch.ones((item_reviews.size(0),item_reviews.size(1))).cuda()
        # for b in range(user_reviews.size(0)):
        #     for k in range(user_reviews.size(1)):
        #         if not torch.any(user_reviews[b][k]):
        #             user_reviews_mask[b][k] = 0

        #     for k in range(item_reviews.size(1)):
        #         if not torch.any(item_reviews[b][k]):
        #             item_reviews_mask[b][k] = 0



        user_embedding = self.dropout(self.user_id_embedding(uids))
        item_embedding = self.dropout(self.item_id_embedding(iids))

        user_reviews = self.word_embs(user_reviews)
        item_reviews = self.word_embs(item_reviews)
        # print(user_reviews.size())
        # input()


        # user review using BILSTM encode
        user_reviews_lstm_list = []
        for i in range(user_reviews.size(1)):
            user_review_feature, _ = self.seq_encoder(user_reviews[:,i,:,:])
            hidden = user_review_feature[:, -1, :]
            user_reviews_lstm_list.append(hidden)
        user_reviews_fea = torch.stack(user_reviews_lstm_list, 1)
        
        # item review using BILSTM encode
        item_reviews_lstm_list = []
        for i in range(item_reviews.size(1)):
            item_review_feature, _ = self.seq_encoder(item_reviews[:,i,:,:])
            hidden = item_review_feature[:, -1, :]
            item_reviews_lstm_list.append(hidden)
        item_reviews_fea = torch.stack(item_reviews_lstm_list, 1)

        # print(user_reviews_fea.size())
        # print(item_reviews_fea.size())
        # input()

        ir_weighted, i_rw, overall_i_rw = self.item_review_att(item_reviews_fea, item_reviews_mask)
        ir_pooled = torch.sum(ir_weighted, 1)

        user_reviews_fea = user_reviews_fea.contiguous()
        user_reviews_fea = F.relu(self.u_linear(user_reviews_fea))

        item_reviews_fea = item_reviews_fea.contiguous()
        item_reviews_fea = F.relu(self.i_linear(item_reviews_fea))
        # print(user_reviews_fea.size())
        # print(item_reviews_fea.size())
        # input()

        ur_pooled, u_rw = self.coatt_rev(user_reviews_fea, item_reviews_fea, user_reviews_mask,
                                         item_reviews_mask, overall_i_rw, review_level=False)

        # print('final output')
        # print(ir_pooled.size())
        # print(ur_pooled.size())
        # input()

        u_fea = torch.cat([ur_pooled, user_embedding], 1)
        i_fea = torch.cat([ir_pooled, item_embedding], 1)
        # print(u_fea.size())
        # print(i_fea.size())
        # input()


        return u_fea, i_fea

    def reset_para(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)



class Co_Attention_wATT(nn.Module):
    def __init__(self):
        super(Co_Attention_wATT, self).__init__()
        self.proj_matrix = nn.Linear(300, 300, bias=False)
        self.V = nn.Parameter(glorot([300, 300]), requires_grad=True)

    #     config.coatt_hidden_size
    def forward(self, user_input, item_input, u_sent_mask, i_sent_mask, i_sw, review_level=False):

        if review_level:
            item_input = item_input
        else:
            norm_i_sw = i_sw.div(torch.norm(i_sw, dim=-1).unsqueeze(-1).expand_as(i_sw) + 1e-13)
            norm_i_sw = norm_i_sw.unsqueeze(-1).expand_as(item_input)
            item_input = item_input * norm_i_sw.detach()

        project_user = torch.bmm(user_input, self.V.unsqueeze(0).expand(user_input.size(0), -1, -1))
        G = torch.bmm(project_user, item_input.transpose(1, 2))  # bsz * L_u * L_i
        G[G == 0] = -1000

        user_coatt = F.max_pool1d(G, G.size(2)).squeeze(2)
        user_coatt = user_coatt / torch.sqrt(torch.tensor(300.0).cuda())
        # u_sent_mask = torch.ones(user_coatt.size()).cuda()
        u_att_weight = masked_softmax(user_coatt, u_sent_mask).unsqueeze(1)  # bsz * 1 * L_u

        user_rep = torch.bmm(u_att_weight, user_input).squeeze(1)
        return user_rep, u_att_weight


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    return init


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)
