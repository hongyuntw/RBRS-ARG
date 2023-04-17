import time
import random
import math
import fire
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset import *
from framework import Model
import models
import config
import pickle
from ranking.evaluate import NDCG, HitRate
from ranking.metric import RankingMetric
import gensim
# from main import build_score_dict
from utils import *
import pandas as pd
from transformers import AutoTokenizer, BartForConditionalGeneration
from models.bart import myBartForConditionalGeneration
torch.set_printoptions(precision=4)

device = 'cuda'


def gen_attack_review(gen_model, tokenizer, device, item_reviews, item_ratings, max_input_length = 512, max_target_length = 256):
    rating_tokens_map = {
        '1.0' : '<rating_1>',
        '2.0' : '<rating_2>',
        '3.0' : '<rating_3>',
        '4.0' : '<rating_4>',
        '5.0' : '<rating_5>',
    }

    for i in range(len(item_reviews)):
        rating_token = rating_tokens_map[str(item_ratings[i])]
        item_reviews[i] = rating_token + ' ' + item_reviews[i]
    input_text = f' {tokenizer.sep_token} '.join(item_reviews)
    encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
    encoded_input = encoded_input.to(device)

    generated_tokens = gen_model.generate(**encoded_input, 
                                            early_stopping=True, 
                                            max_length=max_target_length, 
                                            num_beams=5,
                                            do_sample=False,
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.5
                                            )

    beams_output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return beams_output_texts


def predict_origin(model, data_loader, opt, origin=True):
    score_dict = {}
    """
    score_dict = {
        user_id : [predict_score, ground_truth]
    }
    """
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)

            test_data = unpack_input(opt, test_data)
            

            uids_batch, iids_batch = test_data[2], test_data[3]

            output = model(test_data)

            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

            for i in range(test_data[0].shape[0]):
                uid = test_data[2][i].cpu().item()
                predict_score = output[i].cpu().item()
                gt_score = scores[i].cpu().item()
                score_dict[uid] = [predict_score, gt_score]


    data_len = len(data_loader.dataset)
    if data_len == 0:
        mse = mae = 0.0
    else:
        mse = total_loss * 1.0 / data_len
        mae = total_maeloss * 1.0 / data_len
    return (total_loss, mse, mae, score_dict)


def predict_add_review(model, data_loader, opt, item_review_count, new_review):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()

    new_review_ids = review2ids(opt, new_review)

    with torch.no_grad():
        score_dict = {}
        """
        score_dict = {
            user_id : [predict_score, ground_truth]
        }
        """

        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)

            uids, iids = list(zip(*test_data))
            uids = list(uids)
            iids = list(iids)
            
            # b_size, #reviews, review_max_length
            user_reviews = opt.users_review_list[uids] 
            # b_size, #reviews
            user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id


            # b_size, doc_max_length
            user_doc = opt.user_doc[uids]

            item_user2id = opt.item2userid_list[iids]  # 检索出该item对应的user id
            item_reviews = opt.items_review_list[iids][0]

            # add new review ids behind all reviews
            item_reviews[item_review_count] = new_review_ids
     
            item_reviews = np.repeat(item_reviews[None], item_user2id.shape[0], axis=0)

            item_doc = opt.item_doc[iids][0]
            pre = item_doc.copy()
            pos = 0
            for i in range(len(item_doc) - 1, -1, -1):
                if item_doc[i] != 0:
                    pos = i
                    break
            if pos > 0:
                item_doc = list(item_doc[:pos + 1]) + new_review_ids 
                if len(item_doc) < opt.doc_len:
                    item_doc += [0] * (opt.doc_len - len(item_doc))
                else:
                    item_doc = item_doc[:opt.doc_len]
                item_doc = np.array(item_doc)

            item_doc = np.repeat(item_doc[None], item_user2id.shape[0], axis=0)
            data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
            test_data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
            
            uids_batch, iids_batch = test_data[2], test_data[3]

            output = model(test_data)

            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

            for i in range(test_data[0].shape[0]):
                uid = test_data[2][i].cpu().item()
                predict_score = output[i].cpu().item()
                gt_score = scores[i].cpu().item()
                score_dict[uid] = [predict_score, gt_score]


        data_len = len(data_loader.dataset)
        if data_len == 0:
            mse = mae = 0.0
        else:
            mse = total_loss * 1.0 / data_len
            mae = total_maeloss * 1.0 / data_len
 
    return (total_loss, mse, mae, score_dict)



def get_prediction_shift(origin_result, pred_result):
    org_score_dict = origin_result[3]
    pred_score_dict = pred_result[3]
    ps = 0.0
    if len(pred_score_dict) == 0:
        return ps
    for k, v in pred_score_dict.items():
        org_pred = org_score_dict[k][0]
        pred = v[0]
        #  most shift
        # ps += abs(pred - org_pred)
        ## most postivie
        ps += pred - org_pred

    ps /= len(pred_score_dict)
    return ps



def logging(fp, msg):
    print(msg, file=fp)


def do_attack(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    opt.load_w2v_model()
    assert(len(opt.pth_path) > 0)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")

    path = os.path.join(opt.data_root, 'test/')
    data = np.load(path + 'Test.npy', encoding='bytes')
    test_index_set = []
    for i, d in enumerate(data):
        uid, iid = d
        test_index_set.append(iid)
    test_index_set = list(dict.fromkeys(test_index_set))


    test_item_review_count = defaultdict(int)
    for test_index in test_index_set:
        max_review_per_item = opt.item2userid_list[test_index].shape[0]
        review_count = (opt.item2userid_list[test_index] != opt.user_num - 1 ).sum()
        test_item_review_count[test_index] = review_count

    test_item_review_count = {k: v for k, v in sorted(test_item_review_count.items(), key=lambda item: item[1], reverse=True)}
    del path, data

    device = 'cuda'
    gen_model_unspervised_name = 'review_generate/unsupervised/checkpoints/bart-base-cnn-rating-tokens/checkpoint-47271'
    tokenizer = AutoTokenizer.from_pretrained(gen_model_unspervised_name)
    


    exp_name = opt.exp_name
    used_epoch = opt.used_epoch
    gen_model_name = f'rl_outputs/checkpoints/{opt.dataset}_{opt.model}/{exp_name}/{used_epoch}'

    print(gen_model_name)
    gen_model = myBartForConditionalGeneration.from_pretrained(gen_model_name).to(device)


    gen_model.eval()
    print_review = False
    max_input_length = 512
    max_target_length = 128


    df_path = f'./dataset/{opt.dataset}/data.csv'
    df = pd.read_csv(df_path)

    total_add_ps = 0.0
    count = 0

    attack_item_ids = []
    attack_reviews = []
    attack_ps = []

    attack_item_indices = list(test_index_set)
    sample_num = 500
    attack_item_indices = random.sample(attack_item_indices, k=sample_num)
    for attack_item_index in attack_item_indices:

        attack_item_id = opt.index2item[attack_item_index]
        item_review_count = test_item_review_count[attack_item_index]
        print(item_review_count)
        print(f'item index = {attack_item_index}', f'item id = {attack_item_id}', end=' ')
        print(f'has {item_review_count} reviews in training set')

        if item_review_count >= max_review_per_item:
            continue

        test_data = AttackDataset(opt.data_root, opt, attack_item_index)
        test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"{now()}: test in the test datset")

        print('-----origin data-----')
        origin_result = predict_origin(model, test_data_loader, opt, origin=True)
        print('---------------------')


        ################ 
        # doing attack #
        ################
        user_indices = [u_idx for u_idx in opt.item2userid_list[attack_item_index] if u_idx != (opt.user_num - 1)]
        item_reviews = []
        item_ratings = []
        for user_index in user_indices:
            user_id = opt.index2user[user_index]
            target_row = df.query(f'user_id=="{user_id}" and item_id=="{attack_item_id}"')
            review = target_row.iloc[0]['reviews']
            rating = target_row.iloc[0]['ratings']
            item_reviews.append(review)
            item_ratings.append(rating)


        attack_review = gen_attack_review(gen_model, tokenizer, device, item_reviews, item_ratings,max_input_length=max_input_length, max_target_length=max_target_length)[0]
 
        added_review_result = predict_add_review(model, test_data_loader, opt, item_review_count, attack_review)
        added_ps = get_prediction_shift(origin_result, added_review_result)
        total_add_ps += added_ps
        print(added_ps)
        count += 1

        attack_item_ids.append(attack_item_id)
        attack_reviews.append(attack_review)
        attack_ps.append(added_ps)

    print(gen_model_name)
    np.set_printoptions(suppress=True)
    print(f'testing on {count} item , avg predition shift : {total_add_ps / count:.6f}')

    exp_df = pd.DataFrame({
        'attack_item_ids':attack_item_ids, 
        'attack_reviews':attack_reviews, 
        'attack_ps':attack_ps})

    csv_path = f'./rl_outputs/df/{opt.dataset}_{opt.model}_{exp_name}_{used_epoch}.csv'
    exp_df.to_csv(csv_path, index=None)
    ps = exp_df.attack_ps.values 
    total = ps.shape[0]
    ps = ps[ps > 0]
    success = ps.shape[0]
    print(total, success, success / total)
    print(f'{ps.mean():f}')


def do_attack_add_random_review(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    opt.load_w2v_model()
    assert(len(opt.pth_path) > 0)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)


    model.load(opt.pth_path)
    model.eval()
    print(f"load model: {opt.pth_path}")

    """
    search test item_id reviews num in training
    """
    path = os.path.join(opt.data_root, 'test/')
    data = np.load(path + 'Test.npy', encoding='bytes')
    test_index_set = set()
    for i, d in enumerate(data):
        uid, iid = d
        test_index_set.add(iid)

    test_item_review_count = defaultdict(int)
    for test_index in test_index_set:
        max_review_per_item = opt.item2userid_list[test_index].shape[0]
        review_count = (opt.item2userid_list[test_index] != opt.user_num - 1 ).sum()
        test_item_review_count[test_index] = review_count

    test_item_review_count = {k: v for k, v in sorted(test_item_review_count.items(), key=lambda item: item[1], reverse=True)}
    del path, data

    count = 0
    total_add_ps = 0.0
    sucess_count = 0
    sucess_ps = 0.0

    sample_item_num = 500
    attack_item_indices = list(test_index_set)
    print(len(attack_item_indices))

    df = pd.read_csv(f'dataset/{opt.dataset}/data.csv')
    filter_df = df[df['ratings'] == 5]
    attack_item_indices = random.choices(attack_item_indices, k=sample_item_num)
    for attack_item_index in attack_item_indices:
        attack_item_id = opt.index2item[attack_item_index]
        item_review_count = test_item_review_count[attack_item_index]
        print(item_review_count)
        print(f'item index = {attack_item_index}', f'item id = {attack_item_id}', end=' ')
        print(f'has {review_count} reviews in training set')

        if item_review_count >= max_review_per_item:
            continue


        test_data = AttackDataset(opt.data_root, opt, attack_item_index)
        test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"{now()}: test in the test datset")

        print('-----origin data-----')
        origin_result = predict_origin(model, test_data_loader, opt, origin=True)
        print('---------------------')


        sample_review_num = 10
        sample_data = filter_df.sample(sample_review_num)

        each_sample_review_ps = 0.0

        for i in range(sample_review_num):
            sample_user_id, sample_item_id, sample_rating, sample_review = sample_data.values[i]
            sample_review = clean_str(sample_review)
            added_review_result = predict_add_review(model, test_data_loader, opt, item_review_count, sample_review)

            added_ps = get_prediction_shift(origin_result, added_review_result)
            each_sample_review_ps += added_ps

        ps = each_sample_review_ps / sample_review_num
        
        total_add_ps += ps
        count += 1

        print(f'prediction shift : {ps}')

        if ps > 0:
            sucess_count += 1
            sucess_ps += ps


        if count >= sample_item_num:
            break

    print('do_attack_add_random_review')
    print(f'{opt.dataset} - {opt.model}')
    print(f'attack success : {sucess_count} / {count}, {sucess_count / count:.2f}')
    print(f'total ps: {total_add_ps / count :f}')
    print(f'sucess ps : {sucess_ps / sucess_count:f}')




