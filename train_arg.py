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
import math

# from main import build_score_dict
from utils import *
import pandas as pd
from transformers import AutoTokenizer
from models.bart import myBartForConditionalGeneration
from transformers import AdamW, AutoModelForCausalLM
import sys
from rouge import Rouge
from attack import predict_origin, get_prediction_shift, predict_add_review
import string
import shutil
from aspect.ABAE import ABAE
import wandb


device = "cuda"
max_input_length = 512
max_target_length = 128
rouge = Rouge()
aspect_pooling = 'similar'
exp_name = 'YOUR_EXP_NAME'

save_every_epochs = False

save_steps = 200
steps = 1

alpha = 0.5
mixed_loss = False
epochs = 40
lr = 1e-5

# writer = None
data=None
scores=None

def get_ppl_score(model, tokenizer, sentence):
    input_dict = tokenizer(sentence, return_tensors='pt')
    input_dict = {k : v.to(device) for k, v in input_dict.items()}
    try:
        with torch.no_grad():
            loss=model(**input_dict, labels=input_dict["input_ids"]).loss
        return math.exp(loss.item())
    except Exception as e:
        return 0

def get_inverse_perplexity(output_texts, lm_model, lm_tokenizer):
    inv_ppl_scores = []
    for text in output_texts:
        ppl_score = get_ppl_score(lm_model, lm_tokenizer, text)
        inv_ppl_scores.append(1.0 / ppl_score)
    inv_ppl_scores = torch.tensor(inv_ppl_scores)
    return  inv_ppl_scores.to(device)


def get_seq_prob(generated_outputs, tokenizer):
    # batch_size * seq_len (ids)
    sequence = generated_outputs['sequences']

    # the first token is <EOS> , there is no score of it
    sequence = sequence[:, 1:]

    # mask = batch_size * seq_len 
    mask = (sequence != tokenizer.pad_token_id).float()

    # scores = tuple , tuple size = seq_len

    # element in tuple size = batch size x vocab_size
    # probs = batch size x seq_len x vocab_size
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)

    # batch_size * seq_len
    gen_probs = torch.gather(probs, 2, sequence[:, :, None]).squeeze(-1)
    # log 0 will cause error
    eps=1e-6
    gen_probs = gen_probs + eps
    log_gen_probs = torch.log(gen_probs)

    return log_gen_probs, mask


def get_rouge(output_texts, input_reviews, rouge_type='rouge-1'):

    scores = []
    all_rouge_scores = []
    output_texts_clean = [clean_str(output_text) for output_text in output_texts]
    for i in range(len(output_texts)):
        item_reviews = []
        for item_review in input_reviews[i].tolist():
            item_review = clean_str(item_review)
            if item_review == '':
                item_review = '<PAD>'
            item_review = item_review[:512]
            item_reviews.append(item_review)
        gen_review = [output_texts_clean[i]] * len(item_reviews)
        rouge_scores = rouge.get_scores(gen_review, item_reviews, avg=False, ignore_empty=True)
        rouge_scores = [r[rouge_type]['f'] for r in rouge_scores]
        all_rouge_scores.append(rouge_scores)
        avg_scores = sum(rouge_scores) / len(rouge_scores)
        scores.append(avg_scores)

    all_rouge_scores = np.array(all_rouge_scores, dtype=object)
    scores = torch.tensor(scores)
    return scores.to(device), all_rouge_scores



def logging(tag, value):
    # writer.add_scalar(tag, value, steps)
    wandb.log({
        tag : value,
        'steps' : steps,
    })
    

def cal_prediction_shift(output_texts, item_indices, model, opt):
    prediction_shift = []
    max_review_per_item = opt.item2userid_list[item_indices[0]].shape[0]
    for i in range(len(output_texts)):
        attack_review = output_texts[i]
        item_index = item_indices[i]

        item_review_count =  (opt.item2userid_list[item_index] != opt.user_num - 1 ).sum()
        if item_review_count >= max_review_per_item:
            prediction_shift.append(0)
            continue


        test_data = AttackDataset(opt.data_root, opt, item_index, data_mode='train', data=data, scores=scores)
        test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

        origin_result = predict_origin(model, test_data_loader, opt, origin=True)
        
        added_review_result = predict_add_review(model, test_data_loader, opt, item_review_count, attack_review)
        ps = get_prediction_shift(origin_result, added_review_result)


        prediction_shift.append(ps)
    prediction_shift = torch.tensor(prediction_shift).float()
    return prediction_shift.to(device)

def min_max_normalize(x, upper, lower):
    x[x > upper] = upper
    x = (x - lower) / (upper - lower)
    return x



def get_rewards(input_reviews, output_texts , 
                item_indices, model, opt, 
                lm_model, lm_tokenizer, 
                mode='sample', rewards_type='all'):

    ps = cal_prediction_shift(output_texts, item_indices, model, opt)
    logging(f'{mode}/ps', ps.mean().item())


    inv_ppl = get_inverse_perplexity(output_texts, lm_model, lm_tokenizer)
    logging(f'{mode}/inv_ppl', inv_ppl.mean().item())

    rouge1_score, all_rouge_scores = get_rouge(output_texts, input_reviews, 'rouge-1')
    logging(f'{mode}/rouge-1', rouge1_score.mean().item())


    inv_ppl = min_max_normalize(inv_ppl, 0.1, 0.0)
    rouge1_score = min_max_normalize(rouge1_score, 0.25, 0.0)

    rewards = ps + inv_ppl + rouge1_score

 
    logging(f'{mode}/rewards', rewards.mean().item())

    rewards = torch.unsqueeze(rewards, 1)  # [bs, 1]
    rewards = torch.unsqueeze(rewards, 2)  # [bs, 1, 1]
    return rewards, all_rouge_scores





def get_aspect_emb(reviews, opt, w2v, abae, mode='input'):
    assert mode in ['input', 'sample']
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

    if mode == 'sample':
        review_ids_list = []
        for review in reviews:
            review_ids = review2ids(opt, w2v, review)
            review_ids = torch.LongTensor(review_ids).to(device)
            review_ids_list.append(review_ids)
        review_ids = torch.stack(review_ids_list, 0)
        
        aspect_emb = abae.get_aspect_emb(review_ids)
    elif mode == 'input':
        # reviews = b_size * # of item reviews
        batch_aspect_emb = []
        for i in range(len(reviews)):
            review_ids_list = []
            for review in reviews[i]:
                review_ids = review2ids(opt, w2v, review)
                review_ids = torch.LongTensor(review_ids).to(device)
                review_ids_list.append(review_ids)
            review_ids = torch.stack(review_ids_list, 0)
            
            aspect_emb = abae.get_aspect_emb(review_ids)
            batch_aspect_emb.append(aspect_emb)
        aspect_emb = torch.stack(batch_aspect_emb, 0)
        

    return aspect_emb.to(device)


def train_rl_ps_aspect(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    opt.load_w2v_model()

    assert(len(opt.pth_path) > 0)

    log_base_folder = f'./rl_outputs/logs/{opt.dataset}_{opt.model}/'
    log_folder = log_base_folder + exp_name
    shutil.rmtree(log_folder, ignore_errors=True)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)


    wandb_exp_name = f'{opt.dataset}_{opt.model}_{exp_name}'
    wandb.init()

    ### load recommender model
    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")

    opt.data_mode = 'train'
    data_mode = opt.data_mode
    print(f'data mode = {data_mode}')

    global data, scores
    path = os.path.join(opt.data_root, f'{data_mode}/')
    data = np.load(path + f'{data_mode.capitalize()}.npy', encoding='bytes')
    scores = np.load(path + f'{data_mode.capitalize()}_Score.npy')
    
    ### ABAE model
    from gensim.models import Word2Vec
    w2v = Word2Vec.load(f'aspect/data/word2vec_{opt.dataset}.model')
    n_aspect = 15
    vocab_size = len(w2v.wv.index_to_key)
    model_path = f'aspect/checkpoints/model_{opt.dataset}_{n_aspect}'
    aspect_params = torch.load(model_path)

    abae = ABAE(opt, vocab_size=vocab_size, n_aspect=n_aspect)
    abae.load_state_dict(aspect_params)
    abae.eval()
    abae.to(device)

    l1_loss = torch.nn.L1Loss()

    ### lm model 
    lm_pretrained_model = 'distilgpt2'
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_pretrained_model, use_fast=True)
    # distilgpt2 dont has pad token
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    # lm_model_path = 'distilgpt2'
    if opt.dataset == 'Musical_Instruments_data':
        lm_model_path = './bias_lm/distilgpt2_128-Musical_Instruments_data_reviews/checkpoint-41272'
    elif opt.dataset == 'Digital_Music_data':
        lm_model_path = './bias_lm/distilgpt2-Digital_Music_data_reviews/checkpoint-7731'
    elif opt.dataset == 'Toys_and_Games_data':
        lm_model_path = './bias_lm/distilgpt2_128-Toys_and_Games_data_reviews/checkpoint-485500'
    elif opt.dataset == 'Video_Games_data':
        lm_model_path = './bias_lm/distilgpt2_128-Video_Games_data_reviews/checkpoint-59257'
    else:
        print('no fine-tuned lm model found on current dataset')
        lm_model_path = 'distilgpt2'

    
    print(lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(lm_model_path).to(device)
    lm_model.eval()

    ### Review generator model & tokenier
    gen_model_unspervised_name = './review_generate/unsupervised/checkpoints/bart-base-cnn-rating-tokens/checkpoint-47271'
    tokenizer = AutoTokenizer.from_pretrained(gen_model_unspervised_name)
    
    gen_model_name = './review_generate/unsupervised/checkpoints/bart-base-cnn-rating-tokens/checkpoint-47271'
    gen_model = myBartForConditionalGeneration.from_pretrained(gen_model_name).to(device)

    base_folder = f'./rl_outputs/checkpoints/{opt.dataset}_{opt.model}/'

    save_reviews_folder = f'./rl_outputs/sample_reviews/{opt.dataset}_{opt.model}/'
    if not os.path.exists(save_reviews_folder):
        os.makedirs(save_reviews_folder)


    save_folder = base_folder + exp_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    global steps
    ### Training RL hyperparams
    
    rl_batch_size = 4

    optimizer = AdamW(gen_model.parameters(), lr=lr, weight_decay=1e-5)
    accumulation_steps = 16
    
    rating_tokens_map = {
        '1.0' : '<rating_1>',
        '2.0' : '<rating_2>',
        '3.0' : '<rating_3>',
        '4.0' : '<rating_4>',
        '5.0' : '<rating_5>',
    }

    ### build dataset
    train_set = RLDataset(opt.data_root, opt, data_mode='train')
    train_loader = DataLoader(train_set, batch_size=rl_batch_size, shuffle=True)

    eval_set = RLDataset(opt.data_root, opt, data_mode='test')
    eval_loader = DataLoader(eval_set, batch_size=rl_batch_size, shuffle=False)

    best_eval_reward = float('-inf')
    ### training RL

    print(log_folder)
    switch = 0
    global rewards_type
    for epoch in range(epochs):
        running_loss = 0.0
        running_sample_rewards = 0.0
        running_greedy_rewards = 0.0
        totals_batch = len(train_loader)
        gen_model.train()
        for i, batch_data in enumerate(train_loader):
            item_ids, ratings, reviews = batch_data
            # item_ids = tuple of item id which size equals batch size
            item_indices = [opt.item2index[item_id] for item_id in item_ids]

            input_reviews = np.array(reviews)
            input_reviews = np.transpose(input_reviews)

            
            batch_reviews = []
            for batch_idx in range(len(reviews[0])):
                input_texts = []
                for user_count in range(len(reviews)):
                    rating = ratings[user_count][batch_idx]
                    if rating == '0.0': 
                        break
                    rating_token = rating_tokens_map[rating]
                    input_texts.append(rating_token + ' ' + reviews[user_count][batch_idx])
                batch_reviews.append(input_texts)

            batch_reviews = [f' {tokenizer.sep_token} '.join(item_reviews) for item_reviews in batch_reviews]
            encoded_input = tokenizer(batch_reviews, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
            encoded_input = encoded_input.to(device)

            greedy_generated_tokens = gen_model.generate(**encoded_input, 
                                                    early_stopping=False, 
                                                    max_length=max_target_length, 
                                                    output_scores=True, 
                                                    num_beams=1,
                                                    do_sample=False,
                                                    remove_invalid_values=True)

            sample_generated_outputs = gen_model.generate_with_grad(**encoded_input, 
                                                    early_stopping=False, 
                                                    max_length=max_target_length, 
                                                    output_scores=True, 
                                                    return_dict_in_generate=True,
                                                    do_sample=True,
                                                    num_beams=1,
                                                    remove_invalid_values=True)

            sample_output_texts = tokenizer.batch_decode(sample_generated_outputs.sequences, skip_special_tokens=True)

            log_prob, mask = get_seq_prob(sample_generated_outputs, tokenizer)
            
            # b_size [string]
            greedy_output_texts = tokenizer.batch_decode(greedy_generated_tokens, skip_special_tokens=True)

            sample_rewards, all_rouge_scores = get_rewards(input_reviews, sample_output_texts, item_indices, model, opt, lm_model, lm_tokenizer, mode='sample', rewards_type=rewards_type)
            greedy_rewards, _ = get_rewards(input_reviews, greedy_output_texts, item_indices, model, opt, lm_model, lm_tokenizer, mode='greedy', rewards_type=rewards_type)
            rewards = sample_rewards - greedy_rewards

            running_sample_rewards += sample_rewards.squeeze().mean().item()
            running_greedy_rewards += greedy_rewards.squeeze().mean().item()
            

            mask = torch.unsqueeze(mask, 2) # [bs, max_length, 1]
            log_prob = torch.unsqueeze(log_prob, 2) # [bs, max_length, 1]
            
            rl_loss = -log_prob * mask * rewards 
            rl_loss = torch.sum(rl_loss) / torch.sum(mask)


            ## ABAE loss
            sample_reviews_aspect_emb = get_aspect_emb(sample_output_texts, opt, w2v, abae, mode='sample')
            input_review_aspect_emb = get_aspect_emb(input_reviews, opt, w2v, abae, mode='input')
 
            
            if aspect_pooling == 'similar':
                max_similar_index = np.argmax(all_rouge_scores, 1)
                target_aspect_embedding = input_review_aspect_emb[torch.arange(input_review_aspect_emb.size(0)), max_similar_index]
            elif aspect_pooling == 'mean':
                target_aspect_embedding = torch.mean(input_review_aspect_emb, 1)
            elif aspect_pooling == 'max':
                target_aspect_embedding , _ = torch.max(input_review_aspect_emb, dim=1)

            if aspect_pooling == 'None':
                loss = rl_loss
                aspect_loss = rl_loss
            else:
                aspect_loss = l1_loss(sample_reviews_aspect_emb, target_aspect_embedding)
                loss = rl_loss + aspect_loss

            running_loss += loss.item()
            
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    'run/rl_loss' : rl_loss.item(),
                    'run/aspect_loss' : aspect_loss.item(),
                    'steps' : steps,
                })
            
                steps += 1

            print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f} , reward_s : {running_sample_rewards / (i+1) :.5f} , reward_g : {running_greedy_rewards / (i+1) :.5f}' , end='' )
        

            if save_steps > 0 and steps  % save_steps == 0 :
                save_path = save_folder + f'/{steps}_steps'
                print()
                print(save_path)
                gen_model.save_pretrained(save_path)
                
        print()

        # Evaluation
        running_loss = 0.0
        running_rewards = 0.0
        totals_batch = len(eval_loader)
        gen_model.eval()
        for i, batch_data in enumerate(eval_loader):
            item_ids, ratings, reviews = batch_data
            item_indices = [opt.item2index[item_id] for item_id in item_ids]

            batch_reviews = []
            for batch_idx in range(len(reviews[0])):
                input_texts = []
                for user_count in range(len(reviews)):
                    rating = ratings[user_count][batch_idx]
                    if rating == '0.0': 
                        break
                    rating_token = rating_tokens_map[rating]
                    input_texts.append(rating_token + ' ' + reviews[user_count][batch_idx])
                batch_reviews.append(input_texts)

            batch_reviews = [f' {tokenizer.sep_token} '.join(item_reviews) for item_reviews in batch_reviews]
            encoded_input = tokenizer(batch_reviews, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
            encoded_input = encoded_input.to(device)

            # evaluation using beam search
            generated_tokens = gen_model.generate(**encoded_input, 
                                                    early_stopping=False, 
                                                    max_length=max_target_length, 
                                                    output_scores=True, 
                                                    num_beams=5,
                                                    do_sample=False,
                                                    remove_invalid_values=True,
                                                    no_repeat_ngram_size=2,
                                                    repetition_penalty=1.5)

            # b_size [string]
            beams_output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            input_reviews = np.array(reviews)
            input_reviews = np.transpose(input_reviews)

        
            beams_rewards, _ = get_rewards(input_reviews, beams_output_texts, item_indices, model, opt, lm_model, lm_tokenizer, mode='beam')

            running_rewards += beams_rewards.squeeze().mean().item()

            
            print(f'\r [Eval] Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, rewards : {running_rewards / (i+1) :.5f}' , end='' )
        print()
        torch.cuda.empty_cache()

        if running_rewards > best_eval_reward:
            best_eval_reward = running_rewards
            save_path = save_folder + '/best_rewards'
            gen_model.save_pretrained(save_path)
            print('save best rewards')

        if epoch == 0 or save_every_epochs or (epoch + 1) % 5 == 0:
            save_path = save_folder + f'/{epoch + 1}_epoch'
            gen_model.save_pretrained(save_path)
        
        