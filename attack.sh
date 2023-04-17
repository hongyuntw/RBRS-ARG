dataset="Musical_Instruments_data" \
model="DeepCoNN"
python3 main.py do_attack \
--dataset ${dataset} \
--pth_path="checkpoints/${model}_${dataset}_default.pth" \
--model=${model}  \
--exp_name="exp_name" \
--used_epoch="2000_steps"
# DeepCoNN
# NARRE
# RMG

# ps_ppl_mean_rouge1_difference_clip
# aspect_mean_pooling_no_ft_ps_ppl_rouge_difference
# aspect_similar_pooling_no_ft_ps_ppl_rouge_difference
# ps_ppl_mean_rouge1_difference_switch
# aspect_ps_ppl_mean_rouge1
# aspect_max_ps_ppl_mean_rouge1_difference

# do_attack_add_random_review
# do_attack

# dataset="Digital_Music_data" 
# dataset="Toys_and_Games_data" 
# dataset="Musical_Instruments_data"
# dataset="Software_data"
# dataset="Office_Products_data"
# dataset="Video_Games_data"
