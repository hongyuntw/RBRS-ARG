dataset="Musical_Instruments_data" \
model="DeepCoNN"
python3 main.py train_arg.py \
--dataset ${dataset} \
--pth_path="checkpoints/${model}_${dataset}_default.pth" \
--model=${model}



# dataset="Digital_Music_data" 
# dataset="Toys_and_Games_data" 
# dataset="Musical_Instruments_data"
# dataset="Software_data"
# dataset="Office_Products_data"
# dataset="Video_Games_data"
