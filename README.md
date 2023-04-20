# Code for ARG

This repository is built on top of [Neu-Review-Rec](https://github.com/ShomyLiu/Neu-Review-Rec), and includes various functionalities such as data preprocessing, modeling, and training RBRSs, among others.


## Training RBRSs
To train RBRSs, please follow the training flow of [Neu-Review-Rec](https://github.com/ShomyLiu/Neu-Review-Rec) and run the following command:

```
python3 main.py train \
--dataset ${dataset} \
--model=${model}
```

## Training ARG

### Training LM on Reviews Corpus
To train the language model on the reviews corpus, please run:

```
python3 lm/train_lm.py
````

### Training Aspect Extractor (ABAE)
The implemented ABAE is based on [this](https://gitee.com/peijie_hfut/u-arm/tree/master/src/amazon_pet_supplies/abae) repository. To train/test the aspect extractor, run:
```
python3 aspect/train_abae.py \
--dataset ${dataset} 

python3 aspect/test_abae.py \
--dataset ${dataset} 
````

### Leave-One-Out Pretraining on Transformers-Based Encoder-Decoder
To perform leave-one-out pretraining, you can obtain the data from Leave one out data can obtained from [SelSum](https://github.com/abrazinskas/SelSum/tree/master/data) and run:
```
python3 leave-one-out/train.py
```

### Train ARG
Finally, to train the ARG, please run:
```
python3 main.py train_arg.py \
--dataset ${dataset} \
--model=${model} \
--pth_path="checkpoints/${model}_${dataset}_default.pth" #RBRS checkpoints
````