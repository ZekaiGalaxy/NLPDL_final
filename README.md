# NLPDL Final Project (2022 Autumn)
## Chinese QG ##

This project is based on pretrained model [**Randeng-BART-139M-QG-Chinese**](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M-QG-Chinese). 

In this project, I implemented ***answer separation*** in order to improve the performance of the QG model, as suggested by [**this paper**](https://arxiv.org/pdf/1809.02393.pdf). To further guide the model's decoding process, I introduced a ***post-selection*** method that selects the generated texts with the highest overlap with the given answer as the output. Upon analyzing the data, I discovered significant differences between the training and testing sets, so I employed a data augmentation technique called "***pseudo_q***" to reduce these gaps. The report is available at [here](https://github.com/violets-blue/NLPDL_final/NLPDL_final_report.pdf)

## Run Model ##

### Preparation ###
```
cd NLPDL_final 
mkdir pretrained_model 
mkdir saved_model 
mkdir output 

# Download pretrained QG model & add to pretrained_model
pretrained_model/bart_zh 
from https://huggingface.co/IDEA-CCNL/Randeng-BART-139M-QG-Chinese

# Download pretrained QA model & add to pretrained_model
pretrained_model/QA 
from https://huggingface.co/uer/roberta-base-chinese-extractive-qa
```

### Preprocess ###
```
python data_preprocess/preprocess_squad_zh.py
python data_preprocess/process_pseudo_q.py
```

### Training_args ###
You can create your own training_args to train model and eval model.

An example is shown below
```
# train_args

{
    "do_train":true,  # set to true when train
    "do_predict":false, # set to true when eval
    "output_dir":"saved_model",
    "save_strategy":"epoch",
    "save_total_limit":5,
    "num_train_epochs":5,
    "per_device_train_batch_size":16,
    "gradient_accumulation_steps":4,
    "logging_steps":10,
    "learning_rate":3e-04,
    "seed":42,
    "ans_attn":"none",
    "load_pretrain":1,  # whether to load the pretrained model or train from scratch
    "low_res":100,  # low resource senario
    "aug":"none", # data augmentation methods from [qa, squad, pseudo_q]
    "sep_ans":1,  # whether to use answer separation
    "model_type":"as", # use bart model or ans-stressed-attention bart model
    "load_path":"", # provided for eval
    "generation_max_length":25,
    "predict_with_generate":true
}

```

### Train & Eval ###
```
python train.py your_train_args.json
python train.py your_eval_args.json
```


