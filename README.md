# NLPDL Final Project (2022 Autumn)
## Chinese QG ##

This project is based on pretrained model [**Randeng-BART-139M-QG-Chinese**](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M-QG-Chinese). 

In this project, I implemented ***answer separation*** in order to improve the performance of the QG model, as suggested by [**this paper**](https://arxiv.org/pdf/1809.02393.pdf). To further guide the model's decoding process, I introduced a ***post-selection*** method that selects the generated texts with the highest overlap with the given answer as the output. Upon analyzing the data, we discovered significant differences between the training and testing sets, so we employed a data augmentation technique called "***pseudo_q***" to reduce these gaps.

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
python 
```
