from transformers import BartForConditionalGeneration
import sys
sys.path.append('/home/zhangzekai/NLPDL_final')
from utils.utils import *
from utils.train_eval_utils import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
    TrainingArguments,
)
from tqdm import tqdm
import wandb
from datasets import Dataset as HDataset
from datasets import DatasetDict
from transformers.models.bart.modeling_bart import *
from model.bart_ans_stressed_attn import load_as_model

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(
        default="normal"
    )  
    ans_attn: Optional[str] = field(
        default='none',
    ) # double, tri, norm
    load_pretrain: Optional[int] = field(
        default=1
    )
    load_path: Optional[str] = field(
        default="",
    )

@dataclass
class DataArguments:
    low_res: Optional[int] = field(
        default=100
    )
    aug: Optional[str] = field(
        default='none',
    ) # squad, multi_q
    sep_ans: Optional[int] = field(
        default=1,
    )

parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments)) 
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

expname = getname(train_args, model_args, data_args)
train_args.output_dir+=f'/{expname}'
train_args.run_name = f'{expname}'
set_seed(train_args.seed)

model_type_name = "ASQG" if model_args.model_type == "as" else "QG"

wandb.init(project="QG", name=expname)

model, tokenizer = load_model(model_args, train_args)

train_data = read_jsonl('/home/zhangzekai/NLPDL_final/data/qg_data/train.jsonl')
train_len = len(train_data)
test_data = read_jsonl('/home/zhangzekai/NLPDL_final/data/qg_data/test.jsonl')
if data_args.low_res in [1,5,10,20]:
    train_data = train_data[:int(train_len*data_args.low_res/100)]

if data_args.aug == 'squad':
    train_data += read_txt('/home/zhangzekai/NLPDL_final/data/squad_zh.txt')
elif data_args.aug == 'qa':
    train_data += read_list('/home/zhangzekai/NLPDL_final/data/QA_aug.pkl')
elif data_args.aug == 'multi_q':
    train_data += read_list('/home/zhangzekai/NLPDL_final/data/pseudo_q.pkl')

train_raw_dataset = HDataset.from_dict({'question':[x[0] for x in train_data],'answer':[x[1] for x in train_data],'context':[x[2] for x in train_data]})
test_raw_dataset = HDataset.from_dict({'question':[x[0] for x in test_data],'answer':[x[1] for x in test_data],'context':[x[2] for x in test_data]})

if model_type_name == "ASQG":
    preprocess_function = preprocess_as_function
else:
    if data_args.sep_ans:
        preprocess_function = preprocess_normal_function_sep
    else:
        preprocess_function = preprocess_normal_function_nosep

data = DatasetDict(
    {
        'train' : train_raw_dataset.map(preprocess_function, batched=True),
        'test' : test_raw_dataset.map(preprocess_function, batched=True)
    }
)

test_dataset = data["test"]
context = [x[0] for x in test_data]

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

if model_type_name == "ASQG":
    MyTrainer = ASSeq2SeqTrainer
else:
    MyTrainer = Seq2SeqTrainer

total_predictions = []
total_references = []

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = [x[3:] for x in predictions]
    references = [x[3:] for x in references]
    if train_args.do_predict:
        total_predictions.extend(predictions)
        total_references.extend(references)

    result = compute_score(predictions, references)
    return result

trainer = MyTrainer(
    model,
    train_args,
    train_dataset=data["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if train_args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()  
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

elif train_args.do_predict:
    predicted = trainer.predict(test_dataset)

    logname = f'/home/zhangzekai/NLPDL_final/output/{model_type_name}'
    f = open(f'{logname}.log','w+')

    for i in range(len(total_predictions)):
        f.write('{')
        f.write(
            f'"pred": ["{total_predictions[i]}"], "refs": ["{total_references[i]}"], "context": ["{context[i]}"]'
        )
        f.write('}\n')
    f.close()
