import sys
sys.path.append('/home/zhangzekai/NLPDL_final')
from utils.utils import *
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
import torch
from transformers.models.bart.modeling_bart import *
from model.bart_ans_stressed_attn import load_as_model

# get exp name using args
def getname(train_args, model_args, data_args):
    if model_args.model_type == "as":
        expname = 'ASQG'
    elif model_args.model_type == "normal":
        expname = 'QG'
    expname += f'_({train_args.learning_rate})'

    expname += f'_({model_args.ans_attn}'
    if model_args.load_pretrain:
        expname += '_*)'
    else:
        expname += ')'
    
    expname += f'_({data_args.low_res}_{data_args.aug}_{data_args.sep_ans})'

    return expname

# load model : model_type, pretrained, load_path
def load_model(model_args, train_args):
    tokenizer = AutoTokenizer.from_pretrained("/home/zhangzekai/NLPDL_final/pretrained_model/bart_zh")
    if model_args.model_type == "normal":
        if train_args.do_predict and model_args.load_path!="":
            model = BartForConditionalGeneration.from_pretrained(model_args.load_path)
        elif model_args.load_pretrain:
            model = BartForConditionalGeneration.from_pretrained('/home/zhangzekai/NLPDL_final/pretrained_model/bart_zh')
        else:
            config = AutoConfig.from_pretrained("/home/zhangzekai/NLPDL_final/pretrained_model/bart_zh")
            model = BartForConditionalGeneration(config)
    elif model_args.model_type == "as":
        model = load_as_model()
        if train_args.do_predict and model_args.load_path!="":
            model.resize_token_embeddings(len(tokenizer))
            model.load_state_dict(torch.load(model_args.load_path,map_location='cuda:0'))
        
    if train_args.do_predict and model_args.load_path!="" and model_args.model_type == "as":
        pass
    else:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def preprocess_as_function(examples):

    inputs = []
    labels = []
    locs = []
    for i in range(len(examples['question'])):
        q = examples['question'][i]
        a = examples['answer'][i]
        c = examples['context'][i]
        input = f"知识：{c} 回答:{a}"
        label = f"问题：{q}"  
        # In which piece the answer appears
        loc0 = getloc(input,a)
        inputs.append(input)
        labels.append(label)
        locs.append(loc0)

    input_ids, input_mask = tokenize_batch(inputs)
    label_ids, label_mask = tokenize_batch(labels)
    label_ids[label_mask==0] = -100
    loc0_mask = torch.zeros(input_mask.shape)
    for i in range(len(locs)):
        loc0 = locs[i]
        if len(loc0)>0:
            if loc0[-1]>255:
                loc0 = []
        # mask which area should be stressed
        loc0_mask[i][loc0] = 1

    model_inputs = {}
    model_inputs['input_ids'] = input_ids
    model_inputs['attention_mask'] = input_mask
    model_inputs['labels'] = label_ids
    model_inputs["loc0_mask"] = loc0_mask

    return model_inputs

# sep: whether to apply ans separation
def preprocess_normal_function_sep(examples):

    inputs = []
    labels = []
    for i in range(len(examples['question'])):
        q = examples['question'][i]
        a = examples['answer'][i]
        c = examples['context'][i]
        input = f"知识：{c.replace(a, ' <ans> ')} 回答:{a}"
        label = f"问题：{q}" 

        inputs.append(input)
        labels.append(label)

    input_ids, input_mask = tokenize_batch(inputs)
    label_ids, label_mask = tokenize_batch(labels)

    model_inputs = {}
    model_inputs['input_ids'] = input_ids
    model_inputs['attention_mask'] = input_mask
    model_inputs['labels'] = label_ids

    return model_inputs

def preprocess_normal_function_nosep(examples):

    inputs = []
    labels = []
    for i in range(len(examples['question'])):
        q = examples['question'][i]
        a = examples['answer'][i]
        c = examples['context'][i]

        input = f"知识：{c} 回答:{a}" 
        label = f"问题：{q}"   

        inputs.append(input)
        labels.append(label)

    input_ids, input_mask = tokenize_batch(inputs)
    label_ids, label_mask = tokenize_batch(labels)

    model_inputs = {}
    model_inputs['input_ids'] = input_ids
    model_inputs['attention_mask'] = input_mask
    model_inputs['labels'] = label_ids

    return model_inputs

# Rewrite the Seq2SeqTrainer for Stressed Attn
class ASSeq2SeqTrainer(Seq2SeqTrainer):
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        # print(self._signature_columns)
        signature_columns = self._signature_columns + ["loc0_mask"]

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"

        columns = [k for k in signature_columns if k in dataset.column_names]

        return dataset.remove_columns(ignored_columns)

    def _compute_loss(self, model, inputs, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        logits = model(**inputs, use_cache=False)[0]
        # print(logits.view(-1, logits.shape[-1]))
        # print(labels.view(-1))
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss, logits

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        loc0_mask = inputs.pop("loc0_mask")
        for layer in model.model.decoder.layers:
            layer.encoder_attn.loc = loc0_mask
        loss, _ = self._compute_loss(model, inputs, labels)
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": 32,
            "num_beams": 4,
        }
        
        # modified here, add attributes of loc0_mask
        loc0_mask = inputs.pop("loc0_mask")
        for layer in model.model.decoder.layers:
            layer.encoder_attn.loc = loc0_mask.repeat(4,1)

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
            # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = inputs.pop("labels")
        for layer in model.model.decoder.layers:
            layer.encoder_attn.loc = loc0_mask.repeat(1,1)

        with torch.no_grad():
            # compute loss on predict data
            loss, logits = self._compute_loss(model, inputs, labels)

        loss = loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None)

        logits = generated_tokens

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, logits, labels)
    
