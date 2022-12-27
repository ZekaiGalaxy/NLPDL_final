import json
import pickle
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import os
import evaluate
import numpy as np
import collections
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline

def load_json(path):
    with open(path, mode='r', encoding='utf8') as f:
        return json.load(f)

def read_txt(path):
    # [Q,A,C]
    f = open(path)
    data = [line.strip().split('\t') for line in f.readlines()]
    dataset = []
    for x in data:
        dataset.append(x[0].split('&&'))
    return dataset

def read_list(path):
    with open(path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def read_jsonl(path):
    # [Q,A,C]
    f = open(path)
    dataset = []
    for line in f.readlines():
        data = json.loads(line)
        dataset.append([data['question'],data['answer'],data['context']])
    return dataset

def write_list(lst,path):
    with open(path, 'wb') as fp:
        pickle.dump(lst, fp)

def read_list(path):
    with open(path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

QA_model = AutoModelForQuestionAnswering.from_pretrained('/home/zhangzekai/NLPDL_final/pretrained_model/QA')
QA_tokenizer = AutoTokenizer.from_pretrained('/home/zhangzekai/NLPDL_final/pretrained_model/QA')
QA = pipeline('question-answering', model=QA_model, tokenizer=QA_tokenizer,device=0)

tokenizer = AutoTokenizer.from_pretrained("/home/zhangzekai/NLPDL_final/pretrained_model/bart_zh")

def tokenize_batch(data):
    tokenized=tokenizer(data,padding=True,truncation=True,max_length=256)
    input_ids=torch.tensor(tokenized['input_ids'])
    attention_mask=torch.tensor(tokenized['attention_mask'])
    return input_ids,attention_mask

def QA_aug():
    # data augmentation using QA

    # train_data = read_jsonl("/home/zhangzekai/NLPDL_final/data/qg_data/train.jsonl")
    # predictions = [x[0] for x in train_data]
    # QA_preds = []
    # aug = []
    # for i in tqdm(range(10)):
    #     # len(train_data)
    #     qa_input = {"question":train_data[i][0],"context":train_data[i][2]}
    #     answer = QA(qa_input)["answer"]
    #     if answer!=train_data[i][1]:
    #         aug.append([train_data[i][0], answer, train_data[i][2]])
    # write_list(aug, "/home/zhangzekai/NLPDL_final/data/train_aug.pkl")

    aug = read_list("/home/zhangzekai/NLPDL_final/data/QA_aug.pkl")
    return aug

class MyDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

def find_first(sent, ans):
    tem = []
    x = [sent]
    for sep in ['，','。','！','？','：','；',' ',',','.']:
        tem = []
        for xx in x:
            tem.extend(xx.split(sep))
        x = tem
    
    for xx in x:
        if ans in xx:
            return xx
    return ""

def context_match_idx(i,x,y):
    for j in range(1,len(y)):
        if x[i+j]!=y[j]:
            return 0
    return 1

def context_match(tc,tp):
    for i in range(len(tc)-len(tp)):
        if context_match_idx(i,tc,tp):
            return i
    return -1

def ans_match(tc):
    for i in range(len(tc)):
        # “回答：”
        if (tc[i]==34274 and tc[i+1]==2350 and tc[i+2]==34291):
            return i+3
    return -1

def getloc(c, a):
    p = find_first(c,a)
    tc = tokenizer(c)['input_ids']
    tp = tokenizer(p)['input_ids'][:-1]

    # get context loc
    context_idx = context_match(tc, tp)
    loc0 = list(range(context_idx,context_idx+len(tp))) if context_idx!=-1 else []

    return loc0

def process_dataset(dataset, mode='train', sep_ans=1):
    processed = []
    all_a = []
    all_c = []
    for data in dataset:
        q = data[0]
        a = data[1]
        c = data[2]
        # needed for extracting the pieces that contain ans
        all_c.append(find_first(c,a))
        all_a.append(a)
        input = ""
        label = ""

        if mode=='train':
            input = f"知识：{c.replace(a, ' <ans> ')} 回答:{a}"
            label = f"问题：{q}"

        elif mode=='test':
            if '#' in a:
                first_a = a.split('#')[0]
                if sep_ans:
                    input = f"知识：{c.replace(first_a, ' <ans> ')} 回答:{first_a}"
                    label = f"问题：{q}"
                else:
                    input = f"知识：{c} 回答:{first_a}"
                    label = f"问题：{q}"
            else:
                if sep_ans:
                    input = f"知识：{c.replace(a, ' <ans> ')} 回答:{a}"
                    label = f"问题：{q}"
                else:
                    input = f"知识：{c} 回答:{a}"
                    label = f"问题：{q}"
        processed.append([input, label])

    return processed, all_a, all_c

def collate_fn_train(batch):
    inputs, labels = zip(*batch)
    inputs, labels = list(inputs), list(labels)
    input_ids, input_mask = tokenize_batch(inputs)
    label_ids, label_mask = tokenize_batch(labels)
    return [input_ids, input_mask, label_ids, label_mask]

def save_model(model, epoch, modelname):
    save_dir = '/home/zhangzekai/NLPDL_final/saved_model'
    if not os.path.exists(save_dir+f'/{modelname}'):
        os.makedirs(save_dir+f'/{modelname}')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_dir+f'/{modelname}/{epoch}.pt')


## evalute ##

sacrebleu = evaluate.load('sacrebleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")
# use mt5_tokenizer for fair comparation
mt5_tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

def get_single_EM(pred, ref):
    return int(pred==ref)

def get_single_F1(pred, ref):
    tokenizer = mt5_tokenizer
    pred = tokenizer.tokenize(pred)
    ref = tokenizer.tokenize(ref)
    common = collections.Counter(pred) & collections.Counter(ref)
    num_same = sum(common.values())
    if len(pred)==0 or len(ref)==0 or num_same==0:
        return 0.0
    else:
        precision = 1.0 * num_same / len(pred)
        recall = 1.0 * num_same / len(ref)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
def get_QA_score(preds, refs):
    EM_score = 0.0
    F1_score = 0.0
    for i in range(len(preds)):
        EM_score += get_single_EM(preds[i], refs[i]) 
        F1_score += get_single_F1(preds[i], refs[i]) 
    result = {}
    result['EM'] = EM_score/len(preds)
    result['F1'] = F1_score/len(preds)

    return result

def test_QA(predictions, purify=0):
    if purify:
        test_data = read_list('/home/zhangzekai/NLPDL_final/data/qg_data/purified_test.pkl')
    else:
        test_data = read_jsonl("/home/zhangzekai/NLPDL_final/data/qg_data/test.jsonl")
    assert len(predictions)==len(test_data)
    QA_preds = []
    QA_refs = []
    for i in tqdm(range(len(test_data))):
        qa_input = {"question":predictions[i],"context":test_data[i][2]}
        try:
            answer = QA(qa_input)["answer"]
        except:
            answer = ""
        QA_preds.append(answer)
        QA_refs.append(test_data[i][1])
    return get_QA_score(QA_preds, QA_refs)

def compute_score(predictions, references, purify=0):
    result = {}
    tokenizer = mt5_tokenizer
    # print(predictions)
    # print(references)
    QA_result = test_QA(predictions, purify)
    result["QA_EM"] = QA_result["EM"]
    result["QA_F1"] = QA_result["F1"]
    print('QA done!')
    result["bleu"] = sacrebleu.compute(
                        predictions=predictions,
                        references=references,
                        tokenize='zh'
                    )["score"]
    score_rouge = rouge.compute(
                        predictions=predictions,
                        references=references,
                        tokenizer=lambda x: tokenizer.tokenize(x)
                    )
    result["R1"] = score_rouge["rouge1"]*100
    result["R2"] = score_rouge["rouge2"]*100
    result["RL"] = score_rouge["rougeL"]*100
    print('Rouge done!')
    score_bert = bertscore.compute(
                        predictions=predictions, 
                        references=references, 
                        lang="zh"
                    )
    result["bertscore"] = np.mean(score_bert["f1"])*100

    print(result)
    return result

def post_selection(preds, ans):
    selected = []
    pred_len = len(preds)
    for i in range(len(preds[0])):
        unselected = []
        for j in range(pred_len):
            unselected.append(preds[j][i])
        ref = ans[i]
        scores = [sacrebleu.compute(
                        predictions=[x],
                        references=[ref],
                        tokenize='zh'
                    )["score"] for x in unselected
                    ]
        max_score = 0
        max_idx = 0
        for j in range(pred_len):
            if scores[j]>max_score:
                max_score = scores[j]
                max_idx = j
        
        selected.append(preds[max_idx][i])
    
    return selected
