import sys
sys.path.append('/home/zhangzekai/NLPDL_final')
from utils.utils import *
from transformers import set_seed
import random

set_seed(0)

train_data = read_jsonl('/home/zhangzekai/NLPDL_final/data/qg_data/train.jsonl')
context = [x[2] for x in train_data]

# 7 types of typical questions
p1,p2,p3,p4,p5,p6,p7 = [],[],[],[],[],[],[]


def split_and_match(sent):
    # first split the sentence
    # match signal words; if match, add to the corresponding pseudo_q list
    pieces = [sent]
    splitted = []
    tem = []

    # split
    for sep in ['，','。','！','？','：','；',' ',',','.']:
        tem = []
        for piece in pieces:
            tem.extend(piece.split(sep))
        pieces = tem
    splitted = tem 

    # match & add to list
    for piece in splitted:
        if '不可以' in piece:
            p1.append([piece.replace("不可以","可以")+"吗？","不可以",sent])
        elif '可以' in piece:
            p2.append([piece+"吗？","可以",sent])
        elif '不需要' in piece:
            p3.append([piece.replace("不需要","需要")+"吗？","不需要",sent])
        elif '需要' in piece:
            p4.append([piece+"吗？","需要",sent])
        elif '不要' in piece:
            p5.append([piece.replace("不要","要")+"吗？","不要",sent])
        elif '要' in piece:
            p6.append([piece+"吗？","要",sent])
        elif '不是' in piece:
            p7.append([piece.replace("不是","是")+"吗？","不是",sent])
        
for sent in context:
    split_and_match(sent)

pseudo_q = []
pseudo_q += random.sample(p1,50)
pseudo_q += random.sample(p2,50)
pseudo_q += random.sample(p3,50)
pseudo_q += random.sample(p4,50)
pseudo_q += random.sample(p5,50)
pseudo_q += random.sample(p6,50)
pseudo_q += random.sample(p7,50)

# write_list(pseudo_q,'/home/zhangzekai/NLPDL_final/data/pseudo_q.pkl')










