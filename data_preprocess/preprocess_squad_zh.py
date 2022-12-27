import sys
sys.path.append('/home/zhangzekai/NLPDL_final')
from utils.utils import *
original_data = load_json('/home/zhangzekai/NLPDL_final/data/squad_zh.json')['data']
data=[]
for paper in original_data:
    for para in paper['paragraphs']:
        context = para['context']
        qas = para['qas']
        for qa in qas:
            # some is impossible to answer
            try:
                data.append([qa['question'],qa['answers'][0]['text'],context])
            except:
                pass

# with open('/home/zhangzekai/NLPDL_final/data/squad_zh.txt','w+') as f:
#     for dat in data:
#         f.write(dat[0])
#         f.write('&&')
#         f.write(dat[1])
#         f.write('&&')
#         f.write(dat[2])
#         f.write('\n')

