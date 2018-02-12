import sys
sys.path.insert(0, './resnet')
import os
import numpy as np
import json
from evaluation_data import *

qa_data, answers_vocab_inv = load_data3()
qid = [262262001,393225001]
imid = [262262,393225]
answers = json.load(open('./Results/d_v3.1_full.json'))
#print(answers[1])
#quit()

print('--------- computed answers -------')
for i in range(len(qid)):
   for j in range(len(answers)):
      if qid[i] == answers[j]['question_id']:
         print(qid[i])
         print(answers[j])
         break


print('--------- original data -------')
for i in range(len(qid)):
   for j in range(len(qa_data['validation'])):
      if qid[i] == qa_data['validation'][j]['questions_id']:
         print(qa_data['validation'][j])
         print(answers_vocab_inv[qa_data['validation'][j]['answer']])
         break


for i in range(len(imid)):
   imageid = str(imid[i])
   while len(imageid) < 6:
      imageid = '0' + imageid
   os.system('cp ./Images/mscoco/val2014/COCO_val2014_000000' + imageid \
      + '.jpg ./pres_images/COCO_val2014_000000' + imageid + '.jpg')
      