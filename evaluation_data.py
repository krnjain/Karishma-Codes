import sys
import os
import numpy as np
import pickle
import json


IM_DIM = 224
MAX_QUESTION_LENGTH = 22


def get_validation_batch(batch_no, batch_size, qa_data):
   qa = qa_data['validation']

   si = (batch_no * batch_size)%len(qa)
   ei = min(len(qa), si + batch_size)
   n = ei - si
   questions = np.zeros([batch_size, MAX_QUESTION_LENGTH])
   question_ids = []
   im_ids = []
   count = 0
   for i in range(si, ei):
     questions[count, :] = qa[i]['question']
     im_ids.append(qa[i]['image_id'])
     question_ids.append(qa[i]['questions_id'])

     count += 1

   im_feats = image_features_validation(im_ids)
   return (questions, im_feats, im_ids, question_ids)


def get_validation_batch_ans(batch_no, batch_size, qa_data):
   qa = qa_data['validation']

   si = (batch_no * batch_size)%len(qa)
   ei = min(len(qa), si + batch_size)
   n = ei - si
   questions = np.zeros([batch_size, MAX_QUESTION_LENGTH])
   question_ids = []
   question_text = []
   im_ids = []
   answer = np.zeros( (n, len(qa_data['answer_vocab'])))
   count = 0
   for i in range(si, ei):
     questions[count, :] = qa[i]['question']
     im_ids.append(qa[i]['image_id'])
     answer[count, qa[i]['answer']] = 1.0
     question_ids.append(qa[i]['questions_id'])
     question_text.append(qa[i]['question_text'])

     count += 1

   im_feats = image_features_validation(im_ids)
   return (questions, im_feats, im_ids, question_ids, answer, question_text)



def image_features_validation(im_ids):
    split='val'

    for i in range(len(im_ids)):
        temp = str(im_ids[i])
        while len(temp) < 6:
            temp = '0' + temp
        im_ids[i] = temp

    features = np.zeros([len(im_ids), 2048])
    im_dir = os.listdir('./im_feat_indiv/' + split + '/')

    for i in range(len(im_ids)):
        filename = 'COCO_' + split + '2014_000000' + im_ids[i] + '.npy'
        feat = np.load('./im_feat_indiv/' + split + '/' + filename)
        features[i,:] = feat
    
    return features


def load_data3():
    #if isfile(qa_data_file):
   with open('../neural-vqa-tensorflow/Data/qa_data_file_1000_qid.pkl') as f:
      data = pickle.load(f)

   answer_vocab = data['answer_vocab']
   answer_vocab_inv = {}
   for key in answer_vocab:
      answer_vocab_inv[answer_vocab[key]] = key

   return data, answer_vocab_inv


def create_ans_out(answers_out, answers_vocab_inv, question_ids, answers_idx):
   for i in range(len(answers_idx)):
      ans_i = {}
      ans_i['answer'] = answers_vocab_inv[answers_idx[i]]
      ans_i['answer_idx'] = answers_idx[i]
      ans_i['question_id'] = question_ids[i]
      answers_out.append(ans_i)
   return answers_out


def generate_json(answers_out, filename):
   with open(filename + '.json', 'w') as outfile:
      json.dump(answers_out, outfile)


def convert_validation_answers():
  #data = json.load(open(filename + '.json'))
  questions = json.load(open('./Questions/OpenEnded_mscoco_val2017_questions.json'))

  data = json.load(open('./Results/d_simple_nonorm.json'))
  data2 = json.load(open('./Results/eval_d_3.2_unifinit.json'))
  data3 = json.load(open('./Results/g_v3.2_unifinit.json'))

  qid = []
  print('generating qid...')
  for i in range(len(data)):
    qid.append(data[i]['question_id'])

  print('getting unrecorded answers...')
  count = 0
  for i in range(len(questions['questions'])):
    if questions['questions'][i]['question_id'] not in qid:
      temp = {}
      temp['question_id'] = questions['questions'][i]['question_id']
      temp['answer_idx'] = -1
      temp['answer'] = ''
      data.append(temp)
      data2.append(temp)
      data3.append(temp)
      count += 1
      print(count)

  generate_json(data, './Results/d_simple_nonorm_full')
  generate_json(data2, './Results/eval_d_3.2_unifinit_full')
  generate_json(data3, './Results/g_v3.2_unifinit_full')

#print('v10')
# load all questions
#qa_data = load_data2()
'''
 qa_data['validation'][1]
{'answer': 487, 'image_id': 262148, 'question_text': u'What are the people in the background doing?', 'question': array([     0.,      0.,      0.,      0.,      0.,      0.,      0.,
            0.,      0.,      0.,      0.,      0.,      0.,      0.,
        13114.,  14150.,   4623.,    798.,   8381.,   4623.,   8481.,
         3315.])}
'''


# err_cnt = 0

# for i in range(len(qa_data['training'])):
#    if qa_data['training'][i]['answer'] != qa_data2['training'][i]['answer'] or \
#       qa_data['training'][i]['image_id'] != qa_data2['training'][i]['image_id'] or \
#       qa_data['training'][i]['question_text'] != qa_data2['training'][i]['question_text'] or \
#       np.linalg.norm(qa_data['training'][i]['question'] - qa_data2['training'][i]['question']) > 1e-8:
#       print('error')
#       err_cnt += 1
#    print(str(i) + '/' + str(len(qa_data['training'])))

# for i in range(len(qa_data['validation'])):
#    if qa_data['validation'][i]['answer'] != qa_data2['validation'][i]['answer'] or \
#       qa_data['validation'][i]['image_id'] != qa_data2['validation'][i]['image_id'] or \
#       qa_data['validation'][i]['question_text'] != qa_data2['validation'][i]['question_text'] or \
#       np.linalg.norm(qa_data['validation'][i]['question'] - qa_data2['validation'][i]['question']) > 1e-8:
#       print('error')
#       err_cnt += 1
#    print(str(i) + '/' + str(len(qa_data['validation'])))
#    # print('------ training ------')
#    # print(qa_data['training'][i])
#    # print(qa_data2['training'][i])
#    # print('------ validation ------')
#    # print(qa_data['validation'][i])
#    # print(qa_data2['validation'][i])
# print(err_cnt)
# quit()
#f = open('./processed_data/vocab_file_1000.pkl')
#vocab = pickle.load(f)
#answer_vocab = vocab['answer_vocab']


# qa_data, answers_vocab_inv = load_data3()

# batch_no = 0
# batch_size = 50
# answers_out = []


# (questions_in, im_feat, im_ids, question_ids) = get_validation_batch(batch_no, batch_size, qa_data)

# ''' 
# run questions_in, im_feat through network
# '''
# g_out = np.random.normal(size=[batch_size,1000])
# answers_idx = np.argmax(g_out, axis=1)

# answers_out = create_ans_out(answers_out, answers_vocab_inv, question_ids, answers_idx)

# print(answers_out)

# generate_json(answers_out, 'test2')

'''
answer json file should have format:
[{"answer": "3", "question_id": 1365590}, ...]
'''