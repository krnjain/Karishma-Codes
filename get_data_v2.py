import numpy as np
from os.path import isfile
import os
import pickle
from scipy.misc import imread, imresize, imsave

IM_DIM = 224
MAX_QUESTION_LENGTH = 22
qa_data_file = './processed_data/qa_data_file.pkl'
qa_data_file2 = './processed_data/qa_data_file_1000.pkl'


def get_training_batch(batch_no, batch_size, qa_data, split='train'):
    qa = None
    if split == 'train':
        qa = qa_data['training']
    else:
        qa = qa_data['validation']

    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    # sentence = []
    questions = np.zeros([batch_size, MAX_QUESTION_LENGTH])
    answer = np.zeros( (n, len(qa_data['answer_vocab'])))

    im_ids = []
    count = 0
    for i in range(si, ei):
        # sentence.append(qa[i]['question'])
        questions[count, :] = qa[i]['question']
        answer[count, qa[i]['answer']] = 1.0
        im_ids.append(qa[i]['image_id'])
        #images[count,:,:,:] = load_image(qa[i]['image_id'], split)

        count += 1

    im_feats = image_features(im_ids)
    return (questions, answer, im_feats)


def image_features(im_ids):
    split='train'

    for i in range(len(im_ids)):
        temp = str(im_ids[i])
        while len(temp) < 6:
            temp = '0' + temp
        im_ids[i] = temp

    features = np.zeros([len(im_ids), 2048])
    im_dir = os.listdir('./im_feat_indiv/' + split + '/')

    not_found = []
    for i in range(len(im_ids)):
        filename = 'COCO_' + split + '2014_000000' + im_ids[i] + '.npy'
        if filename not in im_dir:
            not_found.append(filename)

    not_found = set(not_found)
    not_found = list(not_found)
    print(not_found)

    if len(not_found) > 0:
        command_string = 'python generate_feat_indiv.py'
        for i in range(len(not_found)):
            command_string = command_string + ' ' + not_found[i]
        print(command_string)
        os.system(command_string)

    for i in range(len(im_ids)):
        filename = 'COCO_' + split + '2014_000000' + im_ids[i] + '.npy'
        feat = np.load('./im_feat_indiv/' + split + '/' + filename)
        features[i,:] = feat
    
    return features


def load_image(im_id, split='train'):
    im_id = str(im_id)
    while len(im_id) < 6:
        im_id = '0' + im_id
    
    if split=='train':
        path = './Images/mscoco/train2014/'
        im = imread(path + 'COCO_train2014_000000' + str(im_id) + '.jpg').astype(np.float32)
    else:
        path = './Images/mscoco/test2014/'
        im = imread(path + 'COCO_val2014_000000' + str(im_id) + '.jpg').astype(np.float32)

    if len(im.shape) < 3:
        im = imresize(im, [IM_DIM, IM_DIM])
        im2 = np.zeros([IM_DIM, IM_DIM, 3])
        im2[:,:,0] = im
        im2[:,:,1] = im
        im2[:,:,2] = im
        im = im2

    im = imresize(im, [IM_DIM, IM_DIM, 3])
    im = im - np.mean(im)
    return im



def load_data():
    if isfile(qa_data_file):
        with open(qa_data_file) as f:
            data = pickle.load(f)
            return data


def load_data2():
    if isfile(qa_data_file):
        with open(qa_data_file2) as f:
            data = pickle.load(f)
            return data





# qa_data = load_data()
# batch_no = 22
# batch_size = 50
# while batch_no*batch_size < len(qa_data['training']):
#    print('batch = ' + str(batch_no))
#    (questions, answer, image) = get_training_batch(batch_no, batch_size, qa_data)
#    batch_no += 1

#answertext = qa_data['answer_vocab']
#print(sentence)
#for i in range(10):
#    print('image ' + str(i))
#    for key in answertext:
#        if answer[i,answertext[key]] != 0:
#            print(key)
#    
#    imsave(str(i) + '.png', np.reshape(image[i,:,:,:],[224,224,3]))