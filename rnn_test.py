#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
#import ipdb
import time
import math
#import cv2
import codecs, json
import tensorflow.contrib.rnn as rnn_cell
from sklearn.metrics import average_precision_score
from get_data import * 
#from tensorflow.nn import rnn_cell
#import tensorflow as tf

#####################################################
#                 Global Parameters		    #  
#####################################################
print 'Loading parameters ...'
# Data input setting
input_img_h5 = './data_img.h5'
input_ques_h5 = './data_prepro.h5'
input_json = './data_prepro.json'

# Train Parameters setting
learning_rate = 0.0003			# learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1		# at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 500			# batch_size for each iterations
input_embedding_size = 200		# he encoding size of each token in the vocabulary
rnn_size = 512				# size of the rnn in number of hidden nodes in each layer
rnn_layer = 2				# number of the rnn layer
dim_image = 4096
dim_hidden = 1024 #1024			# size of the common embedding vector
num_output = 1000			# number of output answers
img_norm = 1				# normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083

# Check point
checkpoint_path = 'model_save/'

# misc
gpu_id = 0
max_itr = 150000
n_epochs = 300
max_words_q = 26
num_answer = 1000
#####################################################

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_data():

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    # print('loading image feature...')
    # with h5py.File(input_img_h5,'r') as hf:
    #     # -----0~82459------
    #     tem = hf.get('images_train')
    #     img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
	# convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('question aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    print('Normalizing image feature')
    #if img_norm:
    #    tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
    #    img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, train_data


def rnn_test():
    batch_size = 50
    drop_out_rate = 0.2
    input_embedding_size = 200   # the encoding size of each token in the vocabulary
    vocabulary_size = 15881        
    rnn_size = 512               # size of the rnn in number of hidden nodes in each layer
    rnn_layer = 2                # number of the rnn layer
    max_words_q = 22             # maximum possible number of words in question?
    dim_hidden = 2048            # size of the common embedding vector

    embed_ques_W = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08), name='embed_ques_W')

    embed_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, dim_hidden], -0.08,0.08),name='embed_state_W')
    embed_state_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_state_b')

    lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True, state_is_tuple=False)
    lstm_dropout_1 = rnn_cell.DropoutWrapper(lstm_1, output_keep_prob=1-drop_out_rate)
    lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True, state_is_tuple=False)
    lstm_dropout_2 = rnn_cell.DropoutWrapper(lstm_2, output_keep_prob=1-drop_out_rate)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm_dropout_1, lstm_dropout_2], state_is_tuple=False)

    question = tf.placeholder(tf.int32, [batch_size, max_words_q])

    state = stacked_lstm.zero_state(batch_size, tf.float32)
    loss = 0.0
    for i in range(max_words_q):
        if i==0:
            ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
        else:
            tf.get_variable_scope().reuse_variables()
            ques_emb_linear = tf.nn.embedding_lookup(embed_ques_W, question[:,i-1])

        ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-drop_out_rate)
        ques_emb = tf.tanh(ques_emb_drop)

        output, state = stacked_lstm(ques_emb, state)

    #state_drop = tf.nn.dropout(state, 1-drop_out_rate)
    state_linear = tf.nn.xw_plus_b(state, embed_state_W, embed_state_b)
    # state_emb = tf.tanh(state_linear)
    state_emb = tf.nn.relu(state_linear)

    return state_emb, question



def train():
    print 'loading dataset...'
    #dataset, train_data = get_data()
    #num_train = train_data['question'].shape[0]


    #index = np.random.random_integers(0, num_train-1, batch_size)

    #current_question = train_data['question'][index,:]
    #current_length_q = train_data['length_q'][index]
    #current_answers = train_data['answers'][index]
    #current_img_list = train_data['img_list'][index]
    #vocabulary_size = len(dataset['ix_to_word'].keys())

    #print('current question')
    #print(current_question)
    #print(current_question.shape)
    #print(current_question[1,:])
    #print 'vocabulary_size : ' + str(vocabulary_size)

    sess = tf.Session()


    state_emb, question = rnn_test()
    sess.run(tf.global_variables_initializer())
    qa_data = load_data()
    batch_no = 0
    batch_size = 50
    (questions, answer, image) = get_training_batch(batch_no, batch_size, qa_data)
    test = sess.run(state_emb, feed_dict={question: questions})
    print(test)
    print(test.shape)




if __name__ == '__main__':
    #with tf.device('/gpu:'+str(0)):
    train()
    #with tf.device('/gpu:'+str(1)):
        #test()
    
