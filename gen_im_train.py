import sys
sys.path.insert(0, './resnet')
import os
import tensorflow as tf
import numpy as np
#from resnet152 import get_resnet
# from convert2 import load_image
from generator import combine_embeddings, generator_me
from discriminator import discriminator
from rnn_module import rnn_module
from get_data_v2 import load_data, get_training_batch


######## CONSTANTS #######
loss_vals = []
acc_vals = []
vocabulary_size = 15881
input_embedding_size = 200
rnn_size = 512
rnn_layer = 2
dim_hidden = 2048
epsilon = 1e-3
##########################

sess = tf.InteractiveSession()


print('loading data...\n\n')
qa_data = load_data()
print('done loading data...\n\n')
#batch_no = 0
batch_size = 50
batch_no = 6400
while batch_no*batch_size < len(qa_data['training']):
   print('batch = ' + str(batch_no))
   (questions_in_true, answer_in_true, im_feat_true) = get_training_batch(batch_no, batch_size, qa_data)
   batch_no += 1