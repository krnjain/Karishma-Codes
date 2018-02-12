import sys
sys.path.insert(0, './resnet')
import os
import tensorflow as tf
import numpy as np
#from resnet152 import get_resnet
# from convert2 import load_image
from generator_v3 import combine_embeddings, generator_me
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

load_weights = False
save_ver = '2_nonoise'
import_weight_file = 'weights_g_v1.npy'

if load_weights:
   weights = np.load(import_weight_file).item()

   var_dict = {
      'cefcW1': tf.Variable(tf.truncated_normal([4096,4096])),
      'cefcb1': tf.Variable(tf.constant(0.1, shape=[4096])),
      'gfcW1': tf.Variable(weights['gfcW1']),
      'gfcb1': tf.Variable(weights['gfcb1']),
      'gfcW2': tf.Variable(weights['gfcW2']),
      'gfcb2': tf.Variable(weights['gfcb2']),
      'gfcW3': tf.Variable(weights['gfcW3']),
      'gfcb3': tf.Variable(weights['gfcb3']),
      'gfcW4': tf.Variable(weights['gfcW4']),
      'gfcb4': tf.Variable(weights['gfcb4']),
      'rnnqW': tf.Variable(weights['rnnqW'], name='embed_ques_W'),
      'rnnsW': tf.Variable(weights['rnnsW'], name='embed_state_W'),
      'rnnsb': tf.Variable(weights['rnnsb'], name='embed_state_b'),
      'rnnoutbeta': tf.Variable(weights['rnnoutbeta']),
      'rnnoutscale': tf.Variable(weights['rnnoutscale']),
      'cnnoutbeta': tf.Variable(weights['cnnoutbeta']),
      'cnnoutscale': tf.Variable(weights['cnnoutscale']),
      'featbeta': tf.Variable(weights['featbeta']),
      'featscale': tf.Variable(weights['featscale']),
   }

   #batch_no = weights['batch_no']
   batch_no = 0
else:
   weights = np.load(import_weight_file).item()

   var_dict = {
      'cemcnnfcW1': tf.Variable(weights['cemcnnfcW1']),
      'cemcnnfcb1': tf.Variable(weights['cemcnnfcb1']),
      'ceacnnfcW1': tf.Variable(weights['ceacnnfcW1']),
      'ceacnnfcb1': tf.Variable(weights['ceacnnfcb1']),
      'cemrnnfcW1': tf.Variable(weights['cemrnnfcW1']),
      'cemrnnfcb1': tf.Variable(weights['cemrnnfcb1']),
      'cearnnfcW1': tf.Variable(weights['cearnnfcW1']),
      'cearnnfcb1': tf.Variable(weights['cearnnfcb1']),
      'gfcW1': tf.Variable(tf.truncated_normal([4096, 4096])),
      'gfcb1': tf.Variable(tf.constant(0.1, shape=[4096])),
      'gfcW2': tf.Variable(tf.truncated_normal([4096, 4096*1])),
      'gfcb2': tf.Variable(tf.constant(0.1, shape=[4096*1])),
      'gfcW3': tf.Variable(tf.truncated_normal([4096*1, 3500])),
      'gfcb3': tf.Variable(tf.constant(0.1, shape=[3500])),
      'gfcW4': tf.Variable(tf.truncated_normal([3500, 3000])),
      'gfcb4': tf.Variable(tf.constant(0.1, shape=[3000])),
      'rnnqW': tf.Variable(weights['rnnqW'], name='embed_ques_W'),
      'rnnsW': tf.Variable(weights['rnnsW'], name='embed_state_W'),
      'rnnsb': tf.Variable(weights['rnnsb'], name='embed_state_b'),
      'rnnoutbeta': tf.Variable(weights['rnnoutbeta']),
      'rnnoutscale': tf.Variable(weights['rnnoutscale']),
      'cnnoutbeta': tf.Variable(weights['cnnoutbeta']),
      'cnnoutscale': tf.Variable(weights['cnnoutscale']),
      'featbeta': tf.Variable(tf.zeros([4096])),
      'featscale': tf.Variable(tf.ones([4096])),
      'gbeta': tf.Variable(tf.zeros([3000])),
      'gscale': tf.Variable(tf.ones([3000]))
   }

   batch_no = 0


# placeholder for noise variable to be passed into generator
noise = tf.placeholder(tf.float32, (None, 4096))

# answer placeholrder
answers_true = tf.placeholder(tf.float32, (None, 3000))
cnn_out_true = tf.placeholder(tf.float32, (None, 2048))


# load nlp portion
with tf.variable_scope("rnn_module1"):
   rnn_out_true, questions_true = rnn_module(var_dict)  

cnn_mean, cnn_var = tf.nn.moments(cnn_out_true, [0])
cnn_out_true_n = tf.nn.batch_normalization(cnn_out_true,cnn_mean,cnn_var,var_dict['cnnoutbeta'],
   var_dict['cnnoutscale'],epsilon)

rnn_mean, rnn_var = tf.nn.moments(rnn_out_true, [0])
rnn_out_true_n = tf.nn.batch_normalization(rnn_out_true,rnn_mean,rnn_var,var_dict['rnnoutbeta'],
   var_dict['rnnoutscale'],epsilon)

# combine features from image and question
features_true = combine_embeddings(cnn_out_true_n, rnn_out_true_n, var_dict)

#features = tf.add(features_true, noise)
features = features_true
#feat_mean, feat_var = tf.nn.moments(features_true, [0])
#features_true_n = tf.nn.batch_normalization(features_true,feat_mean,feat_var,var_dict['featbeta'],
#   var_dict['featscale'],epsilon)

# load generator network
g_true = generator_me(features, var_dict)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=g_true, labels=answers_true))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
   train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# init variables
sess.run(tf.global_variables_initializer())




print('loading data...\n\n')
qa_data = load_data()
print('done loading data...\n\n')
#batch_no = 0
batch_size = 50
#while batch_no*batch_size < len(qa_data['training']):
while batch_no < 4000:
   print('batch = ' + str(batch_no))
   (questions_in_true, answer_in_true, im_feat_true) = get_training_batch(batch_no, batch_size, qa_data)

   noise_in = np.random.normal(scale=0.3, size=[batch_size,4096])

   train_step.run(feed_dict={
      noise: noise_in, 
      answers_true: answer_in_true,
      cnn_out_true: im_feat_true,
      questions_true: questions_in_true,
   })

   loss_val = sess.run(loss, feed_dict={
      noise: noise_in, 
      answers_true: answer_in_true,
      cnn_out_true: im_feat_true,
      questions_true: questions_in_true,
   })

   g_out = sess.run(g_true, feed_dict={     
      noise: noise_in, 
      answers_true: answer_in_true,
      cnn_out_true: im_feat_true,
      questions_true: questions_in_true,
   })

   print('loss = ' + str(loss_val))
   loss_vals.append(loss_val)
   np.save('loss_vals_g_v' + save_ver, loss_vals)
   #print(loss_val)

   answers_out = np.argmax(g_out, axis=1)
   answers_idx_true = np.argmax(answer_in_true, axis=1)
   error = float(np.sum(answers_out == answers_idx_true)) / float(batch_size)
   acc_vals.append(error)
   np.save('acc_vals_g_v' + save_ver, acc_vals)
   print('error = ' + str(error))

   if batch_no % 25 == 0:
      weights_save = {}
      for key in var_dict:
         weights_save[key] = var_dict[key].eval()
      weights_save['batch_no'] = batch_no
      np.save('weights_g_v' + save_ver, weights_save)

   batch_no += 1

  