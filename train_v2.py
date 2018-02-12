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

# placeholder for noise variable to be passed into generator
noise = tf.placeholder(tf.float32, (None, 4096))

# answer placeholrder
answers_true = tf.placeholder(tf.float32, (None, 3000))
answers_false = tf.placeholder(tf.float32, (None, 3000))


#cnn_out, images_in = get_resnet(sess)

cnn_out_true = tf.placeholder(tf.float32, (None, 2048))
cnn_out_false = tf.placeholder(tf.float32, (None, 2048))

#cnn_out_true = cnn_out[1:50,:,:,:]
#cnn_out_false = cnn_out[51:100,:,:,:]

# true inputs

# load resnet
#with tf.variable_scope('resnet1'):
#   images_true, cnn_out_true = get_resnet(sess)
#with tf.variable_scope('resnet2'):
#   images_false, cnn_out_false = get_resnet(sess)


# start session
#sess = tf.Session()

load_weights = True
save_ver = '2.1'
import_weight_file = 'weights_v2.npy'

if load_weights:
   weights = np.load(import_weight_file).item()

   var_dict = {
      'gfcW1': tf.Variable(weights['gfcW1']),
      'gfcb1': tf.Variable(weights['gfcb1']),
      'gfcW2': tf.Variable(weights['gfcW2']),
      'gfcb2': tf.Variable(weights['gfcb2']),
      'gfcW3': tf.Variable(weights['gfcW3']),
      'gfcb3': tf.Variable(weights['gfcb3']),
      'gfcW4': tf.Variable(weights['gfcW4']),
      'gfcb4': tf.Variable(weights['gfcb4']),
      'dfcW1': tf.Variable(weights['dfcW1']),
      'dfcb1': tf.Variable(weights['dfcb1']),
      'dbeta1': tf.Variable(weights['dbeta1']),
      'dscale1': tf.Variable(weights['dscale1']),
      'dfcW2': tf.Variable(weights['dfcW2']),
      'dfcb2': tf.Variable(weights['dfcb2']),
      'dbeta2': tf.Variable(weights['dbeta2']),
      'dscale2': tf.Variable(weights['dscale2']),
      'dfcW3': tf.Variable(weights['dfcW3']),
      'dfcb3': tf.Variable(weights['dfcb3']),
      'dbeta3': tf.Variable(weights['dbeta3']),
      'dscale3': tf.Variable(weights['dscale3']),
      'dfcW4': tf.Variable(weights['dfcW4']),
      'dfcb4': tf.Variable(weights['dfcb4']),
      'dbeta4': tf.Variable(weights['dbeta4']),
      'dscale4': tf.Variable(weights['dscale4']),
      'dfcW5': tf.Variable(weights['dfcW5']),
      'dfcb5': tf.Variable(weights['dfcb5']),
      'dbeta5': tf.Variable(weights['dbeta5']),
      'dscale5': tf.Variable(weights['dscale5']),
      'dfcW6': tf.Variable(weights['dfcW6']),
      'dfcb6': tf.Variable(weights['dfcb6']),
      'dbeta6': tf.Variable(weights['dbeta6']),
      'dscale6': tf.Variable(weights['dscale6']),
      'dfcW7': tf.Variable(weights['dfcW7']),
      'dfcb7': tf.Variable(weights['dfcb7']),
      'dbeta7': tf.Variable(weights['dbeta7']),
      'dscale7': tf.Variable(weights['dscale7']),
      'rnnqW': tf.Variable(weights['rnnqW'], name='embed_ques_W'),
      'rnnsW': tf.Variable(weights['rnnsW'], name='embed_state_W'),
      'rnnsb': tf.Variable(weights['rnnsb'], name='embed_state_b'),
      'rnnoutbeta': tf.Variable(weights['rnnoutbeta']),
      'rnnoutscale': tf.Variable(weights['rnnoutscale']),
      'cnnoutbeta': tf.Variable(weights['cnnoutbeta']),
      'cnnoutscale': tf.Variable(weights['cnnoutscale']),
      'featbeta': tf.Variable(weights['featbeta']),
      'featscale': tf.Variable(weights['featscale']),
      'gbeta': tf.Variable(weights['gbeta']),
      'gscale': tf.Variable(weights['gscale'])
   }

   #batch_no = weights['batch_no']
   batch_no = 0
else:
   var_dict = {
      'gfcW1': tf.Variable(tf.truncated_normal([4096*2, 4096*2])),
      'gfcb1': tf.Variable(tf.constant(0.1, shape=[4096*2])),
      'gfcW2': tf.Variable(tf.truncated_normal([4096*2, 4096*1])),
      'gfcb2': tf.Variable(tf.constant(0.1, shape=[4096*1])),
      'gfcW3': tf.Variable(tf.truncated_normal([4096*1, 3500])),
      'gfcb3': tf.Variable(tf.constant(0.1, shape=[3500])),
      'gfcW4': tf.Variable(tf.truncated_normal([3500, 3000])),
      'gfcb4': tf.Variable(tf.constant(0.1, shape=[3000])),
      #'gfcW5': tf.Variable(tf.truncated_normal([4096*2, 4096])),
      #'gfcb5': tf.Variable(tf.constant(0.1, shape=[4096])),
      #'gfcW6': tf.Variable(tf.truncated_normal([4096, 3000])),
      #'gfcb6': tf.Variable(tf.constant(0.1, shape=[3000])),
      'dfcW1': tf.Variable(tf.truncated_normal([7096, 3548])),
      'dfcb1': tf.Variable(tf.constant(0.1, shape=[3548])),
      'dbeta1': tf.Variable(tf.zeros([3548])),
      'dscale1': tf.Variable(tf.ones([3548])),
      'dfcW2': tf.Variable(tf.truncated_normal([3548, 1774])),
      'dfcb2': tf.Variable(tf.constant(0.1, shape=[1774])),
      'dbeta2': tf.Variable(tf.zeros([1774])),
      'dscale2': tf.Variable(tf.ones([1774])),
      'dfcW3': tf.Variable(tf.truncated_normal([1774, 1000])),
      'dfcb3': tf.Variable(tf.constant(0.1, shape=[1000])),
      'dbeta3': tf.Variable(tf.zeros([1000])),
      'dscale3': tf.Variable(tf.ones([1000])),
      'dfcW4': tf.Variable(tf.truncated_normal([1000, 500])),
      'dfcb4': tf.Variable(tf.constant(0.1, shape=[500])),
      'dbeta4': tf.Variable(tf.zeros([500])),
      'dscale4': tf.Variable(tf.ones([500])),
      'dfcW5': tf.Variable(tf.truncated_normal([500,100])),
      'dfcb5': tf.Variable(tf.constant(0.1, shape=[100])),
      'dbeta5': tf.Variable(tf.zeros([100])),
      'dscale5': tf.Variable(tf.ones([100])),
      'dfcW6': tf.Variable(tf.truncated_normal([100, 10])),
      'dfcb6': tf.Variable(tf.constant(0.1, shape=[10])),
      'dbeta6': tf.Variable(tf.zeros([10])),
      'dscale6': tf.Variable(tf.ones([10])),
      'dfcW7': tf.Variable(tf.truncated_normal([10, 1])),
      'dfcb7': tf.Variable(tf.constant(0.1, shape=[1])),
      'dbeta7': tf.Variable(tf.zeros([0])),
      'dscale7': tf.Variable(tf.ones([1])),
      'rnnqW': tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08), name='embed_ques_W'),
      'rnnsW': tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, dim_hidden], -0.08,0.08),name='embed_state_W'),
      'rnnsb': tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_state_b'),
      'rnnoutbeta': tf.Variable(tf.zeros([2048])),
      'rnnoutscale': tf.Variable(tf.ones([2048])),
      'cnnoutbeta': tf.Variable(tf.zeros([2048])),
      'cnnoutscale': tf.Variable(tf.ones([2048])),
      'featbeta': tf.Variable(tf.zeros([4096])),
      'featscale': tf.Variable(tf.ones([4096])),
      'gbeta': tf.Variable(tf.zeros([3000])),
      'gscale': tf.Variable(tf.ones([3000]))
   }


# load nlp portion
with tf.variable_scope("rnn_module1"):
   rnn_out_true, questions_true = rnn_module(var_dict)  
with tf.variable_scope("rnn_module2"): 
   rnn_out_false, questions_false = rnn_module(var_dict)



cnn_mean, cnn_var = tf.nn.moments(cnn_out_true, [0])
cnn_out_true_n = tf.nn.batch_normalization(cnn_out_true,cnn_mean,cnn_var,var_dict['cnnoutbeta'],
   var_dict['cnnoutscale'],epsilon)
cnn_mean, cnn_var = tf.nn.moments(cnn_out_false, [0])
cnn_out_false_n = tf.nn.batch_normalization(cnn_out_false,cnn_mean,cnn_var,var_dict['cnnoutbeta'],
   var_dict['cnnoutscale'],epsilon)

rnn_mean, rnn_var = tf.nn.moments(rnn_out_true, [0])
rnn_out_true_n = tf.nn.batch_normalization(rnn_out_true,rnn_mean,rnn_var,var_dict['rnnoutbeta'],
   var_dict['rnnoutscale'],epsilon)
rnn_mean, rnn_var = tf.nn.moments(rnn_out_false, [0])
rnn_out_false_n = tf.nn.batch_normalization(rnn_out_false,rnn_mean,rnn_var,var_dict['rnnoutbeta'],
   var_dict['rnnoutscale'],epsilon)

#cnn_out_true_norm = tf.nn.local_response_normalization(cnn_out_true)
#cnn_out_false_norm = tf.nn.local_response_normalization(cnn_out_false)

#rnn_out_true_norm = tf.nn.local_response_normalization(rnn_out_true)
#rnn_out_false_norm = tf.nn.local_response_normalization(rnn_out_false)

# combine features from image and question
features_true = combine_embeddings(cnn_out_true_n, rnn_out_true_n)
features_false = combine_embeddings(cnn_out_false_n, rnn_out_false_n)

feat_mean, feat_var = tf.nn.moments(features_true, [0])
features_true_n = tf.nn.batch_normalization(features_true,feat_mean,feat_var,var_dict['featbeta'],
   var_dict['featscale'],epsilon)
feat_mean, feat_var = tf.nn.moments(features_false, [0])
features_false_n = tf.nn.batch_normalization(features_false,feat_mean,feat_var,var_dict['featbeta'],
   var_dict['featscale'],epsilon)

# load generator network
g_true = generator_me(features_true_n, noise, var_dict)

#g_true_norm = tf.nn.local_response_normalization(g_true)
g_mean, g_var = tf.nn.moments(g_true, [0])
g_true_n = tf.nn.batch_normalization(g_true,g_mean,g_var,var_dict['gbeta'],var_dict['gscale'],epsilon)
g_mean, g_var = tf.nn.moments(answers_true, [0])
answers_true_n = tf.nn.batch_normalization(answers_true,g_mean,g_var,var_dict['gbeta'],var_dict['gscale'],epsilon)
g_mean, g_var = tf.nn.moments(answers_false, [0])
answers_false_n = tf.nn.batch_normalization(answers_false,g_mean,g_var,var_dict['gbeta'],var_dict['gscale'],epsilon)
#answers_true_norm = tf.contrib.layers.batch_norm(answers_true)
#answers_false_norm = tf.contrib.layers.batch_norm(answers_false)

# load discriminator network
s_r, r_mean, r_var, fc7, fc7n = discriminator(features_true_n, answers_true_n, var_dict)
s_w, w_mean, w_var, fc7, fc7n = discriminator(features_false_n, answers_true_n, var_dict)
s_f, f_mean, f_var, fc7, fc7n = discriminator(features_true_n, g_true_n, var_dict) #g_true

ones = tf.constant(1.0, shape=[50,1], dtype=tf.float32)
loss = -tf.reduce_mean(tf.log(s_r) + tf.log(tf.subtract(ones,s_w))/2.0 + tf.log(tf.subtract(ones,s_f))/2.0)
l1 = tf.log(s_r)
#print(ones.get_shape())
#print(s_w.get_shape())
l2 = tf.subtract(ones,s_w)
l3 = tf.log(tf.subtract(ones,s_f))

#loss = tf.reduce_mean(tf.subtract(g_true,answers_true))

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
while batch_no*batch_size < len(qa_data['training']):
   print('batch = ' + str(batch_no))
   (questions_in_true, answer_in_true, im_feat_true) = get_training_batch(batch_no, batch_size, qa_data)

   if batch_no*batch_size < len(qa_data['training']) - 1:
      (questions_in_false, answer_in_false, im_feat_false) = get_training_batch(batch_no+1, batch_size, qa_data)
   else:
      (questions_in_false, answer_in_false, im_feat_false) = get_training_batch(0, batch_size, qa_data)

   noise_in = np.random.normal(size=[batch_size,4096])

   train_step.run(feed_dict={
      noise: noise_in, 
      answers_true: answer_in_true,
      answers_false: answer_in_false,
      cnn_out_true: im_feat_true,
      cnn_out_false: im_feat_false,
      questions_true: questions_in_true,
      questions_false: questions_in_false
   })

   loss_val = sess.run(loss, feed_dict={
      noise: noise_in, 
      answers_true: answer_in_true,
      answers_false: answer_in_false,
      cnn_out_true: im_feat_true,
      cnn_out_false: im_feat_false,
      questions_true: questions_in_true,
      questions_false: questions_in_false
   })

   g_out = sess.run(g_true, feed_dict={
      noise: noise_in, 
      answers_true: answer_in_true,
      answers_false: answer_in_false,
      cnn_out_true: im_feat_true,
      cnn_out_false: im_feat_false,
      questions_true: questions_in_true,
      questions_false: questions_in_false
   })

   print('loss = ' + str(loss_val))
   loss_vals.append(loss_val)
   np.save('loss_vals_v' + save_ver, loss_vals)
   print(loss_val)

   answers_out = np.argmax(g_out, axis=1)
   answers_idx_true = np.argmax(answer_in_true, axis=1)
   error = float(np.sum(answers_out == answers_idx_true)) / float(batch_size)
   acc_vals.append(error)
   np.save('acc_vals_v' + save_ver, acc_vals)
   print('error = ' + str(error))

   if batch_no % 25 == 0:
      weights_save = {}
      for key in var_dict:
         weights_save[key] = var_dict[key].eval()
      weights_save['batch_no'] = batch_no
      np.save('weights_v' + save_ver, weights_save)

   batch_no += 1

  