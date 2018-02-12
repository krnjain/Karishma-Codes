import sys
sys.path.insert(0, './resnet')
import os
import tensorflow as tf
import numpy as np
#from resnet152 import get_resnet
# from convert2 import load_image
#from generator import combine_embeddings, generator_me
#from discriminator import discriminator
#from rnn_module import rnn_module
#from get_data_v2 import load_data2, get_training_batch
import tensorflow.contrib.rnn as rnn_cell
from evaluation_data import *
import time


######## CONSTANTS #######
# loss_vals = []
# acc_vals = []
# vocabulary_size = 15881
# input_embedding_size = 200
# rnn_size = 512
# rnn_layer = 2
# dim_hidden = 2048
# epsilon = 1e-3
# n_hidden_1 = 3600 	# 1st layer num features
# n_hidden_2 = 3000	# 2nd layer num features
# n_input = 4096		# Vector data input (img shape: 28*28)



loss_vals = []
lossg_vals = []
acc_vals = []
epsilon = 1e-3

batch_size=50 
dropout_rate=0.2
drop_out_rate=dropout_rate

input_embedding_size=200
vocabulary_size=15881
rnn_size=512
rnn_layer=2
max_words_q=22
dim_hidden=1024
num_output = 1000
##########################

sess = tf.InteractiveSession()


save_ver = './Results/auto_out_v2'

# load nlp portion
#with tf.variable_scope("rnn_module1"):
#   rnn_out_true, questions_true = rnn_module(var_dict) 
   #print ('rnn_out_true ', rnn_out_true.get_shape())
   #print ('questions_true ', questions_true.get_shape())
#with tf.variable_scope("rnn_module2"): 
#   rnn_out_false, questions_false = rnn_module(var_dict)

with tf.variable_scope("rnn_module1"):
   # placeholder for noise variable to be passed into generator
   #noise = tf.placeholder(tf.float32, (None, 4096))

   # answer placeholrder
   #answers_true = tf.placeholder(tf.float32, (None, 1000))
   #answers_false = tf.placeholder(tf.float32, (None, 3000))


   #cnn_out, images_in = get_resnet(sess)

   cnn_out_true = tf.placeholder(tf.float32, (None, 2048))
   #cnn_out_false = tf.placeholder(tf.float32, (None, 2048))
   question = tf.placeholder(tf.int32, [batch_size, max_words_q])

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
   #save_ver = '2_out'
   import_weight_file = 'weights_auto.npy'

   if load_weights:
      #weights = np.load(import_weight_file).item()

      var_dict = {
         'rnnqW': tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08), name='embed_ques_W'),
         'encoder_h1': tf.Variable(tf.random_uniform([4096, 3500], -0.08, 0.08)),
         'encoder_h2': tf.Variable(tf.random_uniform([3500, 2500], -0.08, 0.08)),
         'decoder_h1': tf.Variable(tf.random_uniform([2500, 3500], -0.08, 0.08)),
         'decoder_h2': tf.Variable(tf.random_uniform([3500, 4096], -0.08, 0.08)),
         'encoder_b1': tf.Variable(tf.random_uniform([3500], -0.08, 0.08)),
         'encoder_b2': tf.Variable(tf.random_uniform([2500], -0.08, 0.08)),
         'decoder_b1': tf.Variable(tf.random_uniform([3500], -0.08, 0.08)),
         'decoder_b2': tf.Variable(tf.random_uniform([4096], -0.08, 0.08)),

      }

      #batch_no = weights['batch_no']
      batch_no = 0
   

   question = tf.placeholder(tf.int32, [batch_size, max_words_q])
   lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True, state_is_tuple=False)
   lstm_dropout_1 = rnn_cell.DropoutWrapper(lstm_1, output_keep_prob = 1 - dropout_rate)
   lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True, state_is_tuple=False)
   lstm_dropout_2 = rnn_cell.DropoutWrapper(lstm_2, output_keep_prob = 1 - dropout_rate)
   stacked_lstm = rnn_cell.MultiRNNCell([lstm_dropout_1, lstm_dropout_2], state_is_tuple=False)


   state = stacked_lstm.zero_state(batch_size, tf.float32)
   loss = 0.0
   for i in range(max_words_q):  
      if i==0:
         ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
      else:
         tf.get_variable_scope().reuse_variables()
         ques_emb_linear = tf.nn.embedding_lookup(var_dict['rnnqW'], question[:,i-1])
         #ques_emb_linear = tf.gather(var_dict['rnnqW'], question[:,i-1])

      ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-drop_out_rate)
      ques_emb = tf.tanh(ques_emb_drop)

      output, state = stacked_lstm(ques_emb, state)




   #cnn_mean, cnn_var = tf.nn.moments(cnn_out_true, [0])
   #cnn_out_true_n = tf.nn.batch_normalization(cnn_out_true,cnn_mean,cnn_var,var_dict['cnnoutbeta'],
   #   var_dict['cnnoutscale'],epsilon)
   cnn_out_true_n = cnn_out_true
   #print('cnn_out_true_n ', cnn_out_true_n.get_shape())
   #cnn_mean, cnn_var = tf.nn.moments(cnn_out_false, [0])
   #cnn_out_false_n = tf.nn.batch_normalization(cnn_out_false,cnn_mean,cnn_var,var_dict['cnnoutbeta'],
   #   var_dict['cnnoutscale'],epsilon)

   #rnn_mean, rnn_var = tf.nn.moments(rnn_out_true, [0])
   #rnn_out_true_n = tf.nn.batch_normalization(rnn_out_true,rnn_mean,rnn_var,var_dict['rnnoutbeta'],
   #   var_dict['rnnoutscale'],epsilon)
   rnn_out_true_n = state
   #print('rnn_out_true_n ', rnn_out_true_n.get_shape())
   #rnn_mean, rnn_var = tf.nn.moments(rnn_out_false, [0])
   #rnn_out_false_n = tf.nn.batch_normalization(rnn_out_false,rnn_mean,rnn_var,var_dict['rnnoutbeta'],
   #   var_dict['rnnoutscale'],epsilon)

   #cnn_out_true_norm = tf.nn.local_response_normalization(cnn_out_true)
   #cnn_out_false_norm = tf.nn.local_response_normalization(cnn_out_false)

   #rnn_out_true_norm = tf.nn.local_response_normalization(rnn_out_true)
   #rnn_out_false_norm = tf.nn.local_response_normalization(rnn_out_false)

   # combine features from image and question
   #features_true = combine_embeddings(cnn_out_true_n, rnn_out_true_n)
   #features_false = combine_embeddings(cnn_out_false_n, rnn_out_false_n)

   features_true = tf.concat([cnn_out_true_n, rnn_out_true_n],1)
   #print ('features_true ', features_true.get_shape())
   en_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(features_true, var_dict['encoder_h1']),var_dict['encoder_b1']))
   #print ('en_layer_1 ', en_layer_1.get_shape())
   en_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(en_layer_1, var_dict['encoder_h2']),var_dict['encoder_b2']))
   #print ('en_layer_2 ', en_layer_2.get_shape())
   #de_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(en_layer_2, var_dict['decoder_h1']),var_dict['decoder_b1']))
   #print ('de_layer_1 ', de_layer_1.get_shape())
       # Decoder Hidden layer with sigmoid activation #2
   #de_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(de_layer_1, var_dict['decoder_h2']),var_dict['decoder_b2']))
   #print ('de_layer_2 ', de_layer_2.get_shape())
   #y_pred = de_layer_2
   #y_true = features_true

   #saver = tf.train.Saver()
   #saver.restore(sess, './train_auto_v1')

   var_dict['mlp_h1'] = tf.Variable(tf.random_uniform([2500,1700], -0.1, 0.1))
   var_dict['mlp_b1'] = tf.Variable(tf.constant(0.1, shape=[1700]))
   var_dict['mlp_h2'] = tf.Variable(tf.random_uniform([1700,1000], -0.1, 0.1))
   var_dict['mlp_b2'] = tf.Variable(tf.constant(0.1, shape=[1000]))

   mlp_1 = tf.nn.relu(tf.add(tf.matmul(en_layer_2, var_dict['mlp_h1']), var_dict['mlp_b1']))
   scores_emb = tf.nn.relu(tf.add(tf.matmul(mlp_1, var_dict['mlp_h2']), var_dict['mlp_b2']))

#    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = mlp_2, labels =answers_true)

#    loss = tf.reduce_mean(cross_entropy)
#       #cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
      
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#    # init variables
#    #sess.run(tf.global_variables_initializer())

# uninitialized_vars = []
# for var in tf.all_variables():
#     try:
#         sess.run(var)
#     except tf.errors.FailedPreconditionError:
#         uninitialized_vars.append(var)

# init_new_vars_op = tf.initialize_variables(uninitialized_vars)
# sess.run(init_new_vars_op)


# #saver = tf.train.Saver(var_list=tvars)


# tvars = tf.trainable_variables()
# saver = tf.train.Saver(var_list=tvars)

# print('loading data...\n\n')
# qa_data = load_data2()
# print('done loading data...\n\n')
# #batch_no = 0
# batch_size = 50
# while batch_no < 2000: #len(qa_data['training']):
#    print('batch = ' + str(batch_no))
#    (questions_in_true, answer_in_true, im_feat_true) = get_training_batch(batch_no, batch_size, qa_data)
#    #print ('questions_in_true ', questions_in_true)
#    #print ('answer_in_true ', answer_in_true)
#    #print ('im_feat_true', im_feat_true)

# #   if batch_no*batch_size < len(qa_data['training']) - 1:
# #      (questions_in_false, answer_in_false, im_feat_false) = get_training_batch(batch_no+1, batch_size, qa_data)
# #   else:
# #      (questions_in_false, answer_in_false, im_feat_false) = get_training_batch(0, batch_size, qa_data)

#    #noise_in = np.random.normal(size=[batch_size,4096])

#    train_step.run(feed_dict={
#       #noise: noise_in, 
#       answers_true: answer_in_true,
# #      answers_false: answer_in_false,
#       cnn_out_true: im_feat_true,
# #      cnn_out_false: im_feat_false,
#       question: questions_in_true,
# #      questions_false: questions_in_false
#    })

#    loss_val, class_out = sess.run([loss,mlp_2], feed_dict={
#       #noise: noise_in, 
#       answers_true: answer_in_true,
# #      answers_false: answer_in_false,
#       cnn_out_true: im_feat_true,
# #      cnn_out_false: im_feat_false,
#       question: questions_in_true,
# #      questions_false: questions_in_false
#    })

#    # g_out = sess.run(g_true, feed_dict={
#    #    noise: noise_in, 
#    #    answers_true: answer_in_true,
#    #    answers_false: answer_in_false,
#    #    cnn_out_true: im_feat_true,
#    #    cnn_out_false: im_feat_false,
#    #    questions_true: questions_in_true,
#    #    questions_false: questions_in_false
#    # })

#    print('loss = ' + str(loss_val))
#    loss_vals.append(loss_val)
#    np.save('loss_vals_auto_v' + save_ver, loss_vals)
#    print(loss_val)

#    answers_out = np.argmax(class_out, axis=1)
#    answers_idx_true = np.argmax(answer_in_true, axis=1)
#    error = float(np.sum(answers_out == answers_idx_true)) / float(batch_size)
#    acc_vals.append(error)
#    np.save('acc_vals_auto_v' + save_ver, acc_vals)

#    # answers_out = np.argmax(g_out, axis=1)
#    # answers_idx_true = np.argmax(answer_in_true, axis=1)
#    # error = float(np.sum(answers_out == answers_idx_true)) / float(batch_size)
#    # acc_vals.append(error)
#    # np.save('acc_vals_v' + save_ver, acc_vals)
#    # print('error = ' + str(error))

#    if batch_no % 200 == 0:
#       saver.save(sess, 'train_auto_v' + save_ver)
#       # weights_save = {}
#       # for key in var_dict:
#       #    weights_save[key] = var_dict[key].eval()
#       # weights_save['batch_no'] = batch_no
#       # np.save('weights_v' + save_ver, weights_save)

#    batch_no += 1

  

saver = tf.train.Saver()
saver.restore(sess, "./train_auto_v2_out")


qa_data, answers_vocab_inv = load_data3()
batch_no = 0
batch_size = 50
answers_out = []

print((len(qa_data['validation']) - batch_size)/batch_size)
while batch_no*batch_size < len(qa_data['validation']) - batch_size:
   start = time.time()
   (questions_in, im_feat, im_ids, question_ids) = get_validation_batch(batch_no, batch_size, qa_data)

   g_out = sess.run(scores_emb, feed_dict={
      cnn_out_true: im_feat,
      question: questions_in,
   })
   answers_idx = np.argmax(g_out, axis=1)
   answers_out = create_ans_out(answers_out, answers_vocab_inv, question_ids, answers_idx)
   
   if batch_no % 100 == 0:
      generate_json(answers_out, save_ver)
   stop = time.time()

   print(batch_no)
   print(stop - start)
   batch_no += 1

generate_json(answers_out, save_ver)



