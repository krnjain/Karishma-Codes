import sys
sys.path.insert(0, './resnet')
import os
import tensorflow as tf
import numpy as np
#from resnet152 import get_resnet
# from convert2 import load_image
from generator_v3 import combine_embeddings, generator_me
from discriminator import discriminator
#from rnn_module import rnn_module
#from get_data_v2 import load_data2, get_training_batch
import tensorflow.contrib.rnn as rnn_cell
from evaluation_data import *
import time


######## CONSTANTS #######
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

#save_ver = '3.2.1_pred_notg'
save_ver = './Results/d_3.2.1_pred_notg'


weights = np.load('weights_g_v2_nonoise.npy').item()


with tf.variable_scope("rnn_module1"):
   #tf.get_variable_scope().reuse_variables()
   var_dict = {
      'cemcnnfcW1': tf.Variable(weights['cemcnnfcW1'], name='cemcnnfcW1'),
      'cemcnnfcb1': tf.Variable(weights['cemcnnfcb1'], name='cemcnnfcb1'),
      'ceacnnfcW1': tf.Variable(weights['ceacnnfcW1'], name='ceacnnfcW1'),
      'ceacnnfcb1': tf.Variable(weights['ceacnnfcb1'], name='ceacnnfcb1'),
      'cemrnnfcW1': tf.Variable(weights['cemrnnfcW1'], name='cemrnnfcW1'),
      'cemrnnfcb1': tf.Variable(weights['cemrnnfcb1'], name='cemrnnfcb1'),
      'cearnnfcW1': tf.Variable(weights['cearnnfcW1'], name='cearnnfcW1'),
      'cearnnfcb1': tf.Variable(weights['cearnnfcb1'], name='cearnnfcb1'),
      'gfcW1': tf.Variable(tf.truncated_normal([4096, 4096]), name='gfcW1'),
      'gfcb1': tf.Variable(tf.constant(0.1, shape=[4096]), name='gfcb1'),
      'gfcW2': tf.Variable(tf.truncated_normal([4096, 4096]), name='gfcW2'),
      'gfcb2': tf.Variable(tf.constant(0.1, shape=[4096]), name='gfcb2'),
      'gfcW3': tf.Variable(tf.truncated_normal([4096, 2048]), name='gfcW3'),
      'gfcb3': tf.Variable(tf.constant(0.1, shape=[2048]), name='gfcb3'),
      'gfcW4': tf.Variable(tf.truncated_normal([2048, 1000]), name='gfcW4'),
      'gfcb4': tf.Variable(tf.constant(0.1, shape=[1000]), name='gfcb4'),
      'rnnqW': tf.Variable(weights['rnnqW'], name='embed_ques_W'),
      #'rnnsW': tf.Variable(weights['rnnsW'], name='embed_state_W'),
      #'rnnsb': tf.Variable(weights['rnnsb'], name='embed_state_b'),
      #'rnnoutbeta': tf.Variable(tf.zeros([2048])),
      #'rnnoutscale': tf.Variable(tf.ones([2048])),
      #'cnnoutbeta': tf.Variable(tf.zeros([2048])),
      #'cnnoutscale': tf.Variable(tf.ones([2048])),
      #'featbeta': tf.Variable(tf.zeros([4096])),
      #'featscale': tf.Variable(tf.ones([4096])),
      #'gbeta': tf.Variable(tf.zeros([1000])),
      #'gscale': tf.Variable(tf.ones([1000]))
   }

   # question-embedding
   #embed_ques_W = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08), name='embed_ques_W')

   # encoder: RNN body
   lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True, state_is_tuple=False)
   lstm_dropout_1 = rnn_cell.DropoutWrapper(lstm_1, output_keep_prob = 1 - dropout_rate)
   lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True, state_is_tuple=False)
   lstm_dropout_2 = rnn_cell.DropoutWrapper(lstm_2, output_keep_prob = 1 - dropout_rate)
   stacked_lstm = rnn_cell.MultiRNNCell([lstm_dropout_1, lstm_dropout_2], state_is_tuple=False)


   image = tf.placeholder(tf.float32, [batch_size, 2048])
   question = tf.placeholder(tf.int32, [batch_size, max_words_q])
   #answers_true = tf.placeholder(tf.float32, (batch_size, 1000))
   #noise = tf.placeholder(tf.float32, [batch_size, 4096])

   #answers_false = tf.placeholder(tf.float32, (None, 1000))
   #image_false = tf.placeholder(tf.float32, (None, 2048))
   #question_false = tf.placeholder(tf.int32, [batch_size, max_words_q])

      
   #state = tf.zeros([batch_size, stacked_lstm.state_size])
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


   cnn_mean, cnn_var = tf.nn.moments(image, [0])
   cnn_out_true_n = tf.nn.batch_normalization(image,cnn_mean,cnn_var,None,None,epsilon) #,var_dict['cnnoutbeta'],
      #var_dict['cnnoutscale'],epsilon)

   rnn_mean, rnn_var = tf.nn.moments(state, [0])
   rnn_out_true_n = tf.nn.batch_normalization(state,rnn_mean,rnn_var,None,None,epsilon) #var_dict['rnnoutbeta'],
      #var_dict['rnnoutscale'],epsilon)

   features = combine_embeddings(cnn_out_true_n, rnn_out_true_n, var_dict)
   #features = tf.concat([image,state], 1)
   #features = tf.concat([state,image], 1)

   #features2 = tf.concat([features,noise], 1)
   #features2 = tf.add(features,noise)
   features2 = features
   scores_emb = generator_me(features2, var_dict)

   #v1 = tf.Variable(tf.truncated_normal([2048,1000]))
   #v2 = tf.Variable(tf.constant(0.1, shape=[1000]))
   #scores_emb = tf.nn.relu_layer(state, v1, v2)


   #saver = tf.train.Saver()

   #saver.restore(sess, "./train_g_v3.2.2")

#    weights = np.load('weights_pretrained_d.npy').item()

#    #var_dict['gbeta'] = tf.Variable(tf.zeros([1000]), name='gbeta')
#    #var_dict['gscale'] = tf.Variable(tf.ones([1000]), name='gscale')
#    var_dict['dfcW1'] = tf.Variable(weights['dfcW1'], name='dfcW1')
#    var_dict['dfcb1'] = tf.Variable(weights['dfcb1'], name='dfcb1')
#    var_dict['dbeta1'] = tf.Variable(weights['dbeta1'], name='dbeta1')
#    var_dict['dscale1'] = tf.Variable(weights['dscale1'], name='dscale1')
#    var_dict['dfcW2'] = tf.Variable(weights['dfcW2'], name='dfcW2')
#    var_dict['dfcb2'] = tf.Variable(weights['dfcb2'], name='dfcb2')
#    var_dict['dbeta2'] = tf.Variable(weights['dbeta2'], name='dbeta2')
#    var_dict['dscale2'] = tf.Variable(weights['dscale2'], name='dscale2')
#    var_dict['dfcW3'] = tf.Variable(weights['dfcW3'], name='dfcW3')
#    var_dict['dfcb3'] = tf.Variable(weights['dfcb3'], name='dfcb3')
#    var_dict['dbeta3'] = tf.Variable(weights['dbeta3'], name='dbeta3')
#    var_dict['dscale3'] = tf.Variable(weights['dscale3'], name='dscale3')
#    var_dict['dfcW4'] = tf.Variable(weights['dfcW4'], name='dfcW4')
#    var_dict['dfcb4'] = tf.Variable(weights['dfcb4'], name='dfcb4')
#    var_dict['dbeta4'] = tf.Variable(weights['dbeta4'], name='dbeta4')
#    var_dict['dscale4'] = tf.Variable(weights['dscale4'], name='dscale4')
#    var_dict['dfcW5'] = tf.Variable(weights['dfcW5'], name='dfcW5')
#    var_dict['dfcb5'] = tf.Variable(weights['dfcb5'], name='dfcb5')
#    var_dict['dbeta5'] = tf.Variable(weights['dbeta5'], name='dbeta5')
#    var_dict['dscale5'] = tf.Variable(weights['dscale5'], name='dscale5')
#    var_dict['dfcW6'] = tf.Variable(weights['dfcW6'], name='dfcW6')
#    var_dict['dfcb6'] = tf.Variable(weights['dfcb6'], name='dfcb6')


#    g_mean, g_var = tf.nn.moments(scores_emb, [0])
#    g_true_n = tf.nn.batch_normalization(scores_emb,g_mean,g_var,None,None,epsilon) #var_dict['gbeta'],var_dict['gscale'],epsilon)
#    at_mean, at_var = tf.nn.moments(answers_true, [0])
#    answers_true_n = tf.nn.batch_normalization(answers_true,at_mean,at_var,None,None,epsilon) #var_dict['gbeta'],var_dict['gscale'],epsilon)
#    #answers_true_n = answers_true
#    #af_mean, af_var = tf.nn.moments(answers_false, [0])
#    #answers_false_n = tf.nn.batch_normalization(answers_false,af_mean,af_var,var_dict['gbeta'],var_dict['gscale'],epsilon)


#    state2 = stacked_lstm.zero_state(batch_size, tf.float32)
#    loss = 0.0
#    for i in range(max_words_q):  
#       if i==0:
#          ques_emb_linear2 = tf.zeros([batch_size, input_embedding_size])
#       else:
#          tf.get_variable_scope().reuse_variables()
#          ques_emb_linear2 = tf.nn.embedding_lookup(var_dict['rnnqW'], question_false[:,i-1])
#          #ques_emb_linear = tf.gather(var_dict['rnnqW'], question[:,i-1])

#       ques_emb_drop2 = tf.nn.dropout(ques_emb_linear2, 1-drop_out_rate)
#       ques_emb2 = tf.tanh(ques_emb_drop2)

#       output, state2 = stacked_lstm(ques_emb2, state2)


#    cnn_mean, cnn_var = tf.nn.moments(image_false, [0])
#    cnn_out_true_n_f = tf.nn.batch_normalization(image_false,cnn_mean,cnn_var,None,None,epsilon) #var_dict['cnnoutbeta'],
#       #var_dict['cnnoutscale'],epsilon)

#    rnn_mean, rnn_var = tf.nn.moments(state2, [0])
#    rnn_out_true_n_f = tf.nn.batch_normalization(state2,rnn_mean,rnn_var,None,None,epsilon) #var_dict['rnnoutbeta'],
#       #var_dict['rnnoutscale'],epsilon)

#    features_false = combine_embeddings(cnn_out_true_n_f, rnn_out_true_n_f, var_dict)


#    # load discriminator network
#    s_r, fc6, fc5n, fc4n = discriminator(features, answers_true_n, var_dict)
#    s_w, fc6, fc5n, fc4n = discriminator(features_false, answers_true_n, var_dict)
#    s_f, fc6, fc5n, fc4n = discriminator(features, g_true_n, var_dict) #g_true

#    ones = tf.constant(1.0, shape=[50,1], dtype=tf.float32)
#    loss = -tf.reduce_mean(tf.log(s_r) + tf.log(tf.subtract(ones,s_w) + 1e-5*ones)/2.0 + tf.log(tf.subtract(ones,s_f) + 1e-5*ones)/2.0)
#    lossg = -tf.reduce_mean(tf.log(s_f))

#    #v1 = tf.Variable(tf.truncated_normal([2048,1000]))
#    #v2 = tf.Variable(tf.constant(0.1, shape=[1000]))
#    #scores_emb = tf.nn.relu_layer(state, v1, v2)

#    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores_emb, labels=answers_true))



# tvars = tf.trainable_variables()
# opt_g = tf.train.AdamOptimizer(learning_rate=1e-4)
# opt_d = tf.train.AdamOptimizer(learning_rate=1e-4)

# d_vars = []

# g_var_names = ['rnn_module1/cemcnnfcW1:0','rnn_module1/cemcnnfcb1:0',
#    'rnn_module1/ceacnnfcW1:0','rnn_module1/ceacnnfcb1:0','rnn_module1/cemrnnfcW1:0',
#    'rnn_module1/cemrnnfcb1:0','rnn_module1/cearnnfcW1:0','rnn_module1/cearnnfcb1:0',
#    'rnn_module1/gfcW1:0','rnn_module1/gfcb1:0','rnn_module1/gfcW2:0','rnn_module1/gfcb2:0',
#    'rnn_module1/gfcW3:0','rnn_module1/gfcb3:0','rnn_module1/gfcW4:0','rnn_module1/gfcb4:0',
#    'rnn_module1/embed_ques_W:0','rnn_module1/multi_rnn_cell/cell_0/lstm_cell/weights:0',
#    'rnn_module1/multi_rnn_cell/cell_0/lstm_cell/biases:0',
#    'rnn_module1/multi_rnn_cell/cell_0/lstm_cell/w_f_diag:0',
#    'rnn_module1/multi_rnn_cell/cell_0/lstm_cell/w_i_diag:0',
#    'rnn_module1/multi_rnn_cell/cell_0/lstm_cell/w_o_diag:0',
#    'rnn_module1/multi_rnn_cell/cell_1/lstm_cell/weights:0',
#    'rnn_module1/multi_rnn_cell/cell_1/lstm_cell/biases:0',
#    'rnn_module1/multi_rnn_cell/cell_1/lstm_cell/w_f_diag:0',
#    'rnn_module1/multi_rnn_cell/cell_1/lstm_cell/w_i_diag:0',
#    'rnn_module1/multi_rnn_cell/cell_1/lstm_cell/w_o_diag:0']
# d_var_names = ['rnn_module1/dfcW1:0','rnn_module1/dfcb1:0','rnn_module1/dbeta1:0',
#    'rnn_module1/dscale1:0','rnn_module1/dfcW2:0','rnn_module1/dfcb2:0',
#    'rnn_module1/dbeta2:0','rnn_module1/dscale2:0','rnn_module1/dfcW3:0',
#    'rnn_module1/dfcb3:0','rnn_module1/dbeta3:0','rnn_module1/dscale3:0',
#    'rnn_module1/dfcW4:0','rnn_module1/dfcb4:0','rnn_module1/dbeta4:0',
#    'rnn_module1/dscale4:0','rnn_module1/dfcW5:0','rnn_module1/dfcb5:0',
#    'rnn_module1/dbeta5:0','rnn_module1/dscale5:0','rnn_module1/dfcW6:0',
#    'rnn_module1/dfcb6:0']


# d_vars = []
# g_vars = []

# for i in range(len(tvars)):
#    if tvars[i].name in g_var_names:
#       g_vars.append(tvars[i])
#       print('g - ' + tvars[i].name)
#    elif tvars[i].name in d_var_names:
#       d_vars.append(tvars[i])
#       print('d - ' + tvars[i].name)
#    else:
#       print('error')
# print(d_vars)
# print(g_vars)

# # # gradient clipping
# # gvs = opt.compute_gradients(loss,tvars)

# # #for i in range(len(tvars)):
# # #   print(tvars[i].name)
# # clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
# # train_op = opt.apply_gradients(clipped_gvs)

# # gradient clipping
# gvs_d = opt_d.compute_gradients(loss,d_vars)

# #for i in range(len(tvars)):
# #   print(tvars[i].name)
# clipped_gvs_d = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs_d]
# train_op_d = opt_d.apply_gradients(clipped_gvs_d)


# gvs_g = opt_g.compute_gradients(lossg,g_vars)

# #for i in range(len(tvars)):
# #   print(tvars[i].name)
# clipped_gvs_g = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs_g]
# train_op_g = opt_g.apply_gradients(clipped_gvs_g)



# # for i in range(len(tvars)):
# #    print(tvars[i].name)


# #train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# # init variables
# sess.run(tf.global_variables_initializer())

# # init2 = tf.variables_initializer([var_dict['gbeta'],var_dict['gscale'],var_dict['dfcW1'],
# #    var_dict['dfcb1'],var_dict['dbeta1'],var_dict['dscale1'],var_dict['dfcW2'],
# #    var_dict['dfcb2'],var_dict['dbeta2'],var_dict['dscale2'],var_dict['dfcW3'],
# #    var_dict['dfcb3'],var_dict['dbeta3'],var_dict['dscale3'],var_dict['dfcW4'],
# #    var_dict['dfcb4'],var_dict['dbeta4'],var_dict['dscale4'],var_dict['dfcW5'],
# #    var_dict['dfcb5'],var_dict['dbeta5'],var_dict['dscale5'],var_dict['dfcW6'],
# #    var_dict['dfcb6']])
# # sess.run(init2)


# # uninitialized_vars = []
# # for var in tf.all_variables():
# #     try:
# #         sess.run(var)
# #     except tf.errors.FailedPreconditionError:
# #         uninitialized_vars.append(var)

# # init_new_vars_op = tf.initialize_variables(uninitialized_vars)
# # sess.run(init_new_vars_op)

# saver = tf.train.Saver(var_list=tvars)

# #saver.restore(sess, "./train_d_v3.2.1")


saver = tf.train.Saver()
saver.restore(sess, "./train_d_v3.2.1_pred_notg")


qa_data, answers_vocab_inv = load_data3()
batch_no = 0
batch_size = 50
answers_out = []

print((len(qa_data['validation']) - batch_size)/batch_size)
while batch_no*batch_size < len(qa_data['validation']) - batch_size:
   start = time.time()
   (questions_in, im_feat, im_ids, question_ids) = get_validation_batch(batch_no, batch_size, qa_data)

   g_out = sess.run(scores_emb, feed_dict={
      image: im_feat,
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


