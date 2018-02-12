import sys
sys.path.insert(0, './resnet')
import os
import tensorflow as tf
import numpy as np
#from resnet152 import get_resnet
# from convert2 import load_image
#from generator_v3 import combine_embeddings, generator_me
#from discriminator import discriminator
#from rnn_module import rnn_module
#from get_data_v2 import load_data2, get_training_batch
import tensorflow.contrib.rnn as rnn_cell
from discriminator_nonorm import discriminator
from evaluation_data import *
import time


######## CONSTANTS #######
loss_vals = []
lossg_vals = []
acc_vals = []
vocabulary_size = 15881
input_embedding_size = 200
rnn_size = 512
rnn_layer = 2
dim_hidden = 2048
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

#save_ver = 'nonorm'
save_ver = './Results/d_simple_nonorm'



with tf.variable_scope("rnn_module1"):
   # question-embedding
   embed_ques_W = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08), name='embed_ques_W')

   # encoder: RNN body
   lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True, state_is_tuple=False)
   lstm_dropout_1 = rnn_cell.DropoutWrapper(lstm_1, output_keep_prob = 1 - dropout_rate)
   lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True, state_is_tuple=False)
   lstm_dropout_2 = rnn_cell.DropoutWrapper(lstm_2, output_keep_prob = 1 - dropout_rate)
   stacked_lstm = rnn_cell.MultiRNNCell([lstm_dropout_1, lstm_dropout_2], state_is_tuple=False)

   # state-embedding
   embed_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, dim_hidden], -0.08,0.08),name='embed_state_W')
   embed_state_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_state_b')
      
   # image-embedding
   embed_image_W = tf.Variable(tf.random_uniform([2048, dim_hidden], -0.08, 0.08), name='embed_image_W')
   embed_image_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_image_b')
   # score-embedding
   embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
   embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')




   image = tf.placeholder(tf.float32, [batch_size, 2048])
   question = tf.placeholder(tf.int32, [batch_size, max_words_q])
   #answers_true = tf.placeholder(tf.float32, (None, 1000))

   #image_f = tf.placeholder(tf.float32, [batch_size, 2048])
   #question_f = tf.placeholder(tf.int32, [batch_size, max_words_q])
   #answers_f = tf.placeholder(tf.float32, (None, 1000))

      
   #state = tf.zeros([batch_size, stacked_lstm.state_size])
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


   #saver = tf.train.Saver()
   #new_saver = tf.train.import_meta_graph('./train_simple.meta')
   #new_saver.restore(sess, './train_simple.index')
   #saver.restore(sess, "./train_simple_load_v1.1")

   #var_dict = {}
   #var_dict['rnnoutbeta'] = tf.Variable(tf.zeros([2048]))
   #var_dict['rnnoutscale'] = tf.Variable(tf.ones([2048]))
   #var_dict['cnnoutbeta'] = tf.Variable(tf.zeros([2048]))
   #var_dict['cnnoutscale'] = tf.Variable(tf.ones([2048]))

   #cnn_mean, cnn_var = tf.nn.moments(image, [0])
   #cnn_out_true_n = tf.nn.batch_normalization(image,cnn_mean,cnn_var,None,None,epsilon) #,var_dict['cnnoutbeta'],
     # var_dict['cnnoutscale'],epsilon)
   cnn_out_true_n = image
   #rnn_mean, rnn_var = tf.nn.moments(state, [0])
   #rnn_out_true_n = tf.nn.batch_normalization(state,rnn_mean,rnn_var,None,None,epsilon)#var_dict['rnnoutbeta'],
     # var_dict['rnnoutscale'],epsilon)
   rnn_out_true_n = state

   #cnn_out_true_n = image
   #rnn_out_true_n = state

   # multimodal (fusing question & image)
   state_drop = tf.nn.dropout(rnn_out_true_n, 1-drop_out_rate)
   state_linear = tf.nn.xw_plus_b(state_drop, embed_state_W, embed_state_b)
   state_emb = tf.tanh(state_linear)

   image_drop = tf.nn.dropout(cnn_out_true_n, 1-drop_out_rate)
   image_linear = tf.nn.xw_plus_b(image_drop, embed_image_W, embed_image_b)
   image_emb = tf.tanh(image_linear)

   scores = tf.multiply(state_emb, image_emb)
   scores_drop = tf.nn.dropout(scores, 1-drop_out_rate)
   scores_emb1 = tf.nn.xw_plus_b(scores_drop, embed_scor_W, embed_scor_b) 
   scores_emb = tf.nn.sigmoid(scores_emb1)


   
   #var_dict['gbeta'] = tf.Variable(tf.zeros([1000]), name='gbeta')
   #var_dict['gscale'] = tf.Variable(tf.ones([1000]), name='gscale')
#    var_dict = {}
#    #var_dict['dfcW1'] = tf.Variable(tf.truncated_normal([3048, 2048]), name='dfcW1')
#    #var_dict['dfcb1'] = tf.Variable(tf.constant(0.1, shape=[2048]), name='dfcb1')
#    var_dict['dfcW1'] = tf.Variable(tf.random_uniform([5096, 3048], -0.08,0.08), name='dfcW1')
#    var_dict['dfcb1'] = tf.Variable(tf.random_uniform([3048], -0.08,0.08), name='dfcb1')
#    #var_dict['dbeta1'] = tf.Variable(tf.zeros([2048]), name='dbeta1')
#    #var_dict['dscale1'] = tf.Variable(tf.ones([2048]), name='dscale1')
#    var_dict['dfcW2'] = tf.Variable(tf.random_uniform([3048, 1024], -0.08,0.08), name='dfcW2')
#    var_dict['dfcb2'] = tf.Variable(tf.random_uniform([1024], -0.08,0.08), name='dfcb2')
#    #var_dict['dbeta2'] = tf.Variable(tf.zeros([1024]), name='dbeta2')
#    #var_dict['dscale2'] = tf.Variable(tf.ones([1024]), name='dscale2')
#    var_dict['dfcW3'] = tf.Variable(tf.random_uniform([1024, 512], -0.08,0.08), name='dfcW3')
#    var_dict['dfcb3'] = tf.Variable(tf.random_uniform([512], -0.08,0.08), name='dfcb3')
#    #var_dict['dbeta3'] = tf.Variable(tf.zeros([512]), name='dbeta3')
#    #var_dict['dscale3'] = tf.Variable(tf.ones([512]), name='dscale3')
#    var_dict['dfcW4'] = tf.Variable(tf.random_uniform([512, 128], -0.08,0.08), name='dfcW4')
#    var_dict['dfcb4'] = tf.Variable(tf.random_uniform([128], -0.08,0.08), name='dfcb4')
#    #var_dict['dbeta4'] = tf.Variable(tf.zeros([128]), name='dbeta4')
#    #var_dict['dscale4'] = tf.Variable(tf.ones([128]), name='dscale4')
#    var_dict['dfcW5'] = tf.Variable(tf.random_uniform([128, 32], -0.08,0.08), name='dfcW5')
#    var_dict['dfcb5'] = tf.Variable(tf.random_uniform([32], -0.08,0.08), name='dfcb5')
#    #var_dict['dbeta5'] = tf.Variable(tf.zeros([32]), name='dbeta5')
#    #var_dict['dscale5'] = tf.Variable(tf.ones([32]), name='dscale5')
#    var_dict['dfcW6'] = tf.Variable(tf.random_uniform([32, 1], -0.08,0.08), name='dfcW6')
#    var_dict['dfcb6'] = tf.Variable(tf.random_uniform([1], -0.08,0.08), name='dfcb6')



#    state2 = stacked_lstm.zero_state(batch_size, tf.float32)
#    loss = 0.0
#    for i in range(max_words_q):  
#       if i==0:
#          ques_emb_linear2 = tf.zeros([batch_size, input_embedding_size])
#       else:
#          tf.get_variable_scope().reuse_variables()
#          ques_emb_linear2 = tf.nn.embedding_lookup(embed_ques_W, question_f[:,i-1])
#          #ques_emb_linear = tf.gather(var_dict['rnnqW'], question[:,i-1])

#       ques_emb_drop2 = tf.nn.dropout(ques_emb_linear2, 1-drop_out_rate)
#       ques_emb2 = tf.tanh(ques_emb_drop2)

#       output, state2 = stacked_lstm(ques_emb2, state2)


#    #cnn_mean, cnn_var = tf.nn.moments(image_f, [0])
#    #cnn_out_true_n_f = tf.nn.batch_normalization(image_f,cnn_mean,cnn_var,None,None,epsilon) #var_dict['cnnoutbeta'],
#       #var_dict['cnnoutscale'],epsilon)
#    cnn_out_true_n_f = image_f

#    #rnn_mean, rnn_var = tf.nn.moments(state2, [0])
#    #rnn_out_true_n_f = tf.nn.batch_normalization(state2,rnn_mean,rnn_var,None,None,epsilon) #var_dict['rnnoutbeta'],
#       #var_dict['rnnoutscale'],epsilon)
#    rnn_out_true_n_f = state2

#    state_drop2 = tf.nn.dropout(rnn_out_true_n_f, 1-drop_out_rate)
#    state_linear2 = tf.nn.xw_plus_b(state_drop2, embed_state_W, embed_state_b)
#    state_emb2 = tf.tanh(state_linear2)

#    image_drop2 = tf.nn.dropout(cnn_out_true_n_f, 1-drop_out_rate)
#    image_linear2 = tf.nn.xw_plus_b(image_drop2, embed_image_W, embed_image_b)
#    image_emb2 = tf.tanh(image_linear2)

#    scores2 = tf.multiply(state_emb2, image_emb2)
#    scores_drop2 = tf.nn.dropout(scores2, 1-drop_out_rate)
#    scores_emb2 = tf.nn.xw_plus_b(scores_drop2, embed_scor_W, embed_scor_b) 

#    s_r, fc6, fc5n, fc4n = discriminator(tf.concat([rnn_out_true_n,cnn_out_true_n], 1), answers_true, var_dict)
#    s_w, fc6, fc5n, fc4n = discriminator(tf.concat([image_f,state2], 1), answers_true, var_dict)
#    s_f, fc6, fc5n, fc4n = discriminator(tf.concat([rnn_out_true_n,cnn_out_true_n], 1), scores_emb, var_dict) #g_true

#    ones = tf.constant(1.0, shape=[50,1], dtype=tf.float32)
#    loss = -tf.reduce_mean(tf.log(s_r) + tf.log(tf.subtract(ones,s_w) + 1e-5*ones)/2.0 + tf.log(tf.subtract(ones,s_f) + 1e-5*ones)/2.0)
#    loss2 = -tf.reduce_mean(tf.log(s_f))




# tvars = tf.trainable_variables()
# opt_g = tf.train.AdamOptimizer(learning_rate=1e-4)
# opt_d = tf.train.AdamOptimizer(learning_rate=1e-4)
# for i in range(len(tvars)):
#    print(tvars[i].name)


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
#    'rnn_module1/multi_rnn_cell/cell_1/lstm_cell/w_o_diag:0','rnn_module1/embed_ques_W:0',
#    'rnn_module1/embed_state_W:0','rnn_module1/embed_state_b:0',
#    'rnn_module1/embed_image_W:0','rnn_module1/embed_image_b:0',
#    'rnn_module1/embed_scor_W:0','rnn_module1/embed_scor_b:0']
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

# #gradient clipping
# gvs_d = opt_d.compute_gradients(loss,d_vars)

# #for i in range(len(tvars)):
# #   print(tvars[i].name)
# clipped_gvs_d = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs_d]
# train_op_d = opt_d.apply_gradients(clipped_gvs_d)


# gvs_g = opt_g.compute_gradients(loss2,g_vars)

# #for i in range(len(tvars)):
# #   print(tvars[i].name)
# clipped_gvs_g = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs_g]
# train_op_g = opt_g.apply_gradients(clipped_gvs_g)

# #train_op_d = tf.train.AdamOptimizer().minimize(loss, var_list=d_vars)
# #train_op_g = tf.train.AdamOptimizer().minimize(loss2, var_list=g_vars)

# # opt = tf.train.AdamOptimizer(learning_rate=1e-4)
# # # gradient clipping
# # gvs = opt.compute_gradients(loss,tvars)
# # clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
# # train_op = opt.apply_gradients(clipped_gvs)

# #sess.run(tf.global_variables_initializer())

# #print(tvars)





# uninitialized_vars = []
# for var in tf.all_variables():
#     try:
#         sess.run(var)
#     except tf.errors.FailedPreconditionError:
#         uninitialized_vars.append(var)

# init_new_vars_op = tf.initialize_variables(uninitialized_vars)
# sess.run(init_new_vars_op)


# saver = tf.train.Saver(var_list=tvars)

#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# init variables
#sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
saver.restore(sess, "./train_d_simple_vnonorm")


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


