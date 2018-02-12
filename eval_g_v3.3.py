import sys
sys.path.insert(0, './resnet')
import os
import tensorflow as tf
import numpy as np
#from resnet152 import get_resnet
# from convert2 import load_image
from generator_v3 import combine_embeddings, generator_me
#from discriminator import discriminator
#from rnn_module import rnn_module
#from get_data_v2 import load_data2, get_training_batch
import tensorflow.contrib.rnn as rnn_cell
from evaluation_data import *
import time

######## CONSTANTS #######
loss_vals = []
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

#save_ver = '3.3'
save_ver = './Results/g_v3.3'

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
      'gfcW1': tf.Variable(tf.truncated_normal([4096+2048, 4096]), name='gfcW1'),
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
      #'rnnoutbeta': tf.Variable(weights['rnnoutbeta']),
      #'rnnoutscale': tf.Variable(weights['rnnoutscale']),
      #'cnnoutbeta': tf.Variable(weights['cnnoutbeta']),
      #'cnnoutscale': tf.Variable(weights['cnnoutscale']),
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
   #answers_true = tf.placeholder(tf.int32, (batch_size, 1000))
   noise = tf.placeholder(tf.float32, [batch_size, 2048])

      
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

   print(state.get_shape())
   features = combine_embeddings(image, state, var_dict)
   #features = tf.concat([image,state], 1)
   #features = tf.concat([state,image], 1)

   features2 = tf.concat([features,noise], 1)
   scores_emb = generator_me(features2, var_dict)

   #v1 = tf.Variable(tf.truncated_normal([2048,1000]))
   #v2 = tf.Variable(tf.constant(0.1, shape=[1000]))
   #scores_emb = tf.nn.relu_layer(state, v1, v2)

   #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores_emb, labels=answers_true))



# tvars = tf.trainable_variables()
# opt = tf.train.AdamOptimizer(learning_rate=1e-4)
# # gradient clipping
# gvs = opt.compute_gradients(loss,tvars)
# for i in range(len(tvars)):
#    print(tvars[i].name)
# clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
# train_op = opt.apply_gradients(clipped_gvs)



# for i in range(len(tvars)):
#    print(tvars[i].name)


#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# init variables
#sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(sess, "./train_g_v3.3")
#quit()


qa_data, answers_vocab_inv = load_data3()
batch_no = 0
batch_size = 50
answers_out = []

print((len(qa_data['validation']) - batch_size)/batch_size)
while batch_no*batch_size < len(qa_data['validation']) - batch_size:
   start = time.time()
   (questions_in, im_feat, im_ids, question_ids) = get_validation_batch(batch_no, batch_size, qa_data)

   noise_in = np.random.normal(scale=0.3, size=[batch_size,2048])

   g_out = sess.run(scores_emb, feed_dict={
      noise: noise_in,
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


