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
from get_data_v2 import load_data2, get_training_batch
import tensorflow.contrib.rnn as rnn_cell


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

save_ver = '3.2'


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
   answers_true = tf.placeholder(tf.int32, (batch_size, 1000))
   noise = tf.placeholder(tf.float32, [batch_size, 4096])

      
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

   features2 = tf.add(features,noise)
   scores_emb = generator_me(features2, var_dict)

   #v1 = tf.Variable(tf.truncated_normal([2048,1000]))
   #v2 = tf.Variable(tf.constant(0.1, shape=[1000]))
   #scores_emb = tf.nn.relu_layer(state, v1, v2)

   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores_emb, labels=answers_true))



tvars = tf.trainable_variables()
opt = tf.train.AdamOptimizer(learning_rate=1e-4)
# gradient clipping
gvs = opt.compute_gradients(loss,tvars)
for i in range(len(tvars)):
   print(tvars[i].name)
clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
train_op = opt.apply_gradients(clipped_gvs)



for i in range(len(tvars)):
   print(tvars[i].name)


#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# init variables
sess.run(tf.global_variables_initializer())

v1 = var_dict['rnnqW'].eval()

saver = tf.train.Saver()

saver.restore(sess, "./train_g_v3.2")

# tvars2 = tf.trainable_variables()


# v2 = var_dict['rnnqW'].eval()

# save_vars = {}
# for i in range(len(tvars2)):
#    key = str(tvars2[i].name)
#    save_vars[key] = tvars2[i].eval()
#    print('i = ' + str(i) + ', key = ' + key)
# print(save_vars['rnn_module1/multi_rnn_cell/cell_1/lstm_cell/w_o_diag:0'])
# np.save('train_g_v3.2_weights', save_vars)



print('answers 1000....')
print('loading data...\n\n')
qa_data = load_data2()
print('done loading data...\n\n')
batch_size = 50
#while batch_no*batch_size < len(qa_data['training']):
for train_loops in range(10):
   batch_no = 0
   while batch_no*batch_size < len(qa_data['training']):
      print('batch = ' + str(batch_no))
      (questions_in_true, answer_in_true, im_feat_true) = get_training_batch(batch_no, batch_size, qa_data)

      noise_in = np.random.normal(scale=0.3, size=[batch_size,4096])

      _, loss_val, g_out = sess.run([train_op, loss, scores_emb], feed_dict={
         noise: noise_in,
         answers_true: answer_in_true,
         image: im_feat_true,
         question: questions_in_true,
      })

      print(g_out)
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
         #weights_save = {}
         #for key in var_dict:
         #   weights_save[key] = var_dict[key].eval()
         #weights_save['batch_no'] = batch_no
         #np.save('weights_simple_v' + save_ver, weights_save)
         saver.save(sess, 'train_g_v' + save_ver)

      batch_no += 1

     