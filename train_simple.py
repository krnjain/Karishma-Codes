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
from get_data_v2 import load_data2, get_training_batch
import tensorflow.contrib.rnn as rnn_cell


######## CONSTANTS #######
loss_vals = []
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

save_ver = '1'



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
   answers_true = tf.placeholder(tf.int32, (None, 1000))

      
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

   # multimodal (fusing question & image)
   state_drop = tf.nn.dropout(state, 1-drop_out_rate)
   state_linear = tf.nn.xw_plus_b(state_drop, embed_state_W, embed_state_b)
   state_emb = tf.tanh(state_linear)

   image_drop = tf.nn.dropout(image, 1-drop_out_rate)
   image_linear = tf.nn.xw_plus_b(image_drop, embed_image_W, embed_image_b)
   image_emb = tf.tanh(image_linear)

   scores = tf.multiply(state_emb, image_emb)
   scores_drop = tf.nn.dropout(scores, 1-drop_out_rate)
   scores_emb = tf.nn.xw_plus_b(scores_drop, embed_scor_W, embed_scor_b) 

   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores_emb, labels=answers_true))




tvars = tf.trainable_variables()
opt = tf.train.AdamOptimizer(learning_rate=1e-4)
# gradient clipping
gvs = opt.compute_gradients(loss,tvars)
clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
train_op = opt.apply_gradients(clipped_gvs)

print(tvars)

for i in range(len(tvars)):
   print(tvars[i].name)


#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# init variables
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()


print('answers 1000....')
print('loading data...\n\n')
qa_data = load_data2()
print('done loading data...\n\n')
batch_no = 0
batch_size = 50
#while batch_no*batch_size < len(qa_data['training']):
while batch_no < 6500:
   print('batch = ' + str(batch_no))
   (questions_in_true, answer_in_true, im_feat_true) = get_training_batch(batch_no, batch_size, qa_data)

   _, loss_val, g_out = sess.run([train_op, loss, scores_emb], feed_dict={
      answers_true: answer_in_true,
      image: im_feat_true,
      question: questions_in_true,
   })

   print('loss = ' + str(loss_val))
   loss_vals.append(loss_val)
   np.save('loss_vals_simple_v' + save_ver, loss_vals)
   #print(loss_val)

   answers_out = np.argmax(g_out, axis=1)
   answers_idx_true = np.argmax(answer_in_true, axis=1)
   error = float(np.sum(answers_out == answers_idx_true)) / float(batch_size)
   acc_vals.append(error)
   np.save('acc_vals_simple_v' + save_ver, acc_vals)
   print('error = ' + str(error))

   if batch_no % 25 == 0:
      #weights_save = {}
      #for key in var_dict:
      #   weights_save[key] = var_dict[key].eval()
      #weights_save['batch_no'] = batch_no
      #np.save('weights_simple_v' + save_ver, weights_save)
      saver.save(sess, 'train_simple')

   batch_no += 1

  