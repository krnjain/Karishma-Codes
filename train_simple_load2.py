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


save_ver = '1_load'

#saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./train_g_v3.3.meta')
#graph = tf.get_default_graph()

with tf.Session() as sess:
   saver.restore(sess, './train_simple')



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
#sess.run(tf.global_variables_initializer())



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
      #saver.save(sess, 'train_simple')
      weights_save = {}
      for i in range(len(tvars)):
         weights_save[tvars[i].name] = tvars[i].eval()
         np.save('testweights')

   batch_no += 1

  