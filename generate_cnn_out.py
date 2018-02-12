import sys
sys.path.insert(0, './resnet')

import tensorflow as tf
import numpy as np
#from resnet152 import get_resnet
# from convert2 import load_image
#from generator import combine_embeddings, generator_me
#from discriminator import discriminator
#from rnn_module import rnn_module
from get_data import load_data, get_training_batch


sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('./resnet/ResNet-L152.meta')
new_saver.restore(sess, './resnet/ResNet-L152.ckpt')
graph = tf.get_default_graph()
images = graph.get_tensor_by_name("images:0")
out = graph.get_tensor_by_name("avg_pool:0")
#sess.run(tf.global_variables_initializer())


print('loading data...\n\n')
qa_data = load_data()
print('done loading data...\n\n')
batch_no = 0
batch_size = 50
while batch_no*batch_size < len(qa_data['training']):
   print('processing batch ' + str(batch_no))
   (questions_in_true, answer_in_true, image_in_true) = get_training_batch(batch_no, batch_size, qa_data)
   im_feat_true = sess.run(out, feed_dict={images: image_in_true})
   np.save('./im_features/batch'+str(batch_no)+'_50', im_feat_true)
   print('done processing batch ' + str(batch_no))
   batch_no += 1

