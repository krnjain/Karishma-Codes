import sys
sys.path.insert(0, './resnet')

import tensorflow as tf
import numpy as np
from resnet152 import get_resnet
# from convert2 import load_image
from generator import combine_embeddings, generator_me
from discriminator import discriminator
from rnn_module import rnn_module
from get_data import load_data, get_training_batch


sess = tf.InteractiveSession()

print('loading data')
qa_data = load_data()
(questions_in_true, answer_in_true, image_in_true) = get_training_batch(0, 50, qa_data)
print('starting')
im_feat_true = get_resnet(sess, image_in_true)   
print(im_feat_true)
np.save('features_test',im_feat_true)
   #im_feat_true = np.random.normal(size=[batch_size,2048])
   #im_feat_false = np.random.normal(size=[batch_size,2048])


   # get embedding of true input data
   # features_true = sess.run(features, feed_dict={images: image_true, questions: questions_true})
   
   # # get embedding of false input data
   # features_false = sess.run(features, feed_dict={images: image_false, questions: questions_false})

   # # get generated output from true input
   # g_true = sess.run(g, feed_dict={features: features_true, noise: np.random.normal(size=[batch_size,4096])})
   
   # # get discriminator output of correct features, correct answer
   # s_r = sess.run(d, feed_dict={features: features_true, g: answer_true})

   # # get discriminator output of incorrect features, correct answer
   # s_w = sess.run(d, feed_dict={features: features_false, g: answer_true})

   # # get discriminator output of generated features, correct answer
   # s_f = sess.run(d, feed_dict={features: features_true, g: g_true})

   # loss = tf.reduce_mean(tf.add(tf.log(s_r),tf.add(tf.minus(1,s_w),tf.minus(1,s_f))/2))

   # print(loss)

   # h_true = sess.run(features, feed_dict={images: image_true, questions: questions_true})

   # #not sure about this, where are we generating x_hat from mismatching text?
   # stuff = sess.run(g, feed_dict={features: h_true, noise: np.random.normal(size=[batch_size,4096])})

   # x_hat = sess.run(d_features, feed_dict={features: h_true, g: stuff})

   # #yeah I am not sure how to parse these

   # s_r = sess.run(d, feed_dict={d_features: x_hat}) #right, right

   # s_w = sess.run(d, feed_dict={images: images }) #right, wrong h

   # s_f = sess.run(d, feed_dict={}) #wrong x, right



   # test = sess.run(d, feed_dict={images: image_in, noise:np.random.normal(size=[batch_size,4096]), questions: questions_in})


   # # D_loss = -tf.reduce_mean(tf.log)
   # print(test)



# img = load_image("data/cat.jpg")
# batch = img.reshape((1, 224, 224, 3))
# test = sess.run(g, feed_dict={images: batch, noise: np.random.normal(size=[1, 4096])})
#print(test)
#print(test.shape)

#test = sess.run(cnn_out, feed_dict={images: batch})
#print(test)
