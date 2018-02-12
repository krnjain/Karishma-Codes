import sys
#sys.path.insert(0, './resnet')
import os
import tensorflow as tf
import numpy as np
import time
#from resnet152 import get_resnet
# from convert2 import load_image
#from generator import combine_embeddings, generator_me
#from discriminator import discriminator
#from rnn_module import rnn_module
#from get_data_v2 import load_data, get_training_batch
from scipy.misc import imread, imresize, imsave

IM_DIM = 224


def get_image(image_name):
    im = tf.read_file('./Images/mscoco/val2014/' + image_name)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.convert_image_dtype(im, dtype=tf.float32)
    
    im = tf.image.central_crop(im, central_fraction=1)
    im = tf.expand_dims(im, 0)
    im = tf.image.resize_bilinear(im, [IM_DIM, IM_DIM],
                                         align_corners=False)
    im = tf.squeeze(im, [0])
    im = tf.subtract(im, 0.5)
    im = tf.multiply(im, 2.0)
    im = im.eval()
    print(np.mean(im))
    print(np.max(im))
    print(np.min(im))
    return im


#im_dir = os.listdir('./Images/mscoco/train2014')

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('./resnet/ResNet-L152.meta')
new_saver.restore(sess, './resnet/ResNet-L152.ckpt')
graph = tf.get_default_graph()
images = graph.get_tensor_by_name("images:0")
out = graph.get_tensor_by_name("avg_pool:0")

im_dir = os.listdir('./Images/mscoco/val2014')
for i in range(len(im_dir)-8000):
    idx = i + 8000
    filename, file_extension = os.path.splitext(im_dir[idx])
    im = get_image(filename + '.jpg')
    im = np.reshape(im, [1,IM_DIM,IM_DIM,3])
    im_feat = sess.run(out, feed_dict={images: im})
    #np.save('./im_feat_indiv/val/' + filename, im_feat)
    print(im_feat)
    print('filename = ' + filename)
    print(idx)







# count = 0
# for i in range(len(im_dir)):
#     print('count = ' + str(count))
#     start = time.time()
#     im = get_image(im_dir[i])
#     im = np.reshape(im, [1,IM_DIM,IM_DIM,3])
#     im_feat = sess.run(out, feed_dict={images: im})
#     filename, file_extension = os.path.splitext(im_dir[i])
#     np.save('./im_feat_indiv/train/' + filename, im_feat)
#     count += 1
#     stop = time.time()
#     print(stop - start)


# batch_size = 200
# ims = np.zeros([batch_size, IM_DIM, IM_DIM, 3])
# start = time.time()
# for i in range(batch_size):
#     im = get_image(im_dir[i])
#     ims[i,:,:,:] = im
# im_feats = sess.run(out, feed_dict={images: ims})
# for i in range(batch_size):
#     filename, file_extension = os.path.splitext(im_dir[i])
#     np.save('./im_feat_indiv/train/' + filename, im_feats[i,:])
# stop = time.time()
# print(stop - start)


# ~0.5 seconds when running individually
# 39 seconds with 100

# batch_no = int(sys.argv[1])


# #sess.run(tf.global_variables_initializer())


# print('loading data...\n\n')
# qa_data = load_data()
# print('done loading data...\n\n')
# #batch_no = 0
# batch_size = 50
# #while batch_no*batch_size < len(qa_data['training']):
# #print('processing batch ' + str(batch_no))
# (questions_in_true, answer_in_true, image_in_true) = get_training_batch(batch_no, batch_size, qa_data)
# im_feat_true = sess.run(out, feed_dict={images: image_in_true})
# np.save('./im_features/batch'+str(batch_no)+'_50', im_feat_true)
# #print('done processing batch ' + str(batch_no))
# #batch_no += 1

