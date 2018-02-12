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
from get_data_v2 import load_data, get_training_batch
from scipy.misc import imread, imresize, imsave


IM_DIM = 224


def get_image(image_name):
    im = tf.read_file('./Images/mscoco/train2014/' + image_name)
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
    return im


#im_dir = os.listdir('./Images/mscoco/train2014')

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('./resnet/ResNet-L152.meta')
new_saver.restore(sess, './resnet/ResNet-L152.ckpt')
graph = tf.get_default_graph()
images = graph.get_tensor_by_name("images:0")
out = graph.get_tensor_by_name("avg_pool:0")


print('loading data...\n\n')
qa_data = load_data()
print('done loading data...\n\n')
#print(qa_data)
qa = qa_data['training']
split = 'train'

batch_size = 50
batch_no = 7200
im_dir = os.listdir('./im_feat_indiv/train/')
while batch_no*batch_size < len(qa_data['training']):
    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si

    im_ids = []
    count = 0
    for i in range(si, ei):
        im_ids.append(qa[i]['image_id'])
    im_ids = set(im_ids)
    im_ids = list(im_ids)


    for i in range(len(im_ids)):
        temp = str(im_ids[i])
        while len(temp) < 6:
            temp = '0' + temp
        im_ids[i] = temp


    not_found = []
    nf2 = []
    for i in range(len(im_ids)):
        filename = 'COCO_' + split + '2014_000000' + im_ids[i] + '.npy'
        fname2 = 'COCO_' + split + '2014_000000' + im_ids[i]
        if filename not in im_dir:
            not_found.append(filename)
            nf2.append(fname2)

    not_found = set(not_found)
    not_found = list(not_found)

    nf2 = set(nf2)
    nf2 = list(nf2)

    #print(not_found)
    #print(nf2)

    for i in range(len(nf2)):
        filename = nf2[i]
        #print(filename)
        im = get_image(filename + '.jpg')
        im = np.reshape(im, [1,IM_DIM,IM_DIM,3])
        im_feat = sess.run(out, feed_dict={images: im})
        np.save('./im_feat_indiv/train/' + filename, im_feat)
        print('batch_no = ' + str(batch_no) + ', filename = ' + filename)

    batch_no += 1




# im_dir = os.listdir('./Images/mscoco/train2014')
# out_dir = os.listdir('./im_feat_indiv/train/')
# for i in range(len(im_dir)):
#     filename, file_extension = os.path.splitext(im_dir[i])
#     filename2 = filename + '.npy'
#     if filename2 not in out_dir:
#         im = get_image(filename + '.jpg')
#         im = np.reshape(im, [1,IM_DIM,IM_DIM,3])
#         im_feat = sess.run(out, feed_dict={images: im})
#         np.save('./im_feat_indiv/train/' + filename, im_feat)
#         print('not in - filename = ' + filename)
#     else:
#         print('in - filename = ' + filename)
#     print(i)







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

