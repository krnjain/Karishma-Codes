# from convert2 import print_prob, load_image
import tensorflow as tf


# def resnet_experiment():
#     img = load_image("data/cat.jpg")

#     sess = tf.Session()

#     new_saver = tf.train.import_meta_graph('./ResNet-L152.meta')
#     new_saver.restore(sess, './ResNet-L152.ckpt')

#     graph = tf.get_default_graph()
#     #train_writer = tf.summary.FileWriter('./resnet152graph',graph)

#     prob_tensor = graph.get_tensor_by_name("prob:0")
#     images = graph.get_tensor_by_name("images:0")
#     # test = graph.get_tensor_by_name("scale5/block3/Relu:0")
#     test = graph.get_tensor_by_name("avg_pool:0")
#     print(test)

#     #for op in graph.get_operations():
#     #    print "\n"
#     #    print op.name
#     #    print op.values()

#     #init = tf.initialize_all_variables()
#     #sess.run(init)
#     print "graph restored"

#     batch = img.reshape((1, 224, 224, 3))

#     feed_dict = {images: batch}

#     test2 = sess.run(test, feed_dict=feed_dict)
#     print(test2)

#     prob = sess.run(prob_tensor, feed_dict=feed_dict)

#     print_prob(prob[0])


def get_resnet(sess, image_in):	
    #sess = tf.Session()
    new_saver = tf.train.import_meta_graph('./resnet/ResNet-L152.meta')
    new_saver.restore(sess, './resnet/ResNet-L152.ckpt')
    graph = tf.get_default_graph()
    images = graph.get_tensor_by_name("images:0")
    out = graph.get_tensor_by_name("avg_pool:0")
    #sess.run(tf.global_variables_initializer())

    im_features = sess.run(out, feed_dict={images: image_in})
    #sess.close()
    return im_features #out, images

