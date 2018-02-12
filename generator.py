import tensorflow as tf
import numpy as np


def combine_embeddings(cnn, rnn):
    '''
    Takes as input the embeddings of the image and question produced
    by the CNN and RNN. Embeddings must have the same dimensions
    '''
    summed = tf.add(cnn,rnn)
    multiplied = tf.multiply(cnn,rnn)
    return tf.concat([summed, multiplied], 1)


def generator_me(features, noise, var_dict):
    '''
    Takes as input the feature vector produced by combining the 
    embeddings of the image and question and runs them through
    generative network to produce the corresponding answer
    
    Input feature vectors assumed to have dimensions [batch_size, 4096]
    
    noise should be tensor containing random noise that has the same 
    dimensions as features
    
    If weights==None, weights will be randomly initialized. Otherwise
    weights should be a dict containing the weight values to use for
    each layer
    '''
    # if weights==None:
    #     weights = {}
    #     weights["W1"] = tf.truncated_normal([4096*2, 4096*4])
    #     weights["b1"] = tf.constant(0.1, shape=[4096*4])
    #     weights["W2"] = tf.truncated_normal([4096*4, 4096*4])
    #     weights["b2"] = tf.constant(0.1, shape=[4096*4])
    #     weights["W3"] = tf.truncated_normal([4096*4, 4096*3])
    #     weights["b3"] = tf.constant(0.1, shape=[4096*3])
    #     weights["W4"] = tf.truncated_normal([4096*3, 4096*2])
    #     weights["b4"] = tf.constant(0.1, shape=[4096*2])
    #     weights["W5"] = tf.truncated_normal([4096*2, 4096])
    #     weights["b5"] = tf.constant(0.1, shape=[4096])
    #     weights["W6"] = tf.truncated_normal([4096, 3000])
    #     weights["b6"] = tf.constant(0.1, shape=[3000])

    # x = tf.concat([features, tf.random_normal(features.get_shape())], 1)
    x = tf.concat([features, noise], 1)

    #fc1W = tf.Variable(weights["W1"])
    #fc1b = tf.Variable(weights["b1"])
    #fc1W = tf.get_variable("generatorfc1W", initializer=tf.truncated_normal([4096*2, 4096*4]))
    #fc1b = tf.get_variable("gfc1b", initializer=weights["b1"], trainable=True)
    fc1 = tf.nn.relu_layer(x, var_dict['gfcW1'], var_dict['gfcb1'])

    #fc2W = tf.Variable(weights["W2"])
    #fc2b = tf.Variable(weights["b2"])
    #fc2W = tf.get_variable("gfc2W", initializer=weights["W2"], trainable=True)
    #fc2b = tf.get_variable("gfc2b", initializer=weights["b2"], trainable=True)
    fc2 = tf.nn.relu_layer(fc1, var_dict['gfcW2'], var_dict['gfcb2'])

    #fc3W = tf.Variable(weights["W3"])
    #fc3b = tf.Variable(weights["b3"])
    #fc3W = tf.get_variable("gfc3W", initializer=weights["W3"], trainable=True)
    #fc3b = tf.get_variable("gfc3b", initializer=weights["b3"], trainable=True)
    fc3 = tf.nn.relu_layer(fc2, var_dict['gfcW3'], var_dict['gfcb3'])

    #fc4W = tf.Variable(weights["W4"])
    #fc4b = tf.Variable(weights["b4"])
    #fc4W = tf.get_variable("gfc4W", initializer=weights["W4"], trainable=True)
    #fc4b = tf.get_variable("gfc4b", initializer=weights["b4"], trainable=True)
    fc4 = tf.nn.relu_layer(fc3, var_dict['gfcW4'], var_dict['gfcb4'])

    #fc5W = tf.Variable(weights["W5"])
    #fc5b = tf.Variable(weights["b5"])
    #fc5W = tf.get_variable("gfc5W", initializer=weights["W5"], trainable=True)
    #fc5b = tf.get_variable("gfc5b", initializer=weights["b5"], trainable=True)
    #fc5 = tf.nn.relu_layer(fc4, var_dict['gfcW5'], var_dict['gfcb5'])

    #fc6W = tf.Variable(weights["W6"])
    #fc6b = tf.Variable(weights["b6"])
    #fc6W = tf.get_variable("gfc6W", initializer=weights["W6"], trainable=True)
    #fc6b = tf.get_variable("gfc6b", initializer=weights["b6"], trainable=True)
    #fc6 = tf.nn.relu_layer(fc5, var_dict['gfcW6'], var_dict['gfcb6'])

    #theta_g = [fc1W, fc1b, fc2W, fc2b, fc3W, fc3b, fc4W, fc4b, fc5W, fc5b, fc6W, fc6b]

    return fc4 #, theta_g


#c = tf.random_normal([50, 2048])
#r = tf.random_normal([50, 2048])
#features = combine_embeddings(c, r)
#g = generator(features)
#print(g.get_shape())







