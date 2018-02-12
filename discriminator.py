import tensorflow as tf
import numpy as np


def discriminator(features, g, var_dict, epsilon=1e-3, keep_prob=0.85):
    '''
    Takes as input the d_feature vector produced by combining the
    embeddings of the image, question and the generator output and
    runs them through discrimator network to produce a 1 x 1 answer

    Input feature vectors assumed to have dimensions [batch_size, 4096+3000]

    If weights==None, weights will be randomly initialized. Otherwise
    weights should be a dict containing the weight values to use for
    each layer
    '''
    # if weights==None:
    #     weights = {}
    #     weights["W1"] = tf.truncated_normal([7096, 3548])
    #     weights["b1"] = tf.constant(0.1, shape=[3548])
    #     weights["W2"] = tf.truncated_normal([3548, 1774])
    #     weights["b2"] = tf.constant(0.1, shape=[1774])
    #     weights["W3"] = tf.truncated_normal([1774, 1000])
    #     weights["b3"] = tf.constant(0.1, shape=[1000])
    #     weights["W4"] = tf.truncated_normal([1000, 500])
    #     weights["b4"] = tf.constant(0.1, shape=[500])
    #     weights["W5"] = tf.truncated_normal([500,100])
    #     weights["b5"] = tf.constant(0.1, shape=[100])
    #     weights["W6"] = tf.truncated_normal([100, 10])
    #     weights["b6"] = tf.constant(0.1, shape=[10])
    #     weights["W7"] = tf.truncated_normal([10, 1])
    #     weights["b7"] = tf.constant(0.1, shape=[1])

    d_features = tf.concat([features, g], 1)

    #fc1W = tf.Variable(weights["W1"])
    #fc1b = tf.Variable(weights["b1"])
    #fc1W = tf.get_variable("dfc1W", initializer=weights["W1"], trainable=True)
    #fc1b = tf.get_variable("dfc1b", initializer=weights["b1"], trainable=True)
    fc1 = tf.nn.relu_layer(d_features, var_dict['dfcW1'], var_dict['dfcb1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob)
    batch_mean, batch_var = tf.nn.moments(fc1_drop, [0])
    fc1n = tf.nn.batch_normalization(fc1,batch_mean,batch_var,var_dict['dbeta1'],var_dict['dscale1'],epsilon)

    #fc2W = tf.Variable(weights["W2"])
    #fc2b = tf.Variable(weights["b2"])
    #fc2W = tf.get_variable("dfc2W", initializer=weights["W2"], trainable=True)
    #fc2b = tf.get_variable("dfc2b", initializer=weights["b2"], trainable=True)
    fc2 = tf.nn.relu_layer(fc1n, var_dict['dfcW2'], var_dict['dfcb2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob)
    batch_mean, batch_var = tf.nn.moments(fc2_drop, [0])
    fc2n = tf.nn.batch_normalization(fc2,batch_mean,batch_var,var_dict['dbeta2'],var_dict['dscale2'],epsilon)

    #fc3W = tf.Variable(weights["W3"])
    #fc3b = tf.Variable(weights["b3"])
    #fc3W = tf.get_variable("dfc3W", initializer=weights["W3"], trainable=True)
    #fc3b = tf.get_variable("dfc3b", initializer=weights["b3"], trainable=True)
    fc3 = tf.nn.relu_layer(fc2n, var_dict['dfcW3'], var_dict['dfcb3'])
    fc3_drop = tf.nn.dropout(fc3, keep_prob=keep_prob)
    batch_mean, batch_var = tf.nn.moments(fc3_drop, [0])
    fc3n = tf.nn.batch_normalization(fc3,batch_mean,batch_var,var_dict['dbeta3'],var_dict['dscale3'],epsilon)

    #fc4W = tf.Variable(weights["W4"])
    #fc4b = tf.Variable(weights["b4"])
    #fc4W = tf.get_variable("dfc4W", initializer=weights["W4"], trainable=True)
    #fc4b = tf.get_variable("dfc4b", initializer=weights["b4"], trainable=True)
    fc4 = tf.nn.relu_layer(fc3n, var_dict['dfcW4'], var_dict['dfcb4'])
    fc4_drop = tf.nn.dropout(fc4, keep_prob=keep_prob)
    batch_mean, batch_var = tf.nn.moments(fc4_drop, [0])
    fc4n = tf.nn.batch_normalization(fc4,batch_mean,batch_var,var_dict['dbeta4'],var_dict['dscale4'],epsilon)

    #fc5W = tf.Variable(weights["W5"])
    #fc5b = tf.Variable(weights["b5"])
    #fc5W = tf.get_variable("dfc5W", initializer=weights["W5"], trainable=True)
    #fc5b = tf.get_variable("dfc5b", initializer=weights["b5"], trainable=True)
    fc5 = tf.nn.relu_layer(fc4n, var_dict['dfcW5'], var_dict['dfcb5'])
    fc5_drop = tf.nn.dropout(fc5, keep_prob=keep_prob)
    batch_mean, batch_var = tf.nn.moments(fc5_drop, [0])
    fc5n = tf.nn.batch_normalization(fc5,batch_mean,batch_var,var_dict['dbeta5'],var_dict['dscale5'],epsilon)

    #fc6W = tf.Variable(weights["W6"])
    #fc6b = tf.Variable(weights["b6"])
    #fc6W = tf.get_variable("dfc6W", initializer=weights["W6"], trainable=True)
    #fc6b = tf.get_variable("dfc6b", initializer=weights["b6"], trainable=True)
    fc6 = tf.nn.xw_plus_b(fc5n, var_dict['dfcW6'], var_dict['dfcb6'])
    #batch_mean, batch_var = tf.nn.moments(fc6, [0])
    #fc6n = tf.nn.batch_normalization(fc6,batch_mean,batch_var,var_dict['dbeta6'],var_dict['dscale6'],epsilon)

    #fc7W = tf.Variable(weights["W7"])
    #fc7b = tf.Variable(weights["b7"])
    #fc7W = tf.get_variable("dfc7W", initializer=weights["W7"], trainable=True)
    #fc7b = tf.get_variable("dfc7b", initializer=weights["b7"], trainable=True)
    #fc7 = tf.nn.relu_layer(fc6n, var_dict['dfcW7'], var_dict['dfcb7'])

    #batch_mean, batch_var = tf.nn.moments(fc7, [0])
    #fc7n = tf.nn.batch_normalization(fc7,batch_mean,batch_var,var_dict['dbeta7'],var_dict['dscale7'],epsilon)

    d_prob = tf.nn.sigmoid(fc6)

    #theta_d = [fc1W, fc1b, fc2W, fc2b, fc3W, fc3b, fc4W, fc4b, fc5W, fc5b, fc6W, fc6b, fc7W, fc7b]

    return d_prob, fc6, fc5n, fc4n #, batch_mean, batch_var, fc7, fc7n #, theta_d


#c = tf.random_normal([50, 2048])
#r = tf.random_normal([50, 2048])
#features = combine_embeddings(c, r)
#g = generator(features)
#print(g.get_shape())
