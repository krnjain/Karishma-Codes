#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn_cell



def rnn_module(var_dict, batch_size=50, dropout_rate=0.2, input_embedding_size=200, vocabulary_size=15881, rnn_size=512,
    rnn_layer=2, max_words_q=22, dim_hidden=2048):
    '''
    Inputs:     batch_size - size of bath
                dropout_rate - fraction of nodes (of LSTM) to apply dropout to
                input_embedding_size - the encoding size of each token in the vocabulary
                vocabulary_size - number of wors in the question vocabulary
                rnn_size - size of the rnn in number of hidden nodes in each layer
                rnn_layer - number of layers in the rnn
                max_words_q - maximum possible number of words in question
                dim_hidden - size of the common embedding vector
    '''
    #embed_ques_W = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08), name='embed_ques_W')

    #embed_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, dim_hidden], -0.08,0.08),name='embed_state_W')
    #embed_state_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_state_b')

    embed_ques_W = var_dict['rnnqW']
    embed_state_W = var_dict['rnnsW']
    embed_state_b = var_dict['rnnsb']

    lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True, state_is_tuple=False)
    lstm_dropout_1 = rnn_cell.DropoutWrapper(lstm_1, output_keep_prob=1-dropout_rate)
    lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True, state_is_tuple=False)
    lstm_dropout_2 = rnn_cell.DropoutWrapper(lstm_2, output_keep_prob=1-dropout_rate)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm_dropout_1, lstm_dropout_2], state_is_tuple=False)

    question = tf.placeholder(tf.int32, [batch_size, max_words_q])

    state = stacked_lstm.zero_state(batch_size, tf.float32)
    loss = 0.0
    for i in range(max_words_q):
        if i==0:
            ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
        else:
            tf.get_variable_scope().reuse_variables()
            ques_emb_linear = tf.nn.embedding_lookup(embed_ques_W, question[:,i-1])

        ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-dropout_rate)
        ques_emb = tf.tanh(ques_emb_drop)

        output, state = stacked_lstm(ques_emb, state)

    #state_drop = tf.nn.dropout(state, 1-dropout_rate)
    state_linear = tf.nn.xw_plus_b(state, embed_state_W, embed_state_b)
    # state_emb = tf.tanh(state_linear)
    state_emb = tf.nn.relu(state_linear)

    return state_emb, question


# from get_data import * 
# sess = tf.Session()
# state_emb, question = rnn_test()
# sess.run(tf.global_variables_initializer())
# qa_data = load_data()
# batch_no = 0
# batch_size = 50
# (questions, answer, image) = get_training_batch(batch_no, batch_size, qa_data)
# test = sess.run(state_emb, feed_dict={question: questions})
# print(test)
# print(test.shape)
