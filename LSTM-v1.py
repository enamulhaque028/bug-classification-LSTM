# -*- coding:utf-8 -*-
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import f1_score
from sklearn.utils import shuffle


def split_data(data):
    s_data, l_data = [], []
    for d in data:
        s_data.append(d[1])
        l_data.append(d[2])
    s_data = np.asarray(s_data, dtype="int64")
    l_data = np.asarray(l_data, dtype="int64")
    return s_data, l_data


def LSTM(x, num_hidden, num_layer, keep_prob, weights, biases):
    # single layer version
    lstm_cell = rnn.BasicLSTMCell(num_hidden)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    """
    # multiple layers version
    def LSTM_cell(num_hidden, keep_prob):
        return rnn.DropoutWrapper(cell=rnn.BasicLSTMCell(num_hidden), input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cell = rnn.MultiRNNCell([LSTM_cell(num_hidden, keep_prob) for _ in range(num_layer)])
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    """

    # Linear activation, using rnn inner loop last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.matmul(tf.reduce_mean(outputs, axis=0), weights['out']) + biases['out']


def BiLSTM(x, num_hidden, num_layer, keep_prob, weights, biases):
    # single layer version
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden)
    lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                 dtype=tf.float32)

    """
    # multiple layers version
    def LSTM_cell(num_hidden, keep_prob):
        return rnn.DropoutWrapper(cell=rnn.BasicLSTMCell(num_hidden), input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_fw_cell = rnn.MultiRNNCell([LSTM_cell(num_hidden, keep_prob) for _ in range(num_layer)])
    lstm_bw_cell = rnn.MultiRNNCell([LSTM_cell(num_hidden, keep_prob) for _ in range(num_layer)])
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                 dtype=tf.float32)
    """

    # Linear activation, using rnn inner loop last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.matmul(tf.reduce_mean(outputs, axis=0), weights['out']) + biases['out']


def train_model(n_epoch, batch_size, sequence_length, num_hidden, num_layer,
                num_classes, non_static, project_name):
    # load data
    print("load data")
    train_data = pickle.load(open(project_name + "/train_nn.pkl", "rb"))
    valid_data = pickle.load(open(project_name + "/valid_nn.pkl", "rb"))
    test_data = pickle.load(open(project_name + "/test_nn.pkl", "rb"))
    print("train data :", len(train_data))
    print("valid data :", len(valid_data))
    print("test data :", len(test_data))
    index2vec = pickle.load(open(project_name + "/index2vec.pkl", 'rb'))
    # index2vec = pickle.load(open(project_name + "/index2vec_pt.pkl", 'rb'))
    index2vec = np.asarray(index2vec, dtype='float32')
    train_sentence, train_label = split_data(train_data)
    valid_sentence, valid_label = split_data(valid_data)
    test_sentence, test_label = split_data(test_data)
    train_label_ohe = np.eye(2)[train_label]
    n_train_batches = train_sentence.shape[0] // batch_size + 1
    n_valid_batches = valid_sentence.shape[0] // batch_size + 1
    n_test_batches = test_sentence.shape[0] // batch_size + 1

    # build the model
    print("build the model")
    # tf Graph input
    X = tf.placeholder(tf.int64, [None, sequence_length])
    Y = tf.placeholder(tf.int64, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    # Define weights
    weights = {
        # for LSTM
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        # for BiLSTM
        # Hidden layer weights => 2 * n_hidden because of forward + backward cells
        # 'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    word_embedding = tf.get_variable(name='embedding', shape=index2vec.shape, dtype=tf.float32,
                                     initializer=tf.constant_initializer(index2vec),
                                     trainable=non_static)
    X_embedding = tf.nn.embedding_lookup(word_embedding, X)
    X_embedding = tf.unstack(X_embedding, axis=1)
    # for LSTM
    logits = LSTM(X_embedding, num_hidden, num_layer, keep_prob, weights, biases)
    # for BiLSTM
    # logits = BiLSTM(X_embedding, num_hidden, num_layer, keep_prob, weights, biases)
    predict_op = tf.argmax(tf.nn.softmax(logits), axis=1)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    train_op = tf.train.AdamOptimizer().minimize(loss_op)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # train the model
    print("train the model")
    epoch = 0
    best_valid_score = None
    while epoch < n_epoch:
        epoch += 1
        batch_cost = 0.
        cur_train_sentence, cur_train_label_ohe = shuffle(train_sentence, train_label_ohe, random_state=epoch)
        for batch_index in range(n_train_batches):
            batch_start = batch_size * batch_index
            batch_end = min(batch_size * (batch_index + 1), train_sentence.shape[0])
            s_batch = cur_train_sentence[batch_start: batch_end]
            l_batch = cur_train_label_ohe[batch_start: batch_end]
            _, cost = sess.run([train_op, loss_op], feed_dict={X: s_batch, Y: l_batch, keep_prob: 0.5})
            batch_cost += cost
            if (batch_index + 1) % 20 == 0:
                print(("epoch %i/%i, batch %d/%d, average batch cost %f") % (
                    epoch, n_epoch, batch_index + 1, n_train_batches, batch_cost / 20))
                batch_cost = 0.
        if n_train_batches % 20 != 0:
            print(("epoch %i/%i, batch %d/%d, average batch cost %f") % (
                epoch, n_epoch, batch_index + 1, n_train_batches, batch_cost / (n_train_batches % 20)))
        valid_prediction = []
        for batch_index in range(n_valid_batches):
            batch_start = batch_size * batch_index
            batch_end = min(batch_size * (batch_index + 1), valid_sentence.shape[0])
            s_batch = valid_sentence[batch_start: batch_end]
            cur_pred = sess.run(predict_op, feed_dict={X: s_batch, keep_prob: 1})
            valid_prediction += list(cur_pred)
        cur_score = f1_score(valid_label, valid_prediction)
        print("valid f1_score :", cur_score)
        if best_valid_score is None or best_valid_score < cur_score:
            print("better score on validation set")
            best_valid_score = cur_score
            test_prediction = []
            for batch_index in range(n_test_batches):
                batch_start = batch_size * batch_index
                batch_end = min(batch_size * (batch_index + 1), test_sentence.shape[0])
                s_batch = test_sentence[batch_start: batch_end]
                cur_pred = sess.run(predict_op, feed_dict={X: s_batch, keep_prob: 1})
                test_prediction += list(cur_pred)
            test_score = f1_score(test_label, test_prediction)
            print("test f1_score :", test_score)
    print("=====")
    print("valid f1_score :", best_valid_score)
    print("test f1_score :", test_score)


if __name__ == "__main__":
    train_model(n_epoch=5,
                batch_size=32,
                sequence_length=100,
                num_hidden=100,
                num_layer=3,
                num_classes=2,
                non_static=True,
                project_name=sys.argv[1])
