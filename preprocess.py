# -*- coding:utf-8 -*-
import os
import re
import gensim
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize


def index2vector(word2index_file, index2vec_out, dim, scale, seed=0):
    print("index2vec", word2index_file)
    word2index = pickle.load(open(word2index_file, "rb"))
    vocab_size = len(word2index)
    index2vec = np.zeros((vocab_size + 1, dim), dtype="float32")
    index2vec[0] = np.zeros(dim)  # 0 used for padding
    np.random.seed(seed)
    for word in word2index:
        index = word2index[word]
        index2vec[index] = np.random.uniform(-scale, scale, dim)
    pickle.dump(np.asarray(index2vec), open(index2vec_out, "wb"))


def index2vector_pretained(word2index_file, index2vec_out, dim, scale, seed=0):
    print("index2vector_pretained", word2index_file)
    word_emb = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    word2index = pickle.load(open(word2index_file, "rb"))
    vocab_size = len(word2index)
    index2vec = np.zeros((vocab_size + 1, dim), dtype="float32")
    index2vec[0] = np.zeros(dim)  # 0 used for padding
    np.random.seed(seed)
    unknown_words = 0
    for word in word2index:
        index = word2index[word]
        try:
            cur_vec = word_emb.get_vector(word)
        except Exception:
            cur_vec = np.random.uniform(-scale, scale, dim)
            unknown_words += 1
        index2vec[index] = cur_vec
    print("total words : ", vocab_size)
    print("unknown words : ", unknown_words)
    pickle.dump(np.asarray(index2vec), open(index2vec_out, "wb"))


def pad_forward(in_file, out_file, data_length):
    print("pad_forward", in_file)
    nn_data = []
    for data in pickle.load(open(in_file, 'rb')):
        sent = data[1]
        if len(sent) >= data_length:
            sent = sent[:data_length]
        else:
            pad = [0] * (data_length - len(sent))
            sent = pad + sent
        nn_data.append([data[0], sent, data[2]])
    pickle.dump(nn_data, open(out_file, 'wb'))


def preprocess_project(project_name):
    if not os.path.exists(project_name):
        os.mkdir(project_name)

    input_file = open("dataset/" + project_name + ".pkl", "rb")
    info = pickle.load(input_file)
    input_file.close()

    key = []
    key_id = []
    summary = []
    description = []
    for i in range(len(info)):
        key.append(info[i]['key'])
        cur_index = info[i]['key'].find("-")
        key_id.append(int(info[i]['key'][cur_index + 1:]))
        summary.append(info[i]['fields']['summary'])
        description.append(info[i]['fields']['description'])
    key = np.array(key)
    key_id = np.array(key_id)
    summary = np.array(summary)
    description = np.array(description)

    indicies = np.argsort(key_id)
    key = key[indicies]
    summary = summary[indicies]
    description = description[indicies]

    summary_descriptions = []
    for i in range(len(summary)):
        cur_summary = summary[i]
        cur_description = description[i]
        summary_words = list(word_tokenize(cur_summary))
        summary_words = [word.lower() for word in summary_words]
        try:
            description_words = list(word_tokenize(cur_description))
            description_words = [word.lower() for word in description_words]
        except Exception:
            description_words = []
        summary_descriptions.append(summary_words + description_words)
    summary_descriptions = np.array(summary_descriptions)

    processed_summary_descriptions = []
    pattern1 = re.compile("^[0-9\.]+$")
    for i in range(len(summary_descriptions)):
        cur_summary_description = []
        for word in summary_descriptions[i]:
            if pattern1.match(word) is not None:
                cur_summary_description.append("<NUM>")
                continue
            cur_summary_description.append(word)
        processed_summary_descriptions.append(cur_summary_description)
    processed_summary_descriptions = np.array(processed_summary_descriptions)

    info = pd.read_csv("dataset/" + project_name + "_classification_vs_type.csv")
    label = list((info['CLASSIFIED'] == "BUG").astype(int))
    label = np.array(label)
    label = label[indicies]

    word2index = {}
    index_label = []
    index = 1  # 0 used for padding
    for i in range(len(processed_summary_descriptions)):
        sent_index = []
        for word in processed_summary_descriptions[i]:
            if word not in word2index:
                word2index[word] = index
                sent_index.append(index)
                index += 1
            else:
                sent_index.append(word2index[word])
        index_label.append([key[i], sent_index, label[i]])
    pickle.dump(word2index, open(project_name + "/word2index.pkl", 'wb'))
    pickle.dump(index_label, open(project_name + "/index_label.pkl", 'wb'))

    index2vector(project_name + "/word2index.pkl", project_name + "/index2vec.pkl", dim=100, scale=0.1)
    index2vector_pretained(project_name + "/word2index.pkl", project_name + "/index2vec_pt.pkl", dim=300, scale=0.1)
    pad_forward(project_name + "/index_label.pkl", project_name + "/index_label_nn.pkl", data_length=100)

    input_file = open(project_name + "/index_label_nn.pkl", "rb")
    data = pickle.load(input_file)
    input_file.close()
    train_valid_data = data[:int(len(data) * 0.9)]
    train_valid_data = shuffle(train_valid_data, random_state=0)
    train_data = train_valid_data[:int(len(train_valid_data) * 0.9)]
    valid_data = train_valid_data[int(len(train_valid_data) * 0.9):]
    test_data = data[int(len(data) * 0.9):]
    pickle.dump(train_data, open(project_name + "/train_nn.pkl", 'wb'))
    pickle.dump(valid_data, open(project_name + "/valid_nn.pkl", 'wb'))
    pickle.dump(test_data, open(project_name + "/test_nn.pkl", 'wb'))


def preprocess_all():
    if not os.path.exists("all"):
        os.mkdir("all")

    word2index = {}
    index_label = []
    index = 1  # 0 used for padding

    for project_name in ["jackrabbit", "lucene", "httpclient"]:
        input_file = open("dataset/" + project_name + ".pkl", "rb")
        info = pickle.load(input_file)
        input_file.close()

        key = []
        summary = []
        description = []
        for i in range(len(info)):
            key.append(info[i]['key'])
            summary.append(info[i]['fields']['summary'])
            description.append(info[i]['fields']['description'])

        summary_descriptions = []
        for i in range(len(summary)):
            cur_summary = summary[i]
            cur_description = description[i]
            summary_words = list(word_tokenize(cur_summary))
            summary_words = [word.lower() for word in summary_words]
            try:
                description_words = list(word_tokenize(cur_description))
                description_words = [word.lower() for word in description_words]
            except Exception:
                description_words = []
            summary_descriptions.append(summary_words + description_words)
        summary_descriptions = np.array(summary_descriptions)

        processed_summary_descriptions = []
        pattern1 = re.compile("^[0-9\.]+$")
        for i in range(len(summary_descriptions)):
            cur_summary_description = []
            for word in summary_descriptions[i]:
                if pattern1.match(word) is not None:
                    cur_summary_description.append("<NUM>")
                    continue
                cur_summary_description.append(word)
            processed_summary_descriptions.append(cur_summary_description)
        processed_summary_descriptions = np.array(processed_summary_descriptions)

        info = pd.read_csv("dataset/" + project_name + "_classification_vs_type.csv")
        label = list((info['CLASSIFIED'] == "BUG").astype(int))
        for i in range(len(processed_summary_descriptions)):
            sent_index = []
            for word in processed_summary_descriptions[i]:
                if word not in word2index:
                    word2index[word] = index
                    sent_index.append(index)
                    index += 1
                else:
                    sent_index.append(word2index[word])
            index_label.append([key[i], sent_index, label[i]])
    pickle.dump(word2index, open("all/word2index.pkl", 'wb'))
    pickle.dump(index_label, open("all/index_label.pkl", 'wb'))

    index2vector("all/word2index.pkl", "all/index2vec.pkl", dim=100, scale=0.1)
    index2vector_pretained("all/word2index.pkl", "all/index2vec_pt.pkl", dim=300, scale=0.1)
    pad_forward("all/index_label.pkl", "all/index_label_nn.pkl", data_length=100)

    input_file = open("all/index_label_nn.pkl", "rb")
    data = pickle.load(input_file)
    input_file.close()
    info = []
    for project_name in ['jackrabbit', 'lucene', 'httpclient']:
        input_file = open("dataset/" + project_name + ".pkl", "rb")
        info += pickle.load(input_file)
        input_file.close()
    key_id = []
    for i in range(len(info)):
        key_id.append(info[i]['id'])
    key_id = np.array(key_id).astype(int)
    indicies = np.argsort(key_id)
    train_valid_indicies = indicies[:int(len(indicies) * 0.9)]
    test_indicies = indicies[int(len(indicies) * 0.9):]
    train_valid_data = [data[i] for i in train_valid_indicies]
    test_data = [data[i] for i in test_indicies]
    train_valid_data = shuffle(train_valid_data, random_state=0)
    train_data = train_valid_data[:int(len(train_valid_data) * 0.95)]
    valid_data = train_valid_data[int(len(train_valid_data) * 0.95):]
    pickle.dump(train_data, open("all/train_nn.pkl", 'wb'))
    pickle.dump(valid_data, open("all/valid_nn.pkl", 'wb'))
    pickle.dump(test_data, open("all/test_nn.pkl", 'wb'))


if __name__ == "__main__":
    print("preprocess jackrabbit")
    preprocess_project("jackrabbit")
    print("preprocess lucene")
    preprocess_project("lucene")
    print("preprocess httpclient")
    preprocess_project("httpclient")
    print("preprocess all")
    preprocess_all()
