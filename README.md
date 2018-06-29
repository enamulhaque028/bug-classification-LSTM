# bug-classification-LSTM
Classifying Bug Reports into Bugs and Non-bugs Using LSTM.

Please unzip the datasets in dataset folder before running the programme.

preprocess.py: script to process the dataset.

LSTM-v1.py: script for LSTM-based method without pre-trained word embedding.

LSTM-v2.py: script for LSTM-based method with pre-trained word embedding.
Need to download pre-trained word embedding from https://code.google.com/archive/p/word2vec/.

First, run preprocess.py to preprocess the data.
Then, run LSTM-v1.py/LSTM-v2.py with project name (jackrabbit/lucene/heepclient/all) to train and evaluate the model.
