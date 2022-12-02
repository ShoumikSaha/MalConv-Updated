import os
import time
import pickle
import argparse
import pandas as pd
from keras_preprocessing.sequence import pad_sequences

def preprocess(fn_list, max_len=250000):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    for fn in fn_list:
        if not os.path.isfile(fn):
            print(fn, 'not exist')
        else:
            with open(fn, 'rb') as f:
                corpus.append(f.read())

    corpus = [[byte for byte in doc] for doc in corpus]
    len_list = [len(doc) for doc in corpus]
    # print(len(corpus), len(corpus[0]))
    # print(corpus[0][:100])
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, len_list


def preprocess_single_file(file, max_len = 250000):
    if not os.path.isfile(file):
        print(file, 'not exist')
    else:
        with open(file, 'rb') as f:
            data = f.read()
            data = [[byte for byte in data]]
    data_len = len(data)
    seq = pad_sequences(data, maxlen=max_len, padding='post', truncating='post')
    return seq, data_len
