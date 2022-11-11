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