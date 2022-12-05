from preprocess import preprocess
from random import random
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA



def get_emb(model, input):
    emb_layer = model.layers[1]
    inp_emb = emb_layer(tf.convert_to_tensor(input))

    return inp_emb

def get_input(input_file_list, max_len):
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    data, len_list = preprocess(filenames, max_len)
    return data, len_list

def calc_pca(data_values):
    #print(data_values[0])
    pca = PCA(n_components=2)
    temp = pca.fit_transform(data_values)
    
    print(temp)
    #print(pca.singular_values_)

def pca(model, input_file_list):
    data, len_list = get_input(input_file_list, 250000)

    data_values = []
    #input = np.reshape(data[0], (1, 250000))
    #input_emb = tf.convert_to_tensor(get_emb(model, input))

    for i in range (0, 5): #change to 300 in final
        input = np.reshape(data[i], (1, 250000))
        input_emb = tf.convert_to_tensor(get_emb(model, input))

        with tf.GradientTape() as g:
            g.watch(input_emb)
            d1 = model.layers[2](input_emb)
            d2 = model.layers[3](input_emb)
            d3 = model.layers[4](d2)
            d4 = model.layers[5]([d1, d3])
            d5 = model.layers[6](d4)
            d6 = model.layers[7](d5)

        data_values.append(d6.numpy()[0])


    calc_pca(data_values)
