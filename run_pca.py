from preprocess import preprocess
from random import random
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt


def get_emb(model, input):
    emb_layer = model.layers[1]
    inp_emb = emb_layer(tf.convert_to_tensor(input))

    return inp_emb

def get_input(input_file_list, max_len):
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    #print(label)
    data, len_list = preprocess(filenames, max_len)
    return data, len_list, label

def calc_pca(data_values, labels):
    # print(data_values[0])
    pca = PCA(n_components=2)
    temp = pca.fit_transform(data_values)
    
    plot_graph(temp, labels)
    #print(temp)
    #print(pca.singular_values_)

def plot_graph(data_values, data_labels):
    
    #fig = plt.figure(figsize = (8,8))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.set_xlabel('Principal Component 1', fontsize = 15)
    #ax.set_ylabel('Principal Component 2', fontsize = 15)
    #ax.set_title('2 component PCA', fontsize = 20)

    #targets = ['0', '1', '2']
    #colors = ['r', 'g', 'b']
    #print(data_values[:, 0])
    #print(data_values)
    plt.scatter(data_values[:, 0], data_values[:, 1], c=data_labels, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2 Component PCA")
    plt.legend()
    #plt.show()
    plt.savefig("my_plot.png")

def pca(model, input_file_list):
    data, len_list, labels = get_input(input_file_list, 250000)

    data_values = []
    #input = np.reshape(data[0], (1, 250000))
    #input_emb = tf.convert_to_tensor(get_emb(model, input))

    for i in range (0, 298): #change to 300 in final
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
        #print(d6.numpy())
        data_values.append(d6.numpy()[0])


    calc_pca(data_values, labels)
    #plot_graph(data_values, label)
