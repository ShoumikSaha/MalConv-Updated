import tensorflow as tf
import keras
from tensorflow import convert_to_tensor

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation

def create_model(max_len=250000, win_size=500, vocab_size=256):
    inp = Input((max_len,))
    emb = Embedding(vocab_size, 8)(inp)

    conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid')(conv2)

    mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(mul)
    p = GlobalMaxPool1D()(a)
    d = Dense(64)(p)
    out = Dense(1, activation='sigmoid')(d)

    model = Model(inp, out)

    return model