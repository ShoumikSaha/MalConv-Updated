import numpy as np
from preprocess import preprocess_single_file
import tensorflow as tf
import keras
from tensorflow import convert_to_tensor
from preprocess import preprocess
import utils
import pandas as pd

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


def train_model(model, input_file_list, ab_len, epoch, max_len=250000):
    # tf.keras.backend.clear_session()
    # model = create_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], run_eagerly=True)

    # input = 'DikeDataset/data_label_200.csv'
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    data = preprocess(filenames, max_len)[0]
    print(data.shape)
    dataset_sample = []
    dataset_label = []
    for i, sample in enumerate(data):
        ablated_sample, labels = create_ablations(sample, label[i], ab_len)
        dataset_sample.append(ablated_sample)
        dataset_label.append(labels)
    del data

    dataset_sample = np.array(dataset_sample)
    dataset_label = np.array(dataset_label)
    dataset_sample = np.reshape(dataset_sample, (-1, max_len))
    dataset_label = np.reshape(dataset_label, (-1))
    print(dataset_sample.shape, dataset_label.shape)
    x_train, x_test, y_train, y_test = utils.train_test_split(dataset_sample, dataset_label)
    del dataset_sample, dataset_label

    # epoch = 20
    save_path = 'saved/my_train/vit_model.h5'
    prev_acc = 0

    history = model.fit(x_train, y_train, batch_size=64)
    prev_loss = history.history['loss']
    model.save(save_path)
    for i in range(epoch):
        print("Epoch ", i + 1)
        model = tf.keras.models.load_model(save_path)
        history = model.fit(x_train, y_train, batch_size=64)
        loss = history.history['loss']
        y_pred = model.predict(x_test)
        acc = get_acc(y_pred, y_test)
        print(acc)
        if acc >= prev_acc and loss < prev_loss:
            model.save(save_path)
            prev_acc = acc
            prev_loss = loss
            print("New model saved!")

    y_pred = model.predict(x_test)
    print(get_acc(y_pred, y_test))


def get_acc(y_pred, y_test):
    acc = 0

    for i in range(len(y_pred)):
        if (y_pred[i] > 0.5):
            pred_label = 1
        else:
            pred_label = 0
        if (pred_label == y_test[i]):
            acc += 1
    return acc / len(y_pred)


def create_ablations(input, label, ab_len, max_len=250000):
    ablations = []
    labels = []
    start_idx = 0
    while (start_idx < max_len):
        left_array = np.zeros((start_idx))
        ablate = input[start_idx: start_idx + ab_len]
        # ablate = np.reshape(-1)
        right_array = np.zeros((max(max_len - (start_idx + ab_len), 0)))
        combined_ablate = np.concatenate((left_array, ablate, right_array))
        ablations.append(combined_ablate)
        start_idx += ab_len
        labels.append(label)

    ablations = np.array(ablations)
    labels = np.array(labels)
    # print(labels)
    return ablations, labels


def evaluate_ablations(model, ablations, ab_len, max_len=250000):
    class0 = 0
    class1 = 0
    for ablation in ablations:
        ablation = np.reshape(ablation, (1, max_len))
        pred = model.predict(ablation, verbose=0)
        if (pred < 0.5):
            class0 += 1
        else:
            class1 += 1
            # print("Predicted 1!")

    return 0 if class0 > class1 else 1


def run_model_ablation(model, input_file, label, ab_len, max_len=250000):
    data = preprocess_single_file(input_file, max_len)[0]
    ablations = create_ablations(data[0], label, ab_len)[0]
    pred_class = evaluate_ablations(model, ablations, ab_len)
    print(pred_class)
    return True if pred_class == label else False
