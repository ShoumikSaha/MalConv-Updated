from malconv import create_model
from preprocess import preprocess
import utils
import pandas as pd
import tensorflow as tf
import keras as k

def get_acc(y_pred, y_test):
    acc = 0

    for i in range(len(y_pred)):
        if(y_pred[i]>0.5):
            pred_label = 1
        else:
            pred_label = 0
        if(pred_label==y_test[i]):
            acc += 1
    return acc / len(y_pred)

def train_model(model, input_file_list, max_len, epoch):
    #model = create_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], run_eagerly=True)

    #input = 'DikeDataset/data_label_200.csv'
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    data = preprocess(filenames, max_len)[0]
    x_train, x_test, y_train, y_test = utils.train_test_split(data, label)

    #epoch = 20
    save_path = 'saved/my_train/model.h5'
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
