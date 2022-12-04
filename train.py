from malconv import create_model
from preprocess import preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import utils
import pandas as pd
import tensorflow as tf
import keras as k
import gc
from os.path import join

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
        tf.keras.backend.clear_session()
        del model
        gc.collect()

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


def train_gen_wrapper(model, input_file_list, max_len, epoch):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], run_eagerly=True)

    df = pd.read_csv(input_file_list, header=None)
    data_g, label_g = df[0].values, df[1].values
    xx_train, xx_test, yy_train, yy_test = utils.train_test_split(data_g, label_g)
    print('Train on %d data, test on %d data' % (len(xx_train), len(xx_test)))
    
    history = train_gen(model, xx_train, xx_test, yy_train, yy_test)

    print(xx_test)
    xx_test_data = preprocess(xx_test, max_len)[0]
    yy_pred = model.predict(xx_test_data)
    print(get_acc(yy_pred, yy_test))

def train_gen(model, x_train, x_test, y_train, y_test, max_len=250000, batch_size=16, verbose=True, epochs=3, save_path='../saved/', save_best=True):
    
    # callbacks
    ear = EarlyStopping(monitor='val_acc', patience=5)
    mcp = ModelCheckpoint(join(save_path, 'malconv_gen.h5'), 
                          monitor="val_acc", 
                          save_best_only=save_best, 
                          save_weights_only=False)
    data_gen_output=utils.data_generator(x_train, y_train, max_len, batch_size, shuffle=True)
    print(len(x_train))
    history = model.fit(
        data_gen_output,
        batch_size=batch_size,
        epochs=epochs,
        verbose='auto',
        
        callbacks=[ear, mcp],
        steps_per_epoch=len(x_train)//batch_size + 1,
        validation_data=utils.data_generator(x_test, y_test, max_len, batch_size),
        validation_steps=len(x_test)//batch_size + 1,
        )
    return history