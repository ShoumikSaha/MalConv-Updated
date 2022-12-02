from malconv import create_model
from preprocess import preprocess, preprocess_single_file
import utils
import pandas as pd
import tensorflow as tf
import numpy as np


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

def evaluate(model, input_file_list, max_len=250000):
    y_preds = []
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    for file in filenames:
        data = preprocess_single_file(file, max_len)[0]
        y_pred = model.predict(data, verbose=0)
        print(y_pred)
        y_preds.append(y_pred)
    acc = get_acc(np.asarray(y_preds), label)

    print(acc)
    return acc

