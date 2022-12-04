# This is a sample Python script.
import os

import pandas as pd

import train
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from malconv import create_model
from train import train_model, train_gen_wrapper
from run_attack import run_attack
from vit_model import run_model_ablation, train_model
import tensorflow as tf
from evaluate import evaluate

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tf.keras.backend.clear_session()
    tf.config.run_functions_eagerly(True)
    max_len = 250000
    # model = create_model(max_len)
    save_path = 'saved/my_train/30epoch_2338m_2338b_64batch_model.h5'
    model = tf.keras.models.load_model(save_path)
    print(model.summary())

    input_file_list = 'FullDataset/output/combined_new_200.csv'
    adv_file_list = 'FullDataset/output/adv_malware.csv'

    ##Use the comment sign for train or attack
    # train_model(model, input_file_list, max_len, epoch=30)
    # train_gen_wrapper(model, input_file_list, max_len, epoch=30)
    # run_attack(model, adv_file_list, max_len)
    # eval_acc = evaluate(model, input_file_list)

    # train_model(model, input_file_list, ab_len=20000, epoch=10)

    model = tf.keras.models.load_model('saved/my_train/vit_model_50k.h5')
    correct = 0

    for file in os.listdir('FullDataset/output/adv_malware'):
        pred_class = run_model_ablation(model, 'FullDataset/output/adv_malware/' + file, 1.0, ab_len=50000)
        if(pred_class==True): correct += 1
    print(correct)
    """
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    print(len(filenames))
    for i, file in enumerate(filenames):
        pred_class = run_model_ablation(model, file, label[i], ab_len=20000)
        if (pred_class == True):   correct += 1
    print(correct)
    """
    tf.keras.backend.clear_session()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
