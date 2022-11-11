from preprocess import preprocess
import utils
import pandas as pd
import tensorflow as tf
from gen_adversarial import non_targeted_attack, iterative_attack
import numpy as np

def run_attack(model, input_file_list, max_len):
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    data = preprocess(filenames, max_len)[0]
    print(data.shape)
    for i, input in enumerate (data):
        print("Attacking on ", filenames[i])
        input = np.reshape(input, (1, max_len))
        #print(input.shape)

        print(label[i])
        adv_sample = iterative_attack(non_targeted_attack, input, [[1.0]], model, 5, 0.01)
        pred_score = model.predict(adv_sample)
        print(pred_score)