from preprocess import preprocess
import utils
import pandas as pd
import tensorflow as tf
from gen_adversarial import fgsm_attack, iterative_attack
from gen_universal_adversarial import fgsm_attack_universal, iterative_attack_universal
import numpy as np

def run_attack(model, input_file_list, max_len):
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    data, len_list = preprocess(filenames, max_len)
    pad_percent = 0.1
    print(data.shape)
    evasion_count = 0
    total_count = 0
    pad_len = 20000
    iterative_attack_universal(fgsm_attack_universal, data[0:20], len_list[0:20], pad_len, [[1.0]], model, iterations=1, e=0.5)

    """
    for i, input in enumerate (data):
        print("Attacking on ", filenames[i])
        input = np.reshape(input, (1, max_len))
        #print(input.shape)

        #print(label[i])
        adv_sample, is_added = iterative_attack(fgsm_attack, input, len_list[i], pad_percent, [[1.0]], model, 50, 0.5)
        pred_score = model.predict(adv_sample)
        print(pred_score)
        if(pred_score<=0.5): evasion_count += 1
        if(is_added==True): total_count += 1
        if(i==50):  break
    """

    print("Total: ", total_count, "Evaded: ", evasion_count)
    print("Evasion Accuracy: ", evasion_count/total_count)