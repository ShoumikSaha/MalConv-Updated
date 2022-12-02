from preprocess import preprocess
import utils
import pandas as pd
import tensorflow as tf
from gen_adversarial import fgsm_attack, iterative_attack
from gen_universal_adversarial import fgsm_attack_universal, iterative_attack_universal, get_emb, \
    modify_the_padding_section_universal, get_input_from_emb_by_matrix
import numpy as np

def create_adv_malware(common_perturb_input, inputs, pad_idx, pad_len=20000, max_len=250000):
    modified_inputs = []
    for i, input in enumerate(inputs):
        input = np.reshape(input, (1, -1))
        input = input[:, 0:pad_idx[i]]
        print(input.shape)
        pad_len_i = max(min(pad_len, max_len - pad_idx[i]), 0)
        input = np.concatenate((input, common_perturb_input[:, 0:pad_len_i]), axis=1)
        print(input.shape)
        a = np.zeros((1, max_len - input.shape[1]))
        input = np.concatenate((input, a), axis=1)
        modified_inputs.append(input)
    return modified_inputs

def run_attack(model, input_file_list, max_len):
    df = pd.read_csv(input_file_list, header=None)
    filenames, label = df[0].values, df[1].values
    data, len_list = preprocess(filenames, max_len)
    pad_percent = 0.1
    print(data.shape)
    evasion_count = 0
    total_count = 0
    pad_len = 20000
    #patch_emb = get_emb(model, tf.convert_to_tensor(np.random.randint(0, 255, max_len)))
    #patch_emb = patch_emb[:, 0:20000, :]

    ##The next section is for universal attack
    adv_data, isSuccess, common_perturb = iterative_attack_universal(fgsm_attack_universal, data[0:150], len_list[0:150], pad_len, [[1.0]], model, iterations=5, e=0.5)
    print(adv_data.shape)

    adv_data = create_adv_malware(common_perturb, data[0:150], len_list[0:150])
    for i, adv in enumerate(adv_data):
        adv = np.reshape(adv, (1, max_len))
        pred = model.predict(adv, verbose=0)
        print(pred)
        if (pred < 0.5):   evasion_count += 1

        adv_file_name = 'FullDataset/output/adv_malware_2/' + str(filenames[i]).split('/')[-1]
        with open(adv_file_name, "wb") as file:
            #adv = adv.numpy()
            file.write(adv.tobytes())
    print(evasion_count)

    adv_data = create_adv_malware(common_perturb, data[150:300], len_list[150:300])
    for i, adv in enumerate(adv_data):
        adv = np.reshape(adv, (1, max_len))
        pred = model.predict(adv, verbose=0)
        print(pred)
        if (pred < 0.5):   evasion_count += 1

        adv_file_name = 'FullDataset/output/adv_malware_2/' + str(filenames[i+150]).split('/')[-1]
        with open(adv_file_name, "wb") as file:
            # adv = adv.numpy()
            file.write(adv.tobytes())
    print(evasion_count)

    """"
    for i, adv in enumerate(adv_data):
        adv = np.reshape(adv, (1, max_len))
        #print(adv.shape)
        adv = tf.convert_to_tensor(adv)
        pred = model.predict(adv, verbose=0)
        print(pred)
        if(pred<0.5):   evasion_count += 1
        
        adv_file_name = 'FullDataset/output/adv_malware/' + str(filenames[i]).split('/')[-1]
        with open(adv_file_name, "wb") as file:
            adv = adv.numpy()
            file.write(adv)
        

    total_count = adv_data.shape[0]
    print("Total: ", total_count, "Evaded: ", evasion_count)
    print("Evasion Accuracy: ", evasion_count / total_count)

    evasion_count = 0
    emb_layer = model.layers[1]
    for i in range(150, 200):
        input = np.reshape(data[i], (1, max_len))
        inp_emb = get_emb(model, input)
        pad_len_i = max(min(pad_len, max_len - len_list[i]), 0)
        inp_emb = modify_the_padding_section_universal(inp_emb.numpy(), common_perturb, len_list[i], pad_len_i)
        inp_emb = tf.convert_to_tensor(inp_emb)
        input = get_input_from_emb_by_matrix(inp_emb, emb_layer, max_len)
        input = tf.convert_to_tensor(input)
        pred = model.predict(input, verbose=0)
        print(pred)
        if (pred < 0.5):   evasion_count += 1

        
        adv_file_name = 'FullDataset/output/adv_malware/' + str(filenames[i]).split('/')[-1]
        with open(adv_file_name, "wb") as file:
            adv = input.numpy()
            file.write(adv)
        
    """
    """
    total_count = 50
    print("Total: ", total_count, "Evaded: ", evasion_count)
    print("Evasion Accuracy: ", evasion_count / total_count)
    """
    ##The next section is for normal attack
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
