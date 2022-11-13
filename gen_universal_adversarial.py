import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors


def fgsm_attack_universal(input_emb, input_label, model, e, loss_function=tf.keras.losses.BinaryCrossentropy()):
    # print(input.shape)
    input_emb = tf.convert_to_tensor(input_emb)
    input_label = tf.convert_to_tensor(input_label)
    tf.config.run_functions_eagerly(True)

    with tf.GradientTape() as g:
        g.watch(input_emb)
        d1 = model.layers[2](input_emb)
        d2 = model.layers[3](input_emb)
        d3 = model.layers[4](d2)
        d4 = model.layers[5]([d1, d3])
        d5 = model.layers[6](d4)
        d6 = model.layers[7](d5)
        d7 = model.layers[8](d6)
        prediction = model.layers[9](d7)

        print("Prediction: ", prediction, "Input Label: ", input_label)
        loss = loss_function(input_label, prediction)

    gradient = g.gradient(loss, input_emb)
    # print("Gradient: ", gradient)
    perturbation = e * np.sign(gradient.numpy())
    # print("Perturbation: ", perturbation.shape, perturbation)
    return perturbation


def get_emb(model, input):
    emb_layer = model.layers[1]

    # inp2emb = tf.keras.backend.function([model.input] + [tf.keras.backend.learning_phase()], [emb_layer.output])  # [function] Map sequence to embedding
    # inp_emb = np.squeeze(np.array(inp2emb([input, False])), 0)

    inp_emb = emb_layer(tf.convert_to_tensor(input))
    # print("Input Embedding: ", inp_emb.shape, inp_emb)
    return inp_emb


def get_input_from_emb_by_matrix(inp_emb, emb_layer, max_len):
    emb_matrix = tf.linalg.matmul(inp_emb, tf.linalg.pinv(emb_layer.get_weights()[0]))
    # out = np.zeros((max_len, ))
    # out = np.argmax(emb_matrix.numpy(), axis=0)
    emb_matrix = emb_matrix.numpy()
    print(emb_matrix.shape)
    out = np.zeros((max_len))
    for i in range(emb_matrix.shape[1]):
        max_idx = np.argmax(emb_matrix[0][i])
        out[i] = max_idx
    out = np.reshape(out, (1, -1))
    print(out.shape)
    return out


def get_input_from_emb(input, inp_emb, neigh_model):
    out = input.copy()
    # print(inp_emb.shape)
    # out = tf.linalg.matmul(inp_emb, tf.linalg.pinv(emb_layer.get_weights()[0]))
    # print(inp_emb.shape, out.shape)

    for idx in range(len(inp_emb)):
        target = inp_emb[idx].reshape(1, -1)
        best_idx = neigh_model.kneighbors(target, 1, False)[0][0]
        out[0][idx] = best_idx

    return out


def modify_the_padding_section(input, perturb, pad_idx, pad_len):
    # print("Perturb matrix: ", perturb[0][pad_idx : pad_idx+pad_len])
    for idx in range(pad_idx, pad_idx + pad_len):
        input[0][idx] += perturb[0][idx]
    return input

def modify_the_padding_section_universal(input, perturb, pad_idx, pad_len):
    # print("Perturb matrix: ", perturb[0][pad_idx : pad_idx+pad_len])
    for idx in range(pad_idx, pad_idx + pad_len):
        input[0][idx] += perturb[0][pad_idx-idx]
    return input

def add_patch_end_of_file(input, patch, pad_idx, pad_len, max_len):
    pad_len = max(min(pad_len, max_len-pad_idx),0)
    #print(input.shape, patch.shape, pad_idx, pad_len)
    input[0][pad_idx:pad_idx+pad_len] = patch[:pad_len]

    return input, pad_len

def get_common_perturb(perturbations, pad_len):
    for perturb in perturbations:
        #perturb = perturb.numpy()
        a = np.zeros((pad_len-perturb.shape[0], 8))
        print("Perturb: ", perturb.shape, "a: ", a.shape)
        perturb = np.concatenate((perturb, a))
        print("Perturb shape after concat:", perturb.shape)
    perturbations = np.array(perturbations)
    print("Perturbations: ", perturbations.shape)
    return np.average(perturbations, axis=0)




def iterative_attack_universal(attack, inputs, pad_idx, pad_len, input_label, model, iterations, e,
                     loss_function=tf.keras.losses.BinaryCrossentropy(), max_len=250000):
    emb_layer = model.layers[1]
    print("Iteration ", iterations)
    perturbations = []
    prev_preds = []
    for i, input in enumerate(inputs):
        input = np.reshape(input, (1, max_len))
        pad_len_i = max(min(pad_len, max_len - pad_idx[i]), 0)
        #input, pad_len_i = add_patch_end_of_file(input, patch, pad_idx[i], pad_len, max_len)
        #print(input.shape)
        inp_emb = get_emb(model, input)
        #inp_emb = modify_the_padding_section_universal(inp_emb.numpy(), patch_emb, pad_idx[i], pad_len_i)
        #print(inp_emb.shape)
        perturb = attack(inp_emb, input_label, model, e)
        #print("Perturb: ", perturb.shape)
        perturbations.append(perturb[0][pad_idx[i]:pad_idx[i]+pad_len_i])
        inp_emb = modify_the_padding_section(inp_emb.numpy(), perturb, pad_idx[i], pad_len_i)
        inp_emb = tf.convert_to_tensor(inp_emb)
        input = get_input_from_emb_by_matrix(inp_emb, emb_layer, max_len)
        prev_pred = model.predict(input)
        print("Prev Prediction: ", prev_pred)
        prev_preds.append((prev_pred))

    common_perturb = get_common_perturb(perturbations, pad_len)
    common_perturb = np.reshape(common_perturb, (1, pad_len, 8))
    print("Common Perturbation: ", common_perturb.shape, common_perturb)
    print("Non zero count in comm pert: ", np.count_nonzero(common_perturb))

    new_preds = []
    for i, input in enumerate(inputs):
        input = np.reshape(input, (1, max_len))
        inp_emb = get_emb(model, input)
        pad_len_i = max(min(pad_len, max_len-pad_idx[i]),0)
        inp_emb = modify_the_padding_section_universal(inp_emb.numpy(), common_perturb, pad_idx[i], pad_len_i)
        inp_emb = tf.convert_to_tensor(inp_emb)
        input = get_input_from_emb_by_matrix(inp_emb, emb_layer, max_len)
        inputs[i] = input
        new_pred = model.predict(input)
        print("New Prediction: ", new_pred)
        new_preds.append(new_pred)

    avg_prev_pred = sum(prev_preds)/len(prev_preds)
    avg_new_pred = sum(new_preds)/len(new_preds)

    print("Avg prev pred: ", avg_prev_pred, "Avg new pred: ", avg_new_pred)
    if(avg_new_pred<0.5):
        return inputs, True
    elif(iterations>=0):
        iterative_attack_universal(attack, inputs, pad_idx, pad_len, input_label, model, iterations-1, e)
    else:
        return inputs, False


    return inputs, True

