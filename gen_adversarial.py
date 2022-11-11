import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors


def non_targeted_attack(input_emb, input_label, model, e=0.01, loss_function=tf.keras.losses.BinaryCrossentropy()):
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

        print("Prediction: ", prediction[0][0], "Input Label: ", input_label)
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


def get_input_from_emb(input, inp_emb, neigh_model):
    out = input.copy()
    # print(inp_emb.shape)
    for idx in range(len(inp_emb)):
        target = inp_emb[idx].reshape(1, -1)
        best_idx = neigh_model.kneighbors(target, 1, False)[0][0]
        out[0][idx] = best_idx
    return out


def modify_the_padding_section(input, perturb, pad_idx, pad_len):
    for idx in range(pad_idx, pad_idx + pad_len):
        input[0][idx] += perturb[0][idx]
    return input


def iterative_attack(attack, input, pad_idx, pad_percent, input_label, model, iterations, e,
                     loss_function=tf.keras.losses.BinaryCrossentropy(), max_len=250000):
    emb_layer = model.layers[1]
    emb_weight = emb_layer.get_weights()[0]
    neigh = NearestNeighbors(n_neighbors=1).fit(emb_weight)

    print("Pad Index: ", pad_idx)
    pad_len = max(min(int(pad_idx * pad_percent), max_len - pad_idx), 0)
    print("Pad Length: ", pad_len)
    if (pad_len <= 0):
        print("Exceed length!")
        return input, False
    # inp_emb = get_emb(model, input)
    # print("Input Embedding: ", inp_emb.shape, inp_emb)
    # print("Initial Prediction: ", model.layers[2](tf.convert_to_tensor(inp_emb)))
    prev_pred = model.predict(input)
    print("Initial prediction: ", prev_pred)
    for i in range(iterations):
        print("Iteration ", i)
        # inp_emb = get_emb(model, input)
        inp_emb = get_emb(model, input)
        perturb = attack(inp_emb, input_label, model, e, loss_function)
        # inp_emb += perturb
        inp_emb = modify_the_padding_section(inp_emb.numpy(), perturb, pad_idx, pad_len)
        inp_emb = tf.convert_to_tensor(inp_emb)
        # print("Input Embedding: ", inp_emb.shape, inp_emb)

        input = get_input_from_emb(input, inp_emb.numpy()[0], neigh)
        new_pred = model.predict(input)
        print("Prediction: ", new_pred)
        if (new_pred <= 0.5):
            break
        elif (new_pred < prev_pred):
            prev_pred = new_pred
        else:
            break
        # inp_emb = np.clip(inp_emb, input - e, input + e)
        # adv_input += n
        # adv_input = np.clip(adv_input, input - e, input + e)
    # adv_input = np.clip(adv_input, 0, 1)
    # implement knn to convert the inp_emb to adv_input
    # print("Final Prediction: ", model.layers[2]((tf.convert_to_tensor(inp_emb))))

    # adv_inp = get_input_from_emb(input, inp_emb.numpy()[0], neigh)
    # adv_predict = model.predict(input)
    # print("Adversarial Prediction: ", adv_predict)

    return input, True
