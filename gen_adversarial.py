import tensorflow as tf
import numpy as np

def non_targeted_attack(input_emb, input_label, model, e=0.01, loss_function = tf.keras.losses.BinaryCrossentropy()):
    #print(input.shape)
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
    print("Gradient: ", gradient)
    perturbation = e * np.sign(gradient.numpy())
    print("Perturbation: ", perturbation.shape, perturbation)
    return perturbation

def get_emb(model, input):
    emb_layer = model.layers[1]
    emb_weight = emb_layer.get_weights()[0]
    #inp2emb = tf.keras.backend.function([model.input] + [tf.keras.backend.learning_phase()], [emb_layer.output])  # [function] Map sequence to embedding
    #inp_emb = np.squeeze(np.array(inp2emb([input, False])), 0)

    inp_emb = emb_layer(tf.convert_to_tensor(input))
    print("Input Embedding: ", inp_emb.shape, inp_emb)
    return inp_emb

def iterative_attack(attack, input, input_label, model, iterations, e, loss_function=tf.keras.losses.BinaryCrossentropy()):
    #adv_input = input.copy()
    inp_emb = get_emb(model, input)
    #print("Initial Prediction: ", model.layers[2](tf.convert_to_tensor(inp_emb)))
    for i in range(iterations):
        print("Iteration ", i)
        #inp_emb = get_emb(model, input)
        perturb = attack(inp_emb, input_label, model, e, loss_function)
        inp_emb += perturb
        print("Input Embedding: ", inp_emb.shape, inp_emb)
        #inp_emb = np.clip(inp_emb, input - e, input + e)
        #adv_input += n
        #adv_input = np.clip(adv_input, input - e, input + e)
    #adv_input = np.clip(adv_input, 0, 1)
    #implement knn to convert the inp_emb to adv_input
    #print("Final Prediction: ", model.layers[2]((tf.convert_to_tensor(inp_emb))))
    return inp_emb


