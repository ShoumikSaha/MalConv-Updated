import tensorflow as tf
import numpy as np

def non_targeted_attack(input, input_label, model, e=0.01, loss_function = tf.keras.losses.BinaryCrossentropy()):
    #print(input.shape)
    input = tf.convert_to_tensor(input)
    input_label = tf.convert_to_tensor(input_label)

    with tf.GradientTape() as g:
        g.watch(input)
        prediction = model(input)
        print(prediction, input_label)
        loss = loss_function(input_label, prediction)

    gradient = g.gradient(loss, input)
    print(g)
    perturbation = e * np.sign(gradient.numpy())

    return perturbation


def iterative_attack(attack, input, input_label, model, iterations, e, loss_function=tf.keras.losses.BinaryCrossentropy()):
    adv_input = input.copy()
    for i in range(iterations):
        print("Iteration ", i)
        n = attack(adv_input, input_label, model, e, loss_function)
        adv_input += n
        adv_input = np.clip(adv_input, input - e, input + e)
    adv_input = np.clip(adv_input, 0, 1)
    return adv_input


