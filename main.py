# This is a sample Python script.
import train
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from malconv import create_model
from train import train_model
from run_attack import run_attack
import tensorflow as tf
tf.config.run_functions_eagerly(True)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    tf.config.run_functions_eagerly(True)
    max_len = 250000
    #model = create_model(max_len)
    save_path = 'saved/my_train/model.h5'
    model = tf.keras.models.load_model(save_path)
    print(model.summary())

    input_file_list = 'DikeDataset/data_label_200.csv'
    adv_file_list = 'DikeDataset/adv_label_new.csv'
    #train_model(model, input_file_list, max_len, epoch=30)
    run_attack(model, adv_file_list, max_len)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
