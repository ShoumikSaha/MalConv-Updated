import csv
import pandas as pd
from preprocess import preprocess

file_path = 'DikeDataset/data_label.csv'
max_len = 250000
df = pd.read_csv(file_path, header=None)
filenames, label = df[0].values, df[1].values
# _, len_list = preprocess(filenames, max_len)


with open('DikeDataset/adv_label_new.csv', mode='w') as file:
        file_writer = csv.writer(file, delimiter=',')
        for i, file_name in enumerate(filenames):
            if (label[i] == 0):
                continue
            else:
                _, len_list = preprocess([file_name], max_len)
                print(len_list[0])
                if (len_list[0] < max_len):
                    file_writer.writerow([str(file_name), str(label[i])])
                    print(file_name, label[i])

