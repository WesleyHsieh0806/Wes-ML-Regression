# -*- coding: utf-8 -*-
import csv
import math
import sys
import pandas as pd
import numpy as np
import os

'''# **Testing**
![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)

載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。
'''

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
test_data = pd.read_csv(sys.argv[1], header=None, encoding='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype=float)

for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)


'''test data normalization and feature selection'''

mean_x = np.mean(test_x, axis=0)  # 18 * 9
std_x = np.std(test_x, axis=0)  # 18 * 9

for i in range(len(test_x)):  # 12 * 471
    for j in range(len(test_x[0])):  # 18 * 9
        if not j in range(81, 90):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
# for column in range(len(test_x[0])):
#     if column in range(0, 18) or column in range(36, 63) or column in range(90, 108) or column in range(117, 152) or column in range(153, 162):
#         test_x[:, column] = 0

test_x = np.concatenate(
    (np.ones([len(test_x), 1]), test_x), axis=1).astype(float)
test_x

"""# **Prediction**
說明圖同上

![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)

有了 weight 和測試資料即可預測 target。
"""

w = np.load('weight.npy')

ans_y = np.dot(test_x, w)
ans_y

"""# **Save Prediction to CSV File**"""
file_dir, file_name = os.path.split(sys.argv[2])

if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']

    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
