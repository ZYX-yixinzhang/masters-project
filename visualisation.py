import torch
import re
import matplotlib.pyplot as plt
import os
import pandas as pd

PATH = 'path to file'
model = torch.load(PATH, weights_only=False)
train = pd.read_csv('./train.csv')
train = train.sample(frac=1)
train = train.set_index('PTID')

test = pd.read_csv('./test.csv')
test = test.sample(frac=1)
test = test.set_index('PTID')

file = './cnngru.txt'
f = open(file, 'r')
pef = f.readlines()
f.close()
train_los = []
test_los = []
train_acc = []
test_acc = []
train_sen = []
test_sen = []

for n in range(200):
    train = pef[n]
    number = re.findall(r"[-+]?\d*\.\d+|\d+", train)
    number = [float(num) if '.' in num else int(num) for num in number]
    if n%2==0:
        train_los.append(number[2])
        train_acc.append(number[3])
        train_sen.append(number[-1])
    else:
        test_los.append(number[1])
        test_acc.append(number[0])
        test_sen.append(number[2])
print(train_los)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.ylim((0, 1))
plt.plot(list(range(100)), train_los, color='sandybrown', label='training loss', linestyle = '--')
plt.plot(list(range(100)), test_los, color='orange', label='test loss')
plt.plot(list(range(100)), train_acc, color='royalblue', label='training accuracy', linestyle = '--')
plt.plot(list(range(100)), test_acc, color='mediumblue', label='test accuracy')
plt.axhline(y=0.73, linestyle = ':', color='purple', linewidth=2)
plt.text(19, y=0.75, s='acc=0.73', verticalalignment='bottom',  color='purple', horizontalalignment='right')
plt.title('(a) Training Loss and Accuracy for CNN Model with customised loss')
plt.legend()
plt.show()