# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../datasets"))

# Any results you write to the current directory are saved as output.

# libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import random
import torch

# dataset
letters = pd.read_csv("../datasets/letter-recognition.csv")

# about the dataset

# dimensions
print("Dimensions: ", letters.shape, "\n")

# data types
print(letters.info())

# head
letters.head()

# a quirky bug: the column names have a space, e.g. 'xbox ', which throws and error when indexed
print(letters.columns)

# let's 'reindex' the column names
letters.columns = ['letter', 'xbox', 'ybox', 'width', 'height', 'onpix', 'xbar',
       'ybar', 'x2bar', 'y2bar', 'xybar', 'x2ybar', 'xy2bar', 'xedge',
       'xedgey', 'yedge', 'yedgex']
print(letters.columns)

order = list(np.sort(letters['letter'].unique()))
print(order)

# basic plots: How do various attributes vary with the letters

plt.figure(figsize=(16, 8))
sns.barplot(x='letter', y='onpix',
            data=letters,
            order=order)

letter_means = letters.groupby('letter').mean()
letter_means.head()

plt.figure(figsize=(18, 10))
sns.heatmap(letter_means)

# average feature values
round(letters.drop('letter', axis=1).mean(), 2)

# splitting into X and y
X = letters.drop("letter", axis = 1)
y = letters['letter']

# scaling the features
X_scaled = scale(X)

# train test split
x, x_test, y, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)

class Network(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.layer1     = nn.Linear(d_in, 128)
        self.layer2     = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 26)
        self.relu       = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.batch_norm(self.relu(self.layer1(x)))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

SEED = 16
N_EPOCH = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

n_sample, d_feat = x.shape
n_label = len(np.unique(y))

clf         = Network(d_feat, n_label)
ce_loss     = nn.CrossEntropyLoss()
optimizer   = torch.optim.Adam(clf.parameters(), lr=1e-2)
#lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

y       = np.asarray([ord(l) - 65 for l in y.values])
y_test  = np.asarray([ord(l) - 65 for l in y_test.values])

BATCH_SIZE = 1000

data        = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

for epoch_idx in range(N_EPOCH):

    clf.train()
    running_loss = 0.

    for feats, labels in data_loader:

        labels_ = clf(feats)

        batch_loss = ce_loss(labels_, labels) * BATCH_SIZE

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()

    #lr_schedular.step(running_loss)

    epoch_loss = running_loss / n_sample

    print("Epoch %4d\tLoss : %s" % (epoch_idx + 1, epoch_loss))

    if (epoch_idx + 1) % 1 == 0:
        clf.eval()
        pred_test = torch.argmax(clf(torch.from_numpy(x_test).float()), dim=1)
        print("Acc: %.2f" % (float(torch.sum(pred_test == torch.from_numpy(y_test).long()).item()) / n_sample * 100))