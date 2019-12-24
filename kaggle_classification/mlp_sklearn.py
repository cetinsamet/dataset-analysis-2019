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
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, MinMaxScaler, Normalizer


# dataset
letters = pd.read_csv("../datasets/letter-recognition.csv")

# average feature values
round(letters.drop('letter', axis=1).mean(), 2)

# splitting into X and y
X = letters.drop("letter", axis = 1)
y = letters['letter']


# train test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

### Zero centered
#mean_train  = np.mean(x_train, axis=0)
#std_train   = np.std(x_train, axis=0)

#x_train = (x_train - mean_train) / std_train
#x_test  = (x_test - mean_train) / std_train
### --------------

### Mean normalization
mean_train  = np.mean(x_train, axis=0)
max_train   = np.max(x_train, axis=0)
min_train   = np.min(x_train, axis=0)

x_train = (x_train - mean_train) / (max_train - min_train)
x_test  = (x_test - mean_train) / (max_train - min_train)
### --------------

### Normalizing
#norm = Normalizer(norm='l1')
#x_train = norm.fit_transform(x_train)
#x_test = norm.transform(x_test)
### --------------

### Min-Max Scaling
#mms = MinMaxScaler()
#x_train = mms.fit_transform(x_train)
#x_test = mms.transform(x_test)
### --------------

mlp = MLPClassifier(hidden_layer_sizes=(100, 200), random_state=123)
mlp.fit(x_train, y_train)

pred_test = mlp.predict(x_test)

acc = accuracy_score(y_test, pred_test)
print(acc)
