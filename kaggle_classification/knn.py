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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# dataset
letters = pd.read_csv("../datasets/letter-recognition.csv")

# average feature values
round(letters.drop('letter', axis=1).mean(), 2)

# splitting into X and y
X = letters.drop("letter", axis = 1)
y = letters['letter']

# scaling the features
X_scaled = scale(X)

# train test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

pred_test = knn.predict(x_test)

acc = accuracy_score(y_test, pred_test)
print(acc)
