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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3, random_state=123)
kmeans.fit(X)

x_pred = kmeans.predict(X)

d = dict()
for i in range(len(kmeans.labels_)):
    if str(kmeans.labels_[i]) not in d:
        d[str(kmeans.labels_[i])] = []
    d[str(kmeans.labels_[i])].append(y[i])

for k, v in d.items():


exit()


# dataset
letters = pd.read_csv("../datasets/letter-recognition.csv")

# average feature values
round(letters.drop('letter', axis=1).mean(), 2)

# splitting into X and y
X = letters.drop("letter", axis = 1)
y = letters['letter']

# scaling the features
X_scaled = scale(X)

le = LabelEncoder()
y = le.fit_transform(y)

# train test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)

kmeans = KMeans(n_clusters=26, n_init=25)
kmeans.fit(x_train, y_train)

pred_test = kmeans.predict(x_test)

print(pred_test)
print(y_test)

acc = accuracy_score(y_test, pred_test)
print(acc)
