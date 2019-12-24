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
from sklearn.metrics import accuracy_score

# dataset
letters = pd.read_csv("../datasets/letter-recognition.csv")

# average feature values
round(letters.drop('letter', axis=1).mean(), 2)

# splitting into X and y
X = letters.drop("letter", axis = 1)
y = letters['letter']

# scaling the features
#X_scaled = scale(X)

# train test split
x, x_test, y, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

rfc = RandomForestClassifier(max_depth=100, random_state=16)

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)

n_estimators        = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features        = ['auto', 'sqrt']
max_depth           = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split   = [2, 5, 10]
min_samples_leaf    = [1, 2, 4]
bootstrap           = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model_cv = RandomizedSearchCV(estimator = rfc,
                              param_distributions = random_grid,
                              scoring= 'accuracy',
                              cv = folds,
                              verbose = 1,
                              return_train_score=True)

model_cv.fit(x, y)
best_random = model_cv.best_estimator_


pred_test = best_random.predict(x_test)
acc = accuracy_score(y_test, pred_test)
print(acc)
