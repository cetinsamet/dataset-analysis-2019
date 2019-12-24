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
from sklearn.preprocessing import scale
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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
mean_train  = np.mean(x_train, axis=0)
std_train   = np.std(x_train, axis=0)

x_train = (x_train - mean_train) / std_train
x_test  = (x_test - mean_train) / std_train
### --------------

bilbobagging = BaggingClassifier(MLPClassifier(max_iter=500), random_state=123)
bilbobagging.fit(x_train, y_train)

pred_test = bilbobagging.predict(x_test)

acc = accuracy_score(y_test, pred_test)
print(acc)
