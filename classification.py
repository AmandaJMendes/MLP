#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:40:38 2022

@author: amanda
"""

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from mlp import *
from metrics import accuracy
from lossFunctions import CrossEntropy
from optimizers import SGD


i = Input(4)
h = Hidden(ReLU, 64)
h1 = Hidden(ReLU, 32)
o = Output(Softmax, 3)

model = MLP()

model.add_layer(i, h, h1,o)
model.compile_mlp()

iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

enc = OneHotEncoder(sparse=False)
y = enc.fit_transform(y.reshape((-1, 1)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)

model.train(X_train, y_train, CrossEntropy(), SGD(0.001), 300, 8, show = False)

print(model.evaluate(X_test, y_test, accuracy))

