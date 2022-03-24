#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:40:52 2022

@author: amanda
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from mlp import *
from metrics import MSE
from optimizers import SGD

i = Input(13)
h = Hidden(ReLU, 64)
h1 = Hidden(ReLU, 32)
o = Output(Identity, 1)

model = MLP()

model.add_layer(i, h, h1,o)
model.compile_mlp()

dataset = datasets.load_boston()
X = dataset.data; y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc=MinMaxScaler()
scaler = sc.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

sc2=MinMaxScaler()
y_train = y_train.reshape((-1,1))
scaler = sc2.fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test.reshape((-1,1)))

model.train(X_train, y_train, MSE(), SGD(0.01), 500, 16)

print(model.evaluate(X_test, y_test, MSE))