import torch as T
import numpy as np
import pandas as pd

from dso_torch import Dso


X = np.array([[1.], [2.], [3.], [4.], [5.]])
y = np.array([2., 4., 6., 8., 10.])
batch_size = 8
epochs = 100

tokens_lib = {'*': 2, '/': 2, 'cos': 1, 'sin': 1, 'exp': 1, '-': 2, 'x_0':0, 'log':1, '+': 2, 'c': 0}
tokens_indices = {0: '+', 1: '*', 2: '-', 3: '/', 4: 'sin', 5: 'cos', 6: 'log', 7: 'exp', 8: 'x_0', 9: 'c'}

constraints = {
    'cos': ['sin', 'cos'],
    'sin': ['cos', 'sin'],
    'exp': ['exp'], 
    'log': ['log'],
    'no_parent': ['x_0', 'c']
    #'max_level': ['log', 'exp', 'sin', 'cos', '/', '+', '-', '*']
    }

input_dim = len(tokens_lib.keys()) * 2
output_dim = len(tokens_lib.keys())

dso = Dso(X=X, y=y, network_dims=[input_dim, output_dim * 3, output_dim], batch_size=batch_size, num_layers=2, lr=1e-3, tokens_lib=tokens_lib, tokens_indices=tokens_indices, constraints=constraints)

for i in range(epochs):
    # index = np.random.choice(range(len(X)), size=batch_size, replace=False)
    # batch = X[index.astype(int)].astype(dtype=np.float32)
    dso.train(X)
    