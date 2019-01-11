
# coding: utf-8

# In[2]:

import numpy as np
from numpy import linalg
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import seaborn as sns


def Kohonen(X,y):
    # будем использовать евклидово расстояние
    print('Kohonen')
    h = 0.5
    dh = 0.0001 # h должен уменьшаться, шаг выбираем тут
    [m,n] = X.shape
    a = []
    for i in range(2):
        a.append(np.ones([n]))
    w = np.array([elem for elem in a])
    index = 0
    k = 0
    while h > 0:
        if index == len(X):
            index = 0
        i = 0
        min_d = linalg.norm(X[index] - w[0],ord=2) #вычисление евклидова расстояния
        for j in range(1,len(w)):
            if linalg.norm(X[index] - w[j],ord=2) < min_d:
                min_d = linalg.norm(X[index] - w[j],ord=2)
                i = j
        w_new = w[i] + h*(X[index] - w[i])
        if np.all(w_new == w[i]):
            break
        else:
            w[i] = w_new
            h = h - dh
            k = k + 1
            index = index + 1
    plt.scatter(X[:, 0], X[:, 1], marker='.', c=y, s=25, edgecolor='k')
    plt.scatter(w[:, 0], w[:, 1], marker='o', s=25)
    plt.show()
    return w


if __name__ == '__main__':
    sns.set()
    X = np.array([[1,2],[0,2],[1,3],[3,2]])
    y = [0,0,1,1]
    w = Kohonen(X,y)


# In[ ]:



