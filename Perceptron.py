

import numpy as np

def uniform_vector(X,y):
    [m,n] = X.shape
    x_e = np.ones([m,1])
    uniform_X = np.hstack((X,x_e))
    for i in range(len(y)):
        if y[i] == 1:
            uniform_X[i] *=-1
    return uniform_X

def Perceptron(X,N): # X - набор унифицированных вектор-признаков

    [m,n] = X.shape
    k = 0
    w = np.zeros([n],'f')
    n = 0
    true_count = 0
    while 1:
        if true_count == len(X):
            print('Число итераций %i' %(k))
            return w
        if k == N:
            print('Число итераций %i' %(k))
            return w
        if n == len(X):
            n = 0
        if np.all(np.dot(X[n],w) > 0):
            true_count = true_count + 1
        else:
            true_count = 0
            w = w + X[n].T
        k = k + 1
        n = n + 1

if __name__ == "__main__":
    X = np.array([[1,2],[0,2],[1,3],[3,2]])
    y = [0,0,1,1]
    uniform_X = uniform_vector(X,y)
    w = Perceptron(uniform_X,1000)


# In[ ]:



