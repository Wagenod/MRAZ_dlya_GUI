
# coding: utf-8

# In[5]:

import numpy as np


def bipolation(y):
    y[y == 0] = -1
    return y

def sign(y):
    y = np.sign(y)
    y[y == 0] = 1
    return y


def Hebb(X,y,N): # y - биполярный
    [m, n] = X.shape
    x_e = np.ones([m, 1])
    X = np.hstack((X, x_e))
    w = np.zeros([n + 1], 'f')
    k = 0
    index = 0
    while 1:
        if k == N:
            #print('Число итераций %i' %(k))
            return w, k
        if index == len(X):
            index = 0
        w = w + X[index]*y[index]
        if np.all((sign(np.dot(X,w) - 1) == y)):
            #print('Число итераций %i' %(k))
            return w, k
        else:
            k = k + 1
            index = index + 1

#if __name__ == '__main__':
#
#    X = np.array([[3,2],[2,2],[3,3],[4,3]])
#    y = [0,0,1,1]
#    bip_y = bipolation(y)
#    w = Hebb(X,bip_y,1000)
#    print (w)


# In[ ]:



