
# coding: utf-8

# In[1]:

import numpy as np


def Haming(X,w):
    print('Haming')
    [m,n] = X.shape # m эталонных вектров и n признаков. Метод shape возвращает колво строк и столбцов в матрице
    h = 1/n
    e = np.zeros([m])
    Q = np.full([m,m],-h)
    for i in range(m):
        Q[i,i] = 1
    y = n/2 + (1/2)*np.dot(X,w) #вектор мер близостей
    k = 0
    while 1:
        s = np.dot(Q,y)
        s[s < 0] = 0
        if np.all(s == y): # если все  s равны y (сравниваем массивы)
            n_class = np.nonzero(s)[0][0]#берем первую строку и первый столбец
            print('Алгоритм окончил работу')
            print('Номер класса: %i' %(n_class + 1))
            return
        else:
            y = s
            k += 1


if __name__ == '__main__':
    X = np.array([[1,1,1,-1,1,-1],[-1,-1,-1,1,1,1]])
    y = np.array([1,1,1,-1,1,1])
    Haming(X,y)


# In[ ]:



