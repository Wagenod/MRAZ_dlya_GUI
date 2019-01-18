 # -*- coding: utf-8 -*-
import f
import numpy as np
import math
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
#для случая бинарной классификации
def calculate_E_and_D(y,predict_y,c):
    #predict_y = predict_y.T[0]
    p,n = [0,0]
    l = len(y)
    for i in range(len(y)):
        if y[i] == c and predict_y[i] == c:
            p+=1
        elif y[i] !=c and predict_y[i] == c:
            n+=1
    try:
        E = n/(n+p)
    except:
        E = 0
    D = p/l
    return E,D

def calculate_p_and_n(y,predict_y,c):
    P,N,p,n = [0,0,0,0]
    #P = y.count(c)
    for i in range(len(y)):
        if y[i] == c:
            P+=1
    N = len(y) - P
    for i in range(len(y)):
        if y[i] == c and predict_y[i] == c:
            p+=1
        elif y[i] !=c and predict_y[i] == c:
            n+=1
    return P,N,p,n

def calculate_informativ(P,N,p,n):
    def binom_coef(n,k):
        if k > n:
            return 0
        a = math.factorial(n)
        b = math.factorial(k)
        c = math.factorial(n-k)
        result = a/(b*c)
        return result
    Ic = -math.log(binom_coef(P,p)*binom_coef(N,n)/binom_coef(N+P,n+p))
    return Ic

def predict(X,con,c):
    other_c = 0 if c == 1 else 1
    '''
    if len(con) == 1:
        X[con[0][0](X)] = c
    else:
        for i in range(len(X)):
            if con[0][0](X[i]) and con[1][0](X[i]):
                X[i] = c
    X[ X!=c] = other_c
    return X
    '''
    y_predict = []
    for i in range(len(X)):
        flag = True
        for term in con:
            if not term[0](X[i]):
                    flag = False
        if flag:
            y_predict.append(c)
        else:
            y_predict.append(other_c)
    y_predict = np.array(y_predict)
    return y_predict

def KOPA(X,y,V,K,E_max,D_min,T_min,T_max):
    def add_to_list(R,con,T,c):
        if len(R) < T:
            R.append(con)
        else:
            predict_y = predict(X.copy(),con,c)
            P,N,p,n = calculate_p_and_n(y,predict_y,c)
            con_informative = calculate_informativ(P,N,p,n)
            Ic = []
            for r in R:
                predict_y = predict(X.copy(),r,c)
                P,N,p,n = calculate_p_and_n(y,predict_y,c)
                informative = calculate_informativ(P,N,p,n)
                Ic.append(informative)
            Ic = np.array(Ic)
            i = Ic.argmin()
            if con_informative > Ic[i]:
                R.pop(i)
                R.append(con)
        print(f'Длина списка {len(R)} для класса {c}')
    def increase(con,j_s):
        if con is None:
            j_s = -1
            con = []
        for j in range(j_s + 1,len(V)):
            con.append(V[j])
            y0_predict = predict(X.copy(),con,0)
            y1_predict = predict(X.copy(),con,1)
            E0,D0 = calculate_E_and_D(y,y0_predict,0)
            E1,D1 = calculate_E_and_D(y,y1_predict,1)
            if K >= len(con) and ((D0 >= D_min and E_max >= E0) or (D1 >= D_min and E_max >= E1)):
                if (D0 >= D_min and E_max >= E0) and len(R_0) < len(R_1):
                    add_to_list(R_0,con,T_max,0)
                elif (D1 >= D_min and E_max >= E1):
                    add_to_list(R_1,con,T_max,1)
                else:
                    add_to_list(R_0,con,T_max,0)
                con = []
            elif K > len(con) and ((D0 >= D_min) or (D1 >= D_min)):
                j_s = j
                increase(con,j_s)
            else:
                con.pop()
    R_0 = []
    R_1 = []
    h = 0.01
    while 1:
        increase(None,0)
        T = min(len(R_0),len(R_1))
        if T > T_min and T <= T_max:
            print(f'D_min = {D_min}, E_max = {E_max}')
            return R_0, R_1
        if T < T_min:
            D_min = D_min - h
            E_max = E_max + h
        if T > T_max:
            D_min = D_min + h
            E_max = E_max - h
            
def classification(R,X,y,c):
    i = 1
    print(f'Класс {c}')
    for con in R:
        y_predict = predict(X,con,c)
        E,D = calculate_E_and_D(y,y_predict,c)
        print(f'Для коньюнкции {i}: E = {E}, D = {D}')
        print(classification_report(y,y_predict))
        i+=1                        
        
            
if __name__ == '__main__':
    [X, y] = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1)
    print (X)
	print(y)
	V = [] #множество элементарных предикатов
    for i in range(len(X)):
        V.append((f.p_lte(X[i]),X[i],'=<'))
        V.append((f.p_gte(X[i]),X[i],'>='))
    R_1, R_2 = KOPA(X,y,V,2,0.4,0.6,2,4)
    classification(R_2,X,y,1)
        
        
        
        
        