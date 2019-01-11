import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

def Stop_Cond(X, W):
    if np.all(np.matmul(X, W) > 0):
        return True
    return False
def Cond_Insepar(E):
    if np.all(E <= 0) and np.all(E != 0):
        return True
    return False
def comp_E(X, W, b):
    return np.around((np.matmul(X, W) - b).astype(np.double), 6);
def NSKO_alg(Matrix_X, Vector_y, h):
    n_row = Matrix_X.shape[0];
    Matrix_X = np.concatenate((Matrix_X, np.ones((n_row, 1))), axis=1)

    b = np.ones((n_row, 1));
  
    for i in range(n_row):
        Matrix_X[i, :] = Matrix_X[i, :] if Vector_y[i] == 0 else -Matrix_X[i, :]
    del Vector_y

    i = 1;
    Vector_w = np.matmul(np.linalg.pinv(Matrix_X), b);

    while(True):
        E = comp_E(Matrix_X, Vector_w, b);
        if (Stop_Cond(Matrix_X, Vector_w)):
            return Vector_w;
        if (Cond_Insepar(E)):
            return np.zeros(Vector_w.shape);
        b = b + h * (np.multiply(E, np.heaviside(E, 0)));
        Vector_w = np.matmul(np.linalg.pinv(Matrix_X), b);
    return Vector_w;

Matrix_X = np.array([[1, 2], [0, 2], [1, 3], [3, 2]]);  
Vector_y = np.array([[0], [0], [1], [1]]); 
w = NSKO_alg(Matrix_X, Vector_y, h = 1.6)
print(w)
w = np.poly1d(w)
print(np.poly1d(w))
#fig=plt.figure
#ax=fig.add_subplot(111)
#ax.plot(w)