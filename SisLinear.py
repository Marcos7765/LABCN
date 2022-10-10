import numpy as np

def gaussSeidel(A, x, b, iter):
    X = x.copy()
    
    for iter in range(0,iter):
        
        for i in range(0,len(X)):
            
            X[i] = b[i]
            red = 0
            for j in range(0,len(A[i])):
               
                if j == i:
                    continue
                red += X[j]*A[i][j]
            
            X[i] -= red 
            
            X[i] = X[i]/A[i][i]
    
    return X


def exemploSlide():
    A = np.array([[5, 1, 1], [3, 4, 1], [3, 3, 6]], dtype=np.float64)
    x = np.array([0, 0, 0], dtype=np.float64)
    b = np.array([5, 6, 0], dtype=np.float64)
    for i in range(5):
        print(gaussSeidel(A,x,b,i))