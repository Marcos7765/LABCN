import numpy as np
import matplotlib.pyplot as plt

Xnorm = np.array([-0.9985351563, -0.7998046875, -0.5966796875, -0.3984375,
        -0.1997070313, 0, 0.1987304688, 0.3999023438, 0.599609375,
        0.7983398438, 1], dtype=np.float32)

Ynorm = np.array([1.559570313, 1.342773438, 1.189941407, 1.08203125, 
        1.022460938, 1, 1.020996094, 1.086425782, 1.19921875, 1.352050782,
        1.558105469], dtype=np.float32)

V = 2048
Yminesc = 0.7114257813
Xcentro = 2305
h = 3546

def mmqparab(X, Y):
    '''f(x) = a0 + a1*x + a2*x² por vandro'''
    vander = np.array([[1, X[i], X[i]*X[i]] for i in range(0, len(X))],
            dtype=np.float32)
    res = np.linalg.solve(np.transpose(vander)@vander, np.transpose(vander)@Y)
    return res

def exemploSlide():
    '''teste do exemplo do slide 49 do Ajuste de Curvas Parte I usando matriz
    do walter, funciona'''
    X = np.array([-2, -1.5, 0, 1, 2.2, 3.1], dtype=np.float32)
    Y = np.array([-30.5,-20.2,-3.3,8.9,16.8,21.4], dtype=np.float32)
    print(mmqparab(X, Y))

def Yparab(x):
    res = mmqparab(Xnorm, Ynorm)
    return np.sum([np.multiply(res[i],np.power(x,i)) for i in range(len(res))],
     axis=0)

def plot(Y, Xnorm, basico=True):
    if basico:
        plt.plot(Xnorm, Y(Xnorm))
        plt.show()
    else:
        cX = Xnorm*V + Xcentro
        cY = h - (Y(Xnorm) - 1 + Yminesc)*V
        plt.imshow(plt.imread('ptsColetados.png'))
        plt.plot(cX, cY)
        plt.show()

def mmqTaylor(X, Y):
    [np.power(X[i], 2*i) for i in range(len(X))]
    '''f(x) = a0 + a1*x^2 + a2*x^4 por valdo'''
    vander = np.array([np.power([1, X[i], X[i]*X[i]],2) for i in range(0, len(X))],
            dtype=np.float32)
    res = np.linalg.solve(np.transpose(vander)@vander, np.transpose(vander)@Y)
    return res


def Ytay(x):
    res = mmqTaylor(Xnorm, Ynorm)
    return np.sum([np.multiply(res[i],np.power(x,2*i)) for i in range(len(res))],
     axis=0)


def Ycat(x):
    a0, a1, a2 = mmqTaylor(Xnorm, Ynorm)
    a = (a1*2 + np.power(a2*24, 1/3))/2
    b = a0 - a
    print([a, b])
    return a*np.cosh(x/a) + b

plot(Yparab, Xnorm, False)
plot(Ytay, Xnorm, False)
plot(Ycat, Xnorm, False)
