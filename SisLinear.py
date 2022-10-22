import numpy as np
import matplotlib.pyplot as plt

def jacobi(A, x, b, maxIter, prec):
    '''Método iterativo de Jacobi para a matriz de coeficientes A, o vetor 
    solução inicial x, o vetor de resultados b, o máximo de iterações maxIter 
    e a precisão prec. Retorna o vetor solução e uma linha de estado.'''
    X = x.copy()
    Xantigo = X.copy()
    
    for iter in range(0,maxIter):
        
        for i in range(0,len(X)):
            
            X[i] = b[i]
            red = 0
            for j in range(0,len(A[i])):
               
                if j == i:
                    continue
                red += Xantigo[j]*A[i][j]
            
            X[i] -= red 
            
            X[i] = X[i]/A[i][i]
    
        if (np.linalg.norm((X-Xantigo), ord=np.inf)
            /np.linalg.norm(X, ord=np.inf)) < prec:
            return X, "Obtido em "+str(iter)+" iterações."
        Xantigo = X.copy()
    
    return np.array(X, dtype=np.float32), "Obtido com o máximo de iterações."

def gaussSeidel(A, x, b, maxIter, prec):
    '''Método iterativo de Gauss-Seidel para a matriz de coeficientes A,
    o vetor solução inicial x, o vetor de resultados b, o máximo de iterações
    maxIter e a precisão prec. Retorna o vetor solução e uma linha de estado.
    '''
    X = x.copy()
    Xantigo = X.copy()
    
    for iter in range(0,maxIter):
        
        for i in range(0,len(X)):
            
            X[i] = b[i]
            red = 0
            for j in range(0,len(A[i])):
               
                if j == i:
                    continue
                red += X[j]*A[i][j]
            
            X[i] -= red 
            
            X[i] = X[i]/A[i][i]
    
        if (np.linalg.norm((X-Xantigo), ord=np.inf)
            /np.linalg.norm(X, ord=np.inf)) < prec:
            return X, "Obtido em "+str(iter)+" iterações."

        Xantigo = X.copy()
    
    return np.array(X, dtype=np.float32), "Obtido com o máximo de iterações."


def exemploSlide():
    '''Teste do método de Gauss-Seidel com valores do slide da aula.'''
    A = np.array([[5, 1, 1], [3, 4, 1], [3, 3, 6]], dtype=np.float64)
    x = np.array([0, 0, 0], dtype=np.float64)
    b = np.array([5, 6, 0], dtype=np.float64)
    for i in range(5):
        print(gaussSeidel(A,x,b,i))


def descript(metodo, A, x, b, prec, maxIter, **metArgs):
    '''Descriptografa o vetor mensagem b em ascii com a matriz de criptografia
    A usando o método escolhido com um vetor solução inicial x, uma precisão 
    prec, um máximo de iterações maxIter e, caso haja, outros argumentos 
    **metArgs. Retorna um vetor de letras e uma linha de estado.'''
    res = metodo(
                A=np.array(A, dtype=np.float32), 
                x = np.array(x, dtype=np.float32), 
                b = np.array(b, dtype=np.float32), 
                maxIter = maxIter,
                prec = prec,
                **metArgs
                )
    return [chr(e) for e in np.array(np.round(res[0]), dtype=int)], res[1]

def convDiag(A):
    '''Testa se a matriz A tem diagonal estritamente dominante.'''
    k = 0
    for i in (np.sum(np.abs(A),axis=0)):
        if i >= (2*np.abs(A[k][k])):
            k += 1
            return False
    return True

def parte2():
    '''Aplicação das funções para a parte 2 do relatório.'''
    C = [
        [10,0,0,1,0,1,1,1,1,1,1,1],
        [0,10,0,1,1,1,1,1,1,1,1,0],
        [1,0,10,0,1,1,1,1,1,1,1,1],
        [1,1,0,10,1,1,1,1,1,1,0,0],
        [1,1,1,0,10,0,1,1,1,1,1,1],
        [1,1,1,1,0,10,0,1,1,1,1,1],
        [1,1,1,1,1,0,10,0,1,1,1,1],
        [1,1,1,1,1,1,0,10,0,1,1,1],
        [1,1,1,1,1,1,1,0,10,0,1,1],
        [1,1,1,1,1,1,1,1,0,10,0,1],
        [1,1,1,1,1,1,1,1,1,0,10,1],
        [0,1,1,1,1,1,1,1,1,0,0,10]]

    #verificar convergência
    if(convDiag(C)):

        b = [[1938],[1936],[2039],[1839],[2017],[2020],[1318],[2020],[1934],
            [2001],[2072],[2004]]
        x = np.zeros(len(b), dtype=np.float32)
        #aplicar os métodos
        print(descript(gaussSeidel, C, x, b, np.power(10., -4), 100))
        print(descript(jacobi, C, x, b, np.power(10., -4), 100))

def decompLU(A):
    '''Função para decompor a matriz A nas matrizes triângulares L e U, também
    é um exemplo da importância de acessar arrays do numpy através da notação
    [linha, coluna] ao invés de [linha][coluna].'''
    tamanho = len(A[0])
    L = np.identity(tamanho, dtype=np.float64)
    U = np.zeros((tamanho,tamanho), dtype=np.float64)
    print(U.shape)

    for passo in range(tamanho):
        
        #linha U[passo][:]
        for j in range(tamanho):
            if(passo <= j):
                U[passo,j] += (A[passo,j] - L[passo,:passo]@U[:passo,j])

        #coluna L[:][passo]
        for i in range(tamanho):
            if(passo < i):
                L[i][passo] += ((A[i,passo] - L[i,:]@U[:,passo])
                                /U[passo,passo])

    return L, U


def exemploLU():
    '''Teste da decomposição LU no exercício da página 17 
    do slide da Aula 11'''
    A = np.array([[1, -3, 2],[-2, 8, -1],[4,-6,5]],dtype=np.float64)
    print("A:\n", A)
    res = decompLU(A)
    print("L:\n",res[0])
    print("U:\n",res[1])

def rgb2gray(rgb):
    '''Função para obter a imagem cinza, com ruído'''
   rgb = rgb.astype(float); #converte pra float
   return (rgb[:,:,0]+rgb[:,:,1]+rgb[:,:,2])*1/3

img = plt.imread('foto.jpg')
imggray = rgb2gray(img)
    '''Para tirar o ruído, remova as linhas 168 e 169'''
t = imggray.shape
imgruido = imggray + np.random.rand(t[0],t[1])
plt.imshow(imggray, cmap="gray")
plt.show()
