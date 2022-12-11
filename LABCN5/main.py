import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import pandas as pd

Xs = [38, 181, 243, 297, 327, 405, 521,
      583, 649, 708, 823, 892, 957]
Ys = [282, 111, 81, 127, 214, 240, 141,
      100, 154, 242, 272, 252, 153]

class Interpol():
  def __init__(self, x, y):
    self.x = np.array(x, dtype=np.float64)
    self.y = np.array(y, dtype=np.float64)

  def lagrange(self, X):
    #interpolacao lagrangiana [fazer]
    x = self.x
    y = self.y
    res = 0
    for i in range(len(x)):
      prod = 1
      for j in range(len(x)):
        if j == i: continue
        prod *= (X - x[j])/(x[i] - x[j])
      res +=y[i]*prod

    return res

  def newton(self, X, particionada=True):
    #interpolacao newtoniana [fazer]
    x = self.x.copy()
    y = self.y.copy()
    res = 0
    if particionada:
      for p in range(2,len(x), 2):
        if X < x[p]:
          res += y[p-2]
          res += (X-x[p-2])*(y[p-1]-y[p-2])/(x[p-1]-x[p-2])

          res += (X-x[p-2])*(X-x[p-1])*(((y[p]-y[p-1])/(x[p]-x[p-1])-((y[p-1]-y[p-2])/(x[p-1]-x[p-2]))))/(x[p]-x[p-2])
          return res
        #for i in range(len(x))
    return "Erro"

  def adicionarPonto(self, ponto):
    self.x = np.concatenate([self.x, ponto[0]], axis=None)
    self.y = np.concatenate([self.y, ponto[1]], axis=None)

  def derivadaNum(self, x:ArrayLike, h=np.float64(0.001),func=None,
      tabela:bool=False):
    if func == None: func = self.lagrange
    res = np.array((func(x + h) - func(x))/h, ndmin=2, dtype=np.float64)
    #daqui você pode usar os métodos do obj DataFrame pra escrever em 
    #algum arquivo ou msm passar pro ctrl+c
    if tabela: return pd.DataFrame(data=res, dtype=np.float64, 
                      columns=["x"+str(i) for i in range(0, res.size)])
    return res
  
def trapComposto(x, y, pulo=1):

  pass

def simpson(x, y, pulo=1, modo=1):
  pass

def exemploLagrange():
  '''Teste da interpolação de Lagrange através dos exemplo do slide
  30 da apresentação das aulas 16 e 17.'''
  teste = Interpol([0, 1, 2, 3], [1, 2, 9, 28])
  print(teste.lagrange(1.5))

def testLagrangeParc():
  '''Teste básico da interpolação de Newton particionada de acordo com
  o roteiro'''
  teste = Interpol([0,1,2,3,5],[1,2,3,3,5])
  print(teste.newton(1))