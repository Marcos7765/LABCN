import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import pandas as pd

class Interpol():
  def __init__(self, x, y):
    self.x = np.array(x, dtype=np.float64)
    self.y = np.array(y, dtype=np.float64)

  def lagrange(self, X):
    #interpolacao lagrangiana [fazer]
    x = self.x.copy()
    y = self.y.copy()
    res = 0
    return res

  def newton(self, X):
    #interpolacao newtoniana [fazer]
    x = self.x.copy()
    y = self.y.copy()
    res = 0
    return res

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