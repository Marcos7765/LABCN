import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import pandas as pd

Xs = [38, 181, 243, 297, 327, 405, 521,
      583, 649, 708, 823, 892, 957]
Ys = [282, 111, 81, 127, 214, 240, 141,
      100, 154, 242, 272, 252, 153]
PxPraCm = 1/18

class Interpol():
  def __init__(self, x, y):
    self.x = np.array(x, dtype=np.float64)
    self.y = np.array(y, dtype=np.float64)

  @np.vectorize
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

  @np.vectorize
  def newton(self, X, particionada=True):
    #interpolacao newtoniana [fazer]
    x = self.x
    y = self.y
    res = 0
    if particionada:
      for p in range(2,len(x), 2):
        if X <= x[p]:
          res += y[p-2]
          res += (X-x[p-2])*(y[p-1]-y[p-2])/(x[p-1]-x[p-2])

          res += (X-x[p-2])*(X-x[p-1])*(((y[p]-y[p-1])/(x[p]-x[p-1])-((y[p-1]-y[p-2])/(x[p-1]-x[p-2]))))/(x[p]-x[p-2])
          return res
        #for i in range(len(x))
      #caso esteja depois do último ponto
      p = len(x)-1
      res += y[p-2]
      res += (X-x[p-2])*(y[p-1]-y[p-2])/(x[p-1]-x[p-2])
      res += (X-x[p-2])*(X-x[p-1])*(((y[p]-y[p-1])/(x[p]-x[p-1])-((y[p-1]-y[p-2])/(x[p-1]-x[p-2]))))/(x[p]-x[p-2])
      return res

  def adicionarPonto(self, ponto):
    self.x = np.concatenate([self.x, ponto[0]], axis=None)
    self.y = np.concatenate([self.y, ponto[1]], axis=None)

  def derivadaNum(self, x:ArrayLike, h=np.float64(0.001),func=None,
      tabela:bool=False):
    if func == 'newton': func = self.newton
    else: func = self.lagrange
    res = np.array((func(self, X=x + h) - func(self, X=x))/h, ndmin=2, dtype=np.float64)
    #daqui você pode usar os métodos do obj DataFrame pra escrever em 
    #algum arquivo ou msm passar pro ctrl+c
    if tabela: return pd.DataFrame(data=res, dtype=np.float64, 
                      columns=["x"+str(i) for i in range(0, res.size)])
    return res
  
def trapComposto(x, y, pulo=1):
  res = 0
  for i in range(pulo, len(x), pulo):
    res+= (x[i]-x[i-pulo])*(y[i]+y[i-pulo])/2
  return res

def simpson(x, y, pulo=1, modo='1/3'):
  res = 0
  if modo=='3/8':
    for i in range(3*pulo, len(x), 3*pulo):
      res+= 3*((x[i]-x[i-(3*pulo)])/3)*(y[i]+ 3*(y[i-pulo]+y[i-(2*pulo)]) +y[i-(3*pulo)])/8
  else:
    for i in range(2*pulo, len(x), 2*pulo):
      res+= ((x[i]-x[i-(2*pulo)])/2)*(y[i]+(4*y[i-pulo])+y[i-(2*pulo)])/3
  return res

def exemploLagrange():
  '''Teste da interpolação de Lagrange através dos exemplo do slide
  30 da apresentação das aulas 16 e 17.'''
  teste = Interpol([0, 1, 2, 3], [1, 2, 9, 28])
  print(teste.lagrange(1.5))

def testLagrangeParc():
  '''Teste básico da interpolação de Newton particionada de acordo com
  o roteiro'''
  teste = Interpol([0,1,2,3,5],[1,1,1,1,1])
  #print(teste.newton(1))
  #print(teste.derivadaNum([0,1,2,3],func='newton'))
  return teste.x, teste.derivadaNum(teste.x)

def testeTrapz():
  '''Teste básico do método dos trapézios composto para o comprimento
  da interpolação de lagrange de uma reta y=1 no intervalo [0,19].'''
  teste = Interpol([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
  ders = teste.derivadaNum(teste.x)[0]
  print(np.sqrt(1 + ders*ders))
  print(trapComposto(teste.x, np.sqrt(1 + ders*ders),3))

def resultados():
  '''Resultados da integração pelos métodos do trapézio composto, de
  simpsons 1/3 e 3/8 na função interpoladora dos pontos experimentais
  pelos métodos de Newton e de Lagrange para 7 e para 19 pontos 
  igualmente espaçados.'''
  inter = Interpol(Xs, Ys)
  intervalo = np.linspace(inter.x[0],inter.x[-1],19)
  print(f'19 Pontos:\n{intervalo}')
  dersN = inter.derivadaNum(intervalo, np.float64(0.000001),'newton')[0]
  dersL = inter.derivadaNum(intervalo, np.float64(0.000001))[0]
  print(f'Derivadas Newton:\n{dersN}')
  print(f'Derivadas Lagrange:\n{dersL}')
  fcompN = np.sqrt(1 + dersN*dersN)
  fcompL = np.sqrt(1 + dersL*dersL)
  #print(fcomp)
  print(f'Comprimentos:')
  print(f'Trapézio, stride 3, Lagrange:{trapComposto(intervalo, fcompL,3)*PxPraCm}cm')
  print(f'Trapézio, stride 3, Newton:{trapComposto(intervalo, fcompN,3)*PxPraCm}cm')
  print(f'Simpson 1/3, stride 3, Lagrange:{simpson(intervalo, fcompL,3)*PxPraCm}cm')
  print(f'Simpson 1/3, stride 3, Newton:{simpson(intervalo, fcompN,3)*PxPraCm}cm')
  print(f'Simpson 3/8, stride 3, Lagrange:{simpson(intervalo, fcompL,3,"3/8")*PxPraCm}cm')
  print(f'Simpson 3/8, stride 3, Newton:{simpson(intervalo, fcompN,3,"3/8")*PxPraCm}cm')
  print(f'Trapézio, stride 1, Lagrange:{trapComposto(intervalo, fcompL)*PxPraCm}cm')
  print(f'Trapézio, stride 1, Newton:{trapComposto(intervalo, fcompN)*PxPraCm}cm')
  print(f'Simpson 1/3, stride 1, Lagrange:{simpson(intervalo, fcompL,1)*PxPraCm}cm')
  print(f'Simpson 1/3, stride 1, Newton:{simpson(intervalo, fcompN,1)*PxPraCm}cm')
  print(f'Simpson 3/8, stride 1, Lagrange:{simpson(intervalo, fcompL,1,"3/8")*PxPraCm}cm')
  print(f'Simpson 3/8, stride 1, Newton:{simpson(intervalo, fcompN,1,"3/8")*PxPraCm}cm')

def testeSimpson():
  '''Teste da integração por Simpson 1/3 e 3/8 com o exemplo do slide 
  63 da apresentação das aulas 18, 19 e 20.'''
  print(simpson([0, 0.5, 1], [0, 0.479, 0.841]))
  print(simpson([0, 0.3333, 0.6667, 1], [0, 0.3272, 0.6184, 0.8415], modo='3/8'))

resultados()