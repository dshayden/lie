import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
import lie

dof = 1
n = 2

G = np.zeros((dof, n, n))
G[0, 0, 1] = -1
G[0, 1, 0] = 1

def expm(c):
  """ Exponential map of SO(2). """
  t = c[1,0]
  ct = np.cos(t); st = np.sin(t)
  return np.array([[ct, -st], [st, ct]])

def logm(c):
  """ Log map of SO(2). """
  t = np.arctan2(c[1,0], c[0,0])
  return np.array([[0, -t], [t, 0]])

def inv(X):
  return X.T

def Adj(X):
  return np.array([1,])

def alg(c):
  """ Return matrix repr. of lie algebra vector c

  INPUT
    c (ndarray, [1,]): vector of so(2) coefficients

  OUTPUT
    C (ndarray, [2,2]): matrix-valued tangent vector in so(2)
  """
  # return lie.modAngle(c)*G[0]
  return c*G[0]

def algi(C):
  """ Return vector of se(2) coefficients

  INPUT
    C (ndarray, [2,2]): matrix-valued tangent vector in so(2)

  OUTPUT
    c (ndarray, [2,]): vector of so(2) coefficients.
  """
  return C[1,0]
  # return lie.modAngle(C[1,0])
