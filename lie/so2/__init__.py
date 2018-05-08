import numpy as np
from scipy.linalg import expm, logm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn

dof = 1
n = 2

G = np.zeros((dof, n, n))
G[0, 0, 1] = -1
G[0, 1, 0] = 1

def alg(c):
  """ Return matrix repr. of lie algebra vector c

  INPUT
    c (ndarray, [1,]): vector of so(2) coefficients

  OUTPUT
    C (ndarray, [2,2]): matrix-valued tangent vector in so(2)
  """
  return c[0]*G[0]

def algi(C):
  """ Return vector of se(2) coefficients

  INPUT
    C (ndarray, [2,2]): matrix-valued tangent vector in so(2)

  OUTPUT
    c (ndarray, [2,]): vector of so(2) coefficients.
  """
  return np.array([ C[1, 0] ])
