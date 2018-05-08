import numpy as np
from scipy.linalg import expm, logm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn

dof = 3
n = 3

G = np.zeros((dof, n, n))
G[0, 1, 2] = 1
G[0, 2, 1] = -1
G[1, 0, 2] = -1
G[1, 2, 0] = 1
G[2, 0, 1] = 1
G[2, 1, 0] = -1

def alg(c):
  """ Return matrix repr. of lie algebra vector c

  INPUT
    c (ndarray, [3,]): vector of so(3) coefficients

  OUTPUT
    C (ndarray, [3,3]): matrix-valued tangent vector in se(3)
  """
  phi = c[0]*G[0] + c[1]*G[1] + c[2]*G[2]
  return phi

def algi(C):
  """ Return vector of so(3) coefficients

  INPUT
    C (ndarray, [3,3]): matrix-valued tangent vector in so(3)

  OUTPUT
    c (ndarray, [3,]): vector of so(3) coefficients.
  """
  return np.array([ C[2, 1], C[0, 2], C[1, 0] ])

def Adj(X):
  """ Return Adj_X for X an element of SO(3)

  For the function
    f(X, c) = algi(logm(X expm(alg(c)) X^{-1}))
  returns the Jacobian
    frac{ partial f }{ partial c } |_{c=0}
  where c \in R^{dof}, X \in SO(3)

  INPUT
    X (ndarray, [3,3]): Element of SO(3)

  OUTPUT
    A (ndarray, [3,3]): Jacobian of f
  """
  return X

def rvs(mean=None, cov=np.eye(6)):
  """ Sample SO(3) RV.

  INPUTS
    mean (ndarray, [3,3]): mean element in SO(3)
    cov (ndarray, [3,3]): covariance in so(3)

  OUTPUTS
    x (ndarray, [3,3]): RV in SO(3)
  """
  eps = alg(mvn.rvs(np.zeros(dof), cov))
  if mean is None: return expm(eps)
  else: return expm(eps).dot(mean)