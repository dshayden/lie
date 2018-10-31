import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
import lie
from lie import so3

dof = 6
n = 4

# algebra generators
G = np.zeros((dof, n, n))
for i in range(3): G[i,i,3] = 1
G[3:, :3, :3] = so3.G

def alg(c):
  """ Return matrix repr. of lie algebra vector c

  INPUT
    c (ndarray, [6,]): vector of se(3) coefficients, first three are translation

  OUTPUT
    C (ndarray, [4,4]): matrix-valued tangent vector in se(3)
  """
  u = c[0]*G[0] + c[1]*G[1] + c[2]*G[2]
  phi = c[3]*G[3] + c[4]*G[4] + c[5]*G[5]
  return u + phi

def algi(C):
  """ Return vector of se(3) coefficients

  INPUT
    C (ndarray, [4,4]): matrix-valued tangent vector in se(3)

  OUTPUT
    c (ndarray, [6,]): vector of se(3) coefficients.
  """
  u = C[:3, 3]
  phi = so3.algi(C[:3,:3])
  return np.concatenate((u, phi))

def expm(X):
  wx = X[:3,:3]
  wx2 = wx.dot(wx)

  w = so3.algi(wx)
  t2 = w.T.dot(w)
  t = np.sqrt(t2)
  eye = np.eye(3)

  if t2 < 1e-2:
    A = lie.TaylorSinXoverX(t)
    B = lie.TaylorOneMinusCosXOverX2(t)
    C = lie.TaylorOneMinusSinXOverXOverX2(t)
  else:
    st = np.sin(t)
    ct = np.cos(t)
    A = st/t
    B = (1-ct)/t2
    C = (1-A)/t2

  R = eye + A*wx + B*wx2
  V = eye + B*wx + C*wx2
  u = X[:3,3]
  Vu = V.dot(u)

  return np.concatenate((
    np.concatenate(( R, Vu[:,np.newaxis]), axis=1),
    np.array([[0, 0, 0, 1]])))

def logm(X): 
  R = X[:3,:3]
  d = X[:3,3]

  logR = lie.so3.logm(R)
  logR2 = logR.dot(logR)
  w = lie.so3.algi(logR)
  t2 = w.T.dot(w)
  t = np.sqrt(t2)

  if t2 < 1e-2:
    A = lie.TaylorSinXoverX(t)
    B = lie.TaylorOneMinusCosXOverX2(t)
    if t2 == 0: coeff = 1 # should this be 1 or 0 or ... ?
    else: coeff = 1/t2
  else:
    A = np.sin(t)/t
    B = (1 - np.cos(t))/t2
    coeff = 1/t2

  Vi = np.eye(n-1) - 0.5*logR + coeff*(1 - A/(2*B))*logR2
  u = Vi.dot(d)
  return np.concatenate((
    np.concatenate((logR, u[:,np.newaxis]), axis=1),
    np.array([[0, 0, 0, 0]])))

def Adj(X):
  """ Return Adj_X for X an element of SE(3)

  For the function
    f(X, c) = algi(logm(X expm(alg(c)) X^{-1}))
  returns the Jacobian
    frac{ partial f }{ partial c } |_{c=0}
  where c \in R^{dof}, X \in SE(3)

  INPUT
    X (ndarray, [4,4]): Element of SE(3)

  OUTPUT
    A (ndarray, [6,6]): Jacobian of f
  """
  R = X[:3,:3]
  t = X[:3,3]
  tx = t[0]*G[3,:3,:3] + t[1]*G[4,:3,:3] + t[2]*G[5,:3,:3]
  A = np.concatenate((
      np.concatenate((R, tx.dot(R)), axis=1),
      np.concatenate((np.zeros((3,3)), R), axis=1)))
  return A

def rvs(mean=None, cov=np.eye(6)):
  """ Sample SE(3) RV.

  INPUTS
    mean (ndarray, [4,4]): mean element in SE(3)
    cov (ndarray, [6,6]): covariance in se(3)

  OUTPUTS
    x (ndarray, [4,4]): RV in SE(3)
  """
  eps = alg(mvn.rvs(np.zeros(dof), cov))
  if mean is None: return expm(eps)
  else: return expm(eps).dot(mean)

def Rt(X):
  """ Decompose X in SE(3) into rotation and translation components.

  INPUT
    X (ndarray, [n, n]): Element of SE(3)

  OUTPUT
    R (ndarray, [n-1, n-1]): Element of SO(3)
    t (ndarray, [3,]): Translation component
  """
  return X[:3,:3], X[:3, 3]

def inv(X):
  """ Return inverse of X in SE(3) in closed form.

  INPUT
    X (ndarray, [n, n]): Element of SE(3)

  OUTPUT
    Xi (ndarray, [n, n]): Element of SE(3) such that X.dot(Xi) = I
  """
  R, t = Rt(X)
  Ri = R.T
  return np.concatenate((
    np.concatenate((Ri, -Ri.dot(t)[:,np.newaxis]), axis=1),
    np.array([[0, 0, 0, 1]])))

def dist2(X, Y, c=1, d=1):
  """ Squared scale-dependent, left-invariant Riemannian metric.

  INPUT
    X (ndarray, [n, n]): Element of SE(3)
    Y (ndarray, [n, n]): Element of SE(3)
    c (positive float): rotation distance scaling
    d (positive float): translation distance scaling
  """
  R1, t1 = Rt(X)
  R2, t2 = Rt(Y)
  R1i = R1.T
  term1 = np.linalg.norm(so3.algi(so3.logm(R1i.dot(R2))))**2
  term2 = np.linalg.norm(t2 - t1)**2
  return c*term1 + d*term2

def Exp(A, B):
  """ Riemannian exponential map of B in tangent space of A

  INPUT
    A (ndarray, [n, n]): Element of SE(3) whose tangent space we are working in
    B (ndarray, [n, n]): Matrix repr. of se(3) element in tangent space of A
  
  OUTPUT
    C (ndarray, [n, n]): Element of SE(3)
  """
  return A.dot(expm(B))

def Log(A, B):
  """ Riemannian logarithmic map of B to the tangent space of A

  INPUT
    A (ndarray, [n, n]): Element of SE(3) whose tangent space we are considering 
    B (ndarray, [n, n]): Element of SE(3) about A
  
  OUTPUT
    c (ndarray, [n, n]): Matrix repr. of se(3) element in tangent space of A
  """
  return logm(inv(A).dot(B))

def karcher(X, w=None, dst=dist2):
  """ Karcher mean of elements of SE(3)

  INPUT
    X (ndarray, [N, n, n]): N elements of SE(3)
    w (ndarray, [N,]): N weights, default to 1/N if None
    dst (function pointer): Pointer to distance function

  OUTPUT
    mu (ndarray, [n, n]): Karcher mean of elements
  """
  N = X.shape[0]
  if w is None: w = (1/N)*np.ones(N)

  mu = X[np.random.randint(N)]
  xv = np.zeros((N, dof))
  norm = 1e8
  cnt = 0
  while norm > 1e-8:
    assert np.linalg.norm(mu.dot(inv(mu)) - np.eye(4)) < 1e-9, 'Not SE(3)'

    # Compute distances
    D = 0
    for i in range(N):
      D += w[i] * dst(X[i], mu)
      xv[i] = algi(Log(mu, X[i]))

    # Compute new mu, exponentiate
    muAlgebra = np.average(xv, axis=0, weights=w)
    newMu = Exp(mu, alg(muAlgebra))
    norm = np.sqrt(dst(mu, newMu))
    mu = newMu
  return mu
