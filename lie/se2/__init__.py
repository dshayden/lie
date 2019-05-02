import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from lie import so2
import lie

dof = 3
n = 3

G = np.zeros((dof, n, n))
G[0, 0, 2] = 1
G[1, 1, 2] = 1
G[2, :2, :2] = so2.G[0]

def expm(C):
  """ Return exponential map of matrix repr. of algebra vector, c. """
  t = C[1,0]
  u = C[:2, 2]

  if np.abs(t)<1e-2:
    stt = lie.TaylorSinXoverX(t)
    ctt = lie.TaylorOneMinusCosXOverX(t)
  else:
    stt = np.sin(t)/t
    ctt = (1-np.cos(t))/t

  thetaExp = so2.expm(C[:2,:2])

  V = np.array( [[stt, -ctt], [ctt, stt]] )
  Vu = V.dot(u)
  return np.concatenate((
    np.concatenate((thetaExp, Vu[:,np.newaxis]), axis=1),
    np.array([[0, 0, 1]])))

def logm(C):
  """ Return logarithmic map of group element. """
  theta_x = so2.logm(C[:2,:2])
  u = getVi(C).dot(C[:2,2])
  return np.concatenate((
    np.concatenate(( theta_x, u[:,np.newaxis]), axis=1),
    [[0, 0, 0]]
  ))

  # make matrix 
  # tx = so2.logm(C[:2,:2])
  # t = so2.algi(tx)
  # # if np.abs(t)<1e-2:  
  # #   stt = lie.TaylorSinXoverX(t)
  # #   ctt = lie.TaylorOneMinusCosXOverX(t)
  # # else:
  # #   stt = np.sin(t)/t
  # #   ctt = (1-np.cos(t))/t
  # #
  # # A = stt; B = ctt
  # # Vi = (1 / (A**2 + B**2)) * np.array([[A, B], [-B, A]])
  # Vi = getVi(C)
  #
  # d = C[:2,2]
  # u = Vi.dot(d)
  #
  # return alg(np.concatenate(( u, [t] )))

def getVi(C):
  """ Return the V^{-1} matrix computed in log map; C can be SE(2) or SO(2) """
  t = so2.algi(so2.logm(C[:2,:2]))

  if np.abs(t)<1e-2:
    stt = lie.TaylorSinXoverX(t)
    ctt = lie.TaylorOneMinusCosXOverX(t)
  else:
    stt = np.sin(t)/t
    ctt = (1-np.cos(t))/t
  V = np.array( [[stt, -ctt], [ctt, stt]] )
  return np.linalg.inv(V)

  # Vi = np.linalg.inv(V)
  # return Vi

  # tx = so2.logm(C[:2,:2])
  # t = so2.algi(tx)
  # if np.abs(t)<1e-2:  
  #   stt = lie.TaylorSinXoverX(t)
  #   ctt = lie.TaylorOneMinusCosXOverX(t)
  # else:
  #   stt = np.sin(t)/t
  #   ctt = (1-np.cos(t))/t
  # A = stt; B = ctt
  # Vi = (1 / (A**2 + B**2)) * np.array([[A, B], [-B, A]])
  # return Vi

def alg(c):
  """ Return matrix repr. of lie algebra vector c

  INPUT
    c (ndarray, [3,]): vector of se(2) coefficients

  OUTPUT
    C (ndarray, [3,3]): matrix-valued tangent vector in se(2)
  """
  return c[0]*G[0] + c[1]*G[1] + c[2]*G[2]

def algi(C):
  """ Return vector of se(2) coefficients

  INPUT
    C (ndarray, [3,3]): matrix-valued tangent vector in se(2)

  OUTPUT
    c (ndarray, [3,]): vector of se(2) coefficients.
  """
  return np.array([ C[0,2], C[1,2], C[1,0] ])

def Adj(X):
  """ Return Adj_X for X an element of SE(2)

  For the function
    f(X, c) = algi(logm(X expm(alg(c)) X^{-1}))
  returns the Jacobian
    frac{ partial f }{ partial c } |_{c=0}
  where c \in R^{dof}, X \in SE(2)

  INPUT
    X (ndarray, [3,3]): Element of SE(2)

  OUTPUT
    A (ndarray, [3,3]): Jacobian of f
  """
  A = X.copy()
  A[0,2] = X[1,2]
  A[1,2] = -X[0,2]
  return A

def plot(X, colors=None, l=1, origin=None, ax=None):
  """ Plot SE2 transformation X acting on origin.

  INPUT
    X (ndarray, [3,3]): SE(2) transformation
    colors (ndarray, [2,3]): colors of each axis
    l (float): length of axis
    origin (ndarray, [3,3]): origin points in homogeneous 2D coordinates
    ax (matplotlib.axes._subplots.AxesSubplot): axes to plot in
  """
  if origin is not None: pts = origin
  else: pts = np.array([[0, 0, 1], [l, 0, 1], [0, l, 1]]) # 3 x 3
  if ax is None: ax = plt.gca()
  # x = inv(X).dot(pts.T).T
  x = (X.dot(pts.T)).T

  # x = X.dot(pts.T).T

  if colors is None:
    white = np.array([1, 1, 1])
    alpha = 0.3
    red = alpha * np.array([1, 0, 0]) + (1-alpha) * white
    blue = alpha * np.array([0, 0, 1]) + (1-alpha) * white
    red = np.maximum([0, 0, 0], np.minimum(red, [1,1,1]))
    blue = np.maximum([0, 0, 0], np.minimum(blue, [1,1,1]))
    colors = np.stack((red,blue))

  ax.arrow(x[0,0], x[0,1], x[1,0]-x[0,0], x[1,1]-x[0,1], color=colors[0])
  ax.arrow(x[0,0], x[0,1], x[2,0]-x[0,0], x[2,1]-x[0,1], color=colors[1])
  # ax.arrow(x[0,0], x[0,1], x[1,0]-x[0,0], x[1,1]-x[0,1], edgecolor=colors[0],
  #   facecolor='k')
  # ax.arrow(x[0,0], x[0,1], x[2,0]-x[0,0], x[2,1]-x[0,1], edgecolor=colors[1],
  #   facecolor='k')
  plt.scatter(x[:,0], x[:,1], s=0)

  ax.set_aspect('equal', 'box')

def rvs(mean=None, cov=np.eye(3)):
  """ Sample SE(2) RV.

  INPUTS
    mean (ndarray, [3,3]): mean element in SE(2)
    cov (ndarray, [3,3]): covariance in se(2)

  OUTPUTS
    x (ndarray, [3,3]): RV in SE(2)
  """
  eps = alg(mvn.rvs(np.zeros(dof), cov))
  if mean is None: return expm(eps)
  else: return expm(eps).dot(mean)

def Rt(X):
  """ Decompose X in SE(2) into rotation and translation components.

  INPUT
    X (ndarray, [n, n]): Element of SE(2)

  OUTPUT
    R (ndarray, [n-1, n-1]): Element of SO(2)
    t (ndarray, [2,]): Translation component
  """
  return X[:2,:2], X[:2, 2]

def inv(X):
  """ Return inverse of X in SE(2) in closed form.

  INPUT
    X (ndarray, [n, n]): Element of SE(2)

  OUTPUT
    Xi (ndarray, [n, n]): Element of SE(2) such that X.dot(Xi) = I
  """
  R, t = Rt(X)
  Ri = R.T
  return np.concatenate((
    np.concatenate((Ri, -Ri.dot(t)[:,np.newaxis]), axis=1),
    np.array([[0, 0, 1]])))

def Exp(A, B):
  """ Riemannian exponential map of B in tangent space of A

  INPUT
    A (ndarray, [n, n]): Element of SE(2) whose tangent space we are working in
    B (ndarray, [n, n]): Matrix repr. of se(2) element in tangent space of A
  
  OUTPUT
    C (ndarray, [n, n]): Element of SE(2)
  """
  return A.dot(expm(B))

def Log(A, B):
  """ Riemannian logarithmic map of B to the tangent space of A

  INPUT
    A (ndarray, [n, n]): Element of SE(2) whose tangent space we are considering 
    B (ndarray, [n, n]): Element of SE(2) about A
  
  OUTPUT
    c (ndarray, [n, n]): Matrix repr. of se(2) element in tangent space of A
  """
  return logm(inv(A).dot(B))

def dist2(X, Y, c=1, d=1):
  """ Squared scale-dependent, left-invariant Riemannian metric.

  INPUT
    X (ndarray, [n, n]): Element of SE(2)
    Y (ndarray, [n, n]): Element of SE(2)
    c (positive float): rotation distance scaling
    d (positive float): translation distance scaling
  """
  R1, t1 = Rt(X)
  R2, t2 = Rt(Y)
  R1i = R1.T
  # import IPython as ip
  # ip.embed()
  # term1 = np.linalg.norm(so2.algi(logm(R1i.dot(R2))))**2
  term1 = np.linalg.norm(so2.algi(so2.logm(R1i.dot(R2))))**2
  term2 = np.linalg.norm(t2 - t1)**2
  return c*term1 + d*term2

def karcher(X, w=None, dst=dist2):
  """ Karcher mean of elements of SE(2)

  INPUT
    X (ndarray, [N, n, n]): N elements of SE(2)
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
  # while norm > 1e-8:
  while norm > 1e-2:
    assert np.linalg.norm(mu.dot(inv(mu)) - np.eye(n)) < 1e-9, 'Not SE(2)'

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
    cnt = cnt + 1
  return mu
