from . import so2, se2, se3, so3
from scipy.linalg import expm, logm
import numpy as np

# Most of this library is based on
# Lie groups for 2D and 3D Transformations, Ethan Eade, 2017
# http://ethaneade.com/

def BCH(X, Y):
  """ Compute Baker–Campbell–Hausdorff to 4th order

  BCH is a series, defined as (for X, Y in Lie algebra g):
    BCH : g -> g
    BCH(X,Y) = log( exp(X) exp(Y) )

  INPUT
    X,Y (ndarray, [n, n]): Output of alg(x), alg(y) for x, y in R^k, for some
                           Matrix Lie algebra with k degrees of freedom.

  OUTPUT
    Z (ndarray, [n, n]): An element of the algebra.
  """
  # unverified, do not use
  def B(x, y): return x.dot(y) - y.dot(x)
  Bxy = B(X,Y)
  Byx = B(Y,X)
  Bxxy = B(X, Bxy)
  Byyx = B(Y, Byx)
  return X + Y + 0.5*Bxy + (1/12.)*(Bxxy + Byyx) - (1/24.)*B(Y, Bxxy)

# good
def TaylorSinXoverX(x):
  """ Return taylor expansion of sin(x)/x """
  return 1 - x**2/6. + x**4/120.

def TaylorOneMinusCosXOverX(x):
  """ Return 10th order taylor expansion of (1-cos(x))/x """
  return x/2 - x**3/24 + x**5/720

# good
def TaylorXoverTwoSinX(x):
  """ Return 10th order taylor expansion of x/(2*sin(x)) """
  return 0.5 + x**2/12. + (7*x**4)/720. + (31*x**6)/30240. + \
    (127*x**8)/1209600. + (73*x**10)/6842880.

# good
def TaylorOneMinusCosXOverX2(x):
  """ Return 10th order taylor expansion of (1-cos(x))/x^2 """
  return 0.5 - x**2/24. + x**4/720. - x**6/40320. + x**8/3628800. \
    - x**10/479001600. + x**12/87178291200.

def TaylorOneMinusSinXOverXOverX2(x):
  """ Return 10th order taylor expansion of (1-sin(x)/x)/x^2 """
  return 1/6. - x**2/120. + x**4/5040. - x**6/362880. + x**8/39916800. \
    - x**10 / 6227020800. + x**12/1307674368000.

def TaylorXMinusSinXOverX3(x):
  """ Return taylor expansion of (x-sin(x))/x^3 """

def modAngle(a):
  a = np.asarray(a)
  nz = a!=0
  a[nz] = np.mod(a[nz], np.sign(a[nz])*np.pi)
  return a

def sameAngle(a, b):
  a = np.asarray(a)
  b = np.asarray(b)
  modDiff = np.mod(np.mod(a, np.pi) - np.mod(b, np.pi), np.pi)
  return modDiff < 1e-8 or np.pi - modDiff < 1e-8
