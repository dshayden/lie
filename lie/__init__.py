from . import se2, se3, so3
from scipy.linalg import expm, logm

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
  def B(x, y): return x.dot(y) - y.dot(x)
  Bxy = B(X,Y)
  Byx = B(Y,X)
  Bxxy = B(X, Bxy)
  Byyx = B(Y, Byx)
  return X + Y + 0.5*Bxy + (1/12.)*(Bxxy + Byyx) - (1/24.)*B(Y, Bxxy)
