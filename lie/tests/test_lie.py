import unittest
import lie, numpy as np
import scipy.linalg as sla, numpy.linalg as nla
import IPython as ip

class Test(unittest.TestCase):
  def test_SO2(self):
    eps = 1e-4
    pts = np.array([.1, 0.05, -0.001, -.025, np.pi, 2*np.pi, 2*np.pi-.01,
      -4*np.pi, -4*np.pi+.01])
    m = lie.so2

    for idx, v in enumerate(pts):
      V = m.expm(m.alg(v))
      logV = m.logm(V)
      logv = m.algi(logV)
      eye = np.eye(m.n)

      self.assertTrue( np.abs( nla.det(V) - 1) < eps,
        msg='SO2 det(V) != 1 for pts[%d]' % idx )
      self.assertTrue( nla.norm(V.dot(V.T) - eye) < eps,
        msg='SO2 V.dot(V.T) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm(V.T.dot(V) - eye) < eps,
        msg='SO2 V.T.dot(V) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm(m.alg(logv) - logV) < eps,
        msg='SO2 alg(algi(logV)) != logV for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(logV) - V ) < eps,
        msg='SO2 exp(log(V)) != V for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(-logV).dot(V) - eye ) < eps,
        msg='SO2 exp(-log(V)).dot(V) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm( V.dot(m.expm(-logV)) - eye ) < eps,
        msg='SO2 V.dot(exp(-log(V))) != I for pts[%d]' % idx )

      self.assertTrue( nla.norm( V.dot(m.inv(V)) - eye ) < eps,
        msg='SO2 V.dot(inv(V)) != I for pts[%d]' % idx )

      AdjV = m.Adj(V)
      for idx2, p in enumerate(pts):
        ad1 = AdjV.dot(p)
        ad2 = m.algi( V.dot(m.alg(p)).dot(m.inv(V)) )
        self.assertTrue( nla.norm( ad1 - ad2 ) < eps,
          msg='SO2 Adj(V)*p != algi(V*alg(p)*V^{-1}) for (pts[%d], pts[%d])' % \
            (idx, idx2) )

  def test_SE2(self):
    eps = 1e-4
    pts = np.array([
      [0, 0, 0],
      [-1e-2, 1e-3, 1e-5],
      [-1e-2, 1e-3, -1e-5],
      [-1, -2, -.3],
      [1, 2, 3],
      [4, 4, 2*np.pi],
      [.01, .02, 6*np.pi],
      [.01, .02, -4*np.pi],
      [.01, .02, .03]])
    m = lie.se2

    for idx, v in enumerate(pts):
      V = m.expm(m.alg(v))
      logV = m.logm(V)
      logv = m.algi(logV)
      eye = np.eye(m.n)

      self.assertTrue( np.abs( nla.det(m.Rt(V)[0]) - 1) < eps,
        msg='SE2 det(V_rot) != 1 for pts[%d]' % idx )
      self.assertTrue( nla.norm(m.alg(logv) - logV) < eps,
        msg='SE2 alg(algi(logV)) != logV for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(logV) - V ) < eps,
        msg='SE2 exp(log(V)) != V for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(-logV).dot(V) - eye ) < eps,
        msg='SE2 exp(-log(V)).dot(V) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm( V.dot(m.expm(-logV)) - eye ) < eps,
        msg='SE2 V.dot(exp(-log(V))) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm( V.dot(m.inv(V)) - eye ) < eps,
        msg='SE2 V.dot(inv(V)) != I for pts[%d]' % idx )

      AdjV = m.Adj(V)
      for idx2, p in enumerate(pts):
        ad1 = AdjV.dot(p)
        ad2 = m.algi( V.dot(m.alg(p)).dot(m.inv(V)) )
        self.assertTrue( nla.norm( ad1 - ad2 ) < eps,
          msg='SE2 Adj(V)*p != algi(V*alg(p)*V^{-1}) for (pts[%d], pts[%d])' % \
            (idx, idx2) )

  def test_SO3(self):
    m = lie.so3
    eps = 1e-4
    pts = np.array([
      [0, 0, 0],
      [-1e-3, 1e-2, 1e-4],
      [-1e-3, -1e-2, -1e-4],
      [1e-3, 1e-2, 1e-4],
      [-1, -2, -.3],
      [1, 2, 3],
      [4, 4, 2*np.pi],
      [.01, .02, 6*np.pi],
      [.01, .02, -4*np.pi],
      [.01, .02, .03]])

    for idx, v in enumerate(pts):
      V = m.expm(m.alg(v))
      logV = m.logm(V)
      logv = m.algi(logV)
      eye = np.eye(m.n)

      self.assertTrue( np.abs( nla.det(V) - 1) < eps,
        msg='SO3 det(V) != 1 for pts[%d]' % idx )
      self.assertTrue( nla.norm(V.dot(V.T) - eye) < eps,
        msg='SO3 V.dot(V.T) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm(V.T.dot(V) - eye) < eps,
        msg='SO3 V.T.dot(V) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm(m.alg(logv) - logV) < eps,
        msg='SO3 alg(algi(logV)) != logV for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(logV) - V ) < eps,
        msg='SO3 exp(log(V)) != V for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(-logV).dot(V) - eye ) < eps,
        msg='SO3 exp(-log(V)).dot(V) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm( V.dot(m.expm(-logV)) - eye ) < eps,
        msg='SO3 V.dot(exp(-log(V))) != I for pts[%d]' % idx )

      self.assertTrue( nla.norm( V.dot(m.inv(V)) - eye ) < eps,
        msg='SO3 V.dot(inv(V)) != I for pts[%d]' % idx )

      AdjV = m.Adj(V)
      for idx2, p in enumerate(pts):
        ad1 = AdjV.dot(p)
        ad2 = m.algi( V.dot(m.alg(p)).dot(m.inv(V)) )
        self.assertTrue( nla.norm( ad1 - ad2 ) < eps,
          msg='SO3 Adj(V)*p != algi(V*alg(p)*V^{-1}) for (pts[%d], pts[%d])' % \
            (idx, idx2) )

  def test_SE3(self):
    eps = 1e-4
    pts = np.array([
      [1e-2, 1e-1, -1e-2, 1e-4, -1e-2, 1e-3],
      [0, 0, 0, 0, 0, 0],
      [-1, -2, -3, -.1, -.2, -.3],
      [1, 2, 3, 1, 2, 3],
      [4, 4, 4, 2*np.pi, 2*np.pi, 2*np.pi],
      [.01, .02, -.03, np.pi*1.5, -np.pi*1.75, 6*np.pi],
      [.01, .02, .03, -4*np.pi, -4*np.pi, -4*np.pi],
      [.01, .02, .03, 0, 0, 0]])
    m = lie.se3

    for idx, v in enumerate(pts):
      V = m.expm(m.alg(v))
      logV = m.logm(V)
      logv = m.algi(logV)
      eye = np.eye(m.n)

      if np.linalg.norm(v) < 0.25: eps2 = 0.1
      else: eps2 = eps

      self.assertTrue( np.abs( nla.det(m.Rt(V)[0]) - 1) < eps2,
        msg='SE3 det(V_rot) != 1 for pts[%d]' % idx )
      self.assertTrue( nla.norm(m.alg(logv) - logV) < eps,
        msg='SE3 alg(algi(logV)) != logV for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(logV) - V ) < eps2,
        msg='SE3 exp(log(V)) != V for pts[%d]' % idx )
      self.assertTrue( nla.norm( m.expm(-logV).dot(V) - eye ) < eps2,
        msg='SE3 exp(-log(V)).dot(V) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm( V.dot(m.expm(-logV)) - eye ) < eps2,
        msg='SE3 V.dot(exp(-log(V))) != I for pts[%d]' % idx )
      self.assertTrue( nla.norm( V.dot(m.inv(V)) - eye ) < eps2,
        msg='SE3 V.dot(inv(V)) != I for pts[%d]' % idx )

      AdjV = m.Adj(V)
      for idx2, p in enumerate(pts):
        ad1 = AdjV.dot(p)
        ad2 = m.algi( V.dot(m.alg(p)).dot(m.inv(V)) )
        self.assertTrue( nla.norm( ad1 - ad2 ) < eps2,
          msg='SE3 Adj(V)*p != algi(V*alg(p)*V^{-1}) for (pts[%d], pts[%d])' % \
            (idx, idx2) )

if __name__ == "__main__":
  unittest.main()
