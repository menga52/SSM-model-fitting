import numpy as np; copy = np.copy
from Playground.utilities import *
chol = np.linalg.cholesky


class model:
  def __init__(self,A,Q,C,R,T,mu_init=None,Sigma_inv_init=None):
    self.A = A
    self.Q = Q
    self.C = C
    self.R = R
    self.T = T
    self.state_dim = Q.shape[0]
    self.obs_dim = R.shape[0]
    self.mu_init = mu_init
    self.Sigma_inv_init = Sigma_inv_init
  
  def clone(self):
    Ac = copy(self.A)
    Qc = copy(self.Q)
    Cc = copy(self.C)
    Rc = copy(self.R)
    mu_initc = copy(self.mu_init)
    Sigma_inv_initc = copy(self.Sigma_inv_init)
    return model(Ac,Qc,Cc,Rc,self.T,mu_initc,Sigma_inv_initc)

  def initialize(self):
    # sample X, Y|X
    X = sample_normal(Sigma_inv_init)
    Q_chol = chol(self.Q); R_chol = chol(self.R)
    Xs = []; Ys = []
    for t in range(T):
      X = self.A@X + sample_normal_chol(Q_chol)
      Xs.append(np.copy(X))
      Ys.append(self.C@X + sample_normal_chol(R_chol))
    self.Xs = Xs; self.Ys = Ys
    return Xs, Ys

  def expected(self):
    Xs = []; Ys = []
    X = np.copy(self.mu_init)
    for t in range(self.T):
      X = self.A@X
      Xs.append(np.copy(X))
      Ys.append(self.C@X)
    return Xs, Ys
