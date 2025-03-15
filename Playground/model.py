import numpy as np; copy = np.copy
from Playground.utilities import *
chol = np.linalg.cholesky


class model:
  def __init__(self,A,Q,C,R,T,mu_init=None,Sigma_init=None):
    self.A = A
    self.Q = Q
    self.C = C
    self.R = R
    self.T = T
    self.state_dim = Q.shape[0]
    self.obs_dim = R.shape[0]
    if mu_init is None: mu_init = np.zeros((A.shape[0],1))
    self.mu_init = mu_init
    if Sigma_init is None: Sigma_init = np.zeros(Q.shape)
    self.Sigma_init = Sigma_init +0.00001*np.identity(Q.shape[0])
  
  def clone(self):
    Ac = copy(self.A)
    Qc = copy(self.Q)
    Cc = copy(self.C)
    Rc = copy(self.R)
    mu_init = copy(self.mu_init)
    Sigma_init = copy(self.Sigma_init)
    return model(Ac,Qc,Cc,Rc,self.T,mu_init,Sigma_init)

  def initialize(self):
    # sample X, Y|X
    X = sample_normal(self.Sigma_init)
    Q_chol = chol(self.Q); R_chol = chol(self.R)
    Xs = []; Ys = []
    for t in range(self.T):
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

  def expectedYs(self, Xs):
    Ys = []
    for t in range(self.T):
      Ys.append(self.C@Xs[t])
    return Ys
