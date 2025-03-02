import numpy as np
chol = np.linalg.cholesky
rng = np.random.default_rng(13)

def sample_normal(Sigma):
  Sigma_chol = chol(Sigma)
  return sample_normal_chol(Sigma_chol)

def sample_normal_chol(Sigma_chol):
  dim = Sigma_chol.shape[0]
  temp = np.asarray([[rng.normal(0,1)] for i in range(dim)])
  return Sigma_chol @ temp
