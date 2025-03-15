from Playground.true_model import *
from Playground.sampler import *
from Playground.gradient import gradient_R
from Playground.utilities import rng, sample_normal
import numpy as np
chol = np.linalg.cholesky; inv = np.linalg.inv

EXs, EYs = true_model.expected()
true_model.initialize(); true_model.initialize(); true_model.initialize()
Xs, Ys = true_model.initialize()
sample_size = int(10)
num_steps = 20000
learning_rate = 1/(sample_size*true_model.T)/1000

psi_R_wrong = chol(inv(true_model.R)) + 0.1 * rng.uniform(0, 1, size=true_model.R.shape)
wrong_model = true_model.clone()
wrong_model.R = inv(psi_R_wrong @ psi_R_wrong.T)
orig = np.copy(wrong_model.R)
print(R,'R')
print(orig,'orig')

for t in range(num_steps):
  psi_R_wrong = chol(inv(wrong_model.R))
  samps = sample(sample_size,0,true_model.T, Ys, wrong_model, debug=False, use_forward=True)
  #samps = [[np.asarray([[0],[0]]), sample_normal(np.identity(2))] for _ in range(1000)]
  grad_psi_R = gradient_R(samps, Ys, wrong_model)
  # print(grad_psi_Q,'grad')
  psi_R_wrong += learning_rate * grad_psi_R
  wrong_model.R = inv(psi_R_wrong @ psi_R_wrong.T)
  print(wrong_model.R,'R_wrong')
#print(Q,'Q')