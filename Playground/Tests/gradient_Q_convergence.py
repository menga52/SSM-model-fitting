from Playground.true_model import *
from Playground.sampler import *
from Playground.gradient import gradient_Q
from Playground.utilities import rng, sample_normal
import numpy as np
chol = np.linalg.cholesky; inv = np.linalg.inv

EXs, EYs = true_model.expected()
Xs, Ys = true_model.initialize()
sample_size = int(10)
num_steps = 20000
learning_rate = 1/(sample_size*true_model.T)/1000

psi_Q_wrong = chol(inv(true_model.Q)) + 0.1 * rng.uniform(0, 1, size=true_model.R.shape)
wrong_model = true_model.clone()
wrong_model.Q = inv(psi_Q_wrong @ psi_Q_wrong.T)
orig = np.copy(wrong_model.Q)
print(Q,'Q')
print(orig,'orig')

for t in range(num_steps):
  psi_Q_wrong = chol(inv(wrong_model.Q))
  samps = sample(sample_size,0,true_model.T, Ys, wrong_model, debug=False, use_forward=True)
  #samps = [[np.asarray([[0],[0]]), sample_normal(np.identity(2))] for _ in range(1000)]
  grad_psi_Q = gradient_Q(samps, wrong_model)
  # print(grad_psi_Q,'grad')
  psi_Q_wrong += learning_rate*grad_psi_Q
  wrong_model.Q = inv(psi_Q_wrong@psi_Q_wrong.T)
  print(wrong_model.Q,'Q_wrong')
#print(Q,'Q')