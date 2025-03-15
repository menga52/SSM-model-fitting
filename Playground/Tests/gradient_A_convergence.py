from Playground.true_model import *
from Playground.sampler import *
from Playground.gradient import gradient_A
from Playground.utilities import rng

sample_size = int(1000)
num_steps = 1000
learning_rate = 1/(sample_size*true_model.T)/10000

A_wrong = true_model.A + 0.1 * rng.uniform(0, 1, size=true_model.A.shape)
wrong_model = true_model.clone()
wrong_model.A = A_wrong
EXs, EYs = true_model.expected()
orig = np.copy(A_wrong)
print(orig,'orig')

for t in range(2*num_steps):
  samps = sample(sample_size,0,true_model.T, EYs, wrong_model, debug=False, use_forward=False)
  grad_A = learning_rate*gradient_A(samps, wrong_model)
  #print(grad_A,'grad')
  A_wrong += grad_A
  print(wrong_model.A,'A_wrong')
  #print(A,'A')
