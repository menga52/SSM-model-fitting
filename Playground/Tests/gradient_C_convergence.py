from Playground.true_model import *
from Playground.sampler import *
from Playground.gradient import gradient_C
from Playground.utilities import rng
EXs, EYs = true_model.expected()
sample_size = int(1000)
num_steps = 1000
learning_rate = 1/(sample_size*true_model.T)/10000

C_wrong = true_model.C + 0.1 * rng.uniform(0, 1, size=true_model.C.shape)
wrong_model = true_model.clone()
wrong_model.C = C_wrong
orig = np.copy(C_wrong)
print(orig,'orig')

for t in range(num_steps):
  samps = sample(sample_size,0,true_model.T, EYs, wrong_model, debug=False, use_forward=False)
  grad_C = learning_rate*gradient_C(samps, EYs, wrong_model)
  #print(grad_C,'grad')
  C_wrong += grad_C
  print(wrong_model.C,'C_wrong')
print(orig,'orig')
