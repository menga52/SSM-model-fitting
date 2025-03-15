from Playground.true_model import *
from Playground.sampler import *
from Playground.gradient import gradient_A
from Playground.utilities import rng

sample_size = int(30)
num_steps = 1000
learning_rate = 1/(sample_size*true_model.T)/1000
true_model.Q = np.identity(true_model.Q.shape[0])
true_model.Q[1][0] = 0.8; true_model.Q[0][1] = 0.8
A_wrong = true_model.A + 0.03 * rng.uniform(0, 1, size=true_model.A.shape)
wrong_model = true_model.clone()
wrong_model.A = A_wrong
Xs, Ys = true_model.initialize()
orig = np.copy(A_wrong)
print(true_model.A, 'correct')
print(orig,'orig')

start = 10
length = 300
def descent(buff):
  wrong_model.A = np.copy(orig)
  A_wrong = wrong_model.A
  for t in range(num_steps):
    samps = sample(sample_size, start, length, Ys, wrong_model, buffer=buff, debug=False, use_forward=False)
    grad_A = learning_rate * gradient_A(samps, wrong_model)
    # print(grad_A,'grad')
    A_wrong += grad_A
  print(wrong_model.A, 'A_wrong, buffer', buff)

descent(-1)
for buff in range(3,9):
  print()
  descent(buff)
