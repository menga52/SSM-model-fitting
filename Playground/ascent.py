import numpy as np
from Playground.gradient import gradient_A, gradient_Q, gradient_C, gradient_R
from Playground.utilities import rng
from Playground.sampler import sample
chol = np.linalg.cholesky; inv = np.linalg.inv


def sample_info(sample_size, length, Ys, model, buffer, debug, use_forward):
  start = int((model.T-length)*rng.uniform(0,1))
  return sample(sample_size, start, length, Ys, model, buffer, debug, use_forward), start

def optimizer_factory(max_steps, epsilon, step_sizes, length, Ys, sample_size):
  def optimize(model):
    return __optimize__(model, max_steps, epsilon, step_sizes, length, Ys, sample_size)
  return optimize


def __optimize__(model, max_steps, epsilon, step_sizes, length, Ys, sample_size):
  step_size_A=step_sizes[0]; step_size_Q=step_sizes[1]; step_size_C=step_sizes[2]; step_size_R=step_sizes[3]
  m=model.clone();
  diff=np.inf
  psi_Q = chol(inv(m.Q))
  psi_R = chol(inv(m.R))
  for step in range(max_steps):
    grad_A = np.zeros(model.A.shape); grad_psi_Q = np.zeros(model.Q.shape)
    grad_C = np.zeros(model.C.shape); grad_psi_R = np.zeros(model.R.shape)
    if np.linalg.norm(diff) < epsilon: return m
    samples=np.zeros((batches, batch_size, length, m.state_dim, 1));
    starts=np.zeros(batches)
    for batch in range(batches):
      seqs, start=sample_info(batch_size, length, Ys, m, buffer, debug, use_forward)
      samples[batch]=seqs;
      starts[batch]=start
      grad_psi_Q+=gradient_Q(start, seqs, m)
    diff=step_size * grad_psi_Q
    psi_Q+=diff
    m.Q=inv(psi_Q @ psi_Q.T)
    step_size_A = decay(step_size_A); step_size_Q = decay(step_size_Q); step_size_C = decay(step_size_C); step_size_R = decay(step_size_R)
  return m

def optimizer_factory_A(max_steps,epsilon,step_size,length,Ys,batch_size,batches,decay,buffer=-1,debug=False,use_forward=False,verbose=False):
  def optimize(model):
    return __optimize_A__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,decay,buffer,debug,use_forward,verbose)
  return optimize

def __optimize_A__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,decay,buffer,debug,use_forward, verbose):
  m = model.clone(); diff = np.inf
  for step in range(max_steps):
    grad_A = np.zeros(model.A.shape)
    if np.linalg.norm(diff) < epsilon: return m
    samples = np.zeros((batches,batch_size,length,m.state_dim,1)); starts = np.zeros(batches)
    for batch in range(batches):
      seqs, start = sample_info(batch_size,length,Ys,m,buffer,debug,use_forward)
      samples[batch] = seqs; starts[batch] = start
      grad_A += gradient_A(start, seqs, m)
    diff = step_size * grad_A
    step_size = decay(step_size)
    m.A += diff
    if verbose: print(m.A, 'update')
  return m

def optimizer_factory_Q(max_steps,epsilon,step_size,length,Ys,batch_size,batches,decay,buffer=-1,debug=False,use_forward=False,verbose=False):
  def optimize(model):
    return __optimize_Q__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,decay,buffer,debug,use_forward,verbose)
  return optimize

def __optimize_Q__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,decay,buffer,debug,use_forward,verbose):
  m=model.clone(); diff=np.inf
  psi_Q = chol(inv(m.Q))
  for step in range(max_steps):
    grad_psi_Q=np.zeros(model.Q.shape)
    if np.linalg.norm(diff) < epsilon: return m
    samples=np.zeros((batches, batch_size, length, m.state_dim, 1));
    starts=np.zeros(batches)
    for batch in range(batches):
      seqs, start=sample_info(batch_size, length, Ys, m, buffer, debug, use_forward)
      samples[batch]=seqs;
      starts[batch]=start
      grad_psi_Q += gradient_Q(start, seqs, m)
    diff = step_size * grad_psi_Q
    psi_Q += diff
    m.Q = inv(psi_Q@psi_Q.T)
    step_size=decay(step_size)
    if verbose: print(m.Q, 'update')
  return m

def optimizer_factory_C(max_steps,epsilon,step_size,length,Ys,batch_size,batches,decay,buffer=-1,debug=False,use_forward=False,verbose=False):
  def optimize(model):
    return __optimize_C__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,decay,buffer,debug,use_forward,verbose)
  return optimize

def __optimize_C__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,decay,buffer,debug,use_forward,verbose):
  m=model.clone(); diff=np.inf
  for step in range(max_steps):
    grad_C=np.zeros(model.C.shape)
    if np.linalg.norm(diff) < epsilon: return m
    samples=np.zeros((batches, batch_size, length, m.state_dim, 1));
    starts=np.zeros(batches)
    for batch in range(batches):
      seqs, start=sample_info(batch_size, length, Ys, m, buffer, debug, use_forward)
      samples[batch]=seqs;
      starts[batch]=start
      grad_C+=gradient_C(start, seqs, m)
    diff=step_size * grad_C
    m.C+=diff
    step_size=decay(step_size)
    if verbose: print(m.C, 'update')
  return m

def optimizer_factory_R(max_steps,epsilon,step_size,length,Ys,batch_size,batches,decay,buffer=-1,debug=False,use_forward=False,verbose=False):
  def optimize(model):
    return __optimize_R__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,buffer,debug,use_forward,verbose)
  return optimize

def __optimize_R__(model, max_steps, epsilon, step_size, length, Ys, batches, batch_size,decay,buffer,debug,use_forward,verbose):
  m=model.clone(); diff=np.inf
  psi_R=chol(inv(m.R))
  for step in range(max_steps):
    grad_psi_R=np.zeros(model.R.shape)
    if np.linalg.norm(diff) < epsilon: return m
    samples=np.zeros((batches, batch_size, length, m.state_dim, 1));
    starts=np.zeros(batches)
    for batch in range(batches):
      seqs, start=sample_info(batch_size, length, Ys, m, buffer, debug, use_forward)
      samples[batch]=seqs;
      starts[batch]=start
      grad_psi_R+=gradient_R(start, seqs, m)
    diff=step_size * grad_psi_R
    psi_R+=diff
    m.R=inv(psi_R @ psi_R.T)
    step_size=decay(step_size)
    if verbose: print(m.R, 'update')
  return m