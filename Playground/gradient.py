import numpy as np
inv = np.linalg.inv; chol = np.linalg.cholesky

def p(t, T, length):
  # P(t is in minibatch w/ given length, series length T)
  return min(t,T-t+1,length, T-length+1)/(T-length+1)

def p_factory(T,length):
  def prob(t):
    return p(t,T,length)
  return prob

def gradient_A_single(start, length, sample_sequence, model):
  Q_inv = inv(model.Q); A=model.A; x=sample_sequence
  grad_A = np.zeros(A.shape)
  pr = p_factory(model.T,length)
  for t in range(1,length):
    grad_A += pr(t+start) * Q_inv@(x[t]-A@x[t-1])@x[t-1].T
  return grad_A

def gradient_C_single(start, length, sample_sequence, Ys, model):
  R_inv = inv(model.R); C=model.C; x=sample_sequence
  grad_C = np.zeros(C.shape)
  pr = p_factory(model.T, length)
  for t in range(length):
    grad_C += pr(start+t) * R_inv@(Ys[t+start]-C@x[t])@x[t].T
  return grad_C

def gradient_Q_single(start, length, sample_sequence, model):
  Q=model.Q; Q_inv=inv(Q); A=model.A; psi_Q = chol(Q_inv)
  x = sample_sequence
  grad_psi_Q = np.zeros(Q.shape)
  pr = p_factory(model.T, length)
  for t in range(1,length):
    dev = x[t]-A@x[t-1]
    grad_psi_Q += pr(start+t) * (Q-dev@dev.T)@psi_Q
  return grad_psi_Q

def gradient_R_single(start, length, sample_sequence, Ys, model):
  R=model.R; R_inv=inv(R); C=model.C; psi_R = chol(R_inv)
  x = sample_sequence
  grad_psi_R = np.zeros(R.shape)
  pr = p_factory(model.T, length)
  for t in range(length):
    dev = Ys[start+t]-C@x[t]
    grad_psi_R += pr(start+t) * (R - dev@dev.T)@psi_R
  return grad_psi_R

def gradient_A(start, sample, model):
  grad_A = np.zeros(model.A.shape)
  for seq in sample:
    grad_A += gradient_A_single(start, seq.shape[0], seq, model)
  return grad_A

def gradient_C(start, sample, Ys, model):
  grad_C = np.zeros(model.C.shape)
  for seq in sample:
    grad_C += gradient_C_single(start, seq.shape[0], seq, Ys, model)
  return grad_C

def gradient_Q(start, sample, model):
  grad_psi_Q = np.zeros(model.Q.shape)
  for seq in sample:
    grad_psi_Q += gradient_Q_single(start, seq.shape[0], seq, model)
  return grad_psi_Q

def gradient_R(start, sample, Ys, model):
  grad_psi_R = np.zeros(model.Q.shape)
  for seq in sample:
    grad_psi_R += gradient_R_single(start, seq.shape[0], seq, Ys, model)
  return grad_psi_R