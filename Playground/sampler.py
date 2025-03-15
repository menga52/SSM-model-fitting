from Playground.forward_backward import forward, backward
import numpy as np
inv = np.linalg.inv
from Playground.utilities import sample_normal

def samp_one(h_f,Lambda_f,h_b,Lambda_b,f_ind,b_ind):
  S = inv(Lambda_f[f_ind] + Lambda_b[b_ind])
  return S @ (h_f[f_ind] + h_b[b_ind])

def sample(sample_size, start, length, Ys, model, buffer=-1,debug=False,use_forward=False):
  if buffer==-1: buffer = model.T
  if use_forward: sample_one = sample_one_forward
  else:           sample_one = sample_one_backward
  T = model.T
  if start+length > T:
    print("too long sequence, abandoning")
    return
  buffer_forward = min(start, buffer)
  buffer_backward = min(T-(length+start), buffer)
  h_f, Lambda_f = forward(start-buffer_forward, length+buffer_forward, Ys, model)
  h_b, Lambda_b = backward(start+length+buffer_backward-1, length+buffer_backward, Ys, model)
  h_f = h_f[buffer_forward:]; Lambda_f=Lambda_f[buffer_forward:]
  left = 0; right = left + length
  h_b = h_b[left:right]; Lambda_b = Lambda_b[left:right]
  ret = np.zeros((sample_size,length, model.state_dim,1))
  for i in range(sample_size): ret[i] = sample_one(start,length,Ys,h_f,Lambda_f,h_b,Lambda_b,model,debug)
  return ret

def sample_one_forward(start,length,Ys,h_f,Lambda_f,h_b,Lambda_b,model,debug):
  Sigma = inv(Lambda_f[0]+Lambda_b[0])
  # print(Lambda_b[0],'Sigma_0')
  A=model.A; C=model.C; R=model.R; R_inv=inv(R); Q=model.Q; Q_inv=inv(Q)
  CRC = C.T@R_inv@C; CR=C.T@R_inv; QA = Q_inv@A
  mu = Sigma@(h_f[0]+h_b[0])
  X0 = sample_normal(Sigma)+mu
  if debug: X0 = np.copy(mu)
  Xs = np.zeros((length, model.state_dim, 1))
  Xs[0] = np.copy(X0)
  for t in range(start+1,start+length):
    buffer_t = t-start
    Sigma_t = inv(Lambda_b[buffer_t]+CRC+Q_inv)
    mu_t = Sigma_t@(h_b[buffer_t]+CR@Ys[t]+QA@Xs[buffer_t-1])
    X_t = sample_normal(Sigma_t)+mu_t
    if debug: X_t = np.copy(mu_t)
    Xs[buffer_t] = X_t
  return Xs

def sample_one_backward(start,length,Ys,h_f,Lambda_f,h_b,Lambda_b,model,debug):
  Sigma = inv(Lambda_f[-1]+Lambda_b[-1])
  A=model.A; C=model.C; R=model.R; R_inv=inv(R); Q=model.Q; Q_inv=inv(Q)
  AQA = A.T@Q_inv@A; AQ=A.T@Q_inv; QA = Q_inv@A
  mu = Sigma@(h_f[-1]+h_b[-1])
  X_T = sample_normal(Sigma)+mu
  if debug: X_T = np.copy(mu)
  Xs = np.zeros((length,model.state_dim,1))
  Xs[length-1] = X_T
  buffer_t = length-2
  for t in range(start-1,start-length,-1):
    Sigma_t = inv(AQA+Lambda_f[buffer_t])
    mu_t = Sigma_t @ (AQ@Xs[buffer_t+1]+h_f[buffer_t])
    X_t = sample_normal(Sigma_t)+mu_t
    if debug: X_t = np.copy(mu_t)
    Xs[buffer_t] = X_t
    buffer_t -= 1
  return Xs

