from Playground.forward_backward import forward, backward
import numpy as np
inv = np.linalg.inv
from Playground.utilities import sample_normal

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
  h_f, Lambda_f = forward(start, length+buffer_forward, Ys, model)
  h_b, Lambda_b = backward(start+length+buffer_backward-1, length+buffer_backward, Ys, model)
  h_f = h_f[buffer_forward:]; Lambda_f=Lambda_f[buffer_forward:]
  h_b = h_b[:length]; Lambda_b = Lambda_b[:length]
  return [sample_one(start,length,Ys,h_f,Lambda_f,h_b,Lambda_b,model,debug) for _ in range(sample_size)]

def sample_one_forward(start,length,Ys,h_f,Lambda_f,h_b,Lambda_b,model,debug):
  Sigma = inv(Lambda_f[0]+Lambda_b[0])
  print(Sigma,'Sigma_0')
  A=model.A; C=model.C; R=model.R; R_inv=inv(R); Q=model.Q; Q_inv=inv(Q)
  CRC = C.T@R_inv@C; CR=C.T@R_inv; QA = Q_inv@A
  mu = Sigma@(h_f[0]+h_b[0])
  print(mu,'mu_0')
  X0 = sample_normal(Sigma)+mu
  if debug: X0 = np.copy(mu)
  Xs = []
  Xs.append(X0)
  for t in range(start+1,length):
    buffer_t = t-start
    Sigma_t = inv(Lambda_b[buffer_t]+CRC+Q_inv)
    mu_t = Sigma_t@(h_b[buffer_t]+CR@Ys[t]+QA@Xs[buffer_t-1])
    X_t = sample_normal(Sigma_t)+mu_t
    if debug: X_t = np.copy(mu_t)
    Xs.append(X_t)
  return Xs

def sample_one_backward(start,length,Ys,h_f,Lambda_f,h_b,Lambda_b,model,debug):
  Sigma = inv(Lambda_f[-1]+Lambda_b[-1])
  A=model.A; C=model.C; R=model.R; R_inv=inv(R); Q=model.Q; Q_inv=inv(Q)
  AQA = A.T@Q_inv@A; AQ=A.T@Q_inv; QA = Q_inv@A
  mu = Sigma@(h_f[-1]+h_b[-1])
  X_T = sample_normal(Sigma)+mu
  if debug: X_T = np.copy(mu)
  Xs = []; Xs.append(X_T)
  buffer_t = length-2
  for t in range(start-1,start-length,-1):
    Sigma_t = inv(AQA+Lambda_f[buffer_t])
    mu_t = Sigma_t @ (AQ@Xs[-1]+h_f[buffer_t])
    X_t = sample_normal(Sigma_t)+mu_t
    if debug: X_t = np.copy(mu_t)
    Xs.append(X_t)
    buffer_t -= 1
  Xs.reverse(); return Xs

