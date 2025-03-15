import numpy as np
inv = np.linalg.inv

def forward(start, length, Ys, model):
  A=model.A; Q=model.Q; C=model.C; R=model.R; Q_inv=inv(Q)
  state_dim = model.state_dim
  R_inv = inv(R)
  CRC = C.T@R_inv@C
  h = np.copy(model.mu_init)
  h_f = np.zeros((length, model.state_dim, 1))
  Lambda_f = np.zeros((length, state_dim, state_dim))

  for t in range(start, start+length):
    if t==start:
      Lambda = CRC + A@model.Sigma_init@A.T + Q
      h = C.T@R_inv@Ys[t] + Q_inv@A@h
    else:
      Lambda_inv = inv(Lambda)
      t_inv = inv(Q+A@Lambda_inv@A.T)
      h = C.T@R_inv@Ys[t] + t_inv@A@Lambda_inv@h
      Lambda = CRC + t_inv
    h_f[t-start] = np.copy(h)
    Lambda_f[t-start] = np.copy(Lambda)
  return h_f, Lambda_f

def backward(start, length, Ys, model):
  A=model.A; Q=model.Q; C=model.C; R=model.R
  state_dim = model.state_dim
  Q_inv = inv(Q); R_inv = inv(R)
  Lambda = np.zeros((state_dim,state_dim))
  h = np.zeros((state_dim, 1)) # mu should be irrelevant
  Lambda_b = np.zeros((length,state_dim,state_dim)); h_b = np.zeros((length,state_dim,1))
  Lambda_b[length-1] = np.copy(Lambda)
  h_b[length-1] = np.copy(h)
  AQA = A.T@Q_inv@A; AQ=A.T@Q_inv; QA=Q_inv@A
  CRCQ = C.T@R_inv@C+Q_inv; CR=C.T@R_inv
  for t in range(start-1, start-length, -1):
    buffer_t = t-start+length-1
    temp = AQ@inv(CRCQ+Lambda)
    Lambda = AQA-temp@QA
    h = temp@(CR@Ys[t+1]+h)
    Lambda_b[buffer_t] = np.copy(Lambda)
    h_b[buffer_t] = np.copy(h)
  return h_b, Lambda_b
    
