import numpy as np
inv = np.linalg.inv

def forward(start, length, Ys, model):
  A=model.A; Q=model.Q; C=model.C; R=model.R
  Q_inv = inv(Q); R_inv = inv(R)
  CRC = C.T@R_inv@C
  Lambda = CRC + model.Sigma_inv_init
  h = Lambda@(C.T@R_inv@Ys[start]+model.Sigma_inv_init@model.mu_init)
  Lambda_f = []
  h_f = []
  Lambda_f.append(np.copy(Lambda))
  h_f.append(np.copy(h))

  for t in range(start+1, length):
    Lambda_inv = inv(Lambda)
    h = C.T@R_inv@Ys[t]+inv(Q+A@Lambda_inv@A.T)@A@Lambda_inv@h
    Lambda = CRC + inv(Q+A@Lambda_inv@A.T)
    h_f.append(np.copy(h))
    Lambda_f.append(np.copy(Lambda))
  return h_f, Lambda_f

def backward(start, length, Ys, model):
  A=model.A; Q=model.Q; C=model.C; R=model.R
  state_dim = model.state_dim
  Q_inv = inv(Q); R_inv = inv(R)
  Lambda = np.zeros((state_dim,state_dim))
  h = np.zeros((state_dim, 1)) # mu should be irrelevant
  Lambda_b = []; h_b = []
  Lambda_b.append(np.copy(Lambda))
  h_b.append(np.copy(h))
  AQA = A.T@Q_inv@A; AQ=A.T@Q_inv; QA=Q_inv@A
  CRCQ = C.T@R_inv@R+Q_inv; CR=C.T@R_inv
  for t in range(start-1, start-length, -1):
    temp = inv(CRCQ+Lambda)
    Lambda = AQA-AQ@temp@QA
    h = AQ@temp@(CR@Ys[t+1]+h)
    Lambda_b.append(np.copy(Lambda))
    h_b.append(np.copy(h))
  h_b.reverse(); Lambda_b.reverse()
  return h_b, Lambda_b
    
