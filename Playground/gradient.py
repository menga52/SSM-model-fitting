import numpy as np
inv = np.linalg.inv

def gradient_A_single(sample_sequence, model):
  Q_inv = inv(model.Q); A=model.A; x=sample_sequence
  grad_A = np.zeros(A.shape)
  for t in range(1,len(sample_sequence)):
    grad_A += Q_inv@(x[t]-A@x[t-1])@x[t-1].T
  return grad_A

def gradient_A(sample, model):
  grad_A = np.zeros(model.A.shape)
  for seq in sample:
    grad_A += gradient_A_single(seq, model)
  return grad_A
