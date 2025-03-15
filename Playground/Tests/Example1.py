import numpy as np
from Playground.model import *
from Playground.ascent import optimizer_factory_A


nu = np.pi/4; sin=np.sin; cos=np.cos; s=sin(nu); c=cos(nu)
A = 0.7*np.array([[c,-s],[s,c]])
Q = 0.1*np.identity(2)
C = np.identity(2)
R = np.identity(2)
T = 10**6
# comparatively small Q and large R mean more information in prior state than emission
# so buffering is necessary
mu_0 = np.array([[0],[0]])
Sigma_0 = np.zeros((2,2))
A_wrong = A + rng.uniform(0,.1,size=A.shape)
theta = model(A,Q,C,R,T,mu_0,Sigma_0)
theta_wrong = model(A_wrong, Q, C, R, T, mu_0, Sigma_0)
print('initializing')
Xs, Ys = theta.initialize()
max_steps = 1000; epsilon = 0; step_size = 1; length=20; batch_size = 1; batches=3; buffer=10
verbose = True

def decay(step_size):
  if step_size < 1e-5: return step_size
  return step_size/(1+step_size)

optimize = optimizer_factory_A(max_steps,epsilon,step_size,length,Ys,batch_size,batches,decay,buffer,verbose=verbose)
print(A,'A_true')
print(A_wrong, 'A_wrong')
theta_hat = optimize(theta_wrong)
