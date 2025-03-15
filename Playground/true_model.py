import numpy as np
from Playground.model import *

dim = 2
A = np.identity(dim); A[1][1]=0.9; A[0][0]=0.9; A[1][0] = -0.2
A = np.array([[0.7, -0.2],[-0.6, 0.2]])
Q = 1*np.identity(dim)
C = np.identity(dim)
R = np.identity(dim)
T = 1000
X_init = np.asarray([[4],[9]])
Sigma_init = np.zeros((dim,dim))

true_model = model(A,Q,C,R,T,X_init,np.zeros((dim,dim)))

