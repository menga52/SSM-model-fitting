import numpy as np
from Playground.model import *

dim = 2
A = np.identity(dim); A[1][1]=0.9; #A[0][0]=0.9
Q = 1*np.identity(dim)
C = np.identity(dim)
R = 1*np.identity(dim)
T = 10
X_init = np.asarray([[4],[9]])

true_model = model(A,Q,C,R,T,X_init,np.zeros((dim,dim)))

