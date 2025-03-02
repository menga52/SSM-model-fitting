import numpy as np
from Playground.sampler import *
from Playground.true_model import *

EXs, EYs = true_model.expected()
ind = 6
size = 100000
s = sample(size,0,ind+1,EYs,true_model,use_forward=False)
for k in range(ind):
  print(sum(s[i][k] for i in range(len(s)))/len(s), 'avg at',k)
