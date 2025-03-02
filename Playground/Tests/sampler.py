from Playground.sampler import *
from Playground.true_model import *

EXs, EYs = true_model.expected()
samp = sample(1,0,true_model.T, EYs, true_model,debug=True,use_forward=False)
