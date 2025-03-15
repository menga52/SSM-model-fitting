from Playground.true_model import *
from Playground.sampler import *
from Playground.forward_backward import *

EXs, EYs = true_model.expected()

sample_size = 1
Xs, Ys = true_model.initialize()
start = 10
length = 4
print(Xs[40])
useYs = true_model.expectedYs(Xs)
for buff in range(3,15):
    samps = sample(sample_size,start,length, useYs, true_model, buffer=buff, debug=True, use_forward=False)
    avg = []
    diff = []
    for k in range(length):
        avg.append(sum([samps[i][k] for i in range(sample_size)])/sample_size)
        diff.append(Xs[start+k]-avg[k])

    print(diff, 'diff',buff)
# in theory, the difference should decay with increased buffer, but it seems to do so quickly enough to converge immediately


