import utils.readconfig
from utils.waveio import Audio
import numpy as np
from segment.convo import segmentation_algo
import matplotlib.pyplot as plt
from random import random





def random_algo(audio, num_segs=158):
    ''' TODO: move to better place '''
    duration = len(audio.data)/audio.rate
    times = [random()*duration for i in range(num_segs-1)]
    return sorted(times)



def truth_list():
    with open('../../s01-2/s0101a/s0101a.phones') as f:
        text = f.read()

    lines = text.split('\n')
    body = lines[9:]
    times = [float(l.split()[0]) for l in body if l.split()]
    truth = []
    for i in times:
        if (i >= 60 and i <= 75):
            value = i - 60
            truth.append(value)
    #print(truth)
    return truth

def score(A, B):
  difference = abs(len(A) - len(B))
  start = min(B)
  duration = max(B) - start
  numsegs = 1 + len([b for b in B if b not in [0.0, 1.0]])
  A = [(a-start)/duration for a in A]
  B = [(b-start)/duration for b in B]

  u1 = 0
  distA = []
  for i in range(1,len(A)):
    tmp = A[i]-A[i-1]
    distA.append(tmp)
    u1 = u1 + distA[i-1]**2
  u2 = 0
  distB = []
  for i in range(1,len(B)):
    distB.append(B[i]-B[i-1])
    u2 = u2 + distB[i-1]**2
    
#  print "u1: ", u1
#  print "u2: ", u2
    
  combined = A + B
  combined = sorted(combined)
  distC = []
  uC = 0
  for i in range(1,len(combined)):
    distC.append(combined[i]-combined[i-1])
    uC = uC + distC[i-1]**2  
#  print "uC: ", uC
  score = abs(u1 + u2 - 2*uC)
  print "length_true: ", len(A)
  print "length_mysegs: ", len(B)
  print "difference: ", difference

  score = score * duration**2 / numsegs;
  print "score:", score

truth_list = truth_list()
filenm = '../../s01-2/s0101a/test_clip.wav'#utils.readconfig.get('TESTIN')
#filenm = utils.readconfig.get('TESTIN')
X = Audio(filenm)
mysegs = segmentation_algo(X)
#print(mysegs)
# Plot test signal with vertical bars demarcating computed segments.
X.plot(alsoshow=False)
for t in truth_list:
    plt.plot([t, t], [-1.0, +1.0], c='r')
for t in mysegs:
    plt.plot([t, t], [-1.0, +1.0], c='b')
plt.show(block=False)

raw_input('Hi ')
#mysegs = random_algo(X)
#print mysegs
score(truth_list, mysegs)
file = open('output.txt', 'w')
file.write('hihihi')
file.close()
