import numpy

def score(A, B):
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
    
  print "u1: ", u1
  print "u2: ", u2
    
  combined = A + B
  combined = sorted(combined)
  distC = []
  uC = 0
  for i in range(1,len(combined)):
    distC.append(combined[i]-combined[i-1])
    uC = uC + distC[i-1]**2  
  print "uC: ", uC
  score = abs(u1 + u2 - 2*uC)
  print "score: ", score    
  return score;

def score_algo(algo, truth_dir, audio_dir):
    ''' call algo like this: algo(numpy array) --> [numbers between 0.0 to 1.0]
        truth_dir path to directory containing `0001.seg`
        audio_dir path to directory containing `0001.wav`
        
        returns: number 0.0 to 1.0'''
    
    
    from scipy.io.wavfile import read
    #import audio_dir;
    #import truth_dir
    a = read("s0101a.wav")
    a = numpy.array(a[1],dtype=float)
    seg1 = algo(a)
    score_segs = score(seg1, seg2)


x=[.2,.4,.6,.8]
y=[.1,.5,.9]
w=[1,2,3,6]
q=[.21,.61]
#score(x,y)

from scipy.io.wavfile import read
a = read("s0101a.wav")
print a
print numpy.array(a[1],dtype=float)
    
    
