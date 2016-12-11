#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import get_filterbanks
from python_speech_features import get_filterbanks_location
from python_speech_features import sigproc

from scipy.fftpack import idct
from scipy.fftpack import ifft
from scipy.fftpack import fft

import scipy.io.wavfile as wav
import numpy
import math

(rate,sig) = wav.read("english.wav")
print(rate)
(mfcc_feat,energy) = mfcc(sig,rate, nfilt=26)
numpy.set_printoptions(threshold='nan')



# Above calculate the MFCC and print the whold matrix
# Next part will be the inverse of MFCC method for synthesis

# inverse lifter operation
L=22 #by default or change later
print(energy)
mfcc_feat[:,0]=0
nframes,ncoeff=numpy.shape(mfcc_feat)
n= numpy.arange(ncoeff)

lift=1/(1+(L/2)*numpy.sin(numpy.pi*n/L))
cepstra=lift*mfcc_feat
print('the shape of cepstra after inverse lift: ',numpy.shape(cepstra))

# inverse DCT
cepstra=idct(cepstra, type=2, axis=1, norm='ortho')
l,w=numpy.shape(cepstra)
print('the shape of cepstra after idct: ',numpy.shape(cepstra))

# fill the column 14~26 according to symmetry
'''
for i in range (0,13):
	cepstra=numpy.column_stack((cepstra,cepstra[:,12-i]))
	cepstra.resize(l,w+i+1)
'''
# inverse logrithm (use absolute value)
cepstra=numpy.absolute(numpy.exp(cepstra))
print('symmetry transform after idct: ',numpy.shape(cepstra))

# inverse fbank
# some default value
highfreq=rate/2
lowfreq=0
nfilt=26
nfft=512
winlen=0.025
winstep=0.01

# get filterbank
fb = get_filterbanks(nfilt,nfft,rate,lowfreq,highfreq)
(fbl,sz)= get_filterbanks_location(nfilt,nfft,rate,lowfreq,highfreq)
print('the shape of filter bank: ',numpy.shape(fb))
#print(fb)
print('the shape of filter bank location: ', numpy.shape(fbl))
#print(fbl)
print(sz)


new=numpy.zeros((numpy.shape(cepstra)[0],numpy.shape(fb)[1]))
for i in range (0,26):
	for j in range(0, numpy.shape(cepstra)[0]):
		#new[j,fbl[i]]=energy[j]/max(energy)*cepstra[j.i]
		for k in range(-int(sz[i]), int(sz[i])):
			new[j,fbl[i]+k]=energy[j]/max(energy)*(numpy.absolute(int(sz[i])-k))/int(sz[i])*cepstra[j,i]

print('the shape of frames after inverse filter bank: ',numpy.shape(new))


# inverse power spectra to get frames
frames= ifft(new)
print('the shape of frames after inverse powerspec: ',numpy.shape(frames))

# deframe to orginal signal
numframe=numpy.shape(frames)[0]
print(numframe)
siglen=int(math.ceil(numframe*winstep*rate+winlen*rate))
signal= sigproc.deframesig(frames,siglen,numpy.shape(frames)[1],winstep*rate,
						   lambda x:numpy.ones((x,)))
print(numpy.shape(signal))


#inverse preemphasis
for i in range(1,len(signal)):
	signal[i]=signal[i]+0.95*signal[i-1]


sigfft=numpy.log(fft(signal,512))
offt=numpy.log(fft(sig,512))
sigmin=numpy.absolute(ifft(numpy.exp(offt-sigfft)))

sigminsim=numpy.convolve(sigmin,signal)

wav.write("rst.wav",rate, signal)
wav.write("rst_minus.wav",rate, sigminsim)