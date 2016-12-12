from __future__ import division

import sigproc

from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy.fftpack import ifft
from scipy.fftpack import fft

import scipy.io.wavfile as wav
import numpy

numpy.set_printoptions(threshold='nan')

def cepstral(signal, samplerate, winlen=0.025, winstep=0.01, winfunc=lambda x:numpy.ones((x,))):
	frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
	frames = fft(frames)
	frames = numpy.where(frames == 0,numpy.finfo(float).eps,frames)
	frames = numpy.log(frames) #return result
	
	U,s,V = numpy.linalg.svd(frames, full_matrices=False)
	maximum=max(s);
	s=numpy.where(s<0.0075*maximum, 0, s)
	print s
	return U,s,V



def invcepstral(U,s,V, rate, winlen=0.025, winstep=0.01, winfunc=lambda x:numpy.ones((x,))):
	S=numpy.diag(s)
	frames=numpy.dot(U, numpy.dot(S, V))
	frames=numpy.exp(frames)
	frames=ifft(frames)
	numframe=numpy.shape(frames)[0]
	siglen=(numframe-1)*winstep*rate+winlen*rate
	sig= sigproc.deframesig(frames,siglen,numpy.shape(frames)[1], winstep*rate, lambda x:numpy.ones((x,)))
	return sig
#output audio

(rate,signal) = wav.read("english.wav")
U,s,V=cepstral(signal,rate)
sig=invcepstral(U,s,V, rate)
wav.write("rst.wav",rate, sig)