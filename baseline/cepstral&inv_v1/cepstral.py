from __future__ import division

import sigproc

from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy.fftpack import ifft
from scipy.fftpack import fft

import scipy.io.wavfile as wav
import numpy

def cepstral(signal, samplerate, winlen=0.025, winstep=0.01, preemph=0.97, winfunc=lambda x:numpy.ones((x,))):
	frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
	frames = fft(frames)
	frames = numpy.where(frames == 0,numpy.finfo(float).eps,frames)
	frames = numpy.log(frames) #return result
	return frames



def invcepstral(frames, rate, winlen=0.025, winstep=0.01, preemph=0.97, winfunc=lambda x:numpy.ones((x,))):
	frames=numpy.exp(frames)
	frames=ifft(frames)
	numframe=numpy.shape(frames)[0]
	siglen=(numframe-1)*winstep*rate+winlen*rate
	sig= sigproc.deframesig(frames,siglen,numpy.shape(frames)[1], winstep*rate, lambda x:numpy.ones((x,)))
	return signal
#output audio

(rate,signal) = wav.read("english.wav")
coefficients=cepstral(signal,rate)
signal=invcepstral(coefficients, rate)
wav.write("rst.wav",rate, signal)