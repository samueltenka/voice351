''' author: sam tenka
    date:
    descr: 1d convolution for audio processing.
'''

import numpy as np

def get_smooth_hist(signal, sigma, N):
    ''' Return low-pass filtered array. '''
    gauss = np.arange(-3*sigma, +3*sigma, step) 
    gauss = np.exp(-np.multiply(gauss, gauss) / (2*sigma))
    convo = np.convolve(signal, gauss) 
    convo = convo[len(gauss)//2:][:len(hist)]
    return convo

def get_valleys(signal):
    ''' Return generator of strict valleys in signal. '''
    for i, (a, b, c) in enumerate(zip(signal, signal[1:], signal[2:])):
        if b<a and b<c: yield i 
