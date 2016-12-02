''' author: sam tenka
    date: 2016-11-07
    descr: 1d convolution for audio processing.
'''

import utils.readconfig
import utils.waveio
import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt

def get_mag(signal):
    s = signal.astype('f')
    r = np.sqrt((np.square(s[:,0]) + np.square(s[:,1])) / 2)
    return np.swapaxes(np.array([r, r]), 0, 1)

def get_smooth(audio, sigma):
    ''' Return low-pass filtered array.

        `sigma` has units of seconds, and sets the smoothing timescale.
    '''
    signal=audio.data
    sigma*= audio.rate
    gauss = np.arange(-6*sigma, +6*sigma) 
    gauss = np.exp(-np.square(gauss, gauss) / (2*sigma**2))
    gauss/= np.sum(gauss)
    convo = scisig.fftconvolve(signal, gauss)
    convo = np.swapaxes(np.array(convo), 0, 1)
    convo = convo[len(gauss)//2:][:len(signal)]
    return convo

def smooth_silence(signal, sigma=1.0, scale=0.1):
    ''' Return signal whose quietest parts have been smoothed. '''
    smoothed = np.sqrt(get_smooth(np.square(signal), sigma))
    return np.maximum(signal, scale * smoothed)

def get_valleys(signal):
    ''' Return generator of strict valleys in signal.

        Todo: optimize by using relu / minus / product
    '''
    for i, (a, b, c) in enumerate(zip(signal, signal[1:], signal[2:])):
        if (b<a) and (b<c):
            yield i 

def segment(audio, sigmaA=0.005, sigmaB=0.005):
    ''' Return generator of segment boundaries. '''
    signal = audio.data
    Y = get_mag(signal)
    Y = np.sqrt(get_smooth(np.square(Y), sigmaA))
    Y = smooth_silence(Y)
    Y = get_smooth(Y, sigmaB)

    yield 0.0
    for v in get_valleys(Y[:,0]):
        yield float(v)/audio.rate
    yield float(len(signal))/audio.rate

def test_convo():
    ''' Test convo.segment, and hence also .get_valleys, .get_smooth, .get_mag. '''
    filenm = utils.readconfig.get('TESTIN')
    convnm = utils.readconfig.get('TESTOUT')
    X = utils.waveio.read(filenm)

    # 0. Plot test signal with vertical bars demarcating computed segments.
    utils.waveio.plot(X, alsoshow=False)
    for t in segment(X):
        plt.plot([t, t], [-1.0, +1.0], c='b')
    plt.show(block=False)
    
    # 1. (Prompt to) play sound until user presses enter.
    command = 'play'
    while command:
        utils.waveio.play(filenm)
        command = raw_input('enter to exit; key to replay')

if __name__ == '__main__':
    test_convo()
