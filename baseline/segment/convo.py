''' author: sam tenka
    date:
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

def get_smooth(signal, sigma):
    ''' Return low-pass filtered array. '''
    gauss = np.arange(-6*sigma, +6*sigma) 
    gauss = np.exp(-np.multiply(gauss, gauss) / (2*sigma**2))
    gauss/= np.sum(gauss)
    convo = [scisig.fftconvolve(signal[:,i], gauss) for i in range(2)]
    convo = np.swapaxes(np.array(convo), 0, 1)
    convo = convo[len(gauss)//2:][:len(signal)]
    return convo

def get_valleys(signal):
    ''' Return generator of strict valleys in signal. '''
    for i, (a, b, c) in enumerate(zip(signal, signal[1:], signal[2:])):
        if (b<a) and (b<c):
            yield i 

def segment(signal):
    ''' Return generator of segment boundaries. '''
    Y = get_mag(signal)
    Y = np.sqrt(get_smooth(np.square(Y), utils.waveio.RATE//100))
    Ysmooth = np.sqrt(get_smooth(np.square(Y), utils.waveio.RATE//1)) * 0.1
    Y = np.maximum(Y, Ysmooth)
    Y = get_smooth(Y, utils.waveio.RATE//100)

    yield 0.0
    for v in get_valleys(Y[:,0]):
        yield float(v)/utils.waveio.RATE
    yield float(len(signal))/utils.waveio.RATE

def test_convo():
    ''' Test convo.get_smooth, .get_valleys '''
    filenm = utils.readconfig.get('TESTDATA') + '/noah.wav'
    convnm = utils.readconfig.get('TESTDATA') + '/noah_conv.wav'
    X = utils.waveio.read(filenm)

    utils.waveio.plot(X, alsoshow=False)
    for t in segment(X):
        plt.plot([t, t], [-1.0, +1.0], c='b')
    plt.show(block=False)
    
    command = 'play'
    while command:
        utils.waveio.play(filenm)
        command = raw_input('enter to exit; key to replay')

if __name__ == '__main__':
    test_convo()
