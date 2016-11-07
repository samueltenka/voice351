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
            print(a, b, c)
            yield i 

def test_convo():
    ''' Test convo.get_smooth, .get_valleys '''
    filenm = utils.readconfig.get('TESTDATA') + '/noah.wav'
    convnm = utils.readconfig.get('TESTDATA') + '/noah_conv.wav'
    X = utils.waveio.read(filenm)
    X = get_mag(X)
    utils.waveio.plot(X)
    
    Y = X
    Y = np.sqrt(get_smooth(np.square(X), 44100//100))
    Y2 = np.sqrt(get_smooth(np.square(Y), 44100//1)) * 0.1
    utils.waveio.plot(Y2)
    Y = np.maximum(Y, Y2)
    Y = get_smooth(Y, 44100//100)
    #Y = np.square(get_smooth(np.sqrt(Y), 44100//100))
    #Y = get_smooth(Y, 44100//100)
    
    utils.waveio.plot(Y, alsoshow=False)
    for v in get_valleys(Y[:,0]):
        t = float(v)/44100
        plt.plot([t, t], [-1.0, +1.0])
    plt.show(block=False)
    
    utils.waveio.write(convnm, Y)
    utils.waveio.play(filenm)
    utils.waveio.play(convnm)
    plt.show()

if __name__ == '__main__':
    test_convo()
