''' author: sam tenka
    credits: http://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python
             http://stackoverflow.com/questions/6951046/pyaudio-help-play-a-file
             http://code.activestate.com/recipes/521884-play-sound-files-with-pygame-in-a-cross-platform-m/
    date: 2016-11-06
    descr: Utilities for reading, writing, and playing audio files of .wav format. 
'''

import terminal
import readconfig

# import pygame
import scipy.io.wavfile as sciowav
import matplotlib.pyplot as plt
import numpy as np

class Audio:
    def __init__(self, filenm=None, rate=44100, data=None):
        if filenm is not None:
            self.read(filenm)
        else:
            self.rate = rate
            self.data = data

    def read(self, filenm):
        ''' Read from a .wav; if stereo, select only left channel. 
        '''
        self.rate, self.data = sciowav.read(filenm)
        sec1to2 = self.data[32.2*44100 : 32.6*44100]
        rank = len(self.data.shape)
        if 2 <= rank:
            assert(rank==2 and self.data.shape[1]==2)
            print('Stereo detected. Converting to mono...')
            self.data = self.data[:,0]
    
    # def play(self):
    #     '''
    #     '''
    #     pygame.init()
    #     pygame.mixer.init(frequency=self.rate, channels=2)
    #     stereo = np.transpose(np.array([self.data]*2), (1, 0))
    #     sound = pygame.sndarray.make_sound(stereo)
    #     #sound = pygame.mixer.Sound(filenm)
    #     sound.play()
    #     clock = pygame.time.Clock()
    #     while pygame.mixer.get_busy():
    #         clock.tick(60)
    #     pygame.quit()

    def write(self, filenm):
        ''' Write to .wav 
        '''
        sciowav.write(filenm, self.rate, self.data.astype(np.int16))

    def show(self):
        plt.show()

    def duration(self):
        ''' Return duration in seconds '''
        return len(self.data)/self.rate

    def plot(self, alsoshow=True, maxpts=10000):
        ''' Pop up a matlab-style plot of pressure vs time.

            Works only on stereo (i.e. Nx2 arrays); displays
            average and difference of pressures separately.
            Displays at most `maxpts` plot points, by sampling.
        '''
        step = max(1, len(self.data)//maxpts) 
        indices = np.arange(0, len(self.data), step)
        time = indices.astype(float) / self.rate
        sampnorm = self.data[indices].astype('f')/2**15
        print(time.shape, sampnorm.shape)
    
        plt.clf() 
        plt.plot(time, sampnorm, c='g', label='signal')
    
        plt.ylabel('Pressure (speaker maxes)')
        plt.xlabel('Frame Index (seconds)')
        plt.legend()
        plt.gca().set_ylim(-1, 1)
        plt.gca().set_xlim(0.0, self.duration())
        if alsoshow: self.show()

def test_waveio():
    ''' Test utils.waveio.play, .read, and .write. '''
    data_dir = readconfig.get('TESTDATA')
    filenm = data_dir + '/noah.wav'
    copynm = data_dir + '/noah_copy.wav'

    print('Copy test file...') 
    X = Audio(filenm)
    X.write(copynm)
    Y = Audio(copynm)
    X.plot()
    X.show()

    print('Check equal...')
    assert(np.array_equal(X.data, Y.data))
    #X.play()
    #Y.play()
    print('Success!')

if __name__ == '__main__':
    test_waveio()
