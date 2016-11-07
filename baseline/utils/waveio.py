''' author: sam tenka
    credits: http://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python
             http://stackoverflow.com/questions/6951046/pyaudio-help-play-a-file
             http://code.activestate.com/recipes/521884-play-sound-files-with-pygame-in-a-cross-platform-m/
    date: 2016-11-06
    descr: Utilities for reading, writing, and playing audio files of .wav format. 
'''

import terminal
import readconfig

import pygame
import pyaudio
import scipy.io.wavfile as sciowav
import wave
import numpy 

RATE = 44100

def play(filenm):
    ''' Play a specified wave file.
    '''
    pygame.init()
    sound = pygame.mixer.Sound(filenm)
    sound.play()
    clock = pygame.time.Clock()
    while pygame.mixer.get_busy():
        clock.tick(60)
    pygame.quit()
    
def read(filenm):
   ''' Read wav to numpy array
   '''
   rate, data = sciowav.read(filenm)
   assert(rate==RATE)
   return data

def write(filenm, array):
   ''' Write wav from numpy array
   '''
   sciowav.write(filenm, RATE, array)

def test_waveio():
    ''' Test utils.waveio.play, .read, and .write. '''
    data_dir = readconfig.get('TESTDATA')
    filenm = data_dir + '/noah.wav'
    copynm = data_dir + '/noah_copy.wav'

    print('Copy test file...') 
    X = read(filenm)
    write(copynm, X)
    Y = read(copynm)

    print('Check equal...')
    assert(X==Y)
    play(filenm)
    play(copynm)

test_waveio()
