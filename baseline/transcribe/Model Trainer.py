"""
To do:
    -Implement Metrics (Precision, Accuracy, Categorical Cross-Entropy)
    -Look at model predictions
    -Memory Issue?
"""


#Library
import os, string, scipy.io.wavfile, numpy.fft, re, pickle, random
from numpy import zeros, array, concatenate
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, Activation, TimeDistributed
from keras.preprocessing import sequence
from keras.engine.topology import Merge

##import "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\\metrics.py"


#Variables
window_length = 0.06
step = 0.03
output = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\\Test\\"

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks

##Get list of .wav files and associated .phones files in path
def get_files(path):
    file_list = [file for file in os.listdir(path) if file.endswith(".wav")]
    for a in range(len(file_list)):
        phon_file = file_list[a].replace(".wav", ".phones")
        if os.path.isfile(path+phon_file):
            file_list[a] = (file_list[a], phon_file)
        else:
            file_list[a] = (file_list[a], None)
    ##Returns (wav file, phones file/None)
    return file_list
    
def open_segdict(input):
    try:
        with open(input, "rb") as f:
            segdict = pickle.load(f)
            print "Loaded Segment Dictionary"
    except IOError:
        print "No Segment Dictionary found - creating new one..."
        segdict = {}
    return segdict

def save_segdict(segdict, output):
    with open(output, "wb") as f:
        pickle.dump(segdict, f)
    print "Saved Segment Dictionary"
    return
    
##Returns processed phones file
def read_in_transcription(path, phon_file, segdict):
    a =  len(segdict)+1
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
            for b in range(len(transcription)):
                if transcription[b][2] not in segdict:
                    segdict[transcription[b][2]]= a
                    a+=1
                transcription[b][2]=segdict[transcription[b][2]]
        ##Returns (start time, classification [121/2], transcription_index)
        return transcription, segdict
    else:
        return None
        
##Reads in wav file, windows, and computes DFT
def read_in_fft(path, sound, transcription, output_sound=False):
    print "Reading: "+path+sound
    sampling_rate, wav_array = scipy.io.wavfile.read(path+sound)
    num_ticks = int(sampling_rate*window_length)
    a = 0
    trindex = 0
    fft_list = []
    ##If transcription is present, include this information with DFT
    if transcription != None:
        while a < len(wav_array):
            #Window is rectangular (Not necessarily ideal...)
            if trindex<len(transcription) - 1 and float(transcription[trindex][0]) < float(a+num_ticks)/sampling_rate:
                ##Window up to (but not beyond) the end of the segment
                windowed_sound = wav_array[a:int(sampling_rate*float(transcription[trindex][0]))]
                ##scipy.io.wavfile.write(output+str(a)+re.sub('[\/<>?:]',"",transcription[trindex][2])+".wav",sampling_rate, windowed_sound)
                a=int(sampling_rate*float(transcription[trindex][0]))+1
                trindex += 1
            else:
                windowed_sound = wav_array[a:min(a+num_ticks,len(wav_array)-1)]
                ##scipy.io.wavfile.write(output+str(a)+re.sub('[\/<>?:]',"",transcription[trindex][2])+".wav",sampling_rate, windowed_sound)
                a+=int(step*sampling_rate)
            ##Take DFT of windowed audio
            ##Do we want a cepstrum instead of a spectrum?
            fft_list.append((abs(numpy.fft.fft(windowed_sound, n=num_ticks)), transcription[trindex][2]))
        ##Returns [...,(DFT, transcription at this point),...]
        return fft_list
    ##If not, just include DFT on its own
    else:
        while a < len(wav_array):
            #Window is rectangular (Not necessarily ideal...)
            windowed_sound = wav_array[a:min(a+num_ticks,len(wav_array)-1)]
            a+=num_ticks
            ##Take DFT of windowed audio
            ##Do we want a cepstrum instead of a spectrum?
            fft_list.append((numpy.fft.fft(windowed_sound, n=num_ticks), None))
        ##Returns [...,(DFT, None),...]
        return fft_list


if __name__ == "__main__":    
    path = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\\data\\"
    segment_dict = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\\segment_dict.pkl"
    segdict = open_segdict(segment_dict)
    hidden_units = 70
    batch_size = 1000
    segdict = {}
    files = get_files(path)
    for file in files:
        transcription, segdict = read_in_transcription(path, file[1],segdict)
    save_segdict(segdict, segment_dict)
    
    model = Sequential()
    model.add(Bidirectional(LSTM(output_dim=len(segdict), init='uniform', inner_init='uniform',forget_bias_init='one', return_sequences=True, activation='tanh',inner_activation='sigmoid',), merge_mode='sum', input_shape = (batch_size,960)))
    model.add(Dropout(0.7))
    model.add(TimeDistributed(Dense(len(segdict), activation='sigmoid')))
    
    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    
    model.summary()
    for file in files:
        fft = read_in_fft(path, file[0], transcription)
        X_train = numpy.array([numpy.array(item[0]) for item in fft])
        X_train = numpy.array(numpy.array([X_train[x:x+batch_size] for x in range(0, len(X_train), batch_size)]))
        X_train = sequence.pad_sequences(X_train, maxlen=batch_size)
        X_train = numpy.reshape(X_train, (len(X_train), len(X_train[0]), len(X_train[0][0])))
        Y_train_values = [[1 if i == item[1] else e for i, e in enumerate(numpy.zeros(len(segdict)))] for item in fft]
        Y_train = numpy.array([Y_train_values[x:x+batch_size] for x in range(0, len(Y_train_values), batch_size)])
        Y_train = sequence.pad_sequences(Y_train, maxlen=batch_size)
        Y_train = numpy.reshape(Y_train, (len(Y_train), len(Y_train[0]), len(Y_train[0][0])))
        
        hist = model.fit(X_train, Y_train, batch_size=1, nb_epoch=10, validation_split=0.2, verbose=1)
   
    model.save("./SpeechRecognitionModel.h5")
