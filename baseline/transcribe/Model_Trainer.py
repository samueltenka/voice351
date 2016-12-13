"""
To do:
    -Undersampling and oversampling
"""


#Library
#from __future__ import print_function
import os, string, scipy.io.wavfile, re, pickle, python_speech_features
from numpy import zeros, array, reshape, arange
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.engine.topology import Merge
from keras.optimizers import RMSprop, Adadelta
from keras.utils.np_utils import to_categorical
from functools import partial, update_wrapper
from math import sqrt, pow
from random import sample, randint, choice, shuffle
from sklearn.metrics import confusion_matrix
import keras.backend as K
from collections import deque

#Variables
window_length = 0.06
step = 0.03
batch_size = 5
coefficients = 30
tuple_size = 5
output = './transcribe_output/'


##Get list of .wav files and associated .phones files in path
def get_files(path, num=None):
    file_list = [f for f in os.listdir(path) if f.endswith(".wav")]
    if num==None:
        num = len(file_list)
    for a in range(num):
        print(a, file_list[a])
        phon_file = file_list[a].replace(".wav", ".phones")
        if os.path.isfile(path+phon_file):
            file_list[a] = (file_list[a], phon_file)
        else:
            print('Oh no! %s does not exist!' % (path+phon_file))
            file_list[a] = (file_list[a], None)
    ##Returns (wav file, phones file/None)
    return file_list[:num]

##Opens pickle file as dictionary    
def open_dict(input):
    try:
        with open(input, "rb") as f:
            segdict = pickle.load(f)
            print "Loaded Segment Dictionary"
    except IOError:
        print "No Segment Dictionary found - creating new one..."
        segdict = {}
    return segdict

##Saves dictionary as pickle file
def save_dict(dictionary, output):
    with open(output, "wb") as f:
        pickle.dump(dictionary, f)
    print "Saved dictionary to "+output+"."
    return
    
##Reads .phones file and returns a transcription list  
def read_in_transcription(path, phon_file, segdict):
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
            for b in range(len(transcription)):
                ##Exclude non-speech sounds
                if len(transcription[b])<3:
                    print transcription[b]
                else:
                    if (transcription[b][2] == "{B-TRANS}") or \
                    (transcription[b][2] == "{E-TRANS}") or \
                    (transcription[b][1] != "121") and \
                    (transcription[b][2] != "VOCNOISE") and \
                    (transcription[b][2] != "IVER") and \
                    (transcription[b][2] != "SIL"): 
                        if transcription[b][2] not in segdict:
                            segdict[transcription[b][2]]= 0
                        transcription[b][2]=segdict[transcription[b][2]]
                    else:
                        transcription[b][2]=0
        ##Returns (start time, classification [121/2], transcription_index)
        return transcription
    else:
        return None
        
        
##Reads transcription list and returns dictionaries useful for later on
##segdict: assigns unique numbers to segment classes
##freq_dict: counts frequency of each segments
##tuple_dict: counts frequency of frame n-tuples ending in a given segment
def get_important_info(path, phon_file, segdict, freq_dict, tuple_dict):
    a =  len(segdict)+1
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
            numbers = []
            for b in range(len(transcription)):
                    if len(transcription[b]) <3:
                        print transcription[b]
                    else:
                        if transcription[b][2] not in segdict:
                            segdict[transcription[b][2]]= a
                            a+=1
                        transcription[b][2]=segdict[transcription[b][2]]
                        numbers.append(transcription[b][2])
                        if transcription[b][2] not in freq_dict:
                            freq_dict[transcription[b][2]]=0
                        freq_dict[transcription[b][2]]+=1
            number_tuple=[numbers[tuple_size*c:(c+1)*(tuple_size)] for c in range(len(numbers)/tuple_size)]
            for item in number_tuple:
                if item[-1] not in tuple_dict:
                    tuple_dict[item[-1]]=0
                tuple_dict[item[-1]]+=1
                    
        ##Returns (start time, classification [121/2], transcription_index)
        return segdict, freq_dict, tuple_dict
    else:
        return None
        
##Reads in wav file, windows, and computes relavant features for audio
def read_in_features(path, sound, segdict, transcription=None, output_sound=False):
    print "Reading: "+path+sound
    sampling_rate, wav_array = scipy.io.wavfile.read(path+sound)
    num_ticks = int(sampling_rate*window_length)
    a = 0
    trindex = 0
    features_list = []
    ##If transcription is present, include this information with features
    if transcription != None:
        for trindex in  range(1,len(transcription)-1):
            ##Window up to (but not beyond) the end of the segment
            windowed_sound = wav_array[int(sampling_rate*float(transcription[trindex][0])): \
            int(sampling_rate*float(transcription[trindex+1][0]))]
            if len(windowed_sound)>0 and len(transcription[trindex])> 2 and transcription[trindex][2] != "{E-TRANS}":
                features_list= features_list+[(item, transcription[trindex][2]) for \
                item in python_speech_features.mfcc(windowed_sound, samplerate=sampling_rate, \
                winlen=window_length, winstep=step, numcep=coefficients, nfilt=coefficients)] ## <- MFCCs
        ##Returns [...,(MFCC, transcription at this point),...]
        return features_list
    ##If not, just include features on their own
    else:
        ##Get relevant fetaures from windowed audio
        features_list = [(item, None) for item in \
        python_speech_features.mfcc(windowed_sound, samplerate=sampling_rate, \
        winlen=window_length, winstep=step)] ## <- MFCCs        
        ##Returns [...,(DFT, None),...]
        return features_list

if __name__ == "__main__":    
    trainpath = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\Data\\"
    testpath = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\Data\\"
    segment_dict = "./segment_dict.pkl"
    freq_dict = "./frequency_dict.pkl"
    modelpath = "./SpeechRecognitionModel.h5"

    
    files = get_files(path=trainpath, num=None)
    
    ##Build and save dictionaries that store segment identities and frequencies
    segdict = {}
    freqdict = {}
    tupledict = {}
    
    fft= []
   
    for file in files:
        segdict, freqdict, tupledict  = get_important_info(trainpath, file[1], segdict, freqdict, tupledict)
     
    balance_number = 10
    
    """
    ###If we can only load one file at a time...
    for file in files:
        transcription = read_in_transcription(trainpath, file[1], segdict)
        fft = read_in_features(trainpath, file[0], segdict, transcription)
        slices = [fft[a*tuple_size:(a+1)*(tuple_size)] for a in range(len(fft)/tuple_size)]
        final_array = [[] for a in range(len(segdict))]
        for slice in slices:
            possibility = randint(0,balance_number)
            if slice[-1][-1] < len(final_array):
                if len(final_array[slice[-1][-1]]) < possibility:
                    final_array[slice[-1][-1]].append(slice)
            else:
                print slice[-1]
    model_input = []
    for bucket in final_array:
        model_input= model_input+[val for sublist in bucket for val in sublist]
    model_input = array(model_input)
    """
    
    
    save_dict(segdict, segment_dict)
    save_dict(freqdict, freq_dict)
  
    ##Initialize Model
    model = Sequential()
    
    
    ## With RNN
    out = len(segdict)
    model.add(Dense(2000, activation='tanh', input_shape=(coefficients,)))
    model.add(Dropout(0.2))
    model.add(Dense(out, activation='softmax'))

    rms = RMSprop()
    ada = Adadelta()
    
    model.compile(loss="mse",
        optimizer=ada,
        metrics=["accuracy"])

        
    ##model = load_model(modelpath)
    model.summary()

    transcription_list = []
    fft_list = []
    
    ## populate bucket_list 
    bucket_list = [[] for a in range(len(segdict))]
    filenm = 'bucket_list'
    if not os.path.isfile(filenm):
        print('POPULATE BL')
        for file in files:
            transcription_list.append(read_in_transcription(trainpath, file[1], segdict))
            features = read_in_features(trainpath, file[0], segdict, transcription_list[-1])
            slices = [features[a*tuple_size:(a+1)*(tuple_size)] for a in range(len(features)/tuple_size)]
            for sl in slices:
                phon = sl[-1][-1]
                if phon < len(bucket_list):
                    bucket_list[phon].append(sl)
        with open(filenm, 'wb') as f:
            pickle.dump(bucket_list, f)
    else:
        print('LOAD BL')
        with open(filenm, 'rb') as f:
            bucket_list = pickle.load(f)

    ##Train Model
    try:
        for e in range(20):
            selected_slices = []
            for b in range(len(segdict)):
                if len(bucket_list[b])>0:
                    selected_slices.append([choice(bucket_list[b]) for c in range(balance_number)])
            shuffle(selected_slices)
    
            model_input = []
            for bucket in selected_slices:
                model_input = model_input+[val for sublist in bucket for val in sublist if len(val[0])==coefficients]
               # print len(model_input)

            #print [len(input) for input in model_input]
            X_train = array([list(item[0]) for item in model_input])
            #print X_train
            #X_train = array([list(X_train[x:x+batch_size]) for x in range(0, len(X_train), batch_size)])
            #X_train = array([X_train[x: for x in range(0, len(X_train), batch_size)])
            print('X_train has shape', X_train.shape)
            #X_train = sequence.pad_sequences(X_train, maxlen=batch_size)
            #X_train = reshape(X_train, (len(X_train), len(X_train[0]), len(X_train[0][0])))
    
            Y_train_values = [[1 if i == item[1] else e for i, e in enumerate(zeros(len(segdict)))] for item in model_input]
            Y_train = array(Y_train_values)
            #Y_train = array([Y_train_values[x:x+batch_size] for x in range(0, len(Y_train_values), batch_size)])
            #Y_train = array([Y_train_values[x:x+batch_size] for x in range(0, len(Y_train_values), batch_size)])
            #Y_train = sequence.pad_sequences(Y_train, maxlen=batch_size)
            print('Y_train has shape', Y_train.shape)

            model.fit(X_train, Y_train, batch_size=1, nb_epoch=1, validation_split=0.2, verbose=1)
            
                 ##Predict Output    
            predictions = model.predict(X_train, batch_size=1)
            sorted_dict = sorted(segdict.items(), key=lambda x:x[1])
            predicted_output=[]
            for item in predictions:
                segment_list = list(item)
                max_index = segment_list.index(max(segment_list))
                if max_index != 0:
                    predicted_output.append(sorted_dict[max_index-1][0])
                else:
                    predicted_output.append(0)
            print predicted_output
    except KeyboardInterrupt:
        pass
        
        ##Generate Confusion Matrix at end of each training step
        predictions = model.predict(X_train, batch_size=1)
        cmatrix = confusion_matrix(predictions, Y_train)
        
        
        ##Predict Output    
        sorted_dict = sorted(segdict.items(), key=lambda x:x[1])
        predicted_output=[]
        for item in predictions:
            for segment in item:
                print(segment)
                #segment_list = list(segment)
                #max_index = segment_list.index(max(segment_list))
                #if max_index != 0:
                #    predicted_output.append(sorted_dict[max_index-1][0])
                #else:
                #    predicted_output.append(0)
        print predicted_output
        
        
    model.save("./SpeechRecognitionModel.h5")

