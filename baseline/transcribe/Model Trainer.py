"""
To do:
<<<<<<< HEAD
    -Undersampling and oversampling
=======
    -Implement Metrics (Precision, Accuracy, Categorical Cross-Entropy)
    -Look at model predictions
    -Memory Issue?
>>>>>>> 259ca6bc7915ef698c7cd2b0c82eff7dd92902de
"""


#Library
<<<<<<< HEAD
import os, string, scipy.io.wavfile, re, pickle, python_speech_features
from numpy import zeros, array, reshape
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Activation
from keras.preprocessing import sequence
from keras.engine.topology import Merge
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from functools import partial, update_wrapper
from math import sqrt, pow
from random import sample, randint, choice, shuffle
##from scklearn.metrics import confusion_matrix
import keras.backend as K
from collections import deque
=======
import os, string, scipy.io.wavfile, numpy.fft, re, pickle, random
from numpy import zeros, array, concatenate
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, Activation, TimeDistributed
from keras.preprocessing import sequence
from keras.engine.topology import Merge

##import "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\\metrics.py"

>>>>>>> 259ca6bc7915ef698c7cd2b0c82eff7dd92902de

#Variables
window_length = 0.06
step = 0.03
<<<<<<< HEAD
batch_size = 30
coefficients = 13
tuple_size = 4
output = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\Test\\"

=======
output = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\\Test\\"

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks
>>>>>>> 259ca6bc7915ef698c7cd2b0c82eff7dd92902de

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
<<<<<<< HEAD

##Opens pickle file as dictionary    
def open_dict(input):
=======
    
def open_segdict(input):
>>>>>>> 259ca6bc7915ef698c7cd2b0c82eff7dd92902de
    try:
        with open(input, "rb") as f:
            segdict = pickle.load(f)
            print "Loaded Segment Dictionary"
    except IOError:
        print "No Segment Dictionary found - creating new one..."
        segdict = {}
    return segdict

<<<<<<< HEAD
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
                    if (transcription[b][1] != "121") and \
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
=======
def save_segdict(segdict, output):
    with open(output, "wb") as f:
        pickle.dump(segdict, f)
    print "Saved Segment Dictionary"
    return
    
##Returns processed phones file
def read_in_transcription(path, phon_file, segdict):
>>>>>>> 259ca6bc7915ef698c7cd2b0c82eff7dd92902de
    a =  len(segdict)+1
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
<<<<<<< HEAD
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
=======
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
>>>>>>> 259ca6bc7915ef698c7cd2b0c82eff7dd92902de
    print "Reading: "+path+sound
    sampling_rate, wav_array = scipy.io.wavfile.read(path+sound)
    num_ticks = int(sampling_rate*window_length)
    a = 0
    trindex = 0
<<<<<<< HEAD
    features_list = []
    ##If transcription is present, include this information with features
    if transcription != None:
        for trindex in  range(len(transcription)-1):
            ##Window up to (but not beyond) the end of the segment
            windowed_sound = wav_array[int(sampling_rate*float(transcription[trindex][0])): \
            int(sampling_rate*float(transcription[trindex+1][0]))]
            if len(windowed_sound)>0 and len(transcription[trindex])>2:
                features_list= features_list+[(item, transcription[trindex][2]) for \
                item in python_speech_features.mfcc(windowed_sound, samplerate=sampling_rate, \
                winlen=window_length, winstep=step)] ## <- MFCCs
        
        trindex+=1 
        windowed_sound = wav_array[int(sampling_rate*float(transcription[trindex][0]))-1:len(wav_array)]
        ##Get relevant features from windowed audio
        if len(windowed_sound) > 0:
            features_list= features_list+[(item, transcription[trindex][2]) for item in python_speech_features.mfcc(windowed_sound, samplerate=sampling_rate, \
            winlen=window_length, winstep=step)] ## <- MFCCs
        ##Returns [...,(DFT, transcription at this point),...]
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
    trainpath = "F:\\Buckeyes\\"
    testpath = "F:\\Buckeyes\\"
    segment_dict = "./segment_dict.pkl"
    freq_dict = "./frequency_dict.pkl"
    modelpath = "./SpeechRecognitionModel.h5"

    
    files = get_files(trainpath)
    
    ##Build and save dictionaries that store segment identities and frequencies
    segdict = {}
    freqdict = {}
    tupledict = {}
    
    fft= []
   
    for file in files:
        segdict, freqdict, tupledict  = get_important_info(trainpath, file[1], segdict, freqdict, tupledict)
     
    balance_number = 80
    
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
    
    model.add(Bidirectional(LSTM(output_dim=len(segdict), init='uniform', \
     inner_init='uniform',forget_bias_init='one', return_sequences=True, activation='tanh', \
     inner_activation='sigmoid',), merge_mode='sum', input_shape = (batch_size,coefficients)))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(Dense(len(segdict), activation='sigmoid')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(len(segdict), activation='sigmoid')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(len(segdict), activation='sigmoid')))
    model.add(Activation('softmax'))

    
    """   
    ##With plain NN
    model.add(Dense(output_dim=len(segdict), input_shape = (batch_size, coefficients)))
    model.add(Dropout(0.3))
    model.add(Dense(len(segdict)))
    model.add(Dropout(0.2))
    model.add(Dense(len(segidct)))
    model.add(Activation('softmax'))
    """             

    rms = RMSprop()
    
    model.compile(loss="categorical_crossentropy",
        optimizer=rms,
        metrics=["accuracy"])

        
    ##model = load_model(modelpath)
    model.summary()

    transcription_list = []
    fft_list = []
    
    bucket_list  = [[] for a in range(len(segdict))]

    ##          
    for file in files:
        transcription_list.append(read_in_transcription(trainpath, file[1], segdict))
        features = read_in_features(trainpath, file[0], segdict, transcription_list[-1])
        slices = [features[a*tuple_size:(a+1)*(tuple_size)] for a in range(len(features)/tuple_size)]
        for slice in slices:
            bucket_list[slice[-1][-1]].append(slice)

    ##Train Model
    for a in range(20):     
        selected_slices = []
        for b in range(len(segdict)):
            if len(bucket_list[b])>0:
                selected_slices.append([choice(bucket_list[b]) for c in range(balance_number)])
        shuffle(selected_slices)

        model_input = []
        for bucket in selected_slices:
            model_input= model_input+[val for sublist in bucket for val in sublist]
        X_train = array([array(item[0]) for item in model_input])
        X_train = array(array([X_train[x:x+batch_size] for x in range(0, len(X_train), batch_size)]))
        X_train = sequence.pad_sequences(X_train, maxlen=batch_size)
        X_train = reshape(X_train, (len(X_train), len(X_train[0]), len(X_train[0][0])))

        Y_train_values = [[1 if i == item[1] else e for i, e in enumerate(zeros(len(segdict)))] for item in model_input]

        Y_train = array([Y_train_values[x:x+batch_size] for x in range(0, len(Y_train_values), batch_size)])
        Y_train = sequence.pad_sequences(Y_train, maxlen=batch_size)
        
        hist = model.fit(X_train, Y_train, batch_size=1, nb_epoch=20, validation_split=0.2, verbose=1)              
        
        ##Generate Confusion Matrix at end of each training step
        predictions = model.predict(X_train, batch_size=1)
        ##cmatrix = confusion_matrix(predictions, Y_train)
        
        
        ##Predict Output    
        sorted_dict = sorted(segdict.items(), key=lambda x:x[1])
        predicted_output=[]
        for item in predictions:
            for segment in item:
                segment_list = list(segment)
                max_index = segment_list.index(max(segment_list))
                if max_index != 0:
                    predicted_output.append(sorted_dict[max_index-1][0])
                else:
                    predicted_output.append(0)
        print predicted_output
        
        
    model.save("./SpeechRecognitionModel.h5")
=======
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
>>>>>>> 259ca6bc7915ef698c7cd2b0c82eff7dd92902de
