
import numpy as np
import random
import numpy as np
import pandas as pd
from emd import *
import time
import matplotlib.pyplot as plt
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import plot_model

smooth = 1.


def load_data(filepath, seq_len, normalise_window):
    dataset = pd.read_csv(filepath)
    data = np.array(dataset['#Passengers'])
    data = emd(data)
  
    nb_modes = data.shape[0]
    
    X_train = []
    Y_train = []
    X_test  = []
    Y_test  = []
    
 
    
    for mode in range(nb_modes):
        
    
        sequence_length = seq_len + 1
        result = []
    
        for index in range(len(data[mode]) - sequence_length):
            result.append(data[mode, index: index + sequence_length])
        
        result = np.array(result)
  
        
        if normalise_window:
            result = normalise_windows(result)

        
   
        
        result = np.array(result)

        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]
  
        #np.random.shuffle(train)
    
        x_train = train[:, :-1]
        y_train = train[:, -1]
    
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]
    
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)
    
    return [np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test), nb_modes]



def build_model(layers):
    model = Sequential()
    
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def normalise_windows(window_data):
    normalised_data = []
    token = 0
    for window in window_data:
            if window[0]!=0:
         
                normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
                normalised_data.append(normalised_window)
            else:
                normalised_window = [((float(p) / float(window[1])) - 1) for p in window]
                normalised_data.append(normalised_window)
                
    return normalised_data



def predict_point_by_point(models, data, normalize = False):

	prediction = np.zeros(data.shape[1])

	for mode in range(len(models)):
		model = models[mode]

		iterator = 0

		if normalize:
			x_input = np.reshape(data[mode], (data.shape[1], data.shape[2]))
			x_normalized = np.array(normalise_windows(x_input))
			x_normalized = np.reshape(x_normalized, (x_normalized.shape[0], x_normalized.shape[1], 1))
			predicted = model.predict(x_normalized)
			predicted = np.reshape(predicted, (predicted.size,))
			for window in data[mode]:
			    normalizer= window[0]
			    
			    predicted[iterator] = (predicted[iterator]+1)*normalizer 
			    iterator = iterator + 1

		else:
			predicted = model.predict(data)

		prediction += predicted


	return prediction
