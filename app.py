
from flask import Flask,render_template, request,json
import numpy as np
import random
import json
import h5py

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
    data = emd(data)[0,:]
    
    sequence_length = seq_len + 1
    result = []
    
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
        
    if normalise_window:
        result = normalise_windows(result)
        
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    model = Sequential()
    
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[3],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[4]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data




app = Flask(__name__)
 


@app.route("/")
def output():
	return render_template('index.html')


@app.route('/uploadImage', methods=['POST'])
def uploadImage():

    file = request.files['image']

    file.save('static/data/Input.csv')

    epochs  = 30
    seq_len = 5

    print('> Loading data... ')
    X_train, y_train, X_test, y_test = load_data('static/data/Input.csv', seq_len, False)
    print('> Data Loaded. Compiling...')

    model = build_model([1, 50, 100, 100, 1])

    model.fit(
        X_train,
        y_train,
        batch_size=5,
        nb_epoch=epochs,
        validation_split=0.05)
    predicted = predict_point_by_point(model, X_train)
    plt.plot(y_train)
    plt.plot(predicted)
    plt.legend(['Actual','Predicted'])
    plt.savefig('static/images/OutputGraph.png')

    return render_template('index2.html')

 
if __name__ == "__main__":
    app.run("")