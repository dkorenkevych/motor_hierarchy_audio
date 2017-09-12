import scipy.io.wavfile as scw
from os import listdir
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras import regularizers as rgl
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg


import matplotlib
matplotlib.use('TkAgg')
from Tkinter import *
import numpy as np

import matplotlib.pyplot as plt


def create_sine_data():
    freqs = [0.4/(2*np.sqrt(2))**3, 0.4/(2*np.sqrt(2))**2, 0.4/(2*np.sqrt(2)), 0.4]
    #data_size = 10000
    time_steps = 10000
    data = np.zeros((3, time_steps, 2))
    data[0, :, 0] = np.sin(freqs[2] * np.arange(0, time_steps))
    data[0, :, 1] = 0
    data[1, :, 0] = 0
    data[1, :, 1] = np.sin(freqs[2] * np.arange(150, time_steps + 150))
    data[2, :, 0] = np.sin(freqs[2] * np.arange(0, time_steps))
    data[2, :, 1] = np.sin(freqs[2] * np.arange(150, time_steps + 150))

    # for i in range(len(freqs)):
    #     data[i, :, 0] = np.sin(freqs[i]*np.arange(50*i, time_steps + 50*i))
    #     data[i, :, 1] = 0
    # data *= 0.8
    # #data += 0.1
    # plt.figure()
    # plt.plot(data[0, :200, :])
    # plt.show()
    return data



def reshaper(x):
    x = x[:, -1, :]
    return x


def slicer_output_shape(input_shape):
    shape = list(input_shape)
    shape = [shape[0], shape[-1]]
    return tuple(shape)

def build_model():
    temporal_seq_length1 = 4
    temporal_seq_length2 = 5
    temporal_seq_length3 = 6
    temp_data_length = 6000
    encoded_dim = 10
    num_filters = 8
    weight_reg = 1e-8

    input = Input((120, 50))
    decoder_input = Input((encoded_dim,))
    layer2_input = Input((5, 50))

    layer = TimeDistributed(Dense(output_dim=200, activation='relu', bias_regularizer=rgl.l2(weight_reg),
                                  kernel_regularizer=rgl.l2(weight_reg)))(input)
    layer = LSTM(output_dim=200, return_sequences=True, activation='tanh',
                 bias_regularizer=rgl.l2(weight_reg),
                 kernel_regularizer=rgl.l2(weight_reg),
                 recurrent_regularizer=rgl.l2(weight_reg)
                 )(layer)
    layer = LSTM(output_dim=200, return_sequences=True, activation='tanh',
                 bias_regularizer=rgl.l2(weight_reg),
                 kernel_regularizer=rgl.l2(weight_reg), recurrent_regularizer=rgl.l2(weight_reg)
                 )(layer)
    layer = LSTM(output_dim=encoded_dim, return_sequences=False, activation='tanh', activity_regularizer=rgl.l1(1e-8),
                 bias_regularizer=rgl.l2(weight_reg),
                 kernel_regularizer=rgl.l2(weight_reg),
                 recurrent_regularizer=rgl.l2(weight_reg)
                 )(layer)

    shared_layer1 = LSTM(output_dim=200, return_sequences=True, activation='tanh', bias_regularizer=rgl.l2(weight_reg),
                         kernel_regularizer=rgl.l2(weight_reg), recurrent_regularizer=rgl.l2(weight_reg))

    shared_layer2 = TimeDistributed(
        LSTM(output_dim=200, return_sequences=True, activation='tanh', bias_regularizer=rgl.l2(weight_reg),
             kernel_regularizer=rgl.l2(weight_reg), recurrent_regularizer=rgl.l2(weight_reg)))

    shared_layer3 = TimeDistributed(TimeDistributed(
        LSTM(output_dim=200, return_sequences=True, activation='tanh', bias_regularizer=rgl.l2(weight_reg),
             kernel_regularizer=rgl.l2(weight_reg), recurrent_regularizer=rgl.l2(weight_reg))))

    shared_layer4 = TimeDistributed(Dense(output_dim=50, activation='linear', bias_regularizer=rgl.l2(weight_reg),
                                          kernel_regularizer=rgl.l2(weight_reg)))

    encoded = layer
    representation = RepeatVector(temporal_seq_length1)(layer)
    representation = shared_layer1(representation)
    layer1_output = representation
    representation = TimeDistributed(RepeatVector(temporal_seq_length2))(representation)
    representation = shared_layer2(representation)
    layer2_output = representation
    representation = TimeDistributed(TimeDistributed(RepeatVector(temporal_seq_length3)))(representation)
    representation = shared_layer3(representation)
    representation = Reshape((temporal_seq_length1 * temporal_seq_length2 * temporal_seq_length3, 200))(representation)
    output = shared_layer4(representation)
    model = Model(input=[input], output=[output])
    decoder = RepeatVector(temporal_seq_length1)(decoder_input)
    decoder = shared_layer1(decoder)
    decoder = TimeDistributed(RepeatVector(temporal_seq_length2))(decoder)
    decoder = shared_layer2(decoder)
    decoder = TimeDistributed(TimeDistributed(RepeatVector(temporal_seq_length3)))(decoder)
    decoder = shared_layer3(decoder)
    decoder = Reshape((temporal_seq_length1 * temporal_seq_length2 * temporal_seq_length3, 200))(decoder)
    decoder = shared_layer4(decoder)

    decoder_model = Model(input=[decoder_input], output=decoder)
    encoder_model = Model(input=[input], output=[encoded])
    model.compile(optimizer=Adam(lr=1e-3, clipnorm=1.0), loss='mse')

    for i in range(3700):
        r = np.random.randint(20)
        r = 0
        train_data = data
        noisy_data = train_data + np.random.normal(scale=0.005, size=train_data.shape)
        # plt.figure()
        # plt.plot(noisy_data[:, :, 0].T)
        # plt.show()
        result = model.train_on_batch(noisy_data.reshape((3, 120, 50)), train_data.reshape(3, 120, 50))
        print i, result

    # model.load_weights("model_words1.json")
    model.save_weights("model_words.json", overwrite=True)
    decoder_model.save_weights("decoder_words.json", overwrite=True)
    return model, encoder_model, decoder_model

if __name__ == "__main__":
    #data = create_sine_data()
    data1 = np.expand_dims(scw.read("monday.wav")[1][:6000, :1], axis=0)
    data2 = np.expand_dims(scw.read("tuesday.wav")[1][:6000, :1], axis=0)
    data3 = np.expand_dims(scw.read("wednesday.wav")[1][:6000, :1], axis=0)
    data = np.concatenate([data1, data2, data3], axis=0)
    np.random.seed(458965894)

    model, encoder_model, decoder_model = build_model()
    colors = ['r', 'g', 'b', 'y']
    plt.figure(figsize=(20, 12))
    for i in range(3):
        encoded = data[i:i+1, :].reshape((1, 120, 50))
        #print "encoded", encoded
        encoded = encoder_model.predict(encoded)
        print "encoded", encoded
        predictions = decoder_model.predict(encoded).reshape((1, 6000, 1))
        scw.write("predicted_day_sound%s.wav"%(i), 8000, predictions[0, :, 0])
        #layer1_prediction = layer1_model.predict(encoded)
        #layer2_prediction = layer2_model.predict(layer1_prediction)
        #print "error", np.mean((data[i:i+1, :temp_data_length, :] - predictions[0, :])**2)
        plt.subplot(4, 2, 2*i+1)
        plt.plot(encoded[0, :])
        plt.ylim([-1, 1])
        plt.subplot(4, 2, 2*i+2)
        plt.plot(predictions[0, :])
    plt.tight_layout()
    plt.show()
