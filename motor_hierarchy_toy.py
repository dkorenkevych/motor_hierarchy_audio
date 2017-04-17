from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers as rgl
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from Tkinter import *
import numpy as np




class DB_Gui:
    def __init__(self, root, temporal_seq_length, encoded_dim):
        self.temporal_seq_length = temporal_seq_length
        self.encoded_dim = encoded_dim
        self.master = Frame(root)
        self.initialize()
        self.root = root

    def initialize(self):
        self.master.pack()
        left_side = Frame(self.master)
        left_side.grid(row=0, column=0, sticky=S+N+W+E)
        self.scales = []
        for i in range(4):
            w = Scale(left_side, from_=0, to=100, resolution=-1, command=self.updateValue)
            w.pack()
            w.set(50)
            self.scales.append(w)
        right_side = Frame(self.master)
        right_side.grid(row=0, column=1, sticky=S + N + W + E)
        self.f = Figure(figsize=(5, 4), dpi=100)
        self.plot_area = FigureCanvasTkAgg(self.f, right_side)
        self.plot_area.get_tk_widget().grid(column=0, row=0, sticky=W + E + N + S)
        tool_frame = Frame(right_side)
        toolbar = NavigationToolbar2TkAgg(self.plot_area, tool_frame)
        toolbar.update()
        tool_frame.grid(column=0, row=1, sticky=W)
        self.create_model()
        self.update_plot()

    def create_model(self):

        input = Input((self.temporal_seq_length, 2))
        decoder_input = Input((self.encoded_dim,))
        layer = TimeDistributed(Dense(units=500, activation='relu'))(input)
        layer = LSTM(output_dim=self.encoded_dim, return_sequences=False, activation='tanh',
                     activity_regularizer=rgl.l1(5e-4))(layer)
        shared_layer1 = LSTM(output_dim=500, return_sequences=True, activation='tanh', bias_regularizer=rgl.l2(1e-5),
                             kernel_regularizer=rgl.l2(1e-5), recurrent_regularizer=rgl.l2(1e-5))
        shared_layer2 = TimeDistributed(Dense(units=2, activation='linear', bias_regularizer=rgl.l2(1e-5),
                                              kernel_regularizer=rgl.l2(1e-5)))
        encoded = layer
        representation = RepeatVector(self.temporal_seq_length)(layer)
        representation = shared_layer1(representation)
        output = shared_layer2(representation)
        model = Model(input=[input], output=[output])
        decoder = RepeatVector(self.temporal_seq_length)(decoder_input)
        decoder = shared_layer1(decoder)
        decoder = shared_layer2(decoder)
        decoder_model = Model(input=[decoder_input], output=decoder)
        encoder_model = Model(input=[input], output=[encoded])
        model.compile(optimizer=Adam(decay=1e-5), loss='mse')
        self.model = model
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.decoder.load_weights("decoder.json")

    def update_plot(self):
        data = np.zeros((1, 4))
        for i in range(4):
            data[0, i] = self.scales[i].get()/50.0 - 1
        #print data
        output = self.decoder.predict(data)
        self.f.clear()
        self.a = self.f.add_subplot(111)
        self.a.plot(output[0, :, 0])
        self.plot_area.show()

    def updateValue(self, event):
        self.update_plot()

if __name__ == "__main__":
    temporal_seq_length = 50
    encoded_dim = 4
    root = Tk()
    root.title('GYM results database')
    gui = DB_Gui(root, temporal_seq_length, encoded_dim)
    root.mainloop()
