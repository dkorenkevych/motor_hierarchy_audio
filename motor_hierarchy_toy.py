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
import time
from threading import Thread


class DB_Gui:
    def __init__(self, root, temporal_seq_length, encoded_dim):
        self.temporal_seq_length = temporal_seq_length
        self.encoded_dim = encoded_dim
        self.sequence_so_far2 = [0]
        self.sequence_so_far1 = [0]
        self.master = Frame(root)
        self.initialize()
        self.root = root


    def initialize(self):
        self.master.pack()
        left_side = Frame(self.master)
        left_side.grid(row=0, column=0, sticky=S+N+W+E)
        self.scales = []
        for i in range(2):
            w = Scale(left_side, from_=0, to=100, resolution=-1, command=self.updateValue)
            w.pack()
            w.set(0)
            self.scales.append(w)
        self.scales[0].set(100)
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
        #self.drawing_thread = Thread(target=self.draw_curve)
        #self.drawing_thread.start()
        self.draw_curve()
        #self.update_plot()

    def create_model(self):

        weight_reg = 1e-7
        temporal_seq_length1 = 100
        temporal_seq_length2 = 8
        temporal_seq_length3 = 10
        temp_data_length = 100
        input = Input((temp_data_length, 2))
        decoder_input = Input(batch_shape=(1, 1, encoded_dim))


        shared_layer1 = LSTM(output_dim=50, return_sequences=True, stateful=True, activation='tanh',
                             bias_regularizer=rgl.l2(weight_reg),
                             kernel_regularizer=rgl.l2(weight_reg), recurrent_regularizer=rgl.l2(weight_reg))

        shared_layer2 = TimeDistributed(Dense(output_dim=2, activation='linear', bias_regularizer=rgl.l2(weight_reg),
                                              kernel_regularizer=rgl.l2(weight_reg)))

        shared_layer3 = LSTM(output_dim=100, return_sequences=True, stateful=True, activation='tanh',
                             bias_regularizer=rgl.l2(weight_reg),
                             kernel_regularizer=rgl.l2(weight_reg), recurrent_regularizer=rgl.l2(weight_reg))

        shared_layer4 = LSTM(output_dim=100, return_sequences=True, activation='tanh', bias_regularizer=rgl.l2(weight_reg),
                             kernel_regularizer=rgl.l2(weight_reg), recurrent_regularizer=rgl.l2(weight_reg))



        #decoder = RepeatVector(temporal_seq_length1)(decoder_input)
        decoder = shared_layer1(decoder_input)
        # decoder = TimeDistributed(RepeatVector(temporal_seq_length2))(decoder)
        # decoder = Reshape((temporal_seq_length2, 500))(decoder)
        decoder = shared_layer3(decoder)
        # decoder = TimeDistributed(TimeDistributed(RepeatVector(temporal_seq_length3)))(decoder)
        # decoder = shared_layer4(decoder)
        # decoder = Reshape((temporal_seq_length1 * temporal_seq_length2 * temporal_seq_length3, 500))(decoder)
        decoder = shared_layer2(decoder)
        self.decoder = Model(input=[decoder_input], output=decoder)



        self.decoder.load_weights("decoder.json")

    def update_plot(self):
        data = np.zeros((1, 2))
        for i in range(2):
            data[0, i] = self.scales[i].get()/50.0 - 1
        #print data
        output = self.decoder.predict(data.reshape(1, 1, 2))
        self.sequence_so_far1.append(output[0, 0, 0])
        self.sequence_so_far2.append(output[0, 0, 1])
        # self.f.clear()
        # self.a = self.f.add_subplot(111)
        # self.a.plot(output[0, :, 0])
        # self.plot_area.show()

    def updateValue(self, event):
        self.update_plot()

    def draw_curve(self):
        self.update_plot()
        self.f.clear()
        self.a = self.f.add_subplot(111)
        self.a.plot(self.sequence_so_far1)
        self.a.plot(self.sequence_so_far2)
        self.plot_area.show()
        #time.sleep(0.2)
        root.after(200, lambda: self.draw_curve())


if __name__ == "__main__":
    temporal_seq_length = 50
    encoded_dim = 2
    root = Tk()
    root.title('GYM results database')
    gui = DB_Gui(root, temporal_seq_length, encoded_dim)
    root.mainloop()
