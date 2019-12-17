import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd


class MLPAutoEncoder(nn.Module):
    def __init__(self, NN_SIZE = 512):
        super(MLPAutoEncoder, self).__init__()
        self.layer_1 = nn.Linear(4, NN_SIZE)
        self.layer_1_out = nn.LeakyReLU()

        self.encoder = nn.Linear(NN_SIZE, 2)
        self.encoder_out = nn.LeakyReLU()

        self.decoder = nn.Linear(2, NN_SIZE)
        self.decoder_out = nn.LeakyReLU()
        self.output_layer = nn.Linear(NN_SIZE, 4)
        self.output_out = nn.Tanh()

    def autoEncoderOutput(self, input):
        layer_1 = self.layer_1(input)
        layer_1_out = self.layer_1_out(layer_1)

        encoder = self.encoder(layer_1_out)
        encoder_out = self.encoder_out(encoder)
        return encoder_out


    def forward(self, input):
        layer_1 = self.layer_1(input)
        layer_1_out = self.layer_1_out(layer_1)

        encoder = self.encoder(layer_1_out)
        encoder_out = self.encoder_out(encoder)

        decoder = self.decoder(encoder_out)
        decoder_out = self.decoder_out(decoder)
        output = self.output_layer(decoder_out)
        # output_out = self.output_out(output)

        return output

    def saveModel(self, PATH):
        torch.save(self.state_dict(), PATH)

    def loadModel(self, PATH):
        self.load_state_dict(torch.load(PATH))

        self.eval()

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data  # we only take the first two features.
data_label = iris.target
print(data[:5])
print("Data size : ", data.shape)

number = 26
print(mcd.CSS4_COLORS)
colors = ["red", "blue", "green"]
print("target : ", data_label)

data_point_color = []
for target in data_label:
    data_point_color.append(colors[target])



# normalize and scale data...
data = np.array(data, dtype=np.float32)
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data_min = np.min(data, axis=0)
data_max = np.max(data, axis=0)

# standartization...
# data = (data-data_mean)/data_std
# min max
# data = (data-data_min)/(data_max-data_min)
# mean normalization
data = (data-data_mean)/(data_max-data_min)

data = torch.from_numpy(data).cuda()

autoencoder = MLPAutoEncoder(NN_SIZE=2048).cuda()
if os.path.exists("Model/IRISAutoEncoder.pt"):
    autoencoder.loadModel("Model/IRISAutoEncoder.pt")
    autoencoder.train()
else:
    print("No model is found...")
    exit()

DATA_SIZE = len(data)
BATCH_SIZE = 30
loss_function = nn.MSELoss()


color_index = 0
for batch_index in range(0, DATA_SIZE//BATCH_SIZE):
           train_data = data[batch_index*BATCH_SIZE: (batch_index+1)*BATCH_SIZE]
           predicted = autoencoder(train_data)
           encoder_result = autoencoder.autoEncoderOutput(train_data).detach().cpu().numpy()
           for point in encoder_result:

               plt.scatter(point[0], point[1], color=data_point_color[color_index])
               color_index += 1


plt.show()
