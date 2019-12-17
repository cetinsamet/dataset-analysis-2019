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
        self.layer_1 = nn.Linear(16, NN_SIZE)
        self.layer_1_out = nn.LeakyReLU()

        self.encoder = nn.Linear(NN_SIZE, 2)
        self.encoder_out = nn.LeakyReLU()

        self.decoder = nn.Linear(2, NN_SIZE)
        self.decoder_out = nn.LeakyReLU()
        self.output_layer = nn.Linear(NN_SIZE, 16)
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

data_file = open("datasets/letter-recognition.data" , "r")
data = []
data_label = []
for line in data_file:
    if line == '\n':
        continue
    parts = line.split(",")
    data_label.append(parts[0])
    data.append([float(a) for a in parts[1:]])
data_file.close()

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

# converting letter labels to numbers
label_to_number = {}
for i, a in enumerate(set(data_label)):
    label_to_number[a] = i

for i in range(len(data_label)):
    data_label[i] = label_to_number[data_label[i]]
number = 26
colors = [mcd.CSS4_COLORS[list(mcd.CSS4_COLORS.keys())[5*i]] for i in range(number)]

data_point_color = []
for target in data_label:
    data_point_color.append(colors[target])


data = torch.from_numpy(data).cuda()


autoencoder = MLPAutoEncoder(NN_SIZE=2048).cuda()
if os.path.exists("Model/AutoEncoder.pt"):
    autoencoder.loadModel("Model/AutoEncoder.pt")
    autoencoder.train()
else:
    print("No model is found...")
    exit()

DATA_SIZE = len(data)
BATCH_SIZE = 5000
loss_function = nn.MSELoss()

x = []
y= []
color_index = 0
for batch_index in range(0, DATA_SIZE//BATCH_SIZE):
           train_data = data[batch_index*BATCH_SIZE: (batch_index+1)*BATCH_SIZE]
           predicted = autoencoder(train_data)
           encoder_result = autoencoder.autoEncoderOutput(train_data).detach().cpu().numpy()
           for point in encoder_result:

               # plt.scatter(point[0], point[1], color=data_point_color[color_index])
               x.append(point[0])
               y.append(point[1])
               color_index += 1

           print("BATCH Index : ", batch_index)

plt.scatter(x, y, color=data_point_color)
plt.show()
