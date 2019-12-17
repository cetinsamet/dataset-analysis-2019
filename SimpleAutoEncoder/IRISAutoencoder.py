
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import torch.nn as nn
import os


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
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data  # we only take the first two features.
data_label = iris.target
print(data[:5])
print("Data size : ", data.shape)

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

print("Datasize : ", data.shape)
data = torch.from_numpy(data).cuda()

DATA_SIZE = 150
LEARNING_RATE = 0.00001
BATCH_SIZE = 30

autoencoder = MLPAutoEncoder(NN_SIZE=2048).cuda()
if os.path.exists("Model/IRISAutoEncoder.pt"):
    autoencoder.loadModel("Model/IRISAutoEncoder.pt")
    autoencoder.train()


optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
SCHEDULE_GAMMA = 0.99
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = SCHEDULE_GAMMA)
loss_function = nn.MSELoss()

iter_count = 100
global_iteration = 0
while iter_count != 0:
    # for logging required metrics...
    for iteration in range(iter_count):
        global_iteration += 1
        total_loss = 0

        for batch_index in range(0, DATA_SIZE//BATCH_SIZE):
            train_data = data[batch_index*BATCH_SIZE: (batch_index+1)*BATCH_SIZE]
            predicted = autoencoder(train_data)

            optimizer.zero_grad()
            loss = loss_function(train_data, predicted)
            total_loss += loss.item()*BATCH_SIZE
            loss.backward()
            optimizer.step()

        print("Iteration : %d - Loss : %f" %(global_iteration, total_loss/DATA_SIZE))
        # scheduler.step()
    while True:
        try:
            # when iteration finishes, ask user for new iteration number
            # when 0, training stops...
            iter_count = int(input("New Iteration Count : "))
        except:
            print("Input Type Error")
            continue
        break;
autoencoder.saveModel("Model/IRISAutoEncoder.pt")