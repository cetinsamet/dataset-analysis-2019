import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import torch.nn.functional as F


N_EPOCH         = 5
SEED            = 123
DATAPATH 		= "../datasets/letter-recognition.data"
SMALLDATAPATH 	= "../datasets/letter-recognition-visual.data"
COLORS          = [mcd.CSS4_COLORS[list(mcd.CSS4_COLORS.keys())[5 * i]] for i in range(26)]
K               = 2
MODELNAME       = "clf_" + str(K) + ".pt"

def load_data(filename):
	x,y  = [], []
	with open(filename, 'r') as infile:
		for line in infile:
			line_split = line[:-1].split(',')
			y.append(line_split[0])
			x.append(list(map(int, line_split[1:])))
	return np.asarray(x), np.asarray(y)

class Network(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.layer1     = nn.Linear(d_in, K)
        self.layer2     = nn.Linear(K, d_out)
        self.relu       = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(K)

    def forward(self, x):
        a = self.batch_norm(self.relu(self.layer1(x)))
        x = F.sigmoid(self.layer2(a))
        return x, a

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    x, y = load_data(DATAPATH)

    x = normalize(x, norm='l2')
    #x = scale(x)
    x = torch.from_numpy(x).float()

    y = np.asarray([ord(l) - 65 for l in y])
    y = torch.from_numpy(y).long()

    n_sample, d_feat = x.shape
    n_label = len(np.unique(y))

    clf         = Network(d_feat, d_feat)
    #ce_loss     = nn.CrossEntropyLoss(reduction='sum')
    mse_loss    = nn.MSELoss(reduction='sum')
    optimizer   = torch.optim.Adam(clf.parameters(), lr=1e-2)
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    data        = TensorDataset(x, y)
    data_loader = DataLoader(data, batch_size=32, shuffle=True, drop_last=False)

    for epoch_idx in range(N_EPOCH):

        clf.train()
        running_loss = 0.

        for feats, _ in data_loader:

            feats_ = clf(feats)[0]

            batch_loss = mse_loss(feats_, feats)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

        lr_schedular.step(running_loss)

        epoch_loss = running_loss / n_sample

        print("Epoch %4d\tLoss : %s" % (epoch_idx + 1, epoch_loss))


    x_transformed = clf(x)[1].detach().numpy()
    cols = np.asarray(COLORS)[y.detach().numpy()]

    if K == 2:
        plt.scatter(x_transformed[:, 0], x_transformed[:, 1], s=[2 for _ in range(20000)], c=cols)
        plt.show()
    elif K == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_transformed[:, 0], x_transformed[:, 1], x_transformed[:, 2], s=[2 for _ in range(20000)],
                   c=cols, marker='.')

        ax.azim = -100
        ax.elev = 20
        plt.show()
    else:
        raise NotImplementedError




if __name__ == '__main__':
    main()