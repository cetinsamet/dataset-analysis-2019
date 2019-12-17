import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from mpl_toolkits.mplot3d import Axes3D


N_EPOCH         = 200
SEED            = 123
DATAPATH 		= "./datasets/letter-recognition.data"
SMALLDATAPATH 	= "./datasets/letter-recognition-visual.data"
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
        x_transformed   = self.batch_norm(self.relu(self.layer1(x)))
        out             = self.layer2(x_transformed)
        return out, x_transformed

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    x, y = load_data(DATAPATH)

    x = normalize(x, norm='l2')
    x = torch.from_numpy(x).float()

    y = np.asarray([ord(l) - 65 for l in y])
    y = torch.from_numpy(y).long()
    '''
    n_sample, d_feat = x.shape
    n_label = len(np.unique(y))

    clf         = Network(d_feat, n_label)
    ce_loss     = nn.CrossEntropyLoss(reduction='sum')
    optimizer   = torch.optim.Adam(clf.parameters(), lr=1e-2)
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    data        = TensorDataset(x, y)
    data_loader = DataLoader(data, batch_size=64, shuffle=True, drop_last=False)



    for epoch_idx in range(N_EPOCH):

        clf.train()
        running_loss = 0.

        for feats, labels in data_loader:

            labels_ = clf(feats)[0]
            batch_loss = ce_loss(labels_, labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

        lr_schedular.step(running_loss)

        epoch_loss = running_loss / n_sample

        print("Epoch %4d\tLoss : %s" % (epoch_idx + 1, epoch_loss))

        if (epoch_idx+1) % 10 == 0:
            clf.eval()
            y_pred = torch.argmax(clf(x)[0], dim=1)
            print("Acc: %.2f" % (float(torch.sum(y_pred == y).item()) / n_sample * 100))

    torch.save(clf, MODELNAME)
    '''
    clf = torch.load(MODELNAME)
    clf.eval()


    x_transformed = clf(x)[1].detach().numpy()
    cols = np.asarray(COLORS)[y.detach().numpy()]
    if K == 2:
        plt.scatter(x_transformed[:, 0], x_transformed[:, 1], s=[2 for _ in range(20000)], c=cols)
        plt.show()
    elif K == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x_transformed[:, 0], x_transformed[:, 1], x_transformed[:, 2], s=[2 for _ in range(20000)], c=cols, marker='.')
        plt.show()
    else:
        raise NotImplementedError



if __name__ == '__main__':
    main()