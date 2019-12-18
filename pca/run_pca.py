import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


N_CLASS         = 26
N_SAMPLE        = 20000
D_FEATURES      = 16
SEED            = 123
DATAPATH 		= "../datasets/letter-recognition.data"
SMALLDATAPATH 	= "../datasets/letter-recognition-visual.data"
COLORS          = [mcd.CSS4_COLORS[list(mcd.CSS4_COLORS.keys())[5*i]] for i in range(N_CLASS)]
K               = 3


def load_data(filename):
    x,y  = [], []
    with open(filename, 'r') as infile:
        for line in infile:
            line_split = line[:-1].split(',')
            y.append(line_split[0])
            x.append(list(map(int, line_split[1:])))

    x, y = np.asarray(x), np.asarray(y)
    return x, y

def main(args=None):

    phase = "PCA"

    random.seed(SEED)
    np.random.seed(SEED)

    x, y = load_data(DATAPATH)
    y = np.asarray([ord(l) - 65 for l in y])

    # train data will be used for fitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=SEED)

    # MODELPATH = "./model/pca_" + str(K) + "D.pt"
    PLOTPATH = "./plot/pca_" + str(K) + "D.png"

    pca = PCA(n_components=K)
    pca.fit(x_train)  # <- train data is used for fitting

    # joblib.dump(pca, MODELPATH)      # <- save isomap model
    # pca = joblib.load(MODELPATH)     # <- load isomap model

    x_transformed = pca.transform(x)

    c = np.asarray(COLORS)[y]                       # <- define corresponding colors
    s = np.asarray([2 for _ in range(N_SAMPLE)])    # <- define corresponding data point sizes

    if K == 2:      # number of components = 2 (plot 2D)
        for i in range(N_CLASS):
            indices = np.asarray([idx for idx, y_ in enumerate(y) if y_==i])
            plt.scatter(x_transformed[indices, 0], x_transformed[indices, 1],
                        label= (chr(i + 65)),
                        s=s[indices],
                        c=c[i])

    elif K == 3:    # number of components = 3 (plot 3D)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(N_CLASS):
            indices = np.asarray([idx for idx, y_ in enumerate(y) if y_ == i])
            ax.scatter(x_transformed[indices, 0], x_transformed[indices, 1], x_transformed[indices, 2],
                       label= (chr(i + 65)),
                       s=s[indices],
                       c=c[i],
                       marker='.')
    else:
        raise NotImplementedError

    plt.legend(title="Classes", scatterpoints=1, loc='best',ncol=4, fontsize=8, markerscale=3)
    plt.title(phase)
    plt.savefig(PLOTPATH)
    plt.show()

if __name__ == '__main__':
    main()