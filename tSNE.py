import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import pandas as pd
number = 26
print(mcd.CSS4_COLORS)
# colors to be used for drawing...
colors = [mcd.CSS4_COLORS[list(mcd.CSS4_COLORS.keys())[5*i]] for i in range(number)]
# load data...
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

# normalize and scale data...
data = np.array(data, dtype=np.float32)
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data = (data-data_mean)/data_std

# converting letter labels to numbers
label_to_number = {}
for i, a in enumerate(set(data_label)):
    label_to_number[a] = i

for i in range(len(data_label)):
    data_label[i] = label_to_number[data_label[i]]

print("Data shape : ", data.shape)
# for perplex values [10, 20, 30, 40, 50], run tsne...
for perplex in range(10, 51, 10):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplex, n_iter=500)
    Y = tsne.fit_transform(data)
    fig, ax = plt.subplots()

    groups = pd.DataFrame(Y, columns=['x', 'y']).assign(category=data_label).groupby('category')
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, color=colors[name])

    ax.legend()
    plt.title("tSNE on Letter Recognition Dataset- Perplexity : "+str(perplex))
    plt.show()
print("Okay...")
