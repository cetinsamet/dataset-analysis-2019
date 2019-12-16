import numpy as np
import matplotlib._color_data as mcd
import pandas as pd
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot
from numpy import array


data_file = open("datasets/letter-recognition-visual.data" , "r")
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
data = (data-data_mean)/data_std

label_to_number = {}
for i, a in enumerate(set(data_label)):
    label_to_number[a] = i

for i in range(len(data_label)):
    data_label[i] = label_to_number[data_label[i]]

#umap
um = umap.UMAP()
dataIn2d = um.fit_transform(data)
print(dataIn2d)

# plot umap
fig, ax = pyplot.subplots()
groups = pd.DataFrame(dataIn2d, columns=['x', 'y']).assign(category=data_label).groupby('category')
colors = [mcd.CSS4_COLORS[list(mcd.CSS4_COLORS.keys())[5*i]] for i in range(26)]
for name, points in groups:
    ax.scatter(points.x, points.y, label=name, color=colors[name])

ax.legend()
pyplot.gca().set_aspect('equal', 'datalim')
pyplot.title('UMAP Projection', fontsize=24)
pyplot.show()

print("Done")
