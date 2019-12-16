import numpy as np
import matplotlib._color_data as mcd
import pandas as pd
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot
from numpy import array
from sklearn.cluster import DBSCAN

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

#umap
um = umap.UMAP()
dataIn2d = um.fit_transform(data)

# cluster the data into five clusters
dbscan = DBSCAN()
dataDF = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})
model = dbscan.fit(dataDF)
print(model.labels_)
pyplot.scatter(dataIn2d[:, 0], dataIn2d[:, 1], c=model.labels_)
pyplot.gca().set_aspect('equal', 'datalim')
pyplot.title('UMAP Projection with DBScan', fontsize=24)
pyplot.show()



print("Done")
