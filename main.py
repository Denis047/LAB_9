import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('wine_data.csv')
X1 = data[["Alcohol", "Malic acid"]].values


Square_length = []


for n in range(1, 10):
    algorithm = KMeans(
        n_clusters=n,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=0.001,
        random_state=111,
        algorithm='elkan'
    )
    algorithm.fit(X1)
    Square_length.append(algorithm.inertia_ / 100000)


plt.plot(np.arange(1, 10), Square_length, 'o')
plt.plot(np.arange(1, 10), Square_length, '-', alpha=0.4)
plt.grid(True)
plt.xlabel('Кількість кластерів')
plt.ylabel('Значення інерції')
plt.show()


algorithm = KMeans(
    n_clusters=5,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=111,
    algorithm='elkan'
)
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_


plt.scatter(x='Alcohol', y='Malic acid', data=data, c=labels1, s=50)
plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=200, c='red', alpha=0.75)
plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=20, c='orange', alpha=1)
plt.ylabel('Малинова кислота')
plt.xlabel('Спирт')
plt.show()
