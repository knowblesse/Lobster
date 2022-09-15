"""
WholeSessionUnitPCA
@2022 Knowblesse
PCA analysis using WholeSessionUnitData
"""

import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def wholeSessionUnitDataPCA(
        Tank_Path = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1\#20JUN1-200827-171419_PL'),
        n_cluster = 2,
        drawFigure = True
        ):
    Session_Name = re.search('#.*', str(Tank_Path))[0]
    wholeSessionUnitData_location = [p for p in Tank_Path.glob('*_wholeSessionUnitData.csv')]
    neural_data = np.loadtxt(str(wholeSessionUnitData_location[0]), delimiter=',')

    # Clip neural data
    np.clip(neural_data, -3, 3, out=neural_data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(neural_data)

    kmeans = KMeans(n_clusters=n_cluster)
    kmeans_data = kmeans.fit_predict(pca_data)


    if drawFigure:
        plt.clf()
        plt.scatter(pca_data[kmeans_data == 0, 0], pca_data[kmeans_data == 0, 1], c='r', s=5)
        plt.scatter(pca_data[kmeans_data == 1, 0], pca_data[kmeans_data == 1, 1], c='g', s=5)
        if n_cluster == 3:
            plt.scatter(pca_data[kmeans_data == 2, 0], pca_data[kmeans_data == 2, 1], c='b', s=5)
        plt.title(Session_Name)

    return {'pca_data' : pca_data, 'kmeans_data' : kmeans_data}

