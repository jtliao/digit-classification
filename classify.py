import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
import numpy as np

num_seeds = 3
num_digits = 10

def perform_pca(features):
    pca = PCA(n_components=2)
    after_pca = pca.fit_transform(features)
    # plt.scatter(after_pca[:, 0], after_pca[:, 1])
    # plt.show()
    return after_pca

def find_kmeans(features, seed):
    seed_values = [[None] * num_seeds for _ in range(num_digits)]
    centroid = []
    for i in range(num_digits):
        for j in range(num_seeds):
            seed_values[i][j] = [features[seed[i][j] - 1][0], features[seed[i][j] - 1][1]]
    for z in range(num_digits):
        x = [p[0] for p in seed_values[z][:]]
        y = [p[1] for p in seed_values[z][:]]
        centroid.append([float(sum(x))/num_seeds, float(sum(y))/num_seeds])
    kmeans = KMeans(n_clusters=num_digits, init=np.array(centroid))
    c = kmeans.fit_predict(features)
    with open("out.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        for ind in range(0, 12000):
            writer.writerow([ind + 1, c[ind]])
    # color = []
    # for i in c:
    #     d = {0: 'yellow', 1: 'white', 2: 'violet', 3: 'blue', 4: 'green', 5: 'brown', 6: 'black', 7: 'orange',
    #             8: 'grey', 9: 'pink'}
    #     color.append(d[i])
    #plt.scatter(features[:, 0], features[:, 1], c=color)
    #plt.show()

def main():
    with open("features.csv", "r") as f:
        features = [list(map(float, rec)) for rec in csv.reader(f)]
    # with open("adjacency.csv", "r") as f:
    #     adjacency = [list(map(float, rec)) for rec in csv.reader(f)]
    with open("seed.csv", "r") as f:
        seed = [list(map(int, rec)) for rec in csv.reader(f)]
    pca = perform_pca(features)
    find_kmeans(pca, seed)


if __name__ == "__main__":
    main()