import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
<<<<<<< Updated upstream
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
=======
>>>>>>> Stashed changes
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import math


num_seeds = 3
num_digits = 10


def print_csv(assignments):
    with open("out.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        for ind in range(12000):
            writer.writerow([ind + 1, assignments[ind]])


def preproc(features, dim):
    # norm = normalize(features)
    # rand_proj = [[random.choice([-1/math.sqrt(dim), 1/math.sqrt(dim)]) for _ in range(k)] for _ in range(103)]
    # preprocessed = np.dot(features, rand_proj)
    pca = PCA(n_components=dim)
    preprocessed = pca.fit_transform(features)
    # plt.scatter(after_pca[:, 0], after_pca[:, 1])
    # plt.show()
    return preprocessed


# Gets seed features from features using seed to identify which points are seeds
def get_seed_values(features, seed, dimensions):
    seed_values = [[None] * num_seeds for _ in range(num_digits)]
    # get values of seed indices
    for i in range(num_digits):
        for j in range(num_seeds):
            seed_value = []
            for d in range(dimensions):
                seed_value.append(features[seed[i][j] - 1][d])
            seed_values[i][j] = seed_value
    return seed_values


def find_kmeans(features, seed_values):
    centroid = []
    #find centroid of seed values
    for dig in range(num_digits):
        x = [point[0] for point in seed_values[dig][:]]
        y = [point[1] for point in seed_values[dig][:]]
        z = [point[2] for point in seed_values[dig][:]]
        centroid.append([float(sum(x))/num_seeds, float(sum(y))/num_seeds, float(sum(z))/num_seeds])
    #perform kmeans
    kmeans = KMeans(n_clusters=num_digits, n_init=1, init=np.array(centroid))
    assignments = kmeans.fit_predict(features)
    return assignments


def spectral(features, adjacency, FUCK):
    spec = SpectralClustering(n_clusters=10, affinity="precomputed")
    # spec.fit(adjacency)
    d = spec.fit_predict(adjacency)



    #spec = SpectralEmbedding(n_components=3, affinity="precomputed")
    #d = spec.fit_transform(adjacency)
    print d.shape
    # find_kmeans(d, FUCK)

    #d = [[0 for _ in range(12000)] for _ in range(12000)]
    #for j in range(103):




def svm(features, seeds):
    clf = SVC()
    vector = []
    digit = []
    for i in range(len(seeds)):
        for j in seeds[i]:
            vector.append(j)
            digit.append(i)
    clf.fit(vector, digit)
    return clf.predict(features)


# Creates 2 plots:
# 1st plot plots predicted digit of all of the points
# 2nd plot plots actual digit of all of the seeds
def plot_preds(features, preds, seeds):
    color = []
    d = {0: 'yellow', 1: 'white', 2: 'violet', 3: 'blue', 4: 'green', 5: 'brown', 6: 'black', 7: 'orange',
         8: 'grey', 9: 'pink'}
    for i in preds:
        color.append(d[i])
    #f1 = plt.figure(1)
    plt.scatter(features[:, 0], features[:, 1], c=color)
    #plt.show()

    seed_features = []
    seed_colors = []
    for digit, coord_list in enumerate(seeds):
        for coord in coord_list:
            seed_features.append(coord)
            seed_colors.append(d[digit])
    seed_features = np.array(seed_features)
    #f2 = plt.figure(2)
    plt.scatter(seed_features[:,0], seed_features[:,1], c=color)
    plt.show()

    # Word-around to get 2 plots to stay up at a time
    raw_input()


def spectral(features, adj):
    spec = SpectralEmbedding(n_components = 3, affinity = "precomputed")
    d = spec.fit_transform(adj)
    return d


def main():
    with open("features.csv", "r") as f:
        features = [list(map(float, rec)) for rec in csv.reader(f)]
        features = np.array(features)
    with open("adjacency.csv", "r") as f:
        adjacency = [list(map(float, rec)) for rec in csv.reader(f)]
        adjacency = np.array(adjacency)
    with open("seed.csv", "r") as f:
        seed = [list(map(int, rec)) for rec in csv.reader(f)]
        seed = np.array(seed)

    preprocessed = preproc(features, 3)
    seed_values = get_seed_values(preprocessed, seed, 3)

    # assignments = spectral(features, adjacency, seed_values)
    assignments = find_kmeans(preprocessed, seed_values)
    print_csv(assignments)
    # plot_preds(pca, assignments, seed_values)


if __name__ == "__main__":
    main()