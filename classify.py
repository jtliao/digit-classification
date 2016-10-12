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


# Gets seed features from features using seed to identify which points are seeds
def get_seed_values(features, seed):
    seed_values = [[None] * num_seeds for _ in range(num_digits)]
    # get values of seed indices
    for i in range(num_digits):
        for j in range(num_seeds):
            seed_values[i][j] = [features[seed[i][j] - 1][0], features[seed[i][j] - 1][1]]
    return seed_values


def find_kmeans(features, seed_values):

    centroid = []

    #find centroid of seed values
    for dig in range(num_digits):
        x = [point[0] for point in seed_values[dig][:]]
        y = [point[1] for point in seed_values[dig][:]]
        centroid.append([float(sum(x))/num_seeds, float(sum(y))/num_seeds])
    #perform kmeans
    kmeans = KMeans(n_clusters=num_digits, init=np.array(centroid))
    assignments = kmeans.fit_predict(features)

    with open("out.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        for ind in range(12000):
            writer.writerow([ind + 1, assignments[ind]])

    return assignments


# Creates 2 plots:
# 1st plot plots predicted digit of all of the points
# 2nd plot plots actual digit of all of the seeds
def plot_preds(features, preds, seeds):
    color = []
    d = {0: 'yellow', 1: 'white', 2: 'violet', 3: 'blue', 4: 'green', 5: 'brown', 6: 'black', 7: 'orange',
         8: 'grey', 9: 'pink'}
    for i in preds:
        color.append(d[i])
    f1 = plt.figure(1)
    plt.scatter(features[:, 0], features[:, 1], c=color)
    f1.show()

    seed_features = []
    seed_colors = []
    for digit, coord_list in enumerate(seeds):
        for coord in coord_list:
            seed_features.append(coord)
            seed_colors.append(d[digit])
    seed_features = np.array(seed_features)
    f2 = plt.figure(2)
    plt.scatter(seed_features[:,0], seed_features[:,1], c=color)
    f2.show()

    # Word-around to get 2 plots to stay up at a time
    raw_input()


def main():
    with open("features.csv", "r") as f:
        features = [list(map(float, rec)) for rec in csv.reader(f)]
    # with open("adjacency.csv", "r") as f:
    #     adjacency = [list(map(float, rec)) for rec in csv.reader(f)]
    with open("seed.csv", "r") as f:
        seed = [list(map(int, rec)) for rec in csv.reader(f)]

    pca = perform_pca(features)
    # print(pca)
    seed_values = get_seed_values(pca, seed)
    # print(seed_values)

    assignments = find_kmeans(pca, seed_values)
    plot_preds(pca, assignments, seed_values)


if __name__ == "__main__":
    main()