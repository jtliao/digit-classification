from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.manifold import SpectralEmbedding
from scipy.linalg import eigh
from numpy.linalg import inv
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import math
from copy import copy
from mpl_toolkits.mplot3d import axes3d


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


def preproc_kernel(features, dim):
    # norm = normalize(features)
    # rand_proj = [[random.choice([-1/math.sqrt(dim), 1/math.sqrt(dim)]) for _ in range(k)] for _ in range(103)]
    # preprocessed = np.dot(features, rand_proj)
    pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, n_components=dim)
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


def find_kmeans(features, seed_values, dims):
    centroid = []

    #find centroid of seed values
    for dig in range(num_digits):
        list_of_dims = []
        for i in range(dims):
            list_of_dims.append([point[i] for point in seed_values[dig][:]])

        centroid.append([float(sum(x))/num_seeds for x in list_of_dims])
    print(centroid)
    #perform kmeans
    kmeans = KMeans(n_clusters=num_digits, n_init=1, init=np.array(centroid))
    assignments = kmeans.fit_predict(features)
    return assignments


def spectral(adjacency, seeds):
    d = [[0 for _ in range(12000)] for _ in range(12000)]
    for i in range(12000):
        total = 0
        for j in range(12000):
            total += adjacency[i][j]
        if total == 0:
            d[i][i] = 0
        else:
            d[i][i] = float(1.0/math.sqrt(total))

    identity = [[1 if x == y else 0 for x in range(12000)] for y in range(12000)]
    l = identity - np.dot(np.dot(d, adjacency), d)

    w, v = eigh(l, eigvals=(11997, 11999))
    sed = get_seed_values(v, seeds, 3)
    return v, find_kmeans(v, sed), sed


def cca(features):
    # col_vars = []
    # for i in range(103):
    #     col_values = []
    #     for j in range(12000):
    #         col_values.append(features[j, i])
    #     col_vars.append(np.std(col_values))
    # print col_vars # prints the variance of each col

    view1 = features[:, 0:13]
    view2 = features[:, 100:103]
    concat = np.concatenate((view1, view2), axis=1)
    co = np.cov(concat, rowvar=False)
    eigval, eigvec = eigh(np.dot(np.dot(np.dot(inv(co[0:13, 0:13]), co[0:13, 13:16]), inv(co[13:16, 13:16])),
                                 co[13:16, 0:13]), eigvals=(10, 12))
    # need to reverse the columns of eigvec, since it outputs in ascending order
    reversed_eigvec = copy(eigvec)
    for i in range(13):
        for j in range(3):
            if j == 0:
                reversed_eigvec[i][j] = eigvec[i][2]
            if j == 2:
                reversed_eigvec[i][j] = eigvec[i][0]
    projected = np.dot(view1, reversed_eigvec)
    return projected


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


def do_gmm(features, seed_values):
    # centroid = []

    # find centroid of seed values
    # for dig in range(num_digits):
    #     x = [point[0] for point in seed_values[dig][:]]
    #     y = [point[1] for point in seed_values[dig][:]]
        # centroid.append([float(sum(x)) / num_seeds, float(sum(y)) / num_seeds])

    seeds = []
    for dig in range(num_digits):
        for lst in seed_values[dig]:
            seeds.append(lst)

    # perform gmm
    gmm = GaussianMixture(n_components=10, covariance_type="full")
    gmm.fit(features)


    pred_seeds = gmm.predict(seeds)
    print(pred_seeds)

    mappings = {}

    for i in range(num_digits):
        mappings[pred_seeds[i*3]] = i

    print(mappings)

    # print(gmm.means_)
    # print(gmm.predict(tmp))
    assignments = gmm.predict(features)

    for i in range(len(assignments)):
        assignment = assignments[i]
        assignments[i] = mappings[assignment]

    print(assignments)

    return assignments


# Creates 2 plots:
# 1st plot plots predicted digit of all of the points
# 2nd plot plots actual digit of all of the seeds
def plot_preds(features, preds, seeds, dims=2):
    color = []
    d = {0: 'yellow', 1: 'white', 2: 'violet', 3: 'blue', 4: 'green', 5: 'brown', 6: 'black', 7: 'orange',
         8: 'grey', 9: 'pink'}
    for i in preds:
        color.append(d[i])
    #f1 = plt.figure(1)

    if dims == 2:
        plt.scatter(features[:, 0], features[:, 1], c=color)
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=color)
        plt.show()


    # seed_features = []
    # seed_colors = []
    # for digit, coord_list in enumerate(seeds):
    #     for coord in coord_list:
    #         seed_features.append(coord)
    #         seed_colors.append(d[digit])
    # seed_features = np.array(seed_features)
    # #f2 = plt.figure(2)
    # plt.scatter(seed_features[:,0], seed_features[:,1], c=color)
    # plt.show()

    # Word-around to get 2 plots to stay up at a time
    # raw_input()


def normalize_variances(features):
    features = np.array(features)
    [num_examples, num_features] = np.shape(features)

    means = np.sum(features, 0) / num_examples

    # print(means)
    # print(np.shape(means))
    #
    # print(np.square(features - np.repeat([means], num_examples, 0)))

    normalized = features / (np.repeat([np.sqrt(np.sum(np.square(features - np.repeat([means], num_examples, 0)), 0))], num_examples, 0))
    # print(np.shape(normalized))

    return normalized


def main():
    with open("features.csv", "r") as f:
        features = [list(map(float, rec)) for rec in csv.reader(f)]
        features = np.array(features)
    # with open("adjacency.csv", "r") as f:
    #     adjacency = [list(map(np.float32, rec)) for rec in csv.reader(f)]
    #     adjacency = np.array(adjacency)
    with open("seed.csv", "r") as f:
        seed = [list(map(int, rec)) for rec in csv.reader(f)]

    dim = 3

    # preprocessed = preproc(features, 3)
    ccad = cca(features)
    seed_values = get_seed_values(ccad, seed, dim)

    # data, assignments, seed_values = spectral(features, adjacency, seed)
    # assignments = find_kmeans(ccad, seed_values, dim)
    assignments = do_gmm(ccad, seed_values)
    print_csv(assignments)

    plot_preds(features, assignments, seed, dims=dim)

    #plot_preds(data, assignments, seed_values)


if __name__ == "__main__":
    main()