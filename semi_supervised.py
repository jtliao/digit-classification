import numpy as np
import csv
from scipy.linalg import eigh
from numpy.linalg import inv
from copy import copy
from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
# from sklearn.gaussian_process import RBF


num_seeds = 3
num_digits = 10

original_dim = 103


def perform_pca(features, num_dims):
    pca = PCA(n_components=num_dims)
    after_pca = pca.fit_transform(features)
    # plt.scatter(after_pca[:, 0], after_pca[:, 1])
    # plt.show()
    return after_pca


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

def get_seed_values(features, seed, dim):
    # seed_values = [[None] * num_seeds for _ in range(num_digits)]
    # get values of seed indices
    seed_values = []
    for i in range(num_digits):
        for j in range(num_seeds):
            seed_values.append([features[seed[i][j] - 1][d] for d in range(dim)])
    return np.array(seed_values)


def get_seed_features_and_labels(features, seeds):
    seed_features = []
    seed_labels = []
    for i in range(num_digits):
        for j in range(len(seeds[i])):
            seed_features.append(features[seeds[i][j] - 1])
            seed_labels.append(i)

    print(np.shape(seed_features))
    print(np.shape(seed_labels))

    return seed_features, seed_labels


def run_nn(features, seed_features):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(seed_features)
    distances, indices = nbrs.kneighbors(features)

    # print(list(np.ndarray.flatten(np.sort(distances, axis=0).T)))

    print(distances)
    print(indices)

    confident_set = [[] for _ in range(10)]

    for i in range(np.shape(features)[0]):
        if distances[i] < 1:
            confident_set[indices[i]/3].append(i)

    print(confident_set)
    print(np.shape(confident_set))

    return confident_set



def semi_supervised(known_seeds, features):
    seed_features, seed_labels = get_seed_features_and_labels(features, known_seeds)

    # print seed_features

    clf = GaussianProcessClassifier()
    # clf = SVC(probability=True)
    clf.fit(seed_features, seed_labels)

    # print clf.predict_proba(seed_features)
    # print clf.predict(seed_features)
    for single_features in features:
        print(clf.predict_proba(np.array([single_features]).reshape(1, -1)))
        # print(clf.predict_proba(single_features))


def main():
    with open("features.csv", "r") as f:
        features = [list(map(float, rec)) for rec in csv.reader(f)]
    # with open("adjacency.csv", "r") as f:
    #     adjacency = [list(map(float, rec)) for rec in csv.reader(f)]
    with open("seed.csv", "r") as f:
        seeds = [list(map(int, rec)) for rec in csv.reader(f)]

    # print(seed)

    features = np.array(features)

    cca_features = cca(features)
    seed_values = get_seed_values(cca_features, seeds, 3)

    confident_set = run_nn(cca_features, seed_values)



    semi_supervised(confident_set, features)

    # print(get_seed_values(features, seed))


if __name__ == "__main__":
    main()
