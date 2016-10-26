import numpy as np
import csv

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
# from sklearn.gaussian_process import RBF


def perform_pca(features, num_dims):
    pca = PCA(n_components=num_dims)
    after_pca = pca.fit_transform(features)
    # plt.scatter(after_pca[:, 0], after_pca[:, 1])
    # plt.show()
    return after_pca

def get_seed_values(features, seed):
    features_list = []
    labels_list = []
    for digit, seed_list in enumerate(seed):
        for seed in seed_list:
            labels_list.append(digit)
            features_list.append(features[seed - 1])

    return features_list, labels_list


def semi_supervised(known_seeds, features):
    seed_features, seed_labels = get_seed_values(features, known_seeds)

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
        seed = [list(map(int, rec)) for rec in csv.reader(f)]

    # print(seed)

    features = np.array(features)

    pca_features = perform_pca(features, 3)

    semi_supervised(seed, pca_features)

    # print(get_seed_values(features, seed))


if __name__ == "__main__":
    main()
