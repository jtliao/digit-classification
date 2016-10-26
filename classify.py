from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
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
    centroid = []

    # find centroid of seed values
    for dig in range(num_digits):
        x = [point[0] for point in seed_values[dig][:]]
        y = [point[1] for point in seed_values[dig][:]]
        centroid.append([float(sum(x)) / num_seeds, float(sum(y)) / num_seeds])


    tmp = []
    for dig in range(num_digits):
        for feature in seed_values[dig]:
            tmp.append(feature)

    # perform gmm
    gmm = GaussianMixture(n_components=10, covariance_type="full")
    gmm.fit(features)
    print(gmm.means_)
    print(gmm.predict(tmp))
    assignments = gmm.predict(features)
    print(assignments)

    with open("out_gmm.csv", "wb") as f:
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


def cca(view1, view2, k):
    view1 = np.array(view1)
    view2 = np.array(view2)

    d1 = np.shape(view1)[1]
    d2 = np.shape(view2)[1]

    X = np.concatenate((view1, view2), 1)
    n = np.shape(X)[0]

    SIG = 1 / n * np.matmul(X.T * X)

    SIG11 = SIG[0:d1, 0:d1] + 10 ^ (-8) * np.identity(d1)
    SIG22 = SIG[(d1 + 0):(d1 + d2), (d1 + 0):(d1 + d2)] + 10 ^ (-8) * np.identity(d2)
    SIG12 = SIG[0:d1, (d1 + 0):(d1 + d2)]
    SIG21 = SIG12.T

    [eigvals1, eigvectors1] = np.linalg.eigh(np.inv(SIG11) * SIG12 * np.inv(SIG22) * SIG21, eigvals=(11997,11999));
    # [D, order] = sort(diag(D), 'descend'); %  # sort cols high to low
    # V = V[:, order];
    # W = V(:, 1:k);

    [eigvals2, eigvectors2] = np.linalg.eigh(np.inv(SIG22) * SIG21 * np.inv(SIG11) * SIG12, eigvals=(11997,11999));
    # [DD, order2] = sort(diag(DD), 'descend'); %  # sort cols high to low
    # VV = VV(:, order2);
    # WW = VV(:, 1:k);

    Y = view1 * eigvectors1;
    YY = view2 * eigvectors2;

    return (Y,YY)


def spectral(features, adj):
    spec = SpectralEmbedding(n_components = 3, affinity = "precomputed")
    d = spec.fit_transform(adj)
    return d



def main():
    with open("features.csv", "r") as f:
        features = [list(map(float, rec)) for rec in csv.reader(f)]
        features = np.array(features)
    # with open("adjacency.csv", "r") as f:
    #     adjacency = [list(map(np.float32, rec)) for rec in csv.reader(f)]
    #     adjacency = np.array(adjacency)
    with open("seed.csv", "r") as f:
        seed = [list(map(int, rec)) for rec in csv.reader(f)]

    # print(np.shape(features))
    #
    # normalized = normalize_variances(features)
    # means = np.sum(normalized, 0) / np.shape(normalized)[0]
    # print(np.sqrt(np.sum(np.square(normalized - np.repeat([means], np.shape(normalized)[0], 0)), 0)))


    dims = 2
    preprocessed = preproc(features, dims)

    # preprocessed = normalize_variances(preprocessed)
    seed_values = get_seed_values(preprocessed, seed, dims)

    # assignments = do_gmm(pca, seed_values)
    assignments = find_kmeans(preprocessed, seed_values, dims)
    plot_preds(preprocessed, assignments, seed_values)

    # print(seed_values)
    #
    # assignments = find_kmeans(pca, seed_values)
    # plot_preds(preprocessed, assignments, seed_values)

    # seed = np.array(seed)
    #
    # preprocessed = preproc(features, 3)
    # seed_values = get_seed_values(preprocessed, seed, 3)
    #
    # assignments = spectral(features, adjacency)
    # # assignments = find_kmeans(preprocessed, seed_values)
    # print_csv(assignments)
    # plot_preds(pca, assignments, seed_values)



if __name__ == "__main__":
    main()