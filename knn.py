import csv
from sklearn.neighbors import NearestNeighbors
import numpy as np

num_seeds = 3
num_digits = 10

original_dim = 103

# Gets seed features from features using seed to identify which points are seeds
def get_seed_values(features, seed):
    # seed_values = [[None] * num_seeds for _ in range(num_digits)]
    # get values of seed indices
    seed_values = []
    for i in range(num_digits):
        for j in range(num_seeds):
            seed_values.append([features[seed[i][j] - 1][d] for d in range(103)])
    return np.array(seed_values)


def run_nn(features, seed_features):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(seed_features)
    distances, indices = nbrs.kneighbors(features[1:10,:])
    print(distances)
    print(indices)


def main():
    with open("features.csv", "r") as f:
        features = [list(map(float, rec)) for rec in csv.reader(f)]
    # with open("adjacency.csv", "r") as f:
    #     adjacency = [list(map(float, rec)) for rec in csv.reader(f)]
    with open("seed.csv", "r") as f:
        seed = [list(map(int, rec)) for rec in csv.reader(f)]

    features = np.array(features)
    # print(features[1:10, :])
    seed_values = get_seed_values(features, seed)
    # print(seed_values)

    run_nn(features, seed_values)


if __name__ == "__main__":
    main()