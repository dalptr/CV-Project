import time
from math import sqrt
import random


class KNN:
    def __init__(self):
        pass

    @staticmethod
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    @staticmethod
    def get_neighbors(train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = KNN.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors


def main():
    random.seed(time.time())
    n_random_points = 10
    k_nearest_neighbors = 3
    dataset: list = []
    for i in range(n_random_points):
        dataset.append([random.random(), random.random()])

    nearest_neighbors = KNN.get_neighbors(dataset, dataset[0], k_nearest_neighbors)
    print("All points:")
    for point in dataset:
        print(point)
    print("Nearest neighbors:")
    for neighbor in nearest_neighbors:
        print(neighbor)


main()
