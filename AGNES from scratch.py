# Second Assignment
# Author : Erfan Asadi - 950122680021

# Libraries
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


def update_d(distance_matrix, first_d, tag):
    src_houses = re.split(',', tag)

    for column in distance_matrix.columns:
        if column == tag:
            distance_matrix[column][column] = float("inf")
        else:
            dest_houses = re.split(',', column)
            distance_matrix[tag][column] = distance_matrix[column][tag] = min_distance(first_d, src_houses, dest_houses)
    return distance_matrix


def min_distance(first_d, src_houses, dest_houses):
    min_value = float("inf")
    for s_house in src_houses:
        for d_house in dest_houses:
            if first_d[int(s_house) - 1][int(d_house) - 1] < min_value:
                min_value = first_d[int(s_house) - 1][int(d_house) - 1]
    return min_value


def make_distance_matrix(m, data):
    d = np.zeros((m, m))
    # # distance function = ((x1 - x2)^2 + (y1 - y2) ^ 2) ^ 1/2
    for i in range(len(d)):
        for j in range(len(d)):
            if i == j:
                d[i][j] = 0
            else:
                d[i][j] = ((data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2) ** 0.5

    distance_matrix = pd.DataFrame(d, [i.__str__() for i in range(1, m + 1)],
                                   [i.__str__() for i in range(1, m + 1)])

    # change '0's to inf to find minimum easier
    d[d == 0.0] = float('inf')

    return distance_matrix, d


def agnes(dataset, c=2):
    len_data = len(dataset)

    # 2. creating distance matrix
    distance_matrix, d = make_distance_matrix(len_data, dataset)

    # 3. single link iteration

    while len(distance_matrix.columns) > c:
        # Finding single link
        single_link = min(distance_matrix.min())

        # finding the location of single link in distance matrix(pandas dataframe)
        temp_index = [
            (distance_matrix[col][distance_matrix[col] == single_link].index[i], distance_matrix.columns.get_loc(col))
            for col
            in distance_matrix.columns for i in
            range(len(distance_matrix[col][distance_matrix[col] == single_link].index))]

        index = [temp_index[0][0], distance_matrix.columns[temp_index[0][1]]]

        # updating the indexes
        indexes = np.array(distance_matrix.index)
        new_indexes = [i.__str__() for i in indexes]
        new_indexes[new_indexes.index(index[0].__str__())] = index[0].__str__() + ',' + index[1].__str__()
        distance_matrix = distance_matrix.reindex(new_indexes)

        # updating the columns
        distance_matrix.columns = new_indexes

        # Removing the extra rows and columns
        distance_matrix.drop(columns=index[1], inplace=True)
        distance_matrix.drop(index=index[1], inplace=True)

        # Updating new distance matrix with new distances
        distance_matrix = update_d(distance_matrix, d, index[0].__str__() + ',' + index[1].__str__())

    # Plotting the points with clusters
    cluster_labels = np.zeros(len_data)

    for j in range(len(distance_matrix.columns)):
        for i in re.split(',', distance_matrix.columns[j]):
            cluster_labels[int(i) - 1] = j + 1

    print("Each point cluster is {}".format(cluster_labels))
    return cluster_labels


if __name__ == "__main__":
    # 1. creating 100 points
    length = 100
    points = np.random.randint(0, 15, (length, 2))

    clusters = agnes(points, 5)

    plt.scatter(points[:, 0], points[:, 1], c=clusters,
                cmap="rainbow")
    plt.ylabel("Y", fontsize=17)
    plt.xlabel("X", fontsize=17)
    plt.title("AGNES ALGORITHM ", fontsize=17)
    plt.show()
