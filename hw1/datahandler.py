import pandas as pd
import numpy as np
from scipy import sparse as sps
import math
import csv

def load_data(path="../ml-20m/ratings.csv"):
    df = pd.read_csv(path)
    userIds = df.userId.unique()
    del df
    
    user_data = []
    movie_data = []
    rating_data = []

    ## Missing movie ids in dataset - we construct our own index here
    movieIdMap = {}
    movieIndex = 0

    for i, row in enumerate(csv.reader(open(path, "r")), 1):
        ## Skip header
        if i == 1:
            continue

        if row[1] not in movieIdMap:
            movieIdMap[row[1]] = movieIndex
            movieIndex += 1

    user_data.append(int(row[0])-1)
    movie_data.append(movieIdMap[row[1]])
    rating_data.append(float(row[2]))

    ## Sparse matrix representation
    data = sps.coo_matrix((rating_data, (user_data, movie_data)))

    return data, movieIdMap

## This function takes in the sparse user-item matrix and efficiently splits into equal-sized training and test sets with identical dimensions as the original matrix
def sparse_train_test_split(data):

    n_users = data.shape[0]
    n_items = data.shape[1]

    # Split users into A and B groups
    users_a = np.random.choice(np.arange(0, n_users), math.ceil(n_users/2), replace=False)
    users_a.sort()
    users_b = np.array(list(set(np.arange(0, n_users)) - set(users_a)))
    users_b.sort()

    # Split items into A and B groups
    items_a = np.random.choice(np.arange(0, n_items), math.ceil(n_items/2), replace=False)
    items_a.sort()
    items_b = np.array(list(set(np.arange(0, n_items)) - set(items_a)))
    items_b.sort()

    # Training set: A-users' ratings for A-items, B-users' ratings for B-items
    train_a = data.tocsc()[:,items_a].tocsr()[users_a,:].tocoo()
    train_b = data.tocsc()[:,items_b].tocsr()[users_b,:].tocoo()

    # Testing set: A-users' ratings for B-items, B-users' ratings for A-items
    test_a = data.tocsc()[:,items_b].tocsr()[users_a,:].tocoo()
    test_b = data.tocsc()[:,items_a].tocsr()[users_b,:].tocoo()

    # Concatenate row/col indices for train and test data
    train_inds = (
        np.concatenate((users_a[train_a.row], users_b[train_b.row])),
        np.concatenate((items_a[train_a.col], items_b[train_b.col]))
    )
    test_inds = (
        np.concatenate((users_a[test_a.row], users_b[test_b.row])),
        np.concatenate((items_b[test_a.col], items_a[test_b.col]))
    )

    # Concatenate values for train/test data, construct sparse matrices
    train = sps.coo_matrix((np.concatenate((train_a.data, train_b.data)), train_inds))
    test = sps.coo_matrix((np.concatenate((test_a.data, test_b.data)), test_inds))

    return train, test