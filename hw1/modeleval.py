import numpy as np
from scipy import sparse as sps

# Calculate root mean squared error on given dataset
# Parameters:
## data: sparse matrix containing actual ratings
## v: user matrix
## w: item matrix
## mu: global bias
## b_v: user bias
## b_w: item bias 
# 
# Output: 
## RMSE
def rmse(data, v, w, mu, b_v, b_w):
    locs = np.array(list(zip(data.row, data.col)))
    inds = np.arange(len(data.data))
    rmse = 0.
    for ind in inds:
        i, j = locs[ind]
        r = data.data[ind]
        rmse += (r - ( mu + b_v[i] + b_w[j] + v[i].dot(w[j].T)))**2
    return (rmse/len(inds))**0.5


# Calculate mean reciprocal ranking for all users
# Parameters:
## data: sparse matrix over which to calculate MRR (i.e. test or training fold)
## v: user matrix
## w: item matrix
## mu: global bias
## b_v: user bias
## b_w: item bias
# 
# Output:
## Array of MRR scores (size is number of users in data)
def mrr(data, v, w, mu, b_v, b_w):
    locs = np.array(list(zip(data.row, data.col)))
    inds = np.arange(len(data.data))
    pred = np.zeros(len(data.data))

    ## For each rating in the dataset,
    ## calculate its predicted estimate using the fitted matrix factorization parameters
    for ind in inds:
        i, j = locs[ind]
        r = data.data[ind]
        pred[ind] = v[i].dot(w[j].T) + mu + b_v[i] + b_w[j]

    ## Convert to sparse matrix (CSR format)
    pred_matrix = sps.coo_matrix((pred, (data.row, data.col))).tocsr()
    ## MRR vector
    n = pred_matrix.shape[0]
    mrr = np.zeros(n)

    ## Convert actual data to CSR format for fast row-indexing
    data = data.tocsr()

    for k in range(n):
        ## Get indices of highly-ranked items (score at least 3) by user k
        inds = np.where([data[k,:].data >= 3.0])[1]
        ## Pass if no such items exist (MRR is NaN)
        if len(inds) == 0:
            mrr[k] = None
            continue

        ## Convert estimated ratings for movies in user k's test set
        pred_row = pred_matrix[k,:].data.flatten()
        ranks = np.flip(np.argsort(pred_row) + 1, axis=0)

        ## Calculate MRR for user k
        mrr[k] = sum(1./ranks[inds])/len(inds)
