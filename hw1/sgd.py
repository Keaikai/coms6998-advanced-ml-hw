import numpy as np
from scipy import sparse as sps
import time

# Throw exceptions for all numerical errors (i.e. overflow/underflow)
np.seterr(all="raise")

# SGD with Monetum. Single pass/epoch
# Parameters:
## data: Actual ratings
## W: item matrix
## V_veloc: velocity/momentum for user matrix
## W_veloc: velocity/momentum for item matrix
## mu: global bias (empirical mean)
## b_v: user bias
## b_w: item bias
## bv_veloc: velocity/momentum for user bias
## bw_veloc: velocity/momentum for item bias
## eta: learning rate
## gamma: momentum rate
## l2_reg: L2 regularizer (lambda)
#
# Returns:
# V, W, V_veloc, W_veloc, b_v, b_w, bv_veloc, bw_veloc

def sgd_momentum(data, locs, V, W, V_veloc, W_veloc, mu, b_v, b_w, bv_veloc, bw_veloc, eta = 0.01, gamma=0.5, l2_reg = 0.5):
    
    inds = np.random.permutation(len(data.data))
    
    for ind in inds:
        i,j = locs[ind]
        r = data.data[ind]
        
        error = (r - (mu + b_v[i] + b_w[j] + V[i].dot(W[j].T)))
        
        V_veloc[i] = gamma * V_veloc[i] - eta * (error * W[j] - l2_reg * V[i])
        W_veloc[j] = gamma * W_veloc[j] - eta * (error * V[i] - l2_reg * W[j])
        bv_veloc[i] = gamma * bv_veloc[i] - eta * (error - l2_reg * b_v[i])
        bw_veloc[j] = gamma * bw_veloc[j] - eta * (error - l2_reg * b_w[j])
        
        W[j] -= W_veloc[j]
        V[i] -= V_veloc[i]
        b_v[i] -= bv_veloc[i]
        b_w[j] -= bw_veloc[j]
    
    return V, W, V_veloc, W_veloc, b_v, b_w, bv_veloc, bw_veloc

def obj_func(data, locs, V, W, mu, b_v, b_w, l2_reg):
    obj = 0.
    inds = np.arange(len(data.data))
    inds = inds[0:1000000]
    for ind in inds:
        i, j = locs[ind]
        r = data.data[ind]
        obj += (r - (mu + b_v[i] + b_w[j] + V[i].dot(W[j].T)))**2
    print ("Training RMSE (100k samples): {}".format((obj/len(inds))**0.5))
    return obj + l2_reg * (np.square(V).sum() + np.square(W).sum() + np.square(b_v).sum() + np.square(b_w).sum())

# Gradient descent function. 
# Parameters:
## V: user matrix
## W: item matrix
## V_veloc: velocity/momentum for user matrix
## W_veloc: velocity/momentum for item matrix
## mu: global bias (empirical mean)
## b_v: user bias
## b_w: item bias
## bv_veloc: velocity/momentum for user bias
## bw_veloc: velocity/momentum for item bias
## start_eta: Initial learning rate for SGD
## gamma: Initial momentum rate 
## l2_reg: L2 regularizer (lambda)
#
# Returns:
# V, W, b_v, b_w, number of epochs trained for, error status

def gradient_descent(data, 
                     V, W, 
                     V_veloc, W_veloc, 
                     mu, b_v, b_w, 
                     bv_veloc, bw_veloc, 
                     start_eta, gamma, l2_reg):
    
    last_obj = 1e20
    eta = start_eta
    locs = np.array(list(zip(data.row, data.col)))
    epoch_cnt = 1
    
    while True:
        print ("Starting epoch {} ...".format(epoch_cnt))
        start = time.time()
        try:
            V, W, V_veloc, W_veloc, b_v, b_w, bv_veloc, bw_veloc = sgd_momentum(data, locs, 
                                                  V, W, V_veloc, W_veloc, 
                                                  mu, b_v, b_w, bv_veloc, bw_veloc, 
                                                  eta, gamma, l2_reg)
        except FloatingPointError as e:
            print ("Floating point error encountered: {}, aborting SGD.".format(e))
            return V, W, b_v, b_w, epoch_cnt, True
        
        end = time.time()
        print ("Finished epoch {} (Time: {} s)".format(epoch_cnt, round(end-start, 0)))
        
        obj = obj_func(data, locs, V, W, mu, b_v, b_w, l2_reg)
        print ("Cost: {}".format(obj))
        if obj > last_obj:
            print ("Adjusting eta....")
            eta *= 0.1
        elif (last_obj-obj)/last_obj <= 0.001:
            return V, W, b_v, b_w, epoch_cnt, False
        else:
            pass
        
        if eta < 1.0e-4:
            return V, W, b_v, b_w, epoch_cnt, False
        
        last_obj = obj
        gamma *= 1.1
        epoch_cnt += 1
        
        print ("-------------")