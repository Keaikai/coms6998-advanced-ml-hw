import pandas as pd
import numpy as np
from scipy import sparse as sps
import math
import csv
import time

from sgd import *
from modeleval import *
from datahandler import *

np.seterr(all="raise")

# Grid search function
# Log SGD results, times, RMSE/MRR metrics, and error status
# Parameters:
## data: user-item matrix (full dataset)
## lambduh_list: list of lambda values to try
## rank_list: list of rank values to try
## path_to_results

def gridsearch(data, lambduh_list, rank_list, path_to_results="/home/kevinwu103/results/"):
    
    n_users = data.shape[0]
    n_items = data.shape[1]
    
    result_info = []

    n_params = len(lambduh_list) * len(rank_list)
    cnt = 1
    for rank in rank_list:
        for lambduh in lambduh_list:
            print ("GRID SEARCH ROUND {}/{} r={}, lambda={}".format(cnt, n_params, rank, lambduh))
            
            train, test = sparse_train_test_split(data)
        
            V = np.random.randn(n_users, rank)
            W = np.random.randn(n_items, rank)
            b_v = np.random.randn(n_users)
            b_w = np.random.randn(n_items)

            mu = np.mean(train.data)

            V_veloc = np.zeros((n_users, rank))
            W_veloc = np.zeros((n_items, rank))
            bv_veloc = np.zeros(n_users)
            bw_veloc = np.zeros(n_items)

            gd_start_time = time.time()
            V, W, b_v, b_w, epoch_cnt, err_status = gradient_descent(train, 
                                              V, W, 
                                              V_veloc, W_veloc, 
                                              mu, b_v, b_w, 
                                              bv_veloc, bw_veloc, 
                                              start_eta = 0.01, 
                                              gamma = 0.3, 
                                              l2_reg = lambduh)
            gd_end_time = time.time()

            print ("Calculating RMSE....")
            rmse_start_time = time.time()
            train_rmse = rmse(train, V, W, mu, b_v, b_w)
            test_rmse = rmse(test, V, W, mu, b_v, b_w)
            rmse_end_time = time.time()
                
            print ("Calculating MRR....")
            mrr_start_time = time.time()
            train_mrr = mrr(train, V, W, mu, b_v, b_w)
            test_mrr = mrr(test, V, W, mu, b_v, b_w)
            mrr_end_time = time.time()
            
            result_info.append({
                "lambda": lambduh,
                "rank": rank,
                "train_mrr": train_mrr,
                "test_mrr": test_mrr,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "gd_time": round(gd_end_time - gd_start_time, 0),
                "rmse_time": round(rmse_end_time - rmse_start_time, 0),
                "mrr_time": round(mrr_end_time - mrr_start_time, 0),
                "floating_point_error": err_status,
                "epochs_trained": epoch_cnt
            })
            
            print ("Caching results....")
            
            with open(path_to_results+"results_{}.json".format(cnt), "w") as file:
                json.dump(result_info, file)
                      
            print ("\n\n")
            cnt += 1
            
           
    return result_info
        
        
if __name__ == "__main__":

    data, movieIdMap = load_data("../ml-20m/ratings.csv")

    _ = gridsearch(data, 
        lambduh_list=[0.005, 0.01, 0.05, 0.1, 0.5], 
        rank_list=[5, 10, 20, 40])

