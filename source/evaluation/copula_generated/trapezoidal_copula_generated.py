import numpy as np
#from em.online_expectation_maximization import OnlineExpectationMaximization
from em.trapezoidal_expectation_maximization import TrapezoidalExpectationMaximization
from scipy.stats import random_correlation, norm, expon
from evaluation.helpers import *
import time


if __name__ == "__main__":
    scaled_errors = []
    smaes = []
    runtimes = []
    NUM_RUNS = 1
    n = 10
    max_iter = 10
    BATCH_SIZE=3
    WINDOW_SIZE=6
    NUM_ORD_UPDATES = 1
    batch_c = 8
    for i in range(1,NUM_RUNS+1):
        np.random.seed(i)
        print("starting epoch: " + str(i))
        print("\n")
        X=[
            [1,2,3],
            [3,2,1],
            [2,3,4,5,5,5],
            [3,4,5,2,3,4],
            [1,2,4,5,6,5,6,6,7],
            [3,3,3,4,5,2,9,8,6],
            [1,2,4,5,6,5,6,6,7,9, 9, 10],
            [3,3,3,4,5,2,9,8,6,8, 8, 9],
            [1,2,4,4,4,6,7,6,9,10,11,12,13,14,15],
            [4,5,3,5,5,5,6,8,9,9, 7, 10,15,14,13]
        ]
        # for j in range(5,15,1):
        #     # 6-10 columns are binary, 11-15 columns are ordinal with 5 levels
        #     X[:,j] = cont_to_ord(X[:,j], k=2*(j<10)+5*(j>=10))
        #row_sum = np.sum(X, axis=1)
        all_cont_indices = np.array([True] * 5 + [False] * 10)
        all_ord_indices = np.array([False] * 5 + [True] * 10)

        WINDOW_WIDTH = len(X[BATCH_SIZE])
        cont_indices=all_cont_indices[:WINDOW_WIDTH]
        ord_indices=all_ord_indices[:WINDOW_WIDTH]
        tem = TrapezoidalExpectationMaximization(cont_indices, ord_indices, window_size=WINDOW_SIZE,window_width=WINDOW_WIDTH)
        start_time = time.time()
        # X_imp = np.empty(X_masked.shape)
        # Z_imp = np.empty(X_masked.shape)
        X_imp = []
        Z_imp = []
        start=0
        end=BATCH_SIZE
        WINDOW_WIDTH=len(X[0])
        while end <= n:
            indices = np.arange(start, end, 1)
            #decay_coef = batch_c/(j+batch_c)#？？？？？
            X_batch=X[start:end]
            if len(X_batch[-1])>WINDOW_WIDTH:
                WINDOW_WIDTH=len(X_batch[-1])
                cont_indices = all_cont_indices[:WINDOW_WIDTH]
                ord_indices = all_ord_indices[:WINDOW_WIDTH]
            ###X_batch某些地方置空用于填补数据，以该batch的最大维度为标准
            for i,row in enumerate(X_batch):
                now_width=len(row)
                if now_width<WINDOW_WIDTH:
                    row=row+[np.nan for i in range(WINDOW_WIDTH-now_width)]
                    X_batch[i]=row
            X_batch=np.array(X_batch)

            Z_imp_batch,X_imp_batch = tem.partial_fit_and_predict(X_batch,cont_indices,ord_indices,max_workers = 1, decay_coef=0.5)
            Z_imp.append(Z_imp_batch[0])
            X_imp.append(X_imp_batch[0])
            start=start+1
            end=start+BATCH_SIZE
        for i in range(1,BATCH_SIZE):
            Z_imp.append(Z_imp_batch[i])
            X_imp.append(X_imp_batch[i])


    print("mean of scaled errors is: ")
    print(np.mean(np.array(scaled_errors)))
    print("std deviation of scaled errors is: ")
    print(np.std(np.array(scaled_errors)))
    print("\n")
    mean_smaes = np.mean(np.array(smaes),axis=0)
    print("mean cont smaes are: ")
    print(np.mean(mean_smaes[:5]))
    print("mean bin smaes are: ")
    print(np.mean(mean_smaes[5:10]))
    print("mean ord smaes are: ")
    print(np.mean(mean_smaes[10:]))
    print("\n")
    std_dev_smaes = np.std(np.array(smaes),axis=0)
    print("std dev cont smaes are: ")
    print(np.mean(std_dev_smaes[:5]))
    print("std dev bin smaes are: ")
    print(np.mean(std_dev_smaes[5:10]))
    print("std dev ord smaes are: ")
    print(np.mean(std_dev_smaes[10:]))
    print("\n")
    print("mean time for run is: ")
    print(np.mean(np.array(runtimes)))