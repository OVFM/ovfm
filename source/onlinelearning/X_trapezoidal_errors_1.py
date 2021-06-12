import os
from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt
#from ftrl_adp import FTRL_ADP
import matplotlib
from scipy.sparse import coo_matrix, hstack, csr_matrix
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("..")
from onlinelearning.ftrl_adp import FTRL_ADP
from onlinelearning.online_learning import calculate_svm_error
from sklearn import svm

if __name__=="__main__":
    dataset="credit"
    isshuffle=True

    X_input=pd.read_csv("../data2/"+dataset+"/X_trapezoid_zeros.txt",sep=" " ,header=None)
    X_input=X_input.values
    Y_label=pd.read_csv("../data/"+dataset+"/Y_label.txt",sep=' ',header=None)
    Y_label=Y_label.values
    Y_label=Y_label.flatten()
    n = X_input.shape[0]
    temp = np.ones((n, 1))
    X_input = np.hstack((temp, X_input))
    if isshuffle == True:
        perm = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(perm)
        Y_label = Y_label[perm]
    for decay_choice in range(5):
        for contribute_error_rate in [0.01, 0.02, 0.005, 0]:
            errors = []
            decays = []
            predict = []
            mse = []
            classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
            for row in range(n):
                indices = [i for i in range(X_input.shape[1])]
                x = X_input[row]
                y = Y_label[row]
                p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
                error = [int(np.abs(y - p) > 0.5)]
                errors.append(error)
                decays.append(decay)
                predict.append(p)
                mse.append(mean_squared_error(predict[:row + 1], Y_label[:row + 1]))
            nowcatalog ="../data6_X_trapezoid_zeros/"+ dataset+"/decay_choice" + str(decay_choice) + "/contribute_error_rate" + str(contribute_error_rate)
            isExists = os.path.exists(nowcatalog)
            if not isExists:
                os.makedirs(nowcatalog)
            np.savetxt(nowcatalog + '/errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
            np.savetxt(nowcatalog + '/dacays.txt', decays)
            np.savetxt(nowcatalog + '/mse.txt', mse)
            np.savetxt(nowcatalog + '/predict.txt', predict)

    svm_error, best_C = calculate_svm_error(X_input[:, 1:], Y_label, n)
    np.savetxt("../data6_X_trapezoid_zeros/"+ dataset + '/svm_error.txt', [svm_error])
