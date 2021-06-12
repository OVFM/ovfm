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
    dataset="kr-vs-kp"
    window=2
    batch=8
    decay_change_flag=1
    decay_choice=4
    contribute_error_rate=0

    X_input=pd.read_csv("../data/"+dataset+"/X_zeros.txt",sep=" " ,header=None)
    X_input=X_input.values
    Y_label=pd.read_csv("../data/"+dataset+"/Y_label.txt",sep=" ",header=None)
    Y_label=Y_label.values
    Y_label=Y_label.flatten()
    Z_input=pd.read_csv("../data/"+dataset+"/decaychange"+str(decay_change_flag)+"/window"+str(window)+"/batch"+str(batch)+"/Z_imp.txt",sep=" ",header=None)
    Z_input=Z_input.values

    n = X_input.shape[0]
    temp = np.ones((n, 1))
    X_input = np.hstack((temp, X_input))
    Z_input = np.hstack((temp, Z_input))

    z_final_errors=[]
    x_final_errors=[]

    for i in range(10):
        perm = np.arange(n)
        np.random.seed(i)
        np.random.shuffle(perm)
        Y_label = Y_label[perm]
        X_input = X_input[perm]
        Z_input = Z_input[perm]

        X_errors = []
        X_decays = []
        X_predict = []
        X_mse = []
        X_classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
        for row in range(n):
            indices = [i for i in range(X_input.shape[1])]
            x = X_input[row]
            y = Y_label[row]
            p, decay, loss, w = X_classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]
            X_errors.append(error)
            X_decays.append(decay)
            X_predict.append(p)
            X_mse.append(mean_squared_error(X_predict[:row + 1], Y_label[:row + 1]))
        nowcatalog ="../data8_best_set/"+ dataset+"/shuffle" + str(i)
        x_final_errors.append(np.sum(X_errors) / n)
        isExists = os.path.exists(nowcatalog)
        if not isExists:
            os.makedirs(nowcatalog)
        np.savetxt(nowcatalog + '/X_errors.txt', np.cumsum(X_errors) / (np.arange(len(X_errors)) + 1.0))
        np.savetxt(nowcatalog + '/X_dacays.txt', X_decays)
        np.savetxt(nowcatalog + '/X_mse.txt', X_mse)
        np.savetxt(nowcatalog + '/X_predict.txt', X_predict)

        Z_errors = []
        Z_decays = []
        Z_predict = []
        Z_mse = []
        Z_classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])
        for row in range(n):
            indices = [i for i in range(Z_input.shape[1])]
            x = Z_input[row]
            y = Y_label[row]
            p, decay, loss, w = Z_classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]
            Z_errors.append(error)
            Z_decays.append(decay)
            Z_predict.append(p)
            Z_mse.append(mean_squared_error(Z_predict[:row + 1], Y_label[:row + 1]))

        z_final_errors.append(np.sum(Z_errors)/n)
        nowcatalog = "../data8_best_set/" + dataset + "/shuffle" + str(i)
        isExists = os.path.exists(nowcatalog)
        if not isExists:
            os.makedirs(nowcatalog)
        np.savetxt(nowcatalog + '/Z_errors.txt', np.cumsum(Z_errors) / (np.arange(len(Z_errors)) + 1.0))
        np.savetxt(nowcatalog + '/Z_dacays.txt', Z_decays)
        np.savetxt(nowcatalog + '/Z_mse.txt', Z_mse)
        np.savetxt(nowcatalog + '/Z_predict.txt', Z_predict)

    np.savetxt("../data8_best_set/"+dataset+"/Z_final_errors.txt",z_final_errors)
    np.savetxt("../data8_best_set/" + dataset + "/X_final_errors.txt", x_final_errors)