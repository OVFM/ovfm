import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from em.online_expectation_maximization import OnlineExpectationMaximization
from scipy.stats import random_correlation, norm, expon
from evaluation.helpers import *
from onlinelearning.online_learning import *
import math
import os


dataset = "wdbc" # australian,ionosphere,german,diabetes,wdbc,credit,ionosphere

#getting  hyperparameter
contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,shuffle =get_hyperparameter(dataset)
MASK_NUM=1
X = pd.read_csv("../dataset/MaskData/"+dataset+"/X_process.txt",sep=" " ,header=None)
Y_label = pd.read_csv("../dataset/Datalabel/" + dataset + "/Y_label.txt", sep=' ', header=None)
X_masked=mask_types(X,MASK_NUM,seed=1) #arbitrary setting Nan
X = X.values
Y_label = Y_label.values

all_cont_indices=get_cont_indices(X_masked)
all_ord_indices=~all_cont_indices

n = X_masked.shape[0]
feat = X_masked.shape[1]
Y_label = Y_label.flatten()

#setting hyperparameter
max_iter = batch_size_denominator * 2
BATCH_SIZE = math.ceil(n / batch_size_denominator)
WINDOW_SIZE = math.ceil(n / window_size_denominator)
NUM_ORD_UPDATES = 1
batch_c = 8

X_masked=pd.DataFrame(X_masked)
X_zeros = X_masked.fillna(value=0)

temp = np.ones((n, 1))
X_input = np.hstack((temp, X_zeros))
if shuffle == True:
    perm = np.arange(n)
    np.random.seed(1)
    np.random.shuffle(perm)
    Y_label = Y_label[perm]
    X_input = X_input[perm]
newcatalog = "../dataset/MaskData/" + dataset + "/X_zeros"

#
X_Zero_CER,svm_error =  generate_X(n, X_input, Y_label, newcatalog, decay_choice, contribute_error_rate)


min_final_error=1.0
best_window=0
best_batch=0
ischange=-1
best_decay_choice=-1
best_contribute_error_rate=-1

oem = OnlineExpectationMaximization(all_cont_indices, all_ord_indices, window_size=WINDOW_SIZE)
j = 0
X_imp = np.empty(X_masked.shape)
Z_imp = np.empty(X_masked.shape)
X_masked=np.array(X_masked)
while j <= max_iter:
    start = (j * BATCH_SIZE) % n
    end = ((j + 1) * BATCH_SIZE) % n
    if end < start:
        indices = np.concatenate((np.arange(end), np.arange(start, n, 1)))
    else:
        indices = np.arange(start, end, 1)
    if decay_coef_change == 1:
        this_decay_coef = batch_c / (j + batch_c)
    else:
        this_decay_coef = 0.5
    Z_imp[indices, :], X_imp[indices, :] = oem.partial_fit_and_predict(X_masked[indices, :], max_workers=1,
                                                                       decay_coef=this_decay_coef)
    j += 1

temp = np.ones((n, 1))
X_input = np.hstack((temp, Z_imp))
if shuffle == True:
    perm = np.arange(n)
    np.random.seed(1)
    np.random.shuffle(perm)
    X_input = X_input[perm]
online_error = generate(n, dataset, X_input, Y_label, newcatalog, window_size_denominator,
                                batch_size_denominator, decay_coef_change, decay_choice,
                                contribute_error_rate,X_Zero_CER,svm_error)
# if __name__ == '__main__':
#     print("online_error:",online_error)