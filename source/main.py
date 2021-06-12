import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")
from evaluation.helpers import *
from onlinelearning.online_learning import *
from em.online_expectation_maximization import OnlineExpectationMaximization
from em.trapezoidal_expectation_maximization2 import TrapezoidalExpectationMaximization2

# Defining online_trapezoidal imputation
def Trapezoidal_fun():
    dataset = "ionosphere" # Name of dataset

    contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,isshuffle =get_hyperparameter(dataset)

    # adjusting and generating trapezoidal data stream
    all_cont_indices = np.array([False]*11+[True]*12 +[False]*11)
    all_ord_indices = np.array([True]*11 +[False]*12 + [True]*11)
    file = open("../dataset/MaskData/" + dataset + "/X.txt", 'r')
    X = file.readlines()

    n = len(X)
    for i in range(n):
        X[i] = X[i].strip()
        X[i] = X[i].strip("[]")
        X[i] = X[i].split(",")
        X[i] = list(map(float, X[i]))
        narry = np.array(X[i])
        where_are_nan = np.isnan(narry)
        narry[where_are_nan] = 0
        X[i] = narry.tolist()

    BATCH_SIZE = math.ceil(n / batch_size_denominator)
    WINDOW_SIZE = math.ceil(n / window_size_denominator)
    WINDOW_WIDTH = len(X[BATCH_SIZE])
    cont_indices = all_cont_indices[:WINDOW_WIDTH]
    ord_indices = all_ord_indices[:WINDOW_WIDTH]

    Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep=' ', header=None)
    Y_label = Y_label.values
    Y_label = Y_label.flatten()

    tra_data = trapezoidal_data(X, Y_label, 1, 0, 0.01, 0)
    tra_data.fit() #generate the error file of olsf

    isExist=os.path.exists("../Dataset/MaskData/"+dataset+"/X")
    if not isExist:
        os.makedirs("../Dataset/MaskData/"+dataset+"/X")
    np.savetxt("../Dataset/MaskData/"+dataset+"/X/errors.txt",tra_data.yAxis_errorRate)



    NUM_ORD_UPDATES = 1
    batch_c = 8
    tem2 = TrapezoidalExpectationMaximization2(cont_indices, ord_indices, window_size=WINDOW_SIZE,window_width=WINDOW_WIDTH)
    j = 0
    X_imp = []
    Z_imp = []
    start = 0
    end = BATCH_SIZE
    WINDOW_WIDTH = len(X[0])

    while end <= n:
        X_batch = X[start:end]
        if decay_coef_change == 1:
            this_decay_coef = batch_c / (j + batch_c)
        else:
            this_decay_coef = 0.5
        if len(X_batch[-1]) > WINDOW_WIDTH:
            WINDOW_WIDTH = len(X_batch[-1])
            cont_indices = all_cont_indices[:WINDOW_WIDTH]
            ord_indices = all_ord_indices[:WINDOW_WIDTH]

        for i, row in enumerate(X_batch):
            now_width = len(row)
            if now_width < WINDOW_WIDTH:
                row = row + [np.nan for i in range(WINDOW_WIDTH - now_width)]
                X_batch[i] = row
        X_batch = np.array(X_batch)
        Z_imp_batch, X_imp_batch = tem2.partial_fit_and_predict(X_batch, cont_indices, ord_indices,
                                                                max_workers=1, decay_coef=0.5)
        Z_imp.append(Z_imp_batch[0].tolist())
        X_imp.append(X_imp_batch[0].tolist())
        start = start + 1
        end = start + BATCH_SIZE
    for i in range(1, BATCH_SIZE):
        Z_imp.append(Z_imp_batch[i].tolist())
        X_imp.append(X_imp_batch[i].tolist())

    newcatalog = "../dataset/MaskData/" + dataset + "/decaychange" + str(decay_coef_change) + "/window" + str(
        window_size_denominator) + "/batch" + str(batch_size_denominator)
    isExists = os.path.exists(newcatalog)

    if not isExists:
        os.makedirs(newcatalog)
    # print(Z_imp)
    file = open(newcatalog + "/Z_imp.txt", 'w')
    for fp in Z_imp:
        file.write(str(fp))
        file.write('\n')
    file.close()
    file = open(newcatalog + "/X_imp.txt", 'w')
    for fp in X_imp:
        file.write(str(fp))
        file.write('\n')
    file.close()

    X_input = Z_imp
    if isshuffle == True:
        perm = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(perm)
        Y_label = Y_label[perm]

    online_error = generate3(n, dataset, X_input, Y_label, newcatalog, window_size_denominator,
                             batch_size_denominator, decay_coef_change, decay_choice,
                             contribute_error_rate)

    print("online_error:" + str(online_error))

# Defining Online copula imputation
def Online_Main():
    dataset = "credit"

    # getting  hyperparameter
    contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change, decay_choice = get_hyperparameter(dataset)

    isshuffle = True

    MASK_NUM = 1
    X = pd.read_csv("../dataset/MaskData/" + dataset + "/X_process.txt", sep=" ", header=None)
    X_masked = mask_types(X, MASK_NUM, seed=1)

    X = pd.read_csv("../dataset/MaskData/" + dataset + "/X_process.txt", sep=' ', header=None)
    Y_label = pd.read_csv("../dataset/Datalabel/" + dataset + "/Y_label.txt", sep=' ', header=None)
    X = X.values
    Y_label = Y_label.values
    #
    all_cont_indices = get_cont_indices(X_masked)
    all_ord_indices = ~all_cont_indices

    n = X.shape[0]
    feat = X.shape[1]
    Y_label = Y_label.flatten()

    # setting hyperparameter
    max_iter = batch_size_denominator * 2
    BATCH_SIZE = math.ceil(n / batch_size_denominator)
    WINDOW_SIZE = math.ceil(n / window_size_denominator)
    NUM_ORD_UPDATES = 1
    batch_c = 8
    # %%

    X_masked = pd.DataFrame(X_masked)
    X_zeros = X_masked.fillna(value=0)

    temp = np.ones((n, 1))
    X_input = np.hstack((temp, X_zeros))
    if isshuffle == True:
        perm = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(perm)
        Y_label = Y_label[perm]
        X_input = X_input[perm]
    newcatalog = "../dataset/MaskData/" + dataset + "/X_zeros"

    generate_X(n, X_input, Y_label, newcatalog, decay_choice, contribute_error_rate)

    min_final_error = 1.0
    best_window = 0
    best_batch = 0
    ischange = -1
    best_decay_choice = -1
    best_contribute_error_rate = -1

    oem = OnlineExpectationMaximization(all_cont_indices, all_ord_indices, window_size=WINDOW_SIZE)
    j = 0
    X_imp = np.empty(X_masked.shape)
    Z_imp = np.empty(X_masked.shape)
    X_masked = np.array(X_masked)
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
    newcatalog = "../dataset/MaskData/" + dataset + "/decaychange" + str(decay_coef_change) + "/window" + str(
        window_size_denominator) + "/batch" + str(batch_size_denominator)
    isExists = os.path.exists(newcatalog)
    if not isExists:
        os.makedirs(newcatalog)
    np.savetxt(newcatalog + "/Z_imp.txt", Z_imp)
    np.savetxt(newcatalog + "/X_imp.txt", X_imp)
    temp = np.ones((n, 1))
    X_input = np.hstack((temp, Z_imp))
    if isshuffle == False:
        perm = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(perm)
        X_input = X_input[perm]
    online_error = generate(n, dataset, X_input, Y_label, newcatalog, window_size_denominator,
                            batch_size_denominator, decay_coef_change, decay_choice,
                            contribute_error_rate)

if __name__ == "__main__":
    Trapezoidal_fun()
    # Online_Main()
    print("Run OVFM_algo")