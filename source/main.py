import sys
import math
import warnings
import argparse
from onlinelearning.online_learning import *
from em.trapezoidal_expectation_maximization2 import TrapezoidalExpectationMaximization2
warnings.filterwarnings("ignore")


dataset = "australian" # This is Dataset Name
isshuffle = True
all_cont_indices = np.array(
    [False] + [True] * 2 + [False] * 4 + [True] + [False] * 2 + [True] + [False] * 2 + [True] * 2)
all_ord_indices = np.array(
    [True] + [False] * 2 + [True] * 4 + [False] + [True] * 2 + [False] + [True] * 2 + [False] * 2)
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

Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep=' ', header=None)
Y_label = Y_label.values
Y_label = Y_label.flatten()


min_final_error = 1.0
best_window = 0
best_batch = 0
ischange = -1
best_decay_choice = -1
best_contribute_error_rate = -1

if dataset == "australian":
    contribute_error_rate = 0.02
    window_size_denominator = 2
    batch_size_denominator = 20
    decay_coef_change = 0
elif dataset == "ionosphere":
    contribute_error_rate = 0.01
    window_size_denominator = 2
    batch_size_denominator = 8
    decay_coef_change = 0
elif dataset == "kr-vs-kp":
    contribute_error_rate = 0.005
    window_size_denominator = 2
    batch_size_denominator = 10
    decay_coef_change = 0
elif dataset == "magic04":
    contribute_error_rate = 0.02
    window_size_denominator = 2
    batch_size_denominator = 8
    decay_coef_change = 0
elif dataset == "wdbc":
    contribute_error_rate = 0
    window_size_denominator = 2
    batch_size_denominator = 8
    decay_coef_change = 0

BATCH_SIZE = math.ceil(n / batch_size_denominator)
WINDOW_SIZE = math.ceil(n / window_size_denominator)
WINDOW_WIDTH = len(X[BATCH_SIZE])
cont_indices = all_cont_indices[:WINDOW_WIDTH]
ord_indices = all_ord_indices[:WINDOW_WIDTH]
NUM_ORD_UPDATES = 1
batch_c = 8

tem2 = TrapezoidalExpectationMaximization2(cont_indices, ord_indices, window_size=WINDOW_SIZE,
                                           window_width=WINDOW_WIDTH)
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
#             from sklearn.metrics import mean_squared_error
#             mse=mean_squared_error(X,X_imp)
#             print("mse:")
#             print(mse)

#             temp = np.ones((n, 1))
#             X_input =np.hstack((temp,Z_imp))
X_input = Z_imp
if isshuffle == True:
    perm = np.arange(n)
    np.random.seed(1)
    np.random.shuffle(perm)
    Y_label = Y_label[perm]
    # X_input=X_input[perm
decay_choice = 1

online_error = generate3(n, dataset, X_input, Y_label, newcatalog, window_size_denominator,
                         batch_size_denominator, decay_coef_change, decay_choice,
                         contribute_error_rate)

print("best_window:1/" + str(best_window) + "  best_batch:1/" + str(best_batch) + "  ischange:" + str(
    ischange) + "   best_decay_choice:" + str(best_decay_choice) + "  best_contribute_error_rate:" + str(
    best_contribute_error_rate))
print("min_error:" + str(min_final_error))

# %%
if __name__ == "__main__":
    print("Run OVFM_algo")