import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack, csr_matrix
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys

sys.path.append("..")
from onlinelearning.ftrl_adp import FTRL_ADP
from onlinelearning.online_learning import calculate_svm_error




def ensemble(n,dataset,X_input,Z_input,Y_label,catalog_Z,window_size_denominator,batch_size_denominator,decay_coef_change,decay_choice,contribute_error_rate):
    errors=[]
    predict=[]
    mse=[]
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss=0
    z_loss=0
    lamda=0.5
    eta = 8 * np.sqrt(1/np.log(n))
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]
        p_x, decay_x,loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)
#         print("p_x,loss_x")
#         print(p_x,loss_x)
        z = Z_input[row]
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)
#         print("p_z,loss_z")
#         print(p_z,loss_z)
        
        p=sigmoid(lamda*np.dot(w_x,x)+(1.0-lamda)*np.dot(w_z,z))
#        print(str(p)+"="+str(lamda)+"*"+str(p_x)+"+"+str(1-lamda)+"*"+str(p_z))
        #x_loss+=logistic_loss(p_x,y)
        x_loss+=loss_x
#         print("X累积损失，loss")
#         print(x_loss,logistic_loss(p_x,y))
        #z_loss+=logistic_loss(p_z,y)
        z_loss+=loss_z
#         print("Z累积损失，loss")
#         print(z_loss,logistic_loss(p_z,y))
        lamda=np.exp(-eta*x_loss)/(np.exp(-eta*x_loss)+np.exp(-eta*z_loss))
#         print("lamda")
#         print(lamda)
        
        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
        predict.append(p)
        #print("预测",predict)
        #print("标签",Y_label)
        mse.append(mean_squared_error(predict[:row + 1], Y_label[:row + 1]))

    nowcatalog = catalog_Z + "/decay_choice" + str(decay_choice) + "/contribute_error_rate" + str(contribute_error_rate)
    np.savetxt(nowcatalog + '/ensemble_errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog + '/ensemble_mse.txt', mse)
    np.savetxt(nowcatalog + '/ensemble_predict.txt', predict)
    final_error = np.sum(errors) / (n + 0.0)

    # print("预测值：",predict)
    # print("标签",Y_label)

    ######画错误率图#################
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 1.0))
    plt.ylabel("errors")  # y轴上的名字

    y = pd.read_csv(nowcatalog+ '/errors.txt', sep='\t', header=None)
    plt.xlim((0, n))
    x = range(n)
    plt.plot(x, y, color='green')
    svm_error, best_C = calculate_svm_error(Z_input, Y_label, n)
    y = [svm_error] * n
    plt.plot(x, y, color='green')

    readcatalog = "../data/" + dataset + "/X_zeros" + "/decay_choice" + str(decay_choice) + "/contribute_error_rate" + str(contribute_error_rate)
    y = pd.read_csv(readcatalog + '/errors.txt', sep='\t', header=None)
    plt.plot(x, y, color='blue')
    svm_error = np.loadtxt(readcatalog + "/svm_error.txt")
    y = [svm_error] * n
    plt.plot(x, y, color='blue')

    y = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    plt.plot(x, y, color='orange')

    if decay_coef_change == 1:
        ischange = " online copula  change decay_coef"
    else:
        ischange = " online copula  fixed decay_coef"

    plt.title("windowsize:1/" + str(window_size_denominator) + "  batchsize:1/" + str(batch_size_denominator)
              + ischange + "  decay_choice:" + str(decay_choice) + "  contribute_error_rate:" + str(contribute_error_rate))
    plt.savefig(nowcatalog + '/ensemble_errors.png')
    plt.show()
    plt.clf()

    ########画mse图#########
    plt.figure(figsize=(10, 8))
    plt.xlim((0, n))
    plt.ylabel("mse")
    plt.plot(x, mse, color='orange')
    y = pd.read_csv(readcatalog + '/mse.txt', sep='\t', header=None)
    plt.plot(x, y, color='blue')
    y = pd.read_csv(nowcatalog+ '/mse.txt', sep='\t', header=None)
    plt.plot(x, y, color='green')

    plt.title("windowsize:1/" + str(window_size_denominator) + "  batchsize:1/" + str(
        batch_size_denominator) + ischange + "  decay_choice:" + str(
        decay_choice) + "  contribute_error_rate:" + str(contribute_error_rate))
    plt.savefig(nowcatalog + '/mse_ensemble.png')
    plt.show()
    plt.clf()

    return final_error


def ensemble2(n, dataset, X_input, Z_input, Y_label, catalog_Z, window_size_denominator, batch_size_denominator,
             decay_coef_change, decay_choice, contribute_error_rate):
    errors = []
    errors_x_weight=[]
    errors_z_weight=[]

    predict = []
    predict_x_weight=[]
    predict_z_weight=[]

    mse = []
    mse_x_weight=[]
    mse_z_weight=[]

    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 8 * np.sqrt(1 / np.log(n))
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]
        p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y, decay_choice, contribute_error_rate)
        #         print("p_x,loss_x")
        #         print(p_x,loss_x)
        z = Z_input[row]
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)
        #         print("p_z,loss_z")
        #         print(p_z,loss_z)

        p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))
        #        print(str(p)+"="+str(lamda)+"*"+str(p_x)+"+"+str(1-lamda)+"*"+str(p_z))
        p_x_weight = sigmoid(lamda * np.dot(w_x, x))
        p_z_weight = sigmoid((1.0 - lamda) * np.dot(w_z, z))

        # x_loss+=logistic_loss(p_x,y)
        x_loss += loss_x
        #         print("X累积损失，loss")
        #         print(x_loss,logistic_loss(p_x,y))
        # z_loss+=logistic_loss(p_z,y)
        z_loss += loss_z
        #         print("Z累积损失，loss")
        #         print(z_loss,logistic_loss(p_z,y))
        lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))
        #         print("lamda")
        #         print(lamda)

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
        predict.append(p)

        error_x_weight = [int(np.abs(y - p_x_weight) > 0.5)]
        errors_x_weight.append(error_x_weight)
        predict_x_weight.append(p_x_weight)

        error_z_weight = [int(np.abs(y - p_z_weight) > 0.5)]
        errors_z_weight.append(error_z_weight)
        predict_z_weight.append(p_z_weight)

        #print("预测", predict)
        #print("标签", Y_label)
        mse.append(mean_squared_error(predict[:row + 1], Y_label[:row + 1]))
        mse_x_weight.append(mean_squared_error(predict_x_weight[:row + 1], Y_label[:row + 1]))
        mse_z_weight.append(mean_squared_error(predict_z_weight[:row + 1], Y_label[:row + 1]))

    nowcatalog = catalog_Z + "/decay_choice" + str(decay_choice) + "/contribute_error_rate" + str(contribute_error_rate)
    np.savetxt(nowcatalog + '/ensemble_errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog + '/ensemble_mse.txt', mse)
    np.savetxt(nowcatalog + '/ensemble_predict.txt', predict)
    final_error = np.sum(errors) / (n + 0.0)

    np.savetxt(nowcatalog + '/x_weight_errors.txt', np.cumsum(errors_x_weight) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog + '/x_weight_mse.txt', mse_x_weight)
    np.savetxt(nowcatalog + '/x_weight_predict.txt', predict_x_weight)

    np.savetxt(nowcatalog + '/z_weight_errors.txt', np.cumsum(errors_z_weight) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog + '/z_weight_mse.txt', mse_z_weight)
    np.savetxt(nowcatalog + '/z_weight_predict.txt', predict_z_weight)

    ######画错误率图#################
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 1.0))
    plt.ylabel("errors")  # y轴上的名字

    y = pd.read_csv(nowcatalog + '/errors.txt', sep='\t', header=None)
    plt.xlim((0, n))
    x = range(n)
    plt.plot(x, y, color='green')
    svm_error, best_C = calculate_svm_error(Z_input, Y_label, n)
    y = [svm_error] * n
    plt.plot(x, y, color='green')

    readcatalog = "../data/" + dataset + "/X_zeros" + "/decay_choice" + str(
        decay_choice) + "/contribute_error_rate" + str(contribute_error_rate)
    y = pd.read_csv(readcatalog + '/errors.txt', sep='\t', header=None)
    plt.plot(x, y, color='blue')
    svm_error = np.loadtxt(readcatalog + "/svm_error.txt")
    y = [svm_error] * n
    plt.plot(x, y, color='blue')

    y = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    plt.plot(x, y, color='orange')

    y = np.cumsum(errors_z_weight) / (np.arange(len(errors_z_weight)) + 1.0)
    plt.plot(x, y,  '--',color='green')

    y = np.cumsum(errors_x_weight) / (np.arange(len(errors_x_weight)) + 1.0)
    plt.plot(x, y,  '--',color='blue')

    if decay_coef_change == 1:
        ischange = " online copula  change decay_coef"
    else:
        ischange = " online copula  fixed decay_coef"

    plt.title("windowsize:1/" + str(window_size_denominator) + "  batchsize:1/" + str(batch_size_denominator)
              + ischange + "  decay_choice:" + str(decay_choice) + "  contribute_error_rate:" + str(
        contribute_error_rate))
    plt.savefig(nowcatalog + '/ensemble_errors_2.png')
    plt.show()
    plt.clf()

    ########画mse图#########
    plt.figure(figsize=(10, 8))
    plt.xlim((0, n))
    plt.ylabel("mse")
    plt.plot(x, mse, color='orange')
    plt.plot(x, mse_x_weight,'--', color='blue')
    plt.plot(x, mse_z_weight, '--',color='green')
    y = pd.read_csv(readcatalog + '/mse.txt', sep='\t', header=None)
    plt.plot(x, y, color='blue')
    y = pd.read_csv(nowcatalog + '/mse.txt', sep='\t', header=None)
    plt.plot(x, y, color='green')

    plt.title("windowsize:1/" + str(window_size_denominator) + "  batchsize:1/" + str(
        batch_size_denominator) + ischange + "  decay_choice:" + str(
        decay_choice) + "  contribute_error_rate:" + str(contribute_error_rate))
    plt.savefig(nowcatalog + '/mse_ensemble_2.png')
    plt.show()
    plt.clf()

    return final_error

def logistic_loss(p,y):
    return  (1/np.log(2.0))*(-y*np.log(p)-(1-y)*np.log(1-p))

def sigmoid(x):
    #print("集成算法里的sigmoid输入",x)
    return 1 / (1 + np.exp(-x))