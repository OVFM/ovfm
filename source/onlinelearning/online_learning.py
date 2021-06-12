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
from onlinelearning.ftrl_adp2 import FTRL_ADP2
from sklearn import svm 

def svm_classifier(train_x, train_y,test_x, test_y):
    best_score=0
    best_C=-1
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        clf = svm.LinearSVC(C=C,max_iter=100000)
        clf.fit(train_x,train_y)
        score = clf.score(test_x, test_y)
        if score>best_score:
            best_score=score
            best_C=C
    return  best_score,best_C

def calculate_svm_error(X_input, Y_label,n):
    length=int(0.7*n)
    X_train = X_input[:length, :]
    Y_train = Y_label[:length]
    X_test = X_input[length:, :]
    Y_test = Y_label[length:]
    best_score, best_C = svm_classifier(X_train, Y_train, X_test, Y_test)
    #print(best_score)
    #print(best_C)
    error = 1.0 - best_score
    return error,best_C

def generate_X(n,X_input,Y_label,newcatalog,decay_choice,contribute_error_rate):
    errors=[]
    decays=[]
    predict=[]
    mse=[]

    classifier=FTRL_ADP(decay = 1.0, L1=0., L2=0., LP = 1., adaptive =True,n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]
        p, decay,loss,w = classifier.fit(indices, x, y ,decay_choice,contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))
    nowcatalog=newcatalog+"/decay_choice"+str(decay_choice)+"/contribute_error_rate"+str(contribute_error_rate)
    isExists=os.path.exists(nowcatalog)
    if not isExists:
        os.makedirs(nowcatalog)
    np.savetxt(nowcatalog+'/errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog+'/dacays.txt', decays)
    np.savetxt(nowcatalog+'/mse.txt', mse)
    np.savetxt(nowcatalog+'/predict.txt', predict)
    svm_error,best_C=calculate_svm_error(X_input[:,1:], Y_label, n)
    np.savetxt(nowcatalog+'/svm_error.txt', [svm_error])
    
    
def generate(n,dataset,X_input,Y_label,newcatalog,window_size_denominator,batch_size_denominator,decay_coef_change,decay_choice,contribute_error_rate):

    errors=[]
    decays=[]
    predict=[]
    mse=[]

    classifier=FTRL_ADP(decay = 1.0, L1=0., L2=0., LP = 1., adaptive =True,n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row].data
        y = Y_label[row]
        p, decay,loss,w= classifier.fit(indices, x, y ,decay_choice,contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))
    nowcatalog=newcatalog+"/decay_choice"+str(decay_choice)+"/contribute_error_rate"+str(contribute_error_rate)
    isExists=os.path.exists(nowcatalog)
    if not isExists:
        os.makedirs(nowcatalog)
    np.savetxt(nowcatalog+'/errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog+'/dacays.txt', decays)
    np.savetxt(nowcatalog+'/mse.txt', mse)
    np.savetxt(nowcatalog+'/predict.txt', predict)
    final_error=np.sum(errors)/(n+0.0)
    
    #readcatalog="../data/"+dataset+"/X_zeros"+"/decay_choice"+str(decay_choice)+"/contribute_error_rate"+str(contribute_error_rate)
    readcatalog ="X_zeros" + "/decay_choice" + str(decay_choice) + "/contribute_error_rate" + str(contribute_error_rate)
    ######画错误率图#################
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 1.0))
    plt.ylabel("errors")  # y轴上的名字

    y = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    plt.xlim((0, n))
    x = range(n)
    plt.plot(x, y, color='green')

    svm_error,best_C=calculate_svm_error(X_input[:,1:], Y_label, n)
    y = [svm_error] * n
    plt.plot(x, y, color='green')
    
    y = pd.read_csv(readcatalog+'/errors.txt',sep='\t',header=None)
    plt.plot(x,y,color='blue')
    
    svm_error=np.loadtxt(readcatalog+"/svm_error.txt")
    y = [svm_error] * n
    plt.plot(x, y, color='blue')
    
    if decay_coef_change==1:
        ischange=" online copula  change decay_coef"
    else:
        ischange=" online copula  fixed decay_coef"

    plt.title("windowsize:1/"+str(window_size_denominator)+"  batchsize:1/"+str(batch_size_denominator)+ischange+"  decay_choice:"+str(decay_choice)+"  contribute_error_rate:"+str(contribute_error_rate))
    plt.savefig(nowcatalog+'/errors.png')
    plt.show()
    plt.clf()
    ########画mse图#########
    plt.figure(figsize=(10,8))
    plt.xlim((0,n))
    plt.ylabel("mse")
    plt.plot(x,mse,color='green')
    y = pd.read_csv(readcatalog+'/mse.txt',sep='\t',header=None)
    plt.plot(x,y,color='blue')
    
    plt.title("windowsize:1/"+str(window_size_denominator)+"  batchsize:1/"+str(batch_size_denominator)+ischange+"  decay_choice:"+str(decay_choice)+"  contribute_error_rate:"+str(contribute_error_rate))
    plt.savefig(nowcatalog+'/mse.png')
    plt.show()
    plt.clf()
            
    ########画差值图########
    plt.figure(figsize=(10,8))
    plt.xlim((0,n))
    plt.ylabel("Y_label-predict")
    differ_Z=Y_label-predict
    plt.plot(x,differ_Z,'o',markersize=2.,color='green')
    plt.title("windowsize:1/"+str(window_size_denominator)+"  batchsize:1/"+str(batch_size_denominator)+ischange+"  decay_choice:"+str(decay_choice)+"  contribute_error_rate:"+str(contribute_error_rate))
    plt.savefig(nowcatalog+'/differ.png')
    plt.show()
    plt.clf()
             
    return final_error




def generate2(n,dataset,X_input,Y_label,newcatalog,window_size_denominator,batch_size_denominator,decay_coef_change,decay_choice,contribute_error_rate):

    errors=[]
    decays=[]
    predict=[]
    mse=[]

    classifier=FTRL_ADP(decay = 1.0, L1=0., L2=0., LP = 1., adaptive =True,n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row].data
        y = Y_label[row]
        p, decay,loss =classifier.fit(indices, x, y ,decay_choice,contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))
    nowcatalog=newcatalog+"/decay_choice"+str(decay_choice)+"/contribute_error_rate"+str(contribute_error_rate)
    isExists=os.path.exists(nowcatalog)
    if not isExists:
        os.makedirs(nowcatalog)
    np.savetxt(nowcatalog+'/errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog+'/dacays.txt', decays)
    np.savetxt(nowcatalog+'/mse.txt', mse)
    np.savetxt(nowcatalog+'/predict.txt', predict)
    final_error=np.sum(errors)/(n+0.0)
    
    readcatalog="../dataset/MaskData/"+dataset+"/X"
    ######画错误率图#################
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 1.0))
    plt.ylabel("errors")  # y轴上的名字

    y = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    plt.xlim((0, n))
    x = range(n)
    plt.plot(x, y, color='green')

    
    y = pd.read_csv(readcatalog+'/errors.txt',sep='\t',header=None)
    plt.plot(x,y,color='blue')
    
    if decay_coef_change==1:
        ischange=" online copula  change decay_coef"
    else:
        ischange=" online copula  fixed decay_coef"

    plt.title("windowsize:1/"+str(window_size_denominator)+"  batchsize:1/"+str(batch_size_denominator)+ischange+"  decay_choice:"+str(decay_choice)+"  contribute_error_rate:"+str(contribute_error_rate))
    plt.savefig(nowcatalog+'/errors.png')
    plt.show()
    plt.clf()
             
    return final_error


def generate3(n, dataset, X_input, Y_label, newcatalog, window_size_denominator, batch_size_denominator,
              decay_coef_change, decay_choice, contribute_error_rate):
    errors = []
    decays = []
    predict = []
    mse = []

    classifier = FTRL_ADP2(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label[row]
        p, decay, loss ,w= classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        #print("w",w)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row + 1], Y_label[:row + 1]))
    nowcatalog = newcatalog + "/decay_choice" + str(decay_choice) + "/contribute_error_rate" + str(
        contribute_error_rate)
    isExists = os.path.exists(nowcatalog)
    if not isExists:
        os.makedirs(nowcatalog)
    np.savetxt(nowcatalog + '/errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog + '/dacays.txt', decays)
    np.savetxt(nowcatalog + '/mse.txt', mse)
    np.savetxt(nowcatalog + '/predict.txt', predict)
    final_error = np.sum(errors) / (n + 0.0)

    readcatalog = "../dataset/MaskData/" + dataset + "/X" # update
    ######画错误率图#################
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 1.0))
    plt.ylabel("errors")  # y轴上的名字

    y = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    plt.xlim((0, n))
    x = range(n)
    plt.plot(x, y, color='green')

    y = pd.read_csv(readcatalog + '/errors.txt', sep='\t', header=None)
    plt.plot(x, y, color='blue')

    if decay_coef_change == 1:
        ischange = " online copula  change decay_coef"
    else:
        ischange = " online copula  fixed decay_coef"

    plt.title("windowsize:1/" + str(window_size_denominator) + "  batchsize:1/" + str(
        batch_size_denominator) + ischange + "  decay_choice:" + str(decay_choice) + "  contribute_error_rate:" + str(
        contribute_error_rate))
    plt.savefig(nowcatalog + '/errors.png')
    plt.show()
    plt.clf()

    return final_error

def generate4(n, dataset, X_input, Y_label, newcatalog, window_size_denominator, batch_size_denominator,
              decay_coef_change, decay_choice, contribute_error_rate):
    errors = []
    decays = []
    predict = []
    mse = []

    classifier = FTRL_ADP2(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(X_input[-1])+1)
    for row in range(n):
        indices = [i for i in range(len(X_input[row])+1)]
        x = np.array([1]+X_input[row]).data
        y = Y_label[row]
        p, decay, loss ,w= classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        #print("w",w)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row + 1], Y_label[:row + 1]))
    nowcatalog = newcatalog + "/decay_choice" + str(decay_choice) + "/contribute_error_rate" + str(
        contribute_error_rate)
    isExists = os.path.exists(nowcatalog)
    if not isExists:
        os.makedirs(nowcatalog)
    np.savetxt(nowcatalog + '/errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(nowcatalog + '/dacays.txt', decays)
    np.savetxt(nowcatalog + '/mse.txt', mse)
    np.savetxt(nowcatalog + '/predict.txt', predict)
    final_error = np.sum(errors) / (n + 0.0)

    readcatalog = "../data2/" + dataset + "/X_trapezoid"
    ######画错误率图#################
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 1.0))
    plt.ylabel("errors")  # y轴上的名字

    y = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    plt.xlim((0, n))
    x = range(n)
    plt.plot(x, y, color='green')

    y = pd.read_csv(readcatalog + '/errors.txt', sep='\t', header=None)
    plt.plot(x, y, color='blue')

    if decay_coef_change == 1:
        ischange = " online copula  change decay_coef"
    else:
        ischange = " online copula  fixed decay_coef"

    plt.title("windowsize:1/" + str(window_size_denominator) + "  batchsize:1/" + str(
        batch_size_denominator) + ischange + "  decay_choice:" + str(decay_choice) + "  contribute_error_rate:" + str(
        contribute_error_rate))
    plt.savefig(nowcatalog + '/errors.png')
    plt.show()
    plt.clf()

    return final_error

def generate5(n, dataset, X_input, Y_label, newcatalog, window_size_denominator, batch_size_denominator,
              decay_coef_change, decay_choice, contribute_error_rate):
    errors = []
    decays = []
    predict = []
    mse = []

    classifier = FTRL_ADP2(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(X_input[-1])+1)
    for row in range(n):
        indices = [i for i in range(len(X_input[row])+1)]
        x = np.array([1]+X_input[row]).data
        y = Y_label[row]
        p, decay, loss ,w= classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        #print("w",w)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row + 1], Y_label[:row + 1]))
    np.savetxt(newcatalog + '/trapezoid_errors.txt', np.cumsum(errors) / (np.arange(len(errors)) + 1.0))
    np.savetxt(newcatalog + '/trapezoid_dacays.txt', decays)
    np.savetxt(newcatalog + '/trapezoid_mse.txt', mse)
    np.savetxt(newcatalog + '/trapezoid_predict.txt', predict)
    final_error = np.sum(errors) / (n + 0.0)
    return final_error