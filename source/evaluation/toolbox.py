import numpy as np
import pandas as pd

def DictCompare(W, X, op = 'intersection'):
    '''a and b are two dictionaries, 3 types of operations including:
        1. return the np array includes the features belong to W and intersects with X
        2. return the np array includes the features that X has but W not
        3. return the np array includes the features that W has but X not '''
    if op == 'intersection':
        shared_W_keys = list(W.keys() & X.keys())
        shared_W_keys.sort()
        W = DictToList(W, shared_W_keys)
        X = DictToList(X, shared_W_keys, WorX='X')
        return shared_W_keys, W, X # keys and corresponding values
    if op == 'extraX':
        extra_x_keys = list(set(x_keys) - set(W_keys))
        extra_x_keys.sort()
        extra_x_values = DictToList(X, extra_x_keys, WorX='X')
        return extra_x_keys, extra_x_values
    if op == 'extraW':
        extra_W_keys = np.sort(np.array(list(W.keys() - X.keys())))
        extra_W_values = DictToList(W, extra_W_keys)
        return extra_W_keys, extra_W_values

def DictToList(dict, indexList = None, WorX = 'weight'):
    #df = pd.DataFrame(dict, index=[0])
    df = pd.DataFrame(dict)
    if indexList is not None:
        df_sel = df[indexList]
        if WorX != 'weight':
            return df_sel.values.reshape(1, -1)
        else:
            return df_sel.values.reshape(-1, 1)
    else:
        if WorX != 'weight':
            return df.values.reshape(1, -1)
        else:
            return df.values.reshape(-1, 1)

def MatrixInDict(matrix, dict):
    matrixTemp = matrix.copy()
    '''Always take the full feature space as the dimension of mapped vector'''
    for (r,row) in enumerate(matrix):
        for (c,col) in enumerate(row):
            if dict.get(r) is not None:
                key_new_feature = dict.get(r)
                if key_new_feature is not None:
                    key_new_to_all = key_new_feature.get(c)
                    matrixTemp[r, c] = key_new_to_all
    return matrixTemp

def MatToNestedDict(matrix, row_index, col_index, dict):
    mat = matrix.copy()
    #for in zip() 组队并一一配对
    for x_k, row in zip(row_index, matrix):
        tempDict = {k:v for k, v in zip(col_index, row)}
        tempDict2 = {x_k: tempDict}
        dict.update(tempDict2)

def FindInDict(row_index, col_index, NestDict):
    mat = np.zeros((len(row_index), len(col_index)))
    for (row,i) in enumerate(row_index):
        if NestDict.get(i) is not None:
            dictTemp = NestDict.get(i)
            for (col,j) in enumerate(col_index):
                if dictTemp.get(j) is not None:
                    mat[row][col] = dictTemp.get(j)
    return mat


def IndicatorMatrix(row_index, col_index):
    idDict = {}
    idMat = np.identity(len(list(row_index)))
    MatToNestedDict(idMat, row_index, row_index, idDict)
    for col in col_index:
        if idDict.get(col) is not None:
            for c in col_index:
                if idDict.get(col).get(c) is None:
                    idDict[col][c] = 0
    ZeroMat = np.zeros((len(list(row_index)), len(list(col_index))))
    Indicator = FindInDict(row_index, col_index, idDict)
    return Indicator
import math
def __loss( y, x):
    print("输入x：" + str(x))
    print()
    print (y * np.log(1 + np.exp(-x)) + (1 - y) * np.log(1 + np.exp(x))) * (1 / np.log(2.0))
    return max(x,0)-y*x+np.log(1+np.exp(-abs(x)))


if __name__=="__main__":
    __loss(1.0,1000.0)
