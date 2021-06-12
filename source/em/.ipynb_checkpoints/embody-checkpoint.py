import numpy as np
from scipy.stats import norm, truncnorm

def _em_step_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
    """
    return _em_step_body(*args)

def _em_step_body(Z, r_lower, r_upper, sigma, num_ord_updates=1):
    """
    Iterate the rows over provided matrix
    """
    num, p = Z.shape
    Z_imp = np.copy(Z)
    C = np.zeros((p,p))
    for i in range(num):
        #c, z_imp, z = _em_step_body_row(Z[i, :], r_lower[i, :], r_upper[i, :], sigma)  #########!!!!!!!!!!
        #    每行做一次更新
        try:
            c, z_imp, z = _em_step_body_row(Z[i,:], r_lower[i,:], r_upper[i,:], sigma)#########!!!!!!!!!!
        except:
            np.savetxt("Z.txt", Z)
            print(Z)
        Z_imp[i,:] = z_imp
        Z[i,:] = z
        C += c
    return C, Z_imp, Z


def _em_step_body_row(Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates=1):
    """
    The body of the em algorithm for each row
    Returns a new latent row, latent imputed row and C matrix, which, when added
    to the empirical covariance gives the expected covariance
    每一行的em算法的主体返回一个新的潜在行、潜在的插补行和C矩阵，当将其添加到经验协方差中时，将得到期望的协方差

    Args:
        Z_row (array): (potentially missing) latent entries for one data point
        r_lower_row (array): (potentially missing) lower range of ordinal entries for one data point
        r_upper_row (array): (potentially missing) upper range of ordinal entries for one data point
        sigma (matrix): estimate of covariance
        num_ord (int): the number of ordinal columns

    Returns:
        C (matrix): results in the updated covariance when added to the empircal covariance
        Z_imp_row (array): Z_row with latent ordinals updated and missing entries imputed 
        Z_row (array): inpute Z_row with latent ordinals updated
    """
    Z_imp_row = np.copy(Z_row)#先将原始的那行数据复制过来
    p = Z_imp_row.shape[0]#p是那一行有多少列
    num_ord = r_upper_row.shape[0]#num_ord是序数值的个数
    C = np.zeros((p,p))#C矩阵初始化大小是特征值数目*特征值数目，值为0
    # 观测值的下标
    obs_indices = np.where(~np.isnan(Z_row))[0] ## doing search twice for basically same thing?
    # 缺失值的下标（明明除了观测值都是缺失值啊，为啥要这么操作呢，可能为了节省运算时间）
    missing_indices = np.setdiff1d(np.arange(p), obs_indices) ## Use set difference to avoid another searching
    ord_in_obs = np.where(obs_indices < num_ord)[0]#序数值在观察到的数据里的下标
    ord_obs_indices = obs_indices[ord_in_obs]#观测到的序数值在完整特征空间里的下标
    # obtain correlation sub-matrices
    # obtain submatrices by indexing a "cartesian-product" of index arrays
    sigma_obs_obs = sigma[np.ix_(obs_indices,obs_indices)]#返回sigma矩阵中行号、列号分别为obs_indices,obs_indices的元素
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]#返回sigma矩阵中行号、列号分别为obs_indices,missing_indices的元素
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]#返回sigma矩阵中行号、列号分别为missing_indices的元素
    # print('sigma_obs_obs', sigma_obs_obs)
    # print('len', len(sigma_obs_obs))
    # print('sigma_obs_missing', sigma_obs_missing)

    if len(missing_indices) > 0:
        tot_matrix = np.concatenate((np.identity(len(sigma_obs_obs)), sigma_obs_missing), axis=1)#将大小为sigma_obs_obs的单位矩阵与sigma_obs_missing拼接起来
        intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)#返回方程组sigma_obs_obs*x=tot_matrix的解，shape与tot_matrix一样
        sigma_obs_obs_inv = intermed_matrix[:, :len(sigma_obs_obs)]#矩阵sigma_obs_obs的逆
        J_obs_missing = intermed_matrix[:, len(sigma_obs_obs):]#返回方程组sigma_obs_obs*x=sigma_obs_missing的解
    else:
        sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))  # 矩阵sigma_obs_obs的逆
        # try:
        #     sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))#矩阵sigma_obs_obs的逆
        # except:
        #     print("Z_row")
        #     print(Z_row)
        #     print("sigma")
        #     print(sigma)
        #     print("sigma_obs_obs")
        #     print(sigma_obs_obs)
    # initialize vector of variances for observed ordinal dimensions
    var_ordinal = np.zeros(p)

    # OBSERVED ORDINAL ELEMENTS
    # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
    #当有一个观察到的序数要插补和另一个观察到的维度时，插补这个序数
    #能观察到的数据大于等于2，能观察到的序数大于等于1
    if len(obs_indices) >= 2 and len(ord_obs_indices) >= 1:
        for update_iter in range(num_ord_updates):
            # used to efficiently compute conditional mean
            sigma_obs_obs_inv_Z_row = np.dot(sigma_obs_obs_inv, Z_row[obs_indices])
            for ind in range(len(ord_obs_indices)):
                j = obs_indices[ind]
                not_j_in_obs = np.setdiff1d(np.arange(len(obs_indices)),ind) 
                v = sigma_obs_obs_inv[:,ind]
                new_var_ij = np.asscalar(1.0/v[ind])#np.asscalar将向量转换成标量
                #new_mean_ij = np.dot(v[not_j_in_obs], Z_row[obs_indices[not_j_in_obs]]) * (-new_var_ij)
                new_mean_ij = Z_row[j] - new_var_ij*sigma_obs_obs_inv_Z_row[ind]
                mean, var = truncnorm.stats(#截断分布是指，限制变量x 取值范围(scope)的一种分布
                    a=(r_lower_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    b=(r_upper_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    loc=new_mean_ij,
                    scale=np.sqrt(new_var_ij),
                    moments='mv')
                if np.isfinite(var):
                    var_ordinal[j] = var
                    if update_iter == num_ord_updates - 1:
                        C[j,j] = C[j,j] + var 
                if np.isfinite(mean):
                    Z_row[j] = mean

    # MISSING ELEMENTS
    Z_obs = Z_row[obs_indices]
    Z_imp_row[obs_indices] = Z_obs
    if len(missing_indices) > 0:
        Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T,Z_obs) 
        # variance expectation and imputation
        if len(ord_obs_indices) >= 1 and len(obs_indices) >= 2 and np.sum(var_ordinal) > 0: 
            cov_missing_obs_ord = J_obs_missing[ord_in_obs].T * var_ordinal[ord_obs_indices]
            C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
            C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing) \
                                                           + np.matmul(cov_missing_obs_ord, J_obs_missing[ord_in_obs])
        else:
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
    return C, Z_imp_row, Z_row