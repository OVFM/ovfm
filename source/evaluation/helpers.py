import numpy as np

def get_cont_indices(X):
    max_ord=14
    indices = np.zeros(X.shape[1]).astype(bool)
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) > max_ord:
            indices[i] = True
    return indices

def cont_to_binary(x):
    # make the cutoff a random sample and ensure at least 10% are in each class
    while True:
        cutoff = np.random.choice(x)    #从x中挑选的一个随机数返回，作为阈值
        if len(x[x < cutoff]) > 0.1*len(x) and len(x[x < cutoff]) < 0.9*len(x):
            break
    return (x > cutoff).astype(int)#astype函数改变数组中的元素类型

def cont_to_ord(x, k):
    # make the cutoffs based on the quantiles
    #if k == 2:
        #return cont_to_binary(x)
    std_dev = np.std(x)
    cuttoffs = np.linspace(np.min(x), np.max(x), k+1)[1:]
    ords = np.zeros(len(x))
    for cuttoff in cuttoffs:
        ords += (x > cuttoff).astype(int)
    return ords.astype(int)



def get_mae(x_imp, x_true, x_obs=None):
    if x_obs is not None:
        loc = np.isnan(x_obs)
        imp = x_imp[loc]
        val = x_true[loc]
        return np.mean(np.abs(imp - val))
    else:
        return np.mean(np.abs(x_imp - x_true))
        

def get_smae(x_imp, x_true, x_obs, Med=None, per_type=False, cont_loc=None, bin_loc=None, ord_loc=None):
    error = np.zeros((x_obs.shape[1],2))
    for i, col in enumerate(x_obs.T):
        test = np.bitwise_and(~np.isnan(x_true[:,i]), np.isnan(col))
        if np.sum(test) == 0:
            error[i,0] = np.nan
            error[i,1] = np.nan
            continue
        col_nonan = col[~np.isnan(col)]
        x_true_col = x_true[test,i]
        x_imp_col = x_imp[test,i]
        if Med is not None:
            median = Med[i]
        else:
            median = np.median(col_nonan)
        diff = np.abs(x_imp_col - x_true_col)
        med_diff = np.abs(median - x_true_col)
        error[i,0] = np.sum(diff)
        error[i,1]= np.sum(med_diff)
    if per_type:
        if not cont_loc:
            cont_loc = [True] * 5 + [False] * 10
        if not bin_loc:
            bin_loc = [False] * 5 + [True] * 5 + [False] * 5 
        if not ord_loc:
            ord_loc = [False] * 10 + [True] * 5
        loc = [cont_loc, bin_loc, ord_loc]
        scaled_diffs = np.zeros(3)
        for j in range(3):
            scaled_diffs[j] = np.sum(error[loc[j],0])/np.sum(error[loc[j],1])
    else:
        scaled_diffs = error[:,0] / error[:,1]
    return scaled_diffs

def get_smae_per_type(x_imp, x_true, x_obs, cont_loc=None, bin_loc=None, ord_loc=None):
    if not cont_loc:
        cont_loc = [True] * 5 + [False] * 10
    if not bin_loc:
        bin_loc = [False] * 5 + [True] * 5 + [False] * 5 
    if not ord_loc:
        ord_loc = [False] * 10 + [True] * 5
    loc = [cont_loc, bin_loc, ord_loc]
    scaled_diffs = np.zeros(3)
    for j in range(3):
        missing = np.isnan(x_obs[:,loc[j]])
        med = np.median(x_obs[:,loc[j]][~missing])
        diff = np.abs(x_imp[:,loc[j]][missing] - x_true[:,loc[j]][missing])
        med_diff = np.abs(med - x_true[:,loc[j]][missing])
        scaled_diffs[j] = np.sum(diff)/np.sum(med_diff)
    return scaled_diffs
    
def get_smae_per_type_online(x_imp, x_true, x_obs, Med):
    for i, col in enumerate(x_obs.T):
        missing = np.isnan(col)
        x_true_col = x_true[np.isnan(col),i]
        x_imp_col = x_imp[np.isnan(col),i]
        median = Med[i]
        diff = np.abs(x_imp_col - x_true_col)
        med_diff = np.abs(median - x_true_col)
        scaled_diffs[i] = np.sum(diff)/np.sum(med_diff)
    return scaled_diffs


def get_rmse(x_imp, x_true, relative=False):
    diff = x_imp - x_true
    mse = np.mean(diff**2.0, axis=0)
    rmse = np.sqrt(mse)
    return rmse if not relative else rmse/np.sqrt(np.mean(x_true**2))

def get_relative_rmse(x_imp, x_true, x_obs):
    loc = np.isnan(x_obs)
    imp = x_imp[loc]
    val = x_true[loc]
    return get_scaled_error(imp, val)

def get_scaled_error(sigma_imp, sigma):
    return np.linalg.norm(sigma - sigma_imp) / np.linalg.norm(sigma)

def mask_types(X, mask_num, seed):
    X_masked = np.copy(X)
    mask_indices = []
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i) # uncertain if this is necessary
        for j in range(num_cols//2):
            rand_idx=np.random.choice(2,mask_num,False)
            for idx in rand_idx:
                X_masked[i,idx+2*j]=np.nan
                mask_indices.append((i, idx+2*j))
    return X_masked

def mask(X, mask_fraction, seed=0, verbose=False):
    complete = False
    count = 0
    X_masked = np.copy(X) 
    obs_indices = np.argwhere(~np.isnan(X))
    total_observed = len(obs_indices)
    while not complete:
        np.random.seed(seed)
        if (verbose): print(seed)
        mask_indices = obs_indices[np.random.choice(len(obs_indices), size=int(mask_fraction*total_observed), replace=False)]
        for i,j in mask_indices:
            X_masked[i,j] = np.nan
        complete = True
        for row in X_masked:
            if len(row[~np.isnan(row)]) == 0:
                seed += 1
                count += 1
                complete = False
                X_masked = np.copy(X)
                break
        if count == 50:
            raise ValueError("Failure in Masking data without empty rows")
    return X_masked, mask_indices, seed

def mask_per_row(X, seed=0, size=1):
    X_masked = np.copy(X)
    n,p = X.shape
    for i in range(n):
        np.random.seed(seed*n+i)
        rand_idx = np.random.choice(p, size)
        X_masked[i,rand_idx] = np.nan
    return X_masked

def _project_to_correlation(covariance):
        """
        Projects a covariance to a correlation matrix, normalizing it's diagonal entries

        Args:
            covariance (matrix): a covariance matrix

        Returns:
            correlation (matrix): the covariance matrix projected to a correlation matrix
        """
        D = np.diagonal(covariance)
        D_neg_half = np.diag(1.0/np.sqrt(D))
        return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def generate_sigma(seed):
    np.random.seed(seed)
    W = np.random.normal(size=(18, 18))
    covariance = np.matmul(W, W.T)#矩阵相乘
    D = np.diagonal(covariance)#D为covariance矩阵的对角线元素
    D_neg_half = np.diag(1.0/np.sqrt(D))#将D中每个元素开根号并取逆，以此作为对角线元素生成对角线矩阵
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def generate_LRGC(rank, sigma, n=500, p_seq=(100,100,100), ord_num=5, cont_type = 'LR', seed=1):
    cont_indices = None
    bin_indices = None
    ord_indices = None
    if p_seq[0] > 0:
        cont_indices = range(p_seq[0])
    if p_seq[1] > 0:
        ord_indices = range(p_seq[0],p_seq[0] + p_seq[1])
    if p_seq[2] > 0:
        bin_indices = range(p_seq[0] + p_seq[1], p_seq[0] + p_seq[1] + p_seq[2])
    p = np.sum(p_seq)
    np.random.seed(seed)
    W = np.random.normal(size=(p,rank))
    # TODO: check everything of this form with APPLY
    for i in range(W.shape[0]):
        W[i,:] = W[i,:]/np.sqrt(np.sum(np.square(W[i,:]))) * np.sqrt(1 - sigma)
    Z = np.dot(np.random.normal(size=(n,rank)), W.T) + np.random.normal(size=(n,p), scale=np.sqrt(sigma))
    X_true = Z
    if cont_indices is not None:
        if cont_type != 'LR':
            X_true[:,cont_indices] = X_true[:,cont_indices]**3
    if bin_indices is not None:
        for bin_index in bin_indices:
            X_true[:,bin_index] = continuous2ordinal(Z[:,bin_index], k=2)
    if ord_indices is not None:
        for ord_index in ord_indices:
            X_true[:,ord_index] = continuous2ordinal(Z[:,ord_index], k=ord_num)
    return X_true, W


def continuous2ordinal(x, k = 2, cutoff = None):
    q = np.quantile(x, (0.05,0.95))
    if k == 2:
        if cutoff is None:
            # random cuttoff from the data between the 5th and 95th percentile
            cutoff = np.random.choice(x[(x > q[0])*(x < q[1])])
        x = (x >= cutoff).astype(int)
    else:
        if cutoff is None:
            std_dev = np.std(x)
            min_cutoff = np.min(x) - 0.1 * std_dev
            cutoff = np.sort(np.random.choice(x[(x > q[0])*(x < q[1])], k-1, False))
            max_cutoff = np.max(x) + 0.1 * std_dev
            cuttoff = np.hstack((min_cutoff, cutoff, max_cutoff))
        x = np.digitize(x, cuttoff)
    return x

def grassman_dist(A,B):
    U1, d1, _ = np.linalg.svd(A, full_matrices = False)
    U2, d2, _ = np.linalg.svd(B, full_matrices = False)
    _, d,_ = np.linalg.svd(np.dot(U1.T, U2))
    theta = np.arccos(d)
    return np.linalg.norm(theta), np.linalg.norm(d1-d2)

def get_hyperparameter(dataset):
    if dataset == "australian":
        contribute_error_rate = 0.005
        window_size_denominator = 4
        batch_size_denominator = 8
        decay_coef_change = 0
        decay_choice = 3
        shuffle = False
    elif dataset == "ionosphere":
        contribute_error_rate = 0
        window_size_denominator = 4
        batch_size_denominator = 8
        decay_coef_change = 0
        decay_choice = 4
        shuffle = False
    elif dataset == "german":
        contribute_error_rate = 0.005
        window_size_denominator = 2
        batch_size_denominator = 8
        decay_coef_change = 0
        decay_choice = 4
        shuffle = False
    elif dataset == "diabetes":
        contribute_error_rate = 0.02
        window_size_denominator = 2
        batch_size_denominator = 8
        decay_coef_change = 0
        decay_choice = 0
        shuffle = True
    elif dataset == "wdbc":
        contribute_error_rate = 0
        window_size_denominator = 2
        batch_size_denominator = 8
        decay_coef_change = 0
        decay_choice = 4
        shuffle = False
    elif dataset == "credit":
        contribute_error_rate = 0.02
        window_size_denominator = 2
        batch_size_denominator = 8
        decay_coef_change = 0
        decay_choice = 3
        shuffle = True
    return contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,shuffle
