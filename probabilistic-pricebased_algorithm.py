from statistics import mean
import random
import math
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal as mvn
from sklearn import preprocessing as pre
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
import os


def weights(K,risk):
    weight = np.zeros(K.shape[1])
    for i in range(K.shape[1]):
        if i == 0:
            weight[i] = 0
        elif K[1,i] > K[1,0]:
            weight[i] = 0
        elif K[1,i] < K[1,0] and K[0,i] > K[0,0] - risk:
            weight[i] = (K[1,0] - K[1,i])/(K[0,0] - K[0,i])
        else:
            weight[i] = 0
            
    if np.sum(weight) == 0:
        return 0
    
    if np.any(np.isnan(weight)):
        return 0
    
    if not np.isfinite(weight).all():
        return 0
    
    weight = weight / np.sum(weight)
    x = np.random.choice(range(len(weight)), p=weight)
    return x

def get_corr(X):
    r = np.corrcoef(X[0,:], X[1,:])
    return r[0,1]


def weights_corr_noisy(K, U, risk, t_p, B, R):
    rho = get_corr(K)
    
    if rho > t_p:
        tag, pred = linear_regress(K, U, risk, B, R)
        x = np.where(U[1,:] == tag)[0][0]
        return x, 1, pred
    else:
        return 0, 0, 0

def linear_regress(K, U, risk, B, R):
    x = K[1,:].reshape((-1,1))
    y = K[0,:]
    model = LinearRegression()
    model.fit(x,y)
    unknown_prices = U[1,:]
    unknown_affordable = unknown_prices[(unknown_prices <= B - R)]
    if len(unknown_affordable) == 0:
        return U.tolist().index(0), 0
    y_pred = np.maximum(0, model.predict(unknown_affordable.reshape((-1,1))))
    y_pred = np.transpose(y_pred)
    max_acc = max(K[0,:])
    
    t = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        if max_acc - y_pred[i] > risk:
            t[i] = 0
        else:
            t[i] = y_pred[i]/(unknown_affordable[i]+ np.finfo(float).eps)
            
    
    if np.any(t == 0) or np.any(np.isnan(t)) or np.any(t < 0):
        j = np.argwhere(y_pred == np.max(y_pred))[0]
        pred = y_pred[j]
        tag = unknown_affordable[j]
    else:
        t = t / np.sum(t)
        j = np.random.choice(range(len(t)), 1, p = t)
        pred = y_pred[j]
        tag = unknown_affordable[j]
        
    
    return tag, pred



def probabilistic_pricebased(K,U,B,R,risk):
    K = K[:, np.argsort(K[0])[::-1]]
    U = U[:, np.argsort(U[1])]
    best_dataset = K[:,weights(K,risk)]
    t_p = 0
    best_accuracy = best_dataset[0]
    reserved_price = best_dataset[1]
    datasets_explored = 0
    
    while B - reserved_price >= R and np.any((U[1,:] < (B - R))): 
        j, use_regression, pred = weights_corr_noisy(K,U,risk,t_p,B,R)
        
        if use_regression:
            if U[0,j] - pred > risk:
                t_p = min(1, t_p + 0.1)
            else if U[0,j] - pred <= risk:
                t_p = max(0, t_p - 0.1)
            
            
        B -= R
        datasets_explored += 1
        if U[0,j] > best_accuracy and U[1,j] <= B:
            best_dataset = U[:,j]
            best_accuracy = best_dataset[0]
            reserved_price = best_dataset[1]
        
        dataset_unveiled = U[:,j].copy()
        K = np.hstack((K, dataset_unveiled[:, np.newaxis]))
        U = np.delete(U, j, axis=1)
            
    return best_accuracy, datasets_explored
