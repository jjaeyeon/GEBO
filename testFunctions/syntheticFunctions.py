
import os
import socket

import numpy as np

from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_boston, fetch_openml, load_wine
from sklearn.neural_network import MLPClassifier 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from testFunctions.robot_push_14d import push_function 


def func2C(ht_list, X):
    # ht is a categorical index
    # X is a continuous variable
    X = X * 2

    assert len(ht_list) == 2
    ht1 = ht_list[0]
    ht2 = ht_list[1]

    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:  # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:  # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:  # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)

    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
    return y.astype(float)

def ackleycC(ht_list, X):
    z = np.concatenate([X, np.array(ht_list) * 0.125 - 1])
    a, b, c = 20, 0.2, 2 * np.pi
    f = -a * np.exp(-b * np.mean(z ** 2) ** 0.5) - np.exp(np.mean(np.cos(c * z))) + a + np.exp(1)

    y = f + 1e-6 * np.random.rand()

    return y.astype(float)/3

def pressure_vessel(ht_list, X):
    x = [None for _ in range(4)]
    x[0] = ht_list[0] + 1
    x[1] = ht_list[1] + 1
    x[2] = 10 + ((X[0] + 1) * (200 - 10))/2
    x[3] = 10 + ((X[1] + 1) * (240 - 10))/2
    
    func_value = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * (x[2] ** 2) + 3.1661 * (x[0] ** 2) * x[3] + 19.84 * (x[0] ** 2) * x[2]
    constraint_value = 0
    constraint_1 = x[0] - 0.0193*x[2]
    constraint_2 = x[1] - 0.00954*x[2]
    constraint_3 = np.pi*(x[2]**2)*x[3] + (4/3)*np.pi*(x[2] ** 3) - 1296000
    constraint_cost = (constraint_1 < 0) * 100 + (constraint_2 < 0) * 100 + (constraint_3 < 0) * 100
#     print(f"func_value : {func_value/1e6}")
    return func_value/1e6

def speed_reducer(ht_list, X):
    x = []
    x.append(2.6 + (X[0]+1)/2)
    x.append(0.7 + ((X[1]+1) * 0.1)/2)
    x.append(ht_list[0] + 17)
    x.append(7.3 + (X[2]+1)/2)
    x.append(7.3 + (X[3]+1)/2)
    x.append(2.9 + (X[4]+1)/2)
    x.append(5 + ((X[5] + 1) * 0.5)/2)
    func_value = 0.7854 * x[0] * (x[1]**2) * (3.333*(x[2]**2) + 14.9334*(x[2]) - 43.0934) \
                    - 1.508 * x[0] * (x[5]**2 + x[6]**2) + 7.4777 * (x[5] ** 3 + x[6] ** 3) \
                    + 0.7854 * (x[3] * (x[5] ** 2) + x[4] * (x[6] ** 2))
    constraint_cost = 0
    if (27/(x[0] * (x[1] ** 2) * x[2]) >=1):
        constraint_cost += 1000
    if (397.5/(x[0] * (x[1] ** 2) * (x[2]**2)) >= 1):
        constraint_cost += 1000
    if (1.93 * (x[3] ** 3)/(x[1] * x[2] * (x[5] ** 4)) >= 1):
        constraint_cost += 1000
    if (1.93 * (x[4] ** 3)/(x[1] * x[2] * (x[6] ** 4)) >= 1):
        constraint_cost += 1000
    if (np.sqrt(((745*x[3]/(x[1]*x[2]))**2) +16.9 * 1e6)/(110*(x[5]**3)) >= 1):
        constraint_cost += 1000
    if (np.sqrt(((745*x[4]/(x[1]*x[2]))**2) +157.5 * 1e6)/(85*(x[6]**3)) >= 1):
        constraint_cost += 1000
    if (x[1]*x[2]/40 >= 1):
        constraint_cost += 1000
    if (5*x[1]/x[0] >=1):
        constraint_cost += 1000
    if (x[0]/(12*x[1]) >= 1):
        constraint_cost += 1000
    if ((1.5*x[5] + 1.9)/x[3] >= 1):
        constraint_cost += 1000
    if ((1.1*x[6] + 1.9)/x[4] >= 1):
        constraint_cost += 1000
    return func_value /2000

def calibrate_env(ht_list, X):
    def c(s, t, M, D, L, tau):
#         print((M*np.exp(-(s**2)/4*D*t))/np.sqrt(4*np.pi*D*t) + ((t > tau) * M * np.exp(-((s-L)**2)/(4*D*(t-tau))))/np.sqrt(4*np.pi*D*(t-tau))) 
        val = (M*np.exp(-(s**2)/4*D*t))/np.sqrt(4*np.pi*D*t)
        if (t > tau):
            val += ((t > tau) * M * np.exp(-((s-L)**2)/(4*D*(t-tau))))/np.sqrt(4*np.pi*D*(t-tau))
        return val
    tau = 30.01 + ht_list[0]/1000 
    M = 7 + ((X[0] + 1)*6)/2
    D = 0.02 + ((X[1] + 1)*0.10)/2
    L = 0.01 + ((X[2] + 1)*2.99)/2
#     print(f"M:{M}, D:{D}, L:{L}, tau:{tau}")
    val = 0.0
    for s in [0, 1, 2.5]:
        for t in [15, 30, 45, 60]:
#             print(f"s:{s}, t:{t}")
            val += (c(s, t, 10, 0.07, 1.505, 30.1525) - c(s, t, M, D, L, tau))**2 
#     print(f"val:{val}")
    return val

def dt_wine(ht_list, X):
    splitter_dict = {0: 'best', 1: 'random'}
    criterion_dict = {0: 'gini', 1: 'entropy'}
    
    splitter = splitter_dict[ht_list[0]]
    criterion = criterion_dict[ht_list[1]]
    min_samples_split = 0.01 + (0.99 - 0.01)*X[0]
    max_features = 0.01 + (0.99 - 0.01)*X[1]

    X, y = load_wine(return_X_y=True)

    n_cv = 5

    cv_acc = np.zeros((n_cv,))
    n_data = X.shape[0]
    n_train = int(n_data * 0.7)
    for cv in range(n_cv):
        inds = np.arange(n_data)
        np.random.RandomState(cv ** 2).shuffle(inds)
        train_inds = inds[:n_train]
        test_inds = inds[n_train:]
        train_X, train_y = X[train_inds], y[train_inds]
        test_X, test_y = X[test_inds], y[test_inds]
        clsf = DecisionTreeClassifier(
                                     splitter=splitter,
                                     criterion=criterion,
                                     min_samples_split=min_samples_split,
                                     max_features=max_features
                                    )
        clsf.fit(train_X, train_y)
        pred_test = clsf.predict(test_X)
        cv_acc[cv] = accuracy_score(test_y, pred_test)

    return np.mean(cv_acc).astype(float)

def svm_boston(ht_list, X):
    kernel_dict = {0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid'}
    gamma_dict = {0: 'scale', 1: 'auto'}
    
    kernel = kernel_dict[ht_list[0]]
    gamma = gamma_dict[ht_list[1]]
    C = max(1e-4, min(10 ** X[0], 10))
    nu = max(1e-6, min(10 ** X[1], 1))

    X, y = load_boston(return_X_y=True)

    n_cv = 5

    cv_rmse = np.zeros((n_cv,))
    n_data = X.shape[0]
    n_train = int(n_data * 0.7)
    for cv in range(n_cv):
        inds = np.arange(n_data)
        np.random.RandomState(cv ** 2).shuffle(inds)
        train_inds = inds[:n_train]
        test_inds = inds[n_train:]
        train_X, train_y = X[train_inds], y[train_inds]
        test_X, test_y = X[test_inds], y[test_inds]
        regr = make_pipeline(StandardScaler(), 
                             NuSVR(kernel=kernel, 
                                   gamma=gamma, 
                                   C=C, 
                                   nu=nu)
                            )
        regr.fit(train_X, train_y)
        pred_test = regr.predict(test_X)
        cv_rmse[cv] = np.mean((pred_test - test_y) ** 2) ** 0.5

    return np.mean(cv_rmse).astype(float)


def nn_ml(ht_list,X):
    hidden_layer_sizes_v = {i:i*20+40 for i in range(14)}
    activation_v = {0:'identity', 1:'logistic', 2:'tanh', 3:'relu'}
    batch_size_v = {i:i*20+40 for i in range(9)} 
    learning_rate_v = {0:'constant', 1:'invscaling', 2:'adaptive'}
    
    hidden_layer_sizes = hidden_layer_sizes_v[ht_list[0]]
    activation = activation_v[ht_list[1]]
    batch_size = batch_size_v[ht_list[2]]
    learning_rate = learning_rate_v[ht_list[3]]
    
    learning_rate_init =  0.001 + ((X[0] + 1)*(1-0.001))/2
    momentum = 0.5 + ((X[1] + 1)*(1-0.5))/2
    alpha = 0.0001 + ((X[2] + 1)*(1-0.0001))/2
    
    # dataset = "segment"
    ds = openml.datasets.get_dataset(dataset_id=40984)
    (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    y_new = np.array([val for val in y.values])
    y_new[y_new == "brickface"] = "0"
    y_new[y_new == "sky"] = "1"
    y_new[y_new == "foliage"] = "2"
    y_new[y_new == "cement"] = "3"
    y_new[y_new == "window"] = "4"
    y_new[y_new == "path"] = "5"
    y_new[y_new == "grass"] = "6"
    y = np.array([int(val) for val in y_new])
    
    # normalize X
    X = MinMaxScaler().fit_transform(X)
    # convert y to integer array
    y = np.array([int(val) for val in y])
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)
    
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                          activation=activation,
                          batch_size=batch_size,
                          learning_rate=learning_rate,
                          learning_rate_init=learning_rate_init,
                          momentum=momentum,
                          alpha=alpha,
                          random_state=2)
    
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    # maximize accuracy
    auc = accuracy_score(test_y, pred_y)
    
    return -auc

def robot_pushing(ht_list, X):
    x = np.hstack((ht_list, X))
    argv = []
    argv.append(x[0]-5)
    argv.append(x[1]-5)
    argv.append(x[4]-10)
    argv.append(x[5]-10)
    argv.append(x[8]+2)
    argv.append(((x[10]+1)*2*np.pi)/2)
    argv.append(x[2]-5)
    argv.append(x[3]-5)
    argv.append(x[6]-10)
    argv.append(x[7]-10)
    argv.append(x[9]+2)
    argv.append(((x[11]+1)*2*np.pi)/2) 
    argv.append(((x[12]+1)*10)/2)
    argv.append(((x[13]+1)*10)/2)
    f = push_function.PushReward()
    reward = f(argv)
    return -1*reward

# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
    X = np.asarray(X)
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300

# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html       
# =============================================================================
def mysixhumpcamp(X):
    X = np.asarray(X)
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10

# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
    X = np.asarray(X) / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50