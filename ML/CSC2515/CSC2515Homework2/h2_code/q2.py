# -*- coding: utf-8 -*-
"""

Complete the implementation of locally reweighted least squares.

Created on Tue Sep 12 20:39:09 2017

Implemented on Oct 3rd, 2019
@CHAO, Daren
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

np.random.seed(0)
np.set_printoptions(suppress=True)

# load boston housing prices dataset
# data(506, 13) feature_names(13, )
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(a, b):
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    a_norm = (a ** 2).sum(axis=1).reshape(a.shape[0], 1)
    b_norm = (b ** 2).sum(axis=1).reshape(1, b.shape[0])
    dist = a_norm + b_norm - 2 * a.dot(b.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """
    dist = l2(x_train, test_datum.reshape(1, x_train.shape[1]))  # N x 1 matrix
    dist = -dist / (2 * tau**2)
    # dist = np.exp(dist) / np.sum(np.exp(dist))
    dist = np.exp(dist - logsumexp(dist))
    a_weights = np.diag(dist.reshape((dist.shape[0],)))  # A
    w_star = x_train.T.dot(a_weights).dot(x_train) + lam * np.identity(d)
    w_star = np.linalg.inv(w_star)
    w_star = w_star.dot(x_train.T).dot(a_weights).dot(y_train)
    # w_star = np.linalg.solve(x_train.T.dot(a_weights).dot(x_train) + lam * np.ones(d),
    #                          x_train.T.dot(a_weights).dot(y_train))
    y_hat = test_datum.T.dot(w_star)
    return y_hat


def run_validation(x_, y_, taus_, val_frac):
    """
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    """
    train_x, val_x, train_y, val_y = \
        train_test_split(x_, y_, test_size=val_frac)
    valid_taus_ = []
    train_losses_ = []
    validation_losses_ = []
    for tau_ in taus_:
        losses_train, count_train = 0, 0
        losses_val, count_val = 0, 0
        valid = True  # check whether this tau is valid
        for i in range(train_x.shape[0]):
            try:
                y_hat = LRLS(train_x[i], train_x, train_y, tau_)
                losses_train += 0.5 * (y_hat - train_y[i])**2
                count_train += 1
            except np.linalg.LinAlgError:
                print("ERROR: Singular matrix")
                valid = False
        for i in range(val_x.shape[0]):
            try:
                y_hat = LRLS(val_x[i], train_x, train_y, tau_)
                losses_val += 0.5 * (y_hat - val_y[i]) ** 2
                count_val += 1
            except np.linalg.LinAlgError:
                print("ERROR: Singular matrix")
                valid = False
        if valid:  # remove invalid taus
            valid_taus_.append(tau_)
            losses_train /= count_train
            train_losses_.append(losses_train)
            losses_val /= count_val
            validation_losses_.append(losses_val)
    return np.array(train_losses_), np.array(validation_losses_), valid_taus_


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value.
    # Feel free to play with lambda as well if you wish.
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses, valid_taus = run_validation(x, y, taus, val_frac=0.3)
    # print(train_losses)
    # print(test_losses)
    
    plt.semilogx(valid_taus, train_losses, label="training set")
    plt.semilogx(valid_taus, test_losses, label="validation set")
    plt.ylim(0, 25)
    plt.xlim(0, )
    plt.legend()
    plt.xlabel("tau")
    plt.ylabel("loss")
    plt.show()
