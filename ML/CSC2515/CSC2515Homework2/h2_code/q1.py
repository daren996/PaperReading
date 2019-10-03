# -*- coding: utf-8 -*-
"""
Question 1, CSC2515

I sampled 50000 samples through y=3x_1 + 5x_2 - 9x_3 + 28x_4 -10x_5 + 0.
Through 1000 iterations, the loss dropped to 0.01896. The predicted w was
[3.000181, 5.00041, -9.00045, 28.000627, -10.000461, 0.000123].
However, I evaluated the efficiency of the algorithms with Huber loss
function or square error loss function, and found that they took 6.6 and
0.24 seconds, respectively. I guessed the np.where function takes up most
of the total time.

@Darren Oct 1st, 2019
"""
import time

import numpy as np


np.set_printoptions(suppress=True)


def batch_gen(data, batch_size):
    idx = 0
    while True:
        if idx + batch_size > data.shape[0]:
            idx = 0
        start = idx
        idx += batch_size
        yield data[start: start + batch_size]


def get_data(w_true_, n=50000, d=5, e=10):
    """
    X is a NxD matrix
    t is a N-dimension vector

    :param n: number of samples of X
    :param d: number of features of X
    :param w_true_: the true w vector
    :param e: cardinality of error
    :return x_, t_
    """
    x_ = np.random.random((n, d)) * 1000 - 500
    x_ = np.concatenate((x_, np.ones((n, 1))), axis=1)
    error_ = np.random.random((n,)) * e - e / 2
    t_ = np.dot(x_, w_true_) + error_
    print("x shape:", x_.shape, "error shape:", error_.shape, "y shape:", t_.shape)
    return x_, t_


def general_gradient_descent(x_, t_, func_, batch_size=100, step_size=0.000015, max_iter_count=1000):
    """
    The general gradient descent algorithm.

    :param x_: samples, a NxD matrix
    :param t_: values of results, D-dimension vector
    :param step_size
    :param max_iter_count
    :param func_: loss function
    :param batch_size: size of batches
    :return:
    """
    # 确定样本数量以及变量的个数初始化theta值
    n, d = x_.shape
    w_ = np.zeros(d)
    loss = 1
    iter_count = 0
    gen_x = batch_gen(x_, batch_size)
    gen_t = batch_gen(t_, batch_size)
    while loss > 0.001 and iter_count < max_iter_count:
        batch_x = next(gen_x)
        batch_t = next(gen_t)
        y_ = np.dot(batch_x, w_)
        w_ = w_ - step_size / batch_x.shape[0] * (np.dot(batch_x.T, y_ - batch_t))
        loss = func_(y_, batch_t)
        iter_count += 1
        # print("%.5f" % loss)
    print("loss: %.5f" % loss)
    return w_


def huber_gradient_descent(x_, t_, batch_size=100, delta=50, step_size=0.000015, max_iter_count=1000):
    """
    The gradient descent algorithm using Huber loss model.
    Without using for loop, we applied the np.where function
    here. First, we generalized three vectors using y-t,
    delta or -delta separately; then we used np.where to
    determine which function should be chosen in each line.

    :param delta: delta in Huber function
    :param x_: samples, a NxD matrix
    :param t_: values of results, D-dimension vector
    :param step_size
    :param max_iter_count
    :param batch_size: size of batches
    :return:
    """
    # 确定样本数量以及变量的个数初始化theta值
    n, d = x_.shape
    w_ = np.zeros(d)
    loss = 1
    iter_count = 0
    gen_x = batch_gen(x_, batch_size)
    gen_t = batch_gen(t_, batch_size)
    while loss > 0.01 and iter_count < max_iter_count:
        batch_x = next(gen_x)
        batch_t = next(gen_t)
        y_ = np.dot(batch_x, w_)
        # temp is (y-t) or delta or -delta
        temp = np.where(np.array([-delta <= yt <= delta for yt in (y_ - batch_t)]),
                        (y_ - batch_t),
                        (delta * np.ones((batch_x.shape[0], ))))
        temp = np.where(np.array([-delta <= yt for yt in (y_ - batch_t)]),
                        temp,
                        (-delta * np.ones((batch_x.shape[0], ))))
        w_ = w_ - step_size / batch_x.shape[0] * (np.dot(batch_x.T, temp))
        # loss = np.sqrt(np.sum(np.square(y_ - batch_t))) / batch_x.shape[0]
        loss1 = 0.5 * np.square(y_ - batch_t)
        loss2 = delta * (np.abs(y_ - batch_t) - 0.5 * delta)
        loss = np.sqrt(np.sum(np.where(np.array([-delta <= yt <= delta for yt in (y_ - batch_t)]),
                                       loss1, loss2))) / batch_x.shape[0]
        iter_count += 1
        print("%.5f" % loss)
    print("loss: %.5f" % loss)
    return w_


if __name__ == '__main__':

    # w_truth = [3, 5, -9, 28, -10], b = 0
    w_truth = np.array([3, 5, -9, 28, -10, 0])
    x, t = get_data(np.array(w_truth))

    # time1 = time.time()
    # w = general_gradient_descent(x, t, func_=lambda y_, t_: np.sqrt(np.sum(np.square(y_ - t_))), batch_size=50000)
    # print(np.concatenate((w.reshape((6, 1)), w_truth.reshape(6, 1)), axis=1))
    # time2 = time.time()

    w = huber_gradient_descent(x, t, batch_size=50000)
    print(np.concatenate((w.reshape((6, 1)), w_truth.reshape(6, 1)), axis=1))

    # time3 = time.time()
    # print(time2 - time1, time3 - time2)
