

import matplotlib.pyplot as plt
import numpy as np


def plot_functions(delta_arr):
    delta_arr = delta_arr
    x = np.linspace(-1000, 1000, 100000)
    y_lse = x ** 2 / 2
    plt.plot(x, y_lse, ls="-", lw=2, label="squared error loss")
    for delta in delta_arr:
        y_lh = x ** 2 / 2 * (np.abs(x) <= delta) + delta * (np.abs(x) - delta / 2) * (np.abs(x) > delta)
        plt.plot(x, y_lh, ls="-", lw=2, label="huber loss (delta = %d)" % delta)
    plt.legend()
    plt.show()


plot_functions([300, 50])



