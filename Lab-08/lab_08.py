import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def ex_1():
    # ----------------- a -----------------
    def trend(t, a=2, b=1, c=0):
        return a * t ** 2 + b * t + c

    def seasonal(f1, f2, t):
        return np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    def cyclical(scale, t):
        return np.random.normal(0, scale, len(t))

    n = 1000
    t = np.linspace(0, 1, n)

    time_series = trend(t) + seasonal(1, 2, t) + cyclical(0.1, t)

    fig, axs = plt.subplots(4, 1)
    plot_titles = ["Seria de timp aleatoare", "Trend", "Sezon", "Zgomot"]
    for i, signal in enumerate([time_series, trend(t), seasonal(1, 2, t), cyclical(0.1, t)]):
        axs[i].plot(t, signal)
        axs[i].set_title(plot_titles[i])

    plt.tight_layout()
    plt.savefig("Ex-1/Figura_1.png")
    plt.savefig("Ex-1/Figura_1.pdf")
    plt.show()
    plt.close()

    # ----------------- b -----------------
    autocorr = np.correlate(time_series, time_series, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    plt.plot(t, autocorr)
    plt.title("Autocorelatia seriei de timp")
    plt.savefig("Ex-1/Figura_2.png")
    plt.savefig("Ex-1/Figura_2.pdf")
    plt.show()
    plt.close()


if __name__ == '__main__':
    ex_1()
