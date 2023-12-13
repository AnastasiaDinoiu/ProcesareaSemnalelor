import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg


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

    # ----------------- c -----------------
    p = 150
    train_size = 800

    train_series = time_series[:train_size]

    model = AutoReg(train_series, lags=p)
    predictions = model.fit().predict(start=train_size, end=n - 1)

    plt.figure(figsize=(10, 5))
    plt.plot(t, time_series, label="Seria de timp")
    plt.plot(t[train_size:], predictions, label="Predictii AR")
    plt.title("Modelul AR")
    plt.legend()
    plt.grid()
    plt.savefig("Ex-1/Figura_3.png")
    plt.savefig("Ex-1/Figura_3.pdf")
    plt.show()
    plt.close()

    # ----------------- d -----------------
    best_p, best_m, max = None, None, np.inf

    for p in range(1, 50):
        for m in range(1, 10):
            model = AutoReg(train_series, lags=p)
            predictions = model.fit().predict(start=train_size, end=train_size + m - 1)

            mse = mean_squared_error(time_series[train_size:train_size + m], predictions)
            if mse < max:
                best_p, best_m, max = p, m, mse

    print(f"Parametrii p:{best_p} si m:{best_m}")


if __name__ == '__main__':
    ex_1()
