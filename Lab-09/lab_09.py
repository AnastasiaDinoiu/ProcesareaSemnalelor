import numpy as np
import random
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
import statsmodels.api as sm

t = np.linspace(0, 1, 1000)


def plot_signals(signals, t, titles, fig_title):
    fig, axs = plt.subplots(len(signals), 1)
    for i, signal in enumerate(signals):
        axs[i].plot(t, signal)
        axs[i].set_title(titles[i])
    plt.tight_layout()
    plt.savefig(f"{fig_title}.png")
    plt.savefig(f"{fig_title}.pdf")
    plt.show()
    plt.close()


def ex_1():
    def trend(t, a=2, b=1, c=0):
        return a * t ** 2 + b * t + c

    def seasonal(f1, f2, t):
        return np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    def noise(scale, t):
        np.random.seed(100)
        return np.random.normal(0, scale, len(t))

    return trend(t) + seasonal(1, 2, t) + noise(0.1, t)


def ex_2(time_series):
    def exponential_smoothing(time_series, alpha):
        smoothed_time_series = np.array([time_series[0]])
        for i in range(1, len(time_series)):
            smoothed_time_series = np.append(smoothed_time_series,
                                             alpha * time_series[i] + (1 - alpha) * smoothed_time_series[i - 1])
        return smoothed_time_series

    random_alpha = round(random.uniform(0, 1), 2)
    best_alpha, min = None, np.inf

    for alpha in np.linspace(0, 1, 100):
        smoothed_time_series = exponential_smoothing(time_series[:-1], alpha)
        mse = mean_squared_error(time_series[1:], smoothed_time_series)
        if mse < min:
            best_alpha, min = alpha, mse

    plot_signals(
        [time_series, exponential_smoothing(time_series, random_alpha), exponential_smoothing(time_series, best_alpha)],
        t,
        ["Seria de timp", f"Mediere exponentiala (random_alpha={random_alpha})",
         f"Mediere exponentiala (best_alpha={round(best_alpha, 2)})"],
        "Ex-1/Figura_1")


def ex_3(time_series):
    def estimate_am_coefficients(time_series, errors, q):
        n = len(time_series)
        Y = np.zeros((n - q, q))
        y = errors[q:]
        for i in range(q):
            Y[:, i] = errors[i:n - q + i]  # y = Yx

        return np.linalg.lstsq(Y, y, rcond=None)[0]

    def generate_ma_model(time_series, coefficients, errors):
        n = len(time_series)
        q = len(coefficients)
        mean = np.mean(time_series)
        ma_model = []
        for t in range(q, n):
            y_t = sum(coefficients[i] * errors[t - i] for i in range(q)) + errors[t] + mean
            ma_model.append(y_t)

        return ma_model

    q = 5
    errors = np.random.normal(0, 0.1, len(time_series))
    coefficients = estimate_am_coefficients(time_series, errors, q)
    ma_model = generate_ma_model(time_series, coefficients, errors)


def ex_4(time_series):
    # Modelul ARMA
    def estimate_ar_coefficients(time_series, p):
        n = len(time_series)
        Y = np.zeros((n - p, p))
        y = time_series[p:]
        for i in range(p):
            Y[:, i] = time_series[i:n - p + i]

        return np.linalg.lstsq(Y, y, rcond=None)[0]

    def estimate_am_coefficients(time_series, errors, q):
        n = len(time_series)
        Y = np.zeros((n - q, q))
        y = errors[q:]
        for i in range(q):
            Y[:, i] = errors[i:n - q + i]

        return np.linalg.lstsq(Y, y, rcond=None)[0]

    def generate_arma_model(time_series, ar_coefficients, am_coefficients, errors, p, q):
        n = len(time_series)
        arma_model = []
        for t in range(max(p, q), n):
            ar_t = sum(ar_coefficients[i] * time_series[t - i] for i in range(p))
            ma_t = sum(am_coefficients[i] * errors[t - i] for i in range(q)) + errors[t]
            arma_model.append(ar_t + ma_t + errors[t])

        return arma_model

    p, q = 5, 4
    errors = np.random.normal(0, 0.1, len(time_series))
    ar_coefficients = estimate_ar_coefficients(time_series, p)
    am_coefficients = estimate_am_coefficients(time_series, errors, q)
    arma_model = generate_arma_model(time_series, ar_coefficients, am_coefficients, errors, p, q)

    plt.plot(t, time_series, label="Seria de timp")
    plt.plot(t[max(p, q):], arma_model, label=f"Model ARMA (p={p} q={q})")
    plt.legend()
    plt.title("Modelul MA")
    plt.show()

    # Gasirea parametrilor optimi p si q pentru modelul ARMA
    best_p, best_q, min = None, None, np.inf
    for p in range(1, 20):
        for q in range(1, 20):
            ar_coefficients = estimate_ar_coefficients(time_series, p)
            am_coefficients = estimate_am_coefficients(time_series, errors, q)
            predictions = generate_arma_model(time_series, ar_coefficients, am_coefficients, errors, p, q)
            mse = mean_squared_error(time_series[max(p, q):], np.array(predictions))
            if mse < min:
                best_p, best_q, min = p, q, mse

    print(f"Parametrii p:{best_p} si q:{best_q}")

    # Modelul ARIMA
    arima_model = sm.tsa.ARIMA(np.diff(time_series), order=(2, 0, 3))
    results = arima_model.fit()


if __name__ == '__main__':
    ex_1()
    ex_2(ex_1())
    ex_3(ex_1())
    ex_4(ex_1())
