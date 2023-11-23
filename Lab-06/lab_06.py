import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.signal as sig


def ex_1():
    n = 100
    x = np.random.random(n)
    signals = [x]
    for i in range(3):
        conv = sig.convolve(signals[0], signals[i])
        signals.append(conv)
    normalized_signals = [signal / np.linalg.norm(signal) for signal in signals]

    fig, axs = plt.subplots(4)
    plt.suptitle("Convolutia repetata a unui semnal aleator")
    for index, signal in enumerate(normalized_signals):
        axs[index].plot(signal)
        axs[index].set_title(f"x^{index + 1}")
        axs[index].grid()
    plt.tight_layout()
    plt.savefig("Ex-1/Figura_1.png")
    plt.savefig("Ex-1/Figura_1.pdf")
    plt.show()
    plt.close()
    # Pe masura ce convolutia se repeta, semnalul se "netezeste" si devine mai asemanator cu o gaussiana


def ex_2():
    def direct_polyn_mult(p, q):
        n = len(p)
        r = np.zeros(2 * n - 1)
        for i in range(n):
            for j in range(n):
                r[i + j] += p[i] * q[j]
        return r

    def fft_polyn_mult(p, q):
        n = 2 ** int(np.ceil(np.log2(2 * len(p) - 1)))
        p_fft = np.fft.fft(p, n)
        q_fft = np.fft.fft(q, n)
        r = np.real(np.fft.ifft(p_fft * q_fft))
        return r[:2 * len(p) - 1]

    n = 100
    p = np.random.randint(0, 10, n + 1)
    q = np.random.randint(0, 10, n + 1)

    direct_polynomial_multiplication = direct_polyn_mult(p, q)
    fft_polynomial_multiplication = fft_polyn_mult(p, q)

    plots = [direct_polynomial_multiplication, fft_polynomial_multiplication,
             np.abs(direct_polynomial_multiplication - fft_polynomial_multiplication)]
    titles = ["Inmultirea directa a polinoamelor",
              "Inmultirea prin FFT a polinoamelor",
              "|direct_polynomial_multiplication - fft_polynomial_multiplication|"]

    fig, axs = plt.subplots(3)
    for i in range(3):
        axs[i].plot(plots[i])
        axs[i].set_title(titles[i])
        axs[i].grid()

    plt.tight_layout()
    plt.savefig("Ex-2/Figura_2.png")
    plt.savefig("Ex-2/Figura_2.pdf")
    plt.show()
    plt.close()


def ex_3():
    def rectangular_window(n):
        return np.ones(n)

    def hanning_window(n):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))

    def sinusoidal_signal(t, a=1, f=10, phi=0):
        return a * np.sin(2 * np.pi * f * t + phi)

    nw = 200
    t = np.linspace(0, 1, nw)
    titles = ["Sinusoida initiala",
              "Sinusoida trecuta prin fereastra dreptunghiulara",
              "Sinusoida trecuta prin fereastra Hanning"]

    fig, axs = plt.subplots(3)
    for i, signal in enumerate([sinusoidal_signal(t),
                                sinusoidal_signal(t) * rectangular_window(nw),
                                sinusoidal_signal(t) * hanning_window(nw)]):
        axs[i].plot(t, signal)
        axs[i].set_title(titles[i])
        axs[i].grid()

    plt.tight_layout()
    plt.savefig("Ex-3/Figura_3.png")
    plt.savefig("Ex-3/Figura_3.pdf")
    plt.show()
    plt.close()


def ex_4():
    # a --------------------------------------------------
    dataframe = pd.read_csv("../Lab-05/Train.csv", dtype={
        "ID": "int64",
        "Datetime": "string",
        "Count": "int64"
    })
    dataframe.Datetime = pd.to_datetime(dataframe.Datetime, format="%d-%m-%Y %H:%M")
    start_index = 16056
    x = dataframe.iloc[start_index:start_index + 73]
    x = x.reset_index(drop=True)

    # b --------------------------------------------------
    plt.plot(x.Count, label="Semnal brut", lw=1.3)
    for w in [5, 9, 13, 17]:
        moving_average_filter = np.convolve(x.Count, np.ones(w), "valid") / w
        plt.plot(moving_average_filter, label=f"Nw = {w}", lw=1)
    plt.legend()
    plt.xlabel("Timp (esantioane)")
    plt.ylabel("Amplitudine")
    plt.xlim([0, 73])
    plt.title("Filtru medie alunecatoare")
    plt.grid()
    plt.savefig("Ex-4/Figura_4.png")
    plt.savefig("Ex-4/Figura_4.pdf")
    plt.show()
    plt.close()

    # c --------------------------------------------------
    ts = (x.Datetime[1] - x.Datetime[0]).total_seconds()  # perioada de esantionare
    fs = 1 / ts  # frecventa de esantionare: 1 / 3600 Hz
    f_nyquist = fs / 2  # frecventa Nyquist: 1 / 7200 Hz
    wn = fs / 4  # frecventa de taiere (Hz) este mai mica decat frecventa Nyquist (~0.0001388 Hz)
    wn_normalized = wn / f_nyquist  # frecventa normalizata

    # d --------------------------------------------------
    n = 5
    rp = 5
    b_butter, a_butter = scipy.signal.butter(n, wn_normalized, btype="low")
    b_cheby, a_cheby = scipy.signal.cheby1(n, rp, wn_normalized, btype="low")

    # e --------------------------------------------------
    butter_filter = scipy.signal.filtfilt(b_butter, a_butter, x.Count)
    cheby_filter = scipy.signal.filtfilt(b_cheby, a_cheby, x.Count)

    labels = ["Filtrat Butterworth: n=5", "Filtrat Chebyshev: n=5 rp=5"]
    plt.plot(x.Count, label="Semnal brut")
    for i, filter in enumerate([butter_filter, cheby_filter]):
        plt.plot(filter, label=labels[i], lw=1)
    plt.legend()
    plt.xlabel("Timp (esantioane)")
    plt.ylabel("Amplitudine")
    plt.title("Filtrele Butterworth si Chebyshev")
    plt.grid()
    plt.savefig("Ex-4/Figura_5.png")
    plt.savefig("Ex-4/Figura_5.pdf")
    plt.show()
    plt.close()
    # As alege filtrul Chebyshev deoarece are o ateunare mai mare a frecventelor de ordin inalt

    # f --------------------------------------------------
    for i, n in enumerate([1, 3, 7, 9, 11]):
        plt.plot(x.Count, label="Semnal brut")

        b_butter, a_butter = scipy.signal.butter(n, wn_normalized, btype="low")
        butter_filter = scipy.signal.filtfilt(b_butter, a_butter, x.Count)
        plt.plot(butter_filter, label=f"Filtrat Butterworth: n={n}", lw=1)

        for rp in [3, 5, 7]:
            b_cheby, a_cheby = scipy.signal.cheby1(n, rp, wn_normalized, btype="low")
            cheby_filter = scipy.signal.filtfilt(b_cheby, a_cheby, x.Count)
            plt.plot(cheby_filter, label=f"Filtrat Chebyshev: n={n} rp={rp}", lw=1)

        plt.legend()
        plt.xlabel("Timp (esantioane)")
        plt.ylabel("Amplitudine")
        plt.title(f"Filtrele Butterworth si Chebyshev: n={n}")
        plt.grid()
        plt.savefig(f"Ex-4/Figura_{i + 6}.png")
        plt.savefig(f"Ex-4/Figura_{i + 6}.pdf")
        plt.show()
        plt.close()
    # Pe masura ce ordinul creste, filtrul Butterworth ramane aproximativ la fel, iar filtrul Chebyshev are o atenuare
    # mai mare a frecventelor de ordin inalt.


if __name__ == '__main__':
    ex_1()
    ex_2()
    ex_3()
    ex_4()
