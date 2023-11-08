import numpy as np
import matplotlib.pyplot as plt
import time

from PIL import Image
from scipy.io import wavfile


def ex_1():
    def dft(x):
        N = x.shape[0]
        X = np.zeros(N, dtype=np.complex128)
        for omega in range(N):
            for n in range(N):
                X[omega] += x[n] * np.exp(-2 * np.pi * 1j * n * omega / N)
        return X

    def composed_signal(amplitudes, f_list, n):
        return np.sum([amplitudes[index] * np.sin(2 * np.pi * f * n) for index, f in enumerate(f_list)], axis=0)

    N = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
    np_dft_seconds_list = np.zeros(N.shape[0])
    my_dft_seconds_list = np.zeros(N.shape[0])

    for index, n in enumerate(N):
        t = np.linspace(0, 1, n)
        x = composed_signal([1], [5], t)
        np_dft_seconds = time.time()
        np.fft.fft(x)
        np_dft_seconds_list[index] = time.time() - np_dft_seconds

        my_dft_seconds = time.time()
        dft(x)
        my_dft_seconds_list[index] = time.time() - my_dft_seconds

    plt.plot(N, np_dft_seconds_list, label="np.fft.fft")
    plt.plot(N, my_dft_seconds_list, label="dft")
    plt.xlabel("N")
    plt.ylabel("Seconds")
    plt.yscale("log")
    plt.title("Timpul de executie al functiilor np.fft.fft si dft")
    plt.grid()
    plt.legend()
    plt.savefig("Ex-1/Figura_1.png")
    plt.savefig("Ex-1/Figura_1.pdf")
    plt.show()
    plt.close()


def ex_2():
    def sinusoidal_signal(f, n):
        return np.sin(2 * np.pi * f * n)

    f = 12
    fs = 6
    n = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, fs)

    fig, axs = plt.subplots(4)
    axs[0].plot(n, sinusoidal_signal(12, n))
    axs[0].set_xlim([0, 1])

    for i, fi in enumerate([12, 7, 2]):
        axs[i + 1].plot(n, sinusoidal_signal(fi, n))
        axs[i + 1].scatter(t, sinusoidal_signal(fi, t), c="red")
        axs[i + 1].scatter(t, sinusoidal_signal(f, t), c="orange")
        axs[i + 1].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig("Ex-2/Figura_2.png")
    plt.savefig("Ex-2/Figura_2.pdf")
    plt.show()
    plt.close(fig)


def ex_3():
    def sinusoidal_signal(f, n):
        return np.sin(2 * np.pi * f * n)

    f = 12
    fs = 15
    n = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, fs)

    fig, axs = plt.subplots(4)
    axs[0].plot(n, sinusoidal_signal(12, n))
    axs[0].set_xlim([0, 1])

    for i, fi in enumerate([12, 7, 2]):
        axs[i + 1].plot(n, sinusoidal_signal(fi, n))
        axs[i + 1].scatter(t, sinusoidal_signal(fi, t), c="red")
        axs[i + 1].scatter(t, sinusoidal_signal(f, t), c="orange")
        axs[i + 1].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig("Ex-3/Figura_3.png")
    plt.savefig("Ex-3/Figura_3.pdf")
    plt.show()
    plt.close(fig)


def ex_4():
    f_max = 200
    fs = 2 * f_max
    print(f"Frecventa minima cu care trebuie esantionat semnalul este: {fs}")


def ex_5():
    spectogram = np.asarray(Image.open('Ex-5/spectograma.png'))
    plt.imshow(spectogram)
    plt.show()
    # Din spectograma se poate observa ca puterea sunetelor scade de-a lungul fiecarei vocale


def ex_6():
    # a
    fs, signal = wavfile.read("Ex-6/vowels.wav")
    n = signal.shape[0]

    # b
    group_size = int(0.01 * fs)  # numarul de esantioane din fiecare grup
    overlap = int(0.5 * group_size)  # numarul de esantioane care se suprapun

    signal_groups = []
    for i in range(0, n - group_size, int(group_size - overlap)):
        signal_groups.append(signal[i:i + group_size])

    # c, d
    spectogram = []
    for i, group in enumerate(signal_groups):
        spectogram.append(abs(np.fft.fft(group)))

    spectogram = np.array(spectogram).T
    # e
    plt.imshow(spectogram, aspect='auto', cmap='inferno')
    plt.title("Spectrograma unui semnal audio")
    plt.xlabel("Timp (s)")
    plt.ylabel("Frecventa (Hz)")
    plt.ylim([0, 20])
    plt.colorbar()
    plt.savefig("Ex-6/Spectograma.png")
    plt.savefig("Ex-6/Spectograma.pdf")
    plt.show()
    plt.close()


def ex_7():
    p_signal = 90  # db
    snr_db = 80
    # snr = p_signal / p_noise
    # snr_db = 10 * np.log10(snr)
    # snr_db = 10 * np.log10(p_signal / p_noise)
    # snr_db = 10 * np.log10(p_signal) - 10 * np.log10(p_noise)
    # p_noise = p_signal - snr_db
    print(f"Puterea zgomotului este: {p_signal - snr_db} dB")


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    # ex_3()
    # ex_4()
    # ex_5()
    # ex_6()
    ex_7()
