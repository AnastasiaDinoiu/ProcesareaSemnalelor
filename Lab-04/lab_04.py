import numpy as np
import matplotlib.pyplot as plt
import time


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

    fig, axs = plt.subplots(4)

    n = np.linspace(0, 1, 100)
    axs[0].plot(n, sinusoidal_signal(12, n))
    axs[0].set_xlim([0, 1])

    fs = 6
    t = np.linspace(0, 1, fs)
    axs[1].plot(n, sinusoidal_signal(12, n))
    axs[1].scatter(t, sinusoidal_signal(12, t), c="orange")
    axs[1].set_xlim([0, 1])

    axs[2].plot(n, sinusoidal_signal(7, n))
    axs[2].scatter(t, sinusoidal_signal(7, t), c="red")
    axs[2].scatter(t, sinusoidal_signal(12, t), c="orange")
    axs[2].set_xlim([0, 1])

    axs[3].plot(n, sinusoidal_signal(2, n))
    axs[3].scatter(t, sinusoidal_signal(2, t), c="red")
    axs[3].scatter(t, sinusoidal_signal(12, t), c="orange")
    axs[3].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig("Ex-2/Figura_2.png")
    plt.savefig("Ex-2/Figura_2.pdf")
    plt.show()
    plt.close(fig)


def ex_3():
    def sinusoidal_signal(f, n):
        return np.sin(2 * np.pi * f * n)

    fig, axs = plt.subplots(4)

    n = np.linspace(0, 1, 100)
    axs[0].plot(n, sinusoidal_signal(12, n))
    axs[0].set_xlim([0, 1])

    fs = 15
    t = np.linspace(0, 1, fs)
    axs[1].plot(n, sinusoidal_signal(12, n))
    axs[1].scatter(t, sinusoidal_signal(12, t), c="orange")
    axs[1].set_xlim([0, 1])

    axs[2].plot(n, sinusoidal_signal(7, n))
    axs[2].scatter(t, sinusoidal_signal(7, t), c="red")
    axs[2].scatter(t, sinusoidal_signal(12, t), c="orange")
    axs[2].set_xlim([0, 1])

    axs[3].plot(n, sinusoidal_signal(2, n))
    axs[3].scatter(t, sinusoidal_signal(2, t), c="red")
    axs[3].scatter(t, sinusoidal_signal(12, t), c="orange")
    axs[3].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig("Ex-3/Figura_3.png")
    plt.savefig("Ex-3/Figura_3.pdf")
    plt.show()
    plt.close(fig)



if __name__ == '__main__':
    # ex_1()
    # ex_2()
    ex_3()
