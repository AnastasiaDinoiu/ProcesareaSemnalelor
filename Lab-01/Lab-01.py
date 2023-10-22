import matplotlib.pyplot as plt
import numpy as np


def ex_1():
    def x(t):
        return np.cos(520 * np.pi * t + np.pi / 3)

    def y(t):
        return np.cos(280 * np.pi * t - np.pi / 3)

    def z(t):
        return np.cos(120 * np.pi * t + np.pi / 3)

    # (a)
    t = np.arange(0, 0.03, 0.0005)  # axa reala timp

    # (b)
    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 1 (b)")
    semnale_t = [x(t), y(t), z(t)]

    for index, semnal in enumerate(semnale_t):
        axs[index].plot(t, semnal)
        axs[index].set_xlabel("Timp")
        axs[index].set_ylabel("Amplitudine")
        axs[index].set_xlim([0, 0.03])

    for ax in axs.flat:
        ax.grid()
        plt.tight_layout()
    plt.show()

    # (c)
    fs = 200  # frecventa de esantionare (Hz)
    T = 1 / fs  # perioada de esantionare

    n = np.arange(0, 1, T)
    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 1 (c)")

    for index, semnal in enumerate([x(n), y(n), z(n)]):
        axs[index].plot(t, semnale_t[index])
        axs[index].stem(n, semnal, 'r')
        axs[index].set_xlabel("Timp")
        axs[index].set_ylabel("Amplitudine")
        axs[index].set_xlim([0, 0.03])

    for ax in axs.flat:
        ax.grid()
        plt.tight_layout()
    plt.show()


def ex_2():
    def plot_semnal(titlu, timp, semnal, xlim):
        plt.plot(timp, semnal)
        plt.title(titlu)
        plt.xlim(xlim)
        plt.grid()
        plt.xlabel("Timp")
        plt.ylabel("Amplitudine")
        plt.show()

    amplitudine = 1
    faza = 0

    # (a)
    frecventa = 400
    n_esantioane = 1600
    timp = np.linspace(0, 0.1, n_esantioane)
    semnal = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza)
    plot_semnal("Semnal sinusoidal de frecventa 400 Hz", timp, semnal, [0, 0.01])

    # (b)
    frecventa = 800
    timp = np.linspace(0, 3, 1000000)
    semnal = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza)
    plot_semnal("Semnal sinusoidal de frecventa 800 Hz", timp, semnal, [0, 0.01])

    # (c)
    frecventa = 240
    timp = np.linspace(0, 1, 100000)
    semnal = timp * frecventa - np.floor(timp * frecventa)
    plot_semnal("Semnal Sawtooth de frecventa 240 Hz", timp, semnal, [0, 0.1])

    # (d)
    frecventa = 300
    timp = np.linspace(0, 1, 100000)
    semnal = pow(-1, np.floor(2 * frecventa * timp))
    plot_semnal("Semnal Square de frecventa 300 Hz", timp, semnal, [0, 0.01])

    # (e)
    semnal_2d_aleator = np.random.rand(128, 128)
    plt.imshow(semnal_2d_aleator)
    plt.colorbar()
    plt.title("Semnal 2D aleator")
    plt.show()

    # (f)
    semnal_2d = np.zeros((128, 128))
    np.fill_diagonal(semnal_2d, 1)
    semnal_2d[(1, -1), :] = 2
    plt.imshow(semnal_2d)
    plt.colorbar()
    plt.title("Semnal 2D")
    plt.show()


def ex_3():
    frecventa = 2000

    # (a)
    t = 1 / frecventa
    print(f"Intervalul de timp intre doua esantioane pentru o frecventa de {frecventa} Hz: {t}")

    # (b)
    n_bytes = (4 * 2000 * 3600) / 8
    print(f"Numarul de bytes pentru a ocupa 1 ora de achizitie: {n_bytes}")


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    ex_3()
