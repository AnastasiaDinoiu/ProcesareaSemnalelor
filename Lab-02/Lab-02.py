import scipy
import numpy as np
import matplotlib.pyplot as plt


def semnal_sinusoidal(amplitudine, frecventa, t, faza):
    return amplitudine * np.sin(2 * np.pi * frecventa * t + faza)


def semnal_cosinusoidal(amplitudine, frecventa, t, faza):
    return amplitudine * np.cos(2 * np.pi * frecventa * t + faza)


def ex_1():
    a = 1
    f = 100
    time = np.linspace(0, 1, f)

    plt.subplot(2, 1, 1)
    plt.plot(time, [semnal_sinusoidal(a, f, t, np.pi) for t in time], color='orange')
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Semnal sinusoidal")

    plt.subplot(2, 1, 2)
    plt.plot(time, [semnal_cosinusoidal(a, f, t, np.pi / 2) for t in time], color='red')
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Semnal cosinusoidal")
    plt.show()


def ex_2():
    f = 100
    time = np.linspace(0, 1, f)
    thetas = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    for theta in thetas:
        plt.plot(time, [semnal_sinusoidal(1, f, t, theta) for t in time])
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Exercitiul 2")
    plt.show()


if __name__ == '__main__':
    # Exercitiul 1
    # ex_1()

    # Exercitiul 2
    ex_2()
