import numpy as np
import matplotlib.pyplot as plt
import math


def ex_1():
    def fourier_matrix(n):
        f = np.zeros((n, n), dtype=np.complex128)
        for m in range(n):
            for k in range(n):
                f[m, k] = np.exp(2 * np.pi * 1j * m * k / n)
        return f

    n = 8
    f = fourier_matrix(n)

    def plot_matrix(dim, f):
        n = np.linspace(0, dim - 1, dim)
        fig, axs = plt.subplots(dim, 2)
        plt.suptitle("Partea reala si partea imaginara")

        for index, semnal in enumerate(n):
            axs[index][0].plot(n, np.real(f[index]))
            axs[index][1].plot(n, np.imag(f[index]), '--')
            axs[index][0].grid()
            axs[index][1].grid()
        plt.savefig("ex_1.png")
        plt.show()
        plt.close(fig)

    plot_matrix(n, f)

    def is_unit_matrix(f):
        return np.iscomplexobj(f) and np.allclose(np.dot(f, np.conj(f).T), np.dot(n, np.identity(n)))

    if is_unit_matrix(f):
        print("Matricea Fourier este unitara")
    else:
        print("Matricea Fourier nu este unitara")


def ex_2():
    pass


def ex_3():
    pass


if __name__ == '__main__':
    # ex_1()
    ex_2()
    ex_3()
