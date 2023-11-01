import numpy as np
import matplotlib.pyplot as plt


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
            axs[index][0].plot(n, np.real(f[index]), 'orange')
            axs[index][1].plot(n, np.imag(f[index]), '--')
            axs[index][0].grid()
            axs[index][1].grid()
        plt.savefig("Ex-1/Matricea_Fourier.png")
        plt.savefig("Ex-1/Matricea_Fourier.pdf")
        plt.show()
        plt.close(fig)

    plot_matrix(n, f)

    def is_unit_matrix(f):
        return np.iscomplexobj(f) and np.allclose(np.dot(f, np.conj(f).T), np.dot(n, np.identity(n)), atol=1e-10)

    if is_unit_matrix(f):
        print("Matricea Fourier este unitara")
    else:
        print("Matricea Fourier nu este unitara")


def ex_2():
    def sinusoidal_signal(f, n):
        return np.sin(2 * np.pi * f * n)

    def y(f, n):
        return sinusoidal_signal(f, n) * np.exp(-2 * np.pi * 1j * n)

    n = np.linspace(0, 1, 10 ** 3)
    f = 5

    def figura_1():
        signal = y(f, n)
        point = 670

        fig, axs = plt.subplots(1, 2)
        fig.suptitle("Reprezentarea unui semnal in planul complex")
        axs[0].scatter(n, sinusoidal_signal(f, n), s=2, lw=0, c=np.abs(sinusoidal_signal(f, n)))
        axs[0].stem(n[point], sinusoidal_signal(f, n[point]), 'r')
        axs[0].set_xlabel('Timp')
        axs[0].set_ylabel('Amplitudine')
        axs[0].axhline(0, c='grey', lw=0.5)

        axs[1].scatter(np.real(signal), np.imag(signal), s=2, lw=0, c=np.abs(signal))
        axs[1].scatter(np.real(y(f, n[point])), np.imag(y(f, n[point])), c='r')
        axs[1].plot((0, np.real(y(f, n[point]))), (0, np.imag(y(f, n[point]))), c='r')
        axs[1].set_aspect('equal')
        axs[1].set_xlabel('Real')
        axs[1].set_ylabel('Imaginar')
        axs[1].axhline(0, c='grey', lw=0.5)
        axs[1].axvline(0, c='grey', lw=0.5)

        plt.tight_layout()
        plt.savefig("Ex-2/Figura_1.png")
        plt.savefig("Ex-2/Figura_1.pdf")
        plt.show()
        plt.close(fig)

    figura_1()

    def figura_2():
        omega_list = np.array([1, 2, 5, 7])
        x = sinusoidal_signal(f, n)
        colors = [np.abs(x * np.exp(-2 * np.pi * 1j * omega * n)) for omega in omega_list]
        signals = [x * np.exp(-2 * np.pi * 1j * omega * n) for omega in omega_list]

        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Reprezentarea transformatei Fourier in planul complex")
        axs[0, 0].scatter(np.real(signals[0]), np.imag(signals[0]), s=2, lw=0, c=colors[0])
        axs[0, 0].set_xlabel('Real')
        axs[0, 0].set_ylabel('Imaginar')
        axs[0, 0].set_aspect('equal')
        axs[0, 0].axhline(0, c='grey', lw=0.5)
        axs[0, 0].axvline(0, c='grey', lw=0.5)
        axs[0, 0].title.set_text("ω = 1")

        axs[0, 1].scatter(np.real(signals[1]), np.imag(signals[1]), s=2, lw=0, c=colors[1])
        axs[0, 1].set_xlabel('Real')
        axs[0, 1].set_ylabel('Imaginar')
        axs[0, 1].set_aspect('equal')
        axs[0, 1].axhline(0, c='grey', lw=0.5)
        axs[0, 1].axvline(0, c='grey', lw=0.5)
        axs[0, 1].title.set_text("ω = 2")

        axs[1, 0].scatter(np.real(signals[2]), np.imag(signals[2]), s=2, lw=0, c=colors[2])
        axs[1, 0].set_xlabel('Real')
        axs[1, 0].set_ylabel('Imaginar')
        axs[1, 0].set_aspect('equal')
        axs[1, 0].axhline(0, c='grey', lw=0.5)
        axs[1, 0].axvline(0, c='grey', lw=0.5)
        axs[1, 0].title.set_text("ω = 5")
        axs[1, 0].set_xlim([-1, 1])
        axs[1, 0].set_ylim([-1, 1])

        axs[1, 1].scatter(np.real(signals[3]), np.imag(signals[3]), s=2, lw=0, c=colors[3])
        axs[1, 1].set_xlabel('Real')
        axs[1, 1].set_ylabel('Imaginar')
        axs[1, 1].set_aspect('equal')
        axs[1, 1].axhline(0, c='grey', lw=0.5)
        axs[1, 1].axvline(0, c='grey', lw=0.5)
        axs[1, 1].title.set_text("ω = 7")

        plt.tight_layout()
        plt.savefig("Ex-2/Figura_2.png")
        plt.savefig("Ex-2/Figura_2.pdf")
        plt.show()
        plt.close(fig)

    figura_2()


def ex_3():
    def dft(x):
        N = x.shape[0]
        X = np.zeros(N, dtype=np.complex128)
        for omega in range(N):
            for n in range(N):
                X[omega] += x[n] * np.exp(-2 * np.pi * 1j * n * omega / N)
        return X

    def composed_signal(amplitudes, f_list, n):
        return np.sum([amplitudes[index] * np.sin(2 * np.pi * f * n) for index, f in enumerate(f_list)], axis=0)

    def figura_3():
        n = np.linspace(0, 1, 10**3)
        f_list = np.array([5, 20, 65])
        x = composed_signal([1, 2, 0.5], f_list, n)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Transformata Fourier pentru un semnal cu 3 componente de frecventa")

        axs[0].plot(n, x, lw=1)
        axs[0].set_xlabel("Timp (s)")
        axs[0].set_ylabel("x(t)")
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([-4, 4])

        axs[1].stem(range(101), np.abs(dft(x)[:101]))
        axs[1].set_xlabel("Frecventa (Hz)")
        axs[1].set_ylabel("|X(ω)|")
        axs[1].set_xlim([0, 100])
        axs[1].set_ylim([0, 1200])

        plt.tight_layout()
        plt.savefig("Ex-3/Figura_3.png")
        plt.savefig("Ex-3/Figura_3.pdf")
        plt.show()
        plt.close(fig)

    figura_3()


if __name__ == '__main__':
    ex_1()
    ex_2()
    ex_3()
