from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt


def ex_1():
    def show_spectogram(x):
        y = np.fft.fft2(x)
        freq_db = 20 * np.log10(abs(y))
        plt.imshow(freq_db)
        plt.colorbar()
        plt.show()

    width = 512
    height = 512
    n1 = np.linspace(0, 1, width)
    n2 = np.linspace(0, 1, height)
    n1, n2 = np.meshgrid(n1, n2)

    x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
    plt.imshow(x1, cmap="gray")
    plt.title("np.sin(2 * np.pi * n1 + 3 * np.pi * n2)")
    plt.show()

    show_spectogram(x1)

    x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
    plt.imshow(x2, cmap="gray")
    plt.title("np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)")
    plt.show()

    show_spectogram(x2)


if __name__ == '__main__':
    ex_1()
