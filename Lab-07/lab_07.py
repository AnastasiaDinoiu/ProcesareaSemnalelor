from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter


def ex_1():
    def show_spectogram_log(x, index):
        y = np.fft.fftshift(np.fft.fft2(x))
        freq_db = 20 * np.log10(abs(y + 1e-20))
        img = axs[1].imshow(freq_db)
        fig.colorbar(img, ax=axs[1])
        axs[0].set_title("Semnalul")
        axs[1].set_title("Spectrul semnalului")
        plt.tight_layout()
        plt.savefig(f"Ex-1/Figura_{index}.png")
        plt.savefig(f"Ex-1/Figura_{index}.pdf")
        plt.show()
        plt.close()

    n = 512
    n1, n2 = np.meshgrid(range(n), range(n))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("np.sin(2πn1 + 3πn2)")
    x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
    img = axs[0].imshow(x1)
    fig.colorbar(img, ax=axs[0])
    show_spectogram_log(x1, 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("np.sin(4πn1) + np.cos(6πn2)")
    x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
    img = axs[0].imshow(x2)
    fig.colorbar(img, ax=axs[0])
    show_spectogram_log(x2, 2)

    def show_spectogram(y, index):
        img = axs[1].imshow(np.abs(np.fft.fftshift(y)) + 1e-15)
        fig.colorbar(img, ax=axs[1])
        axs[0].set_title("Semnalul")
        axs[1].set_title("Spectrul semnalului")
        plt.tight_layout()
        plt.savefig(f"Ex-1/Figura_{index}.png")
        plt.savefig(f"Ex-1/Figura_{index}.pdf")
        plt.show()
        plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Y_0,5 - Y_0,N-5, altfel Y_m1,m2 = 0, oricare m1, m2")
    n = 50
    y1 = np.zeros((n, n))
    y1[0][5] = 1
    y1[0][n - 5] = 1
    img = axs[0].imshow(np.fft.ifft2(y1).real)
    fig.colorbar(img, ax=axs[0])
    show_spectogram(y1, 3)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Y_5,0 - Y_N-5,0, altfel Y_m1,m2 = 0, oricare m1, m2")
    y2 = np.zeros((n, n))
    y2[5][0] = 1
    y2[n - 5][0] = 1
    img = axs[0].imshow(np.fft.ifft2(y2).real)
    fig.colorbar(img, ax=axs[0])
    show_spectogram(y2, 4)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Y_5,5 - Y_N-5,N-5, altfel Y_m1,m2 = 0, oricare m1, m2")
    y3 = np.zeros((n, n))
    y3[5][5] = 1
    y3[n - 5][n - 5] = 1
    img = axs[0].imshow(np.fft.ifft2(y3).real)
    fig.colorbar(img, ax=axs[0])
    show_spectogram(y3, 5)


def ex_2():
    def compress(x, freq_cutoff):
        y = np.fft.fft2(x)
        y_cutoff = y.copy()
        freq_db = 20 * np.log10(abs(y + 1e-20))
        y_cutoff[freq_db > freq_cutoff] = 0
        return np.fft.ifft2(y_cutoff).real

    x = misc.face(gray=True)
    snr = np.inf
    snr_threshold = 0.0075
    freq_cuttof = 200
    x_compressed = misc.face(gray=True)

    while snr > snr_threshold:
        x_compressed = compress(x_compressed, freq_cuttof)
        snr = np.sum(x ** 2) / np.sum(np.abs(x - x_compressed) ** 2)  # semnal / zgomot
        freq_cuttof -= 5

    x_freq = np.fft.fft2(x)
    x_freq_compressed = np.fft.fft2(x_compressed)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Imaginea originala si spectrul")
    axs[0].imshow(x, cmap='gray')
    freq_db = 20 * np.log10(abs(x_freq + 1e-20))
    axs[1].imshow(np.fft.fftshift(freq_db), cmap='gray')
    plt.tight_layout()
    plt.savefig("Ex-2/Figura_6.png")
    plt.savefig("Ex-2/Figura_6.pdf")
    plt.show()
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Imaginea comprimata si spectrul")
    axs[0].imshow(x_compressed, cmap='gray')
    freq_db = 20 * np.log10(abs(x_freq_compressed + 1e-20))
    axs[1].imshow(np.fft.fftshift(freq_db), cmap='gray')
    plt.tight_layout()
    plt.savefig("Ex-2/Figura_7.png")
    plt.savefig("Ex-2/Figura_7.pdf")
    plt.show()
    plt.close()

    snr = np.sum(np.abs(x) ** 2) / np.sum(np.abs(x - x_compressed) ** 2)
    print(f"Raportul SNR dupa comprimare: {snr:.4f}")


def ex_3():
    x = misc.face(gray=True)
    pixel_noise = 200
    noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=x.shape)
    x_noisy = x + noise

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(x, cmap='gray')
    axs[0].set_title("Imaginea initiala")
    axs[1].imshow(x_noisy, cmap='gray')
    axs[1].set_title("Dupa adaugarea zgomotului")

    x_denoised = gaussian_filter(x_noisy, sigma=3)

    axs[2].imshow(x_denoised, cmap='gray')
    axs[2].set_title("Filtru Gaussian (sigma=3)")
    plt.tight_layout()
    plt.savefig("Ex-3/Figura_8.png")
    plt.savefig("Ex-3/Figura_8.pdf")
    plt.show()
    plt.close()

    snr_noisy = np.sum(np.abs(x) ** 2) / np.sum(np.abs(x - x_noisy) ** 2)
    snr_denoised = np.sum(np.abs(x) ** 2) / np.sum(np.abs(x - x_denoised) ** 2)

    print(f"Raportul SNR adaugarea zgomotului: {snr_noisy:.4f}")
    print(f"Raportul SNR dupa eliminarea zgomotului: {snr_denoised:.4f}")


def ex_4():
    def plot_spectogram(signal, fs, title, index):
        plt.specgram(signal, Fs=fs)
        plt.title(title)
        plt.xlabel("Timp (secunde)")
        plt.ylabel("Frecventa (Hz)")
        plt.xticks(np.arange(0, 5.5, 0.5))
        plt.ylim(top=20000)
        plt.colorbar()
        plt.savefig(f"Ex-4/Figura_{index}.png")
        plt.savefig(f"Ex-4/Figura_{index}.pdf")
        plt.show()

    fs, signal = wavfile.read("audio.wav")
    signal = signal[fs * 25:fs * 30]

    _, instrument = wavfile.read("drums.wav")
    instrument = instrument[fs * 25:fs * 30]

    plot_spectogram(signal, fs, "Spectrograma semnalului audio", 9)
    plot_spectogram(instrument, fs, "Spectrograma instrumentului audio", 10)


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    # ex_3()
    ex_4()
