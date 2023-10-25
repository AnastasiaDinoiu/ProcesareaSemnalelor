import scipy
import numpy as np
import matplotlib.pyplot as plt
import sounddevice


def ex_1():
    def semnal_sinusoidal(amplitudine, frecventa, t, faza):
        return amplitudine * np.sin(2 * np.pi * frecventa * t + faza)

    def semnal_cosinusoidal(amplitudine, frecventa, t, faza):
        return amplitudine * np.cos(2 * np.pi * frecventa * t + faza)

    a = 1
    f = 100

    timp = np.linspace(0, 1, f)

    plt.subplot(2, 1, 1)
    plt.plot(timp, [semnal_sinusoidal(a, f, t, np.pi) for t in timp], color='orange')
    plt.grid()
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.title("Semnal sinusoidal")

    plt.subplot(2, 1, 2)
    plt.plot(timp, [semnal_cosinusoidal(a, f, t, np.pi / 2) for t in timp], color='red')
    plt.grid()
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.title("Semnal cosinusoidal")
    plt.tight_layout()
    plt.show()


def ex_2():
    def semnal_sinusoidal(frecventa, t, faza):
        return np.sin(2 * np.pi * frecventa * t + faza)

    f = 100
    timp = np.linspace(0, 1, 10000)

    for theta in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        plt.plot(timp, [semnal_sinusoidal(f, t, theta) for t in timp])
    plt.grid()
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.title("Semnale sinosidale de amplitudine unitara si frecventa 100Hz")
    plt.xlim([0, 0.01])
    plt.show()

    semnal_1 = semnal_sinusoidal(f, timp, 0)
    z = np.random.normal(0, 1, 10000)
    gama = [np.sqrt(np.linalg.norm(semnal_1) ** 2 / (snr * np.linalg.norm(z) ** 2)) for snr in [0.1, 1, 10, 100]]

    fig, axs = plt.subplots(4)
    fig.suptitle("Zgomote aleatoare")

    for index, value in enumerate(gama):
        axs[index].plot(timp, semnal_1 + value * z)
        axs[index].set_xlabel("Timp")
        axs[index].set_ylabel("Amplitudine")
        axs[index].set_xlim([0, 0.03])

    for ax in axs.flat:
        ax.grid()
        plt.tight_layout()
    plt.show()


def ex_3():
    def semnal_a():
        frecventa = 400
        timp = np.linspace(0, 2, 10000)
        return np.sin(2 * np.pi * frecventa * timp)

    def semnal_b():
        frecventa = 800
        timp = np.linspace(0, 3, 10000)
        return np.sin(2 * np.pi * frecventa * timp)

    def semnal_c():
        frecventa = 240
        timp = np.linspace(0, 1, 10000)
        return timp * frecventa - np.floor(timp * frecventa)

    def semnal_d():
        frecventa = 300
        timp = np.linspace(0, 1, 10000)
        return pow(-1, np.floor(2 * frecventa * timp))

    fs = 44100
    sounddevice.play(semnal_a(), fs)
    sounddevice.wait()
    sounddevice.play(semnal_b(), fs)
    sounddevice.wait()
    sounddevice.play(semnal_c(), fs)
    sounddevice.wait()
    sounddevice.play(semnal_d(), fs)
    sounddevice.wait()

    rate = int(10e5)
    scipy.io.wavfile.write("semnal_a.wav", rate, semnal_a())


def ex_4():
    timp = np.linspace(0, 1, 10000)

    def semnal_sinusoidal(timp):
        frecventa = 400
        return np.sin(2 * np.pi * frecventa * timp)

    def semnal_sawtooth(timp):
        frecventa = 240
        return timp * frecventa - np.floor(timp * frecventa)

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 4")

    for index, semnal in enumerate(
            [semnal_sinusoidal(timp), semnal_sawtooth(timp), semnal_sinusoidal(timp) + semnal_sawtooth(timp)]):
        axs[index].plot(timp, semnal)
        axs[index].set_xlabel("Timp")
        axs[index].set_ylabel("Amplitudine")
        axs[index].set_xlim([0, 0.03])

    plt.tight_layout()
    plt.show()


def ex_5():
    def semnal_sinusoidal(frecventa, timp):
        return np.sin(2 * np.pi * frecventa * timp)

    timp = np.linspace(0, 2, 100000)
    semnal = np.concatenate((semnal_sinusoidal(200, timp), semnal_sinusoidal(400, timp)), axis=0)

    fs = 44100
    sounddevice.play(semnal, fs)
    sounddevice.wait()
    # pe masura ce frecventa creste, nota (sunetul) devine mai inalta


def ex_6():
    def semnal(f, timp):
        return np.sin(2 * np.pi * f * timp)

    fs = 1000
    timp = np.linspace(0, 0.2, fs)
    a = semnal(fs / 2, timp)
    b = semnal(fs / 4, timp)
    c = semnal(0, timp)

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 6")

    for index, semnal in enumerate([a, b, c]):
        axs[index].plot(timp, semnal)
        axs[index].stem(timp, semnal)
        axs[index].set_xlabel("Timp")
        axs[index].set_ylabel("Amplitudine")
        axs[index].set_xlim([0, 0.01])

    plt.tight_layout()
    plt.show()
    # Pe masura ce frecventa scade, numarul de oscilatii scade.


def ex_7():
    fs = 1000
    timp = np.linspace(0, 1, fs)
    semnal = np.sin(2 * np.pi * 100 * timp)

    def a():
        timp_decimat_a = timp[::4]
        semnal_decimat_a = semnal[::4]

        fig, axs = plt.subplots(2)
        fig.suptitle("Decimare la 1/4, pornind de la primul element")

        axs[0].plot(timp, semnal)
        axs[0].set_xlabel("Timp")
        axs[0].set_ylabel("Amplitudine")
        axs[0].set_xlim([0, 0.01])

        axs[1].plot(timp_decimat_a, semnal_decimat_a)
        axs[1].set_xlabel("Timp")
        axs[1].set_ylabel("Amplitudine")
        axs[1].set_xlim([0, 0.01])

        plt.tight_layout()
        plt.show()
        # Prin decimare, pastrandu-se mai putine esantioane decat in semnalul initial, forma semnalului se modifica,
        # devenind mai putin precisa.

    def b():
        timp_decimat_a = timp[1::4]
        semnal_decimat_a = semnal[1::4]

        fig, axs = plt.subplots(3)

        axs[0].plot(timp, semnal)
        axs[0].stem(timp, semnal)
        axs[0].set_xlabel("Timp")
        axs[0].set_ylabel("Amplitudine")
        axs[0].set_xlim([0, 0.01])
        axs[0].set_title("Semnal original")

        axs[1].plot(timp[::4], semnal[::4])
        axs[1].stem(timp[::4], semnal[::4])
        axs[1].set_xlabel("Timp")
        axs[1].set_ylabel("Amplitudine")
        axs[1].set_xlim([0, 0.01])
        axs[1].set_title("Semnal decimat la 1/4, pornind cu primul element")

        axs[2].plot(timp_decimat_a, semnal_decimat_a)
        axs[2].stem(timp_decimat_a, semnal_decimat_a)
        axs[2].set_xlabel("Timp")
        axs[2].set_ylabel("Amplitudine")
        axs[2].set_xlim([0, 0.01])
        axs[2].set_title("Semnal decimat la 1/4, pornind cu al doilea element")

        plt.tight_layout()
        plt.show()
        # Deoarece decimarea a inceput de la al doilea element, semnalul va incepe mai tarziu decat primul.

    a()
    b()


def ex_8():
    ##
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 10 ** 5)
    curba_1 = np.sin(alpha)
    curba_2 = alpha

    fig, axs = plt.subplots(3)

    axs[0].plot(alpha, curba_1)
    axs[0].axhline(0, c='black')
    axs[0].axvline(0, c='black')
    axs[0].set_title("Curba sin(alpha)")
    axs[0].grid()

    axs[1].plot(alpha, curba_2)
    axs[1].axhline(0, c='black')
    axs[1].axvline(0, c='black')
    axs[1].set_title("Curba alpha")
    axs[1].grid()

    axs[2].plot(alpha, np.abs(curba_1 - curba_2))
    axs[2].axhline(0, c='black')
    axs[2].axvline(0, c='black')
    axs[2].set_title("Eroarea aproximarii sin(alpha) = alpha")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    ##
    aprox_pade = (alpha - 7 * alpha ** 3 / 60) / (1 + alpha ** 2 / 20)

    fig, axs = plt.subplots(3)

    axs[0].plot(alpha, curba_1)
    axs[0].axhline(0, c='black')
    axs[0].axvline(0, c='black')
    axs[0].set_title("Curba sin(alpha)")
    axs[0].grid()

    axs[1].plot(alpha, aprox_pade)
    axs[1].axhline(0, c='black')
    axs[1].axvline(0, c='black')
    axs[1].set_title("Aproximarea Pade")
    axs[1].grid()

    axs[2].plot(alpha, np.abs(curba_1 - aprox_pade))
    axs[2].axhline(0, c='black')
    axs[2].axvline(0, c='black')
    axs[2].set_title("Eroarea aproximarii Pade")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    #########
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 10 ** 5)
    curba_1 = np.sin(alpha)
    curba_2 = alpha

    fig, axs = plt.subplots(3)

    axs[0].plot(alpha, curba_1)
    axs[0].axhline(0, c='black')
    axs[0].axvline(0, c='black')
    axs[0].set_title("Curba sin(alpha)")
    axs[0].grid()

    axs[1].plot(alpha, curba_2)
    axs[1].axhline(0, c='black')
    axs[1].axvline(0, c='black')
    axs[1].set_title("Curba alpha")
    axs[1].grid()

    axs[2].plot(alpha, np.abs(curba_1 - curba_2))
    axs[2].axhline(0, c='black')
    axs[2].axvline(0, c='black')
    axs[2].set_title("Eroarea aproximarii sin(alpha) = alpha")
    axs[2].grid()

    plt.tight_layout()
    plt.yscale('log')
    plt.show()

    ##
    aprox_pade = (alpha - 7 * alpha ** 3 / 60) / (1 + alpha ** 2 / 20)

    fig, axs = plt.subplots(3)

    axs[0].plot(alpha, curba_1)
    axs[0].axhline(0, c='black')
    axs[0].axvline(0, c='black')
    axs[0].set_title("Curba sin(alpha)")
    axs[0].grid()

    axs[1].plot(alpha, aprox_pade)
    axs[1].axhline(0, c='black')
    axs[1].axvline(0, c='black')
    axs[1].set_title("Aproximarea Pade")
    axs[1].grid()

    axs[2].plot(alpha, np.abs(curba_1 - aprox_pade))
    axs[2].axhline(0, c='black')
    axs[2].axvline(0, c='black')
    axs[2].set_title("Eroarea aproximarii Pade")
    axs[2].grid()

    plt.tight_layout()
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    ex_1()
    # ex_2()
    # ex_3()
    # ex_4()
    # ex_5()
    # ex_6()
    # ex_7()
    # ex_8()
