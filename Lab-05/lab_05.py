import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ex_1():
    # a --------------------------------------------------
    dataframe = pd.read_csv("Train.csv", dtype={
        "ID": "int64",
        "Datetime": "string",
        "Count": "int64"
    })
    n = len(dataframe)
    dataframe.Datetime = pd.to_datetime(dataframe.Datetime, format="%d-%m-%Y %H:%M")
    ts = (dataframe.Datetime[1] - dataframe.Datetime[0]).total_seconds()

    # for i, value in enumerate(dataframe.Datetime[1:]):
    #     if (value - dataframe.Datetime[i]).seconds != ts:
    #         print("Nu este esantionat la aceeasi frecventa")
    #         break

    print(f"Frecventa de esantionare: {1 / ts}")

    # b --------------------------------------------------
    interval = dataframe.Datetime[n - 1] - dataframe.Datetime[0]
    print(f"Intervalul de timp care acopera esantioanele din fisier: {interval}")

    # c --------------------------------------------------
    print(f"Frecventa maxima prezenta in semnal: {1 / (2 * ts)}")  # f = fs / 2 = 1 / (2 * ts)

    # d --------------------------------------------------
    x = np.fft.fft(dataframe.Count)  # Transformata Fourier a semnalului x
    x = np.abs(x / n)
    x = x[:n // 2]
    f = (1 / ts) * np.linspace(0, n // 2, n // 2) / n

    plt.plot(f, x, linewidth=1.7, color="red")
    plt.xlabel("Frecventa (Hz)")
    plt.ylabel("|X(ω)|")
    plt.title("Transformata Fourier inainte de eliminarea componentei continue")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig("Ex-1d/Figura_1.png")
    plt.savefig("Ex-1d/Figura_1.pdf")
    plt.grid()
    plt.show()
    plt.close()

    # e --------------------------------------------------
    # Semnalul prezinta o componenta continua: modulul transformatei Fourier are o valoare semnificativa pentru ω = 0
    print(f"Componenta continua: x[{np.argmax(np.abs(x))}] = {x[0]}")
    x[0] = 0

    plt.plot(f, x, linewidth=1.7, color="red")
    plt.title("Transformata Fourier dupa eliminarea componentei continue")
    plt.xlabel("Frecventa (Hz)")
    plt.ylabel("|X(ω)|")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig("Ex-1e/Figura_2.png")
    plt.savefig("Ex-1e/Figura_2.pdf")
    plt.grid()
    plt.show()
    plt.close()

    # f --------------------------------------------------
    first_frequencies_indexes = np.argsort(x)[::-1][:4]  # Primele 4 f sortate descrescator dupa modulul transformatei
    for index in first_frequencies_indexes:
        print(f"Frecventa: {f[index]} Hz; Modulul transformatei: {x[index]}")

    # ts = 1 / f[first_frequencies_indexes] - secunde
    # ts / 3600 / 24 - zile
    print(1 / f[first_frequencies_indexes] / 3600 / 24)
    # [761.91666667 380.95833333   0.99989064 253.97222222] (o zi - 1, 1 an - 365, 2 ani - 730)
    # [~2 ani,      ~1 an,         o zi,      ~8 luni     ]

    # g --------------------------------------------------
    start_index = 1000
    while True:
        if dataframe.Datetime[start_index].dayofweek == 0:
            break
        start_index += 1
    end_index = 24 * dataframe.Datetime[start_index].daysinmonth + start_index

    print(f"Intervalul ales: {dataframe.Datetime[start_index]} - {dataframe.Datetime[end_index]}")
    index_interval = range(start_index, end_index)

    plt.plot(index_interval, dataframe.Count[index_interval])
    plt.title(f"{dataframe.Datetime[start_index]} - {dataframe.Datetime[end_index]}")
    plt.xlabel("Esantioane")
    plt.ylabel("Numarul de masini")
    plt.xlim(left=start_index, right=end_index)
    plt.ylim(bottom=0)
    plt.savefig("Ex-1g/Figura_3.png")
    plt.savefig("Ex-1g/Figura_3.pdf")
    plt.grid()
    plt.show()
    plt.close()

    # i --------------------------------------------------
    threshold = f.mean()
    x_modified = x[f < threshold]
    f_modified = f[f < threshold]

    # Transformata Fourier dupa eliminarea componentelor cu frecvente mai mari decat media
    plt.plot(f_modified, x_modified)
    plt.xlabel("Frecventa (Hz)")
    plt.ylabel("|X(ω)|")
    plt.savefig("Ex-1i/Figura_4.png")
    plt.savefig("Ex-1i/Figura_4.pdf")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid()
    plt.show()
    plt.close()

    # Semnalul reconstruit dupa eliminarea componentelor cu frecvente mai mari decat media
    plt.plot(np.real(np.fft.ifft(x_modified)), label="Semnal filtrat")
    plt.plot(np.real(np.fft.ifft(x)), label="Semnal original")
    plt.savefig("Ex-1i/Figura_5.png")
    plt.savefig("Ex-1i/Figura_5.pdf")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


if __name__ == '__main__':
    ex_1()
