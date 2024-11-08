from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

from util import format_to_4_digits, get_label, get_vec, extract_features


class PlotData:
    def __init__(self) -> None:
        raise NotImplementedError()


def make_plots(start: int, end: int, mode: str):
    sampling_rate = 44100
    figure_width = 10
    match mode:
        case 'wf':  # waveform
            for i in range(start, end):
                T_index: str = format_to_4_digits(i)

                vec, label = get_vec(i), get_label(i)
                N = len(vec)  # samples
                t = N / sampling_rate
                t = np.linspace(0, t, len(vec))

                plt.figure().set_figwidth(figure_width)
                plt.plot(t, vec)
                plt.xlabel("Time, s")
                plt.ylabel("Amplitude")
                plt.title(f"Waveform of the signal, T{T_index}, class={label}")
                plt.grid(True)

                plt.savefig(f".\\fig\\waveform\\T{T_index}_plot.png")

                plt.clf()
                plt.close()
        case 'fft':  # fft
            for i in range(start, end):
                T_index: str = format_to_4_digits(i)

                vec, label = get_vec(i), get_label(i)

                N = len(vec)  # samples
                t = (N / sampling_rate)
                norm_factor = (2 / N) 
                freq_amp = norm_factor * abs(fft(vec)[1:N//2])
                freq = fftfreq(N, t/N)[1:N//2]

                plt.figure().set_figwidth(figure_width)
                plt.plot(freq, freq_amp, c='r')
                plt.xlabel("Frequency, Hz")
                plt.ylabel("Amplitude")
                plt.title(f"Spectrum of the signal, T{T_index}, class={label}")
                plt.grid(True)

                plt.savefig(f".\\fig\\frequency-domain\\T{T_index}_plot_freq.png")

                plt.clf()
                plt.close()
        case 'ef':  # extracted feature
            for i in range(start, end):
                T_index: str = format_to_4_digits(i)

                step = 100
                vec, label = get_vec(i), get_label(i)

                vec = extract_features(vec, step)
                t = np.linspace(0, 1, len(vec))

                plt.figure().set_figwidth(figure_width)
                plt.plot(t, vec, c='g')
                plt.xlabel("Samples")
                plt.ylabel("Amplitude")
                plt.title(f"Max pooling T{T_index}, class={label}")
                plt.grid(True)

                plt.savefig(f".\\fig\\extract-features\\T{T_index}_plot.png")

                plt.clf()
                plt.close()


if __name__ == "__main__":
    make_plots(1, 50, "ef")

    """ N = 1000

    t = np.linspace(-5, 5, N)
    fun = lambda x: np.cos(2 * np.pi * 3 * x) * np.exp(-np.pi * x ** 2)

    # plt.plot(t, fun(t), c='g')

    yf = (2/N) * abs(fft(fun(t))[1:500])
    xf = fftfreq(N, 10/N)[1:500]
    plt.plot(xf, yf)

    plt.show() """
