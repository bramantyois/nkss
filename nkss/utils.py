import numpy as np


def generate_sine_sweep(start_freq=20, stop_freq=1000, duration=0.1, sample_rate=44100):
    """
    generate sine sweep
    :param start_freq: lowest frequency
    :param stop_freq: highest frequency
    :param duration: duration of generated signal
    :param sample_rate: sample rate
    :return: sine sweep signal with amplitude of 1
    """
    num_samples = int(duration*sample_rate)

    freqs = np.linspace(2 * np.pi * start_freq, 2 * np.pi * stop_freq, num_samples)

    t = np.linspace(0, duration, num_samples)

    return np.sin(freqs * t)


def generate_ascending_sine(start_amp=0, stop_amp=1, duration=0.1, freq=1e3, sample_rate=44100):
    """
    generate ascending sine
    :param start_amp: amplitude at the start
    :param stop_amp: amplitude at the end
    :param duration: duration of generated signal
    :param freq: frequency of generated signal
    :param sample_rate: sample rate
    :return:
    """
    num_samples = int(duration * sample_rate)

    amps = np.linspace(start_amp, stop_amp, num_samples)

    t = np.linspace(0, duration, num_samples)

    return amps*np.sin(2 * np.pi * freq * t)


def sigmoid(x, scale=1):
    """
    sigmoid function
    :param x: float var
    :param scale: scaler
    :return:
    """
    return scale / (1 + np.exp(-x))
