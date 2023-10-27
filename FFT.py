import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000, endpoint = False)
inputSignal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

t2 = np.linspace(0, 2 * np.pi, 1000, endpoint = False)
inputSignal2 = 2 * np.sin(3 * t2) + 1.5 * np.sin(7 * t2) + 0.5 * np.sin(15 * t2)

def fft(signal):
    n = len(signal)
    freq = np.fft.fftfreq(n, 1.0 / n)
    FFTVals = np.fft.fft(signal)
    return freq, FFTVals

def plotFFT(signal, freq, FFTVals):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title('Input Signal')
    plt.plot(signal)

    plt.subplot(2, 1, 2)
    plt.title('FFT Values')
    plt.plot(freq, np.abs(FFTVals))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    
print("input signal 1:")
def main():
    freq, FFTVals = fft(inputSignal)
    plotFFT(inputSignal, freq, FFTVals)

if __name__ == '__main__':
    main()

print("input signal 2:")
def main():
    freq, FFTVals = fft(inputSignal2)
    plotFFT(inputSignal2, freq, FFTVals)

if __name__ == '__main__':
    main()
