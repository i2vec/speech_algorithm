import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('/Users/i2vec/Desktop/xumingjun/speech_algorithm/speech_feature/CantinaBand3.wav')
# plt.plot(y)
# plt.title('Signal')
# plt.xlabel('Time (samples)')
# plt.ylabel('Amplitude')
# plt.savefig('signal.png')

import numpy as np

n_fft = 2048
ft = np.abs(librosa.stft(y[: n_fft], hop_length = n_fft + 1))
# plt.plot(ft)
# plt.title('Spectrum')
# plt.xlabel('Frequency Bin')
# plt.ylabel('Amplitude')
# plt.savefig('spectrum.png')

# spec = np.abs(librosa.stft(y, hop_length=512))
# spec = librosa.amplitude_to_db(spec, ref=np.max)
# librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram')
# plt.savefig('spectrogram.png')

spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.f dB')
plt.savefig('mel_spectorgram.png')