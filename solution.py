import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import signal
from scipy.io import wavfile
import wave
import sys

tone1 = wavfile.read('../audio/maskon_tone.wav')
tone2 = wavfile.read('../audio/maskoff_tone.wav')

maskon_signal = np.array(tone1[1][0:116160], dtype=float)
maskoff_signal = np.array(tone2[1][0:116160], dtype=float)

maskon_signal -= np.mean(maskon_signal)
maskon_signal /= np.abs(maskon_signal).max()
maskoff_signal -= np.mean(maskoff_signal)
maskoff_signal /= np.abs(maskoff_signal).max()


maskon_frames = []
maskoff_frames = []
for i in range(100):
    maskon_frames.append(maskon_signal[i*160: i*160+320])
    maskoff_frames.append(maskoff_signal[i*160: i*160+320])

plt.title("Ramce")
plt.plot(maskon_frames[19])
#plt.plot(maskoff_frames[19])
#plt.legend(["s ruskou", "bez rusky"])
plt.savefig('../ramce.png')
plt.close()



maskon_clipped = []
maskoff_clipped = []


for i in range(100):
    maskoff_frame = maskoff_frames[i]
    maskon_frame = maskon_frames[i]
    ceil_maskon = np.abs(maskon_frame).max() * 0.7
    ceil_maskoff = np.abs(maskoff_frame).max() * 0.7
    clipped_maskon_frame = []
    clipped_maskoff_frame = []
    for j in range(320):
        if(maskoff_frame[j] > ceil_maskoff):
            clipped_maskoff_frame.append(1)
        elif(maskoff_frame[j] < -ceil_maskoff):
            clipped_maskoff_frame.append(-1)
        else:
            clipped_maskoff_frame.append(0)

        if(maskon_frame[j] > ceil_maskon):
            clipped_maskon_frame.append(1)
        elif(maskon_frame[j] < -ceil_maskon):
            clipped_maskon_frame.append(-1)
        else:
            clipped_maskon_frame.append(0)
    maskon_clipped.append(clipped_maskon_frame)
    maskoff_clipped.append(clipped_maskoff_frame)


plt.title("Center clipping")
plt.plot(maskon_clipped[19])
#plt.plot(maskoff_clipped[19])
#plt.legend(["s ruskou", "bez rusky"])
plt.savefig('../centerclipping.png')
plt.close()


correlated_maskon = []
correlated_maskoff = []

for i in range(len(maskon_clipped)):
    tmp = []
    for k in range(len(maskon_clipped[i])):
        s = 0
        for n in range(len(maskon_clipped[i])-1-k):
            s += maskon_clipped[i][n]*maskon_clipped[i][n+k]
        tmp.append(s)
    correlated_maskon.append(tmp)


for i in range(len(maskoff_clipped)):
    tmp = []
    for k in range(len(maskoff_clipped[i])):
        s = 0
        for n in range(len(maskoff_clipped[i])-1-k):
            s += maskoff_clipped[i][n]*maskoff_clipped[i][n+k]
        tmp.append(s)
    correlated_maskoff.append(tmp)


maskoff_mean = np.mean(correlated_maskoff)
maskoff_var = np.var(correlated_maskoff)
maskon_mean = np.mean(correlated_maskon)
maskon_var = np.var(correlated_maskon)

print(maskoff_mean, maskoff_var, maskon_mean, maskon_var)

prah = 10
max_maskon = np.argmax(correlated_maskon[2][prah:])
index_maskon = [prah + max_maskon]
value = np.amax(correlated_maskon[2][prah:])
val = [value]


plt.title("Autokorelacia")
plt.plot(correlated_maskon[2])
#plt.plot(correlated_maskoff[2])
plt.stem(index_maskon, val, linefmt='red', label ='lag')
plt.axvline(prah, color = 'grey', label='prah')
plt.legend()
plt.savefig('../autokorelacia.png')
plt.close()


f0_maskon = []
f0_maskoff = []

for i in range(len(correlated_maskon)):
    f0_maskon.append(16000/(np.argmax(correlated_maskon[i][30:]) + 30))


for i in range(len(correlated_maskoff)):
    f0_maskoff.append(16000/(np.argmax(correlated_maskoff[i][30:]) + 30))


plt.title("Zakladna frekvencia ramcov")
plt.plot(f0_maskon)
plt.plot(f0_maskoff)
plt.legend(["s ruskou", "bez rusky"])
plt.savefig('../test.png')
plt.close()


fft_maskon = 10 * np.log10([np.fft.fft(frame, 1024)[:512] for frame in maskon_frames])
fft_maskoff = 10 * np.log10([np.fft.fft(frame, 1024)[:512] for frame in maskoff_frames])


fft_maskon = np.array(fft_maskon)
fft_maskoff = np.array(fft_maskoff)

plt.title("Spektogram bez ruska")
plt.xlabel('[s]')
plt.ylabel('[Hz]')
plt.imshow(10 * np.log10(np.abs(fft_maskoff**2)).T, origin="lower", aspect='auto',
           extent=[0.0, 0.99, 0, 8000])
bar = plt.colorbar()
bar.set_label('Spektralna hustota vykonu [dB]', rotation=270, labelpad=15)
plt.savefig('../spectogram_mask_off.png')
plt.close()

plt.title("Spektogram s ruskom")
plt.xlabel('[s]')
plt.ylabel('[Hz]')
plt.imshow(10 * np.log10(np.abs(fft_maskon**2)).T, origin="lower", aspect='auto',
           extent=[0.0, 0.99, 0, 8000])
bar = plt.colorbar()
bar.set_label('Spektralna hustota vykonu [dB]', rotation=270, labelpad=15)
plt.savefig('../spectogram_mask_on.png')
plt.close()


Hjw = np.divide(fft_maskon, fft_maskoff)
filter_frequency = np.mean(np.abs(Hjw), axis=0)


plt.title("Frekvencna charakteristika rusky")
plt.xlabel('[s]')
plt.ylabel('[Hz]')
plt.plot(8000*np.linspace(0, 1, len(filter_frequency)),
         10 * np.log10(np.abs(filter_frequency**2)))
plt.savefig('../frekvencnachar.png')
plt.close()


filter_idft = np.fft.ifft(filter_frequency, 1024)[:512]

plt.title("Impluzivna odozva")
plt.plot(filter_idft)
plt.savefig('../odozva.png')
plt.close()



sentence1 = wave.open('../audio/maskon_sentence.wav', "r")
sentence2 = wave.open('../audio/maskoff_sentence.wav', "r")
sentencewithout = wavfile.read('../audio/maskoff_sentence.wav')

maskon_tone = scipy.signal.lfilter(filter_idft.real, [1], tone2[1])
maskon_sentence = scipy.signal.lfilter(filter_idft.real, [1], sentencewithout[1])



wavfile.write("../audio/sim_maskon_tone.wav", 16000, np.int16(maskon_tone))
wavfile.write("../audio/sim_maskon_sentence.wav", 16000, np.int16(maskon_sentence))


signal1 = sentence1.readframes(-1)
signal1 = np.fromstring(signal1, "Int16")
fs1 = sentence1.getframerate()

Time1 = np.linspace(0, len(signal1) / fs1, num=len(signal1))

plt.title("S ruskou")
plt.plot(Time1, signal1)
plt.savefig('../sruskou.png')
plt.close()



signal2 = sentence2.readframes(-1)
signal2 = np.fromstring(signal2, "Int16")
fs2 = sentence2.getframerate()

Time2 = np.linspace(0, len(signal2) / fs2, num=len(signal2))

plt.title("Bez rusky")
plt.plot(Time2, signal2)
plt.savefig('../bezruskou.png')
plt.close()


#sentencefilter = wave.open('../audio/sim_maskon_sentence.wav', "r")

#signal = sentencefilter.readframes(-1)
#signal = np.fromstring(signal, "Int16")
#fs = sentencefilter.getframerate()

#Time = np.linspace(0, len(signal) / fs, num=len(signal))

#plt.title("Simulovana")
#plt.plot(Time, signal)
#plt.savefig('../simulovana.png')
#plt.close()