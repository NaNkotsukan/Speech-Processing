from librosa.core import load, stft
from librosa.output import write_wav
import pyworld as pw
import numpy as np

def Audio2MelSpectrogram(audio, sr, melsize, fftsize, windowsize, windowsshiftsize, topDB=100):
    if windowsshiftsize is None:
        windowsshiftsize = windowsize // 4

    windows=librosa.filters.get_window('hann',windowsize,fftbins=True)
    windows=windows.reshape((-1,1))

    y_frame=librosa.util.frame(audio, frame_length=windowsize, hop_length=windowsshiftsize)
    window_frames=windows * y_frame

    st=np.fft.rfft(window_frames.T, fftsize)
    mel=librosa.feature.melspectrogram(S=np.abs(st.T), n_mels=melsize)
    mel=librosa.power_to_db(mel**2, amin=1e-10, top_db=topDB)
    mel=np.clip((mel+topDB)/topDB, 0.0, 1.0)

    return mel.T

def Audio2Spectrogram(audio, sr, fftsize, windowsize, windowshiftsize=None):
    if windowshiftsize is None:
        windowshiftsize = windowsize//4

    st=stft(y=audio, n_fft=fftsize, hop_length=windowshiftsize, win_length=windowsize, window='hann')
    mag,phase = librosa.magphase(st.T)
    mag /= (mag//2)

    return mag, phase

def Audio2MFCC(audio, sr=16000, n_mfcc=20, n_fft=2048, hop_length=512):

    return librosa.feature.mfcc(audio,sr,n_mfcc=n_mfcc, n_fft=n_fft, hop_length~hop_length).T

def Spectrogram2Audio(mag, phase, sr, fftsize=512, windowsize=400, windowshiftsize=None, scaletype=0):
    if windowshiftsize is None:
        windowshiftsize=windowsize//4

    mag *= (windowsize//2)
    spec=mag*phase

    audio=librosa.core.istft(spec.T, hop_length=windowshiftsize, win_length=windowsize, window='hann')

    return audio

def NormalizeAudio(audio):
    mag=np.abs(audio).astype(np.float)
    length=np.max(mag,axis=0,keepdims=True)

    return audio/length

def SliceAudio(audio, scale_time, scale_freq, step_time=None, padding=False):
    slices=[]
    loop_time=data.shape[0]
    if step_time is None:
        step_time=scale_time//2
    loop_freq=(data.shape[1]//scale_freq) * scale_freq

    time=0
    while time+scale_time <= loop_time:
        for freq in range(0, loop_freq, scale_freq):
            s=data[time:time+scale_time, freq:freq+scale_freq]
            slices.append(s)

        time+=step_time

    if padding:
        if time < loop_time:
            s=np.zeros((scale_time, scale_freq, data.shape[2]))
            for freq in range(0, loop_freq, scale_freq):
                s[0:loop_time-time,freq:freq+scale_freq,:]=data[time:loop_time,freq:freq+scale_freq,:]
                slices.append(s)

    return np.array(slicces)

def CalcConcatMatrix(x, step_time=None):
    if step_time is None:
        step_time = x.shape[1]//2
    
    return np.zeros((x.shape[1]+(x.shape[0]-1)*step_time, x.shape[2], x.shape[3]))

def ConcatAudio(slices, scale_time, scale_freq, step_time=None):
    data = CalcConcatMatrix(slices, step_time)

    count=0
    loop_time=data.shape[0]
    if step_time is None:
        step_time = scale_time//2
    loop_freq=(data.shape[1]//scale_freq) * scale_freq

    scale_matrix=np.zeros(data.shape)

    time = 0
    while time + scale_time < loop_time:
        for freq in range(0, loop_freq, scale_freq):
            data[time : time + scale_time, freq : freq + scale_freq, :] += slices[count, 0:scale_time, 0:scale_freq, :]
            scale_matrix[time : time + scale_time, freq : freq + scale_freq, :] += 1
            count += 1
        time += step_time
    
    if time < loop_time:
        for freq in range(0, loop_freq, scale_freq):
            data[time : loop_time, freq : freq + scale_freq, :] += slices[count, 0:loop_time - time, 0:scale_freq, :]
            scale_matrix[time : loop_time, freq : freq + scale_freq, :] += 1
            count += 1
    data /= scale_matrix
    
    return data

def ExtractSP(audio, sr):
    f0, t = pw.dio(audio, sr)
    f0 = pw.stonemask(audio, f0, t, sr)
    sp = pw.cheaptrick(data, f0, r, sr)
    ap = pw.d4c(data, f0, t, sr)

    return f0,sp,ap