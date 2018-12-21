from librosa.core import load,stft,istft
from librosa.output import write_wav
import numpy as np
import argparse
import os
import string
from pykakasi import kakasi
from audio_process import Audio2MelSpectrogram

chars = string.ascii_lowercase + ',.-\"'

def stopword_check(s):
    for c in s:
        if c not in chars:
            return False

    return True

kakasi = kakasi()
kakasi.setMode('H','a')
kakasi.setMode('K','a')
kakasi.setMode('J','a')
converter = kakasi.getConverter()

def convert(s):
    conv = converter.do(s)
    conv = conv.replace('、',',')
    conv = conv.replace('。','.')
    
    return conv

sampling_rate = 16000

audio_path=''
if not os.path.exists(audio_path):
    print('Audiopath {} does not exist'.format(audio_path))

text_path=''
if not os.path.exists(text_path):
    print('Textpath {} does not exist'.format(text_path))

audio_list = os.listdir(audio_path)
for audio_name in audio_list:
    audio_name = audio_path + audio_name
    text_name = audio_name.split('.')[0] + '.txt'

    with open(text_name, 'r', 'utf8') as f:
        text = f.read()
    f.close()

    text = convert(text)
    valid=stopword_check(text)
    if valid:
        audio,sr = load(audio_list, sr=sampling_rate, mono=True)
        mel = Audio2MelSpectrogram(audio, sr, melsize=128, fftsize=512, windowsize=400, windowsshiftsize=160)
        mel = (mel / np.max(mel)) ** 0.6

        li = []
        for c in text:
            li.append(char)