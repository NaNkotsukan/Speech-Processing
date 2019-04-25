import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, serializers
import numpy as np
import os
import argparse
import pylab
from model import Generator, Discriminator
from lain.audio.navi import load, save
from lain.audio.layer.converter import audio2af, encode_sp, decode_sp, af2audio
from lain.audio.layer.augmentation import random_crop
from librosa.display import specshow
import soundfile as sf

def normalize(x, epsilon=1e-8):
    x_mean, x_std = np.mean(x, axis=1, keepdims=True), np.std(x, axis=1, keepdims=True)

    return (x - x_mean) / x_std, x_mean, x_std

xp = cuda.cupy
cuda.get_device(0).use()

x_sp_path = './Dataset/jsut_ver1.1/basic5000/sp/'
x_wav_path = './jsut_ver1.1/basic5000/wav16/'
x_list = os.listdir(x_wav_path)
x_len = len(x_list)

#x_wav_path = './Dataset/miria_trim/'
#x_list = os.listdir(x_wav_path)
#x_len = len(x_list)

generator_xy = Generator()
generator_xy.to_gpu()
serializers.load_npz('./generator_xy.model', generator_xy)
#serializers.load_npz('./generator_mtok.model', generator_xy)

x_sp_box = []

rnd = np.random.randint(x_len)
wav = x_wav_path + x_list[rnd]
#wav = './Dataset/jsut_ver1.1/basic5000/wav16/BASIC5000_3895.wav'
print(wav)
xx = load(wav, sampling_rate=16000)
f0, sp, ap = audio2af(xx)

f0_tmp = np.exp((np.ma.log(f0) - 5.35) / 0.24 * 0.27 + 5.67)
#f0_tmp = np.exp((np.ma.log(f0) - 6.20) / 0.29 * 0.27 + 5.67)

sp, mean, std = normalize(encode_sp(sp))
sp_y = np.zeros_like(sp, dtype=np.float64)
length = sp.shape[0]

for index in range(int(sp.shape[0]/128) + 1):
    if 128 * (index + 1) > length:
        sp_tmp = sp[128 * index : length]
        sp_tmp = np.pad(sp_tmp, ((0, 128 - sp_tmp.shape[0]),(0, 0)), 'constant', constant_values=0)

    else:
        sp_tmp = sp[128 * index : 128 * (index + 1)]

    x_sp_box.append(sp_tmp[np.newaxis, :])

x = chainer.as_variable(xp.array(x_sp_box).astype(xp.float32))

with chainer.using_config('train', False):
    y = generator_xy(x)
y = y.data.get()

for index in range(int(length/128) + 1):
    if 128 * (index + 1) > length:
        sp_y[128 * index : length] = y[index][0][0 : length - 128 * index]
    
    else:
        sp_y[128 * index : 128 * (index + 1)] = y[index][0]

print(sp_y.shape)

y = sp_y * std + mean
y = decode_sp(y)
audio = af2audio(f0_tmp, y, ap, sr=16000)
audio = audio.astype(np.float32)
sf.write('./test.wav', audio, 16000, subtype='PCM_16')