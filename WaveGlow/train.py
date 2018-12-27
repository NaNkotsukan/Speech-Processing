import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
from model import WaveGlow
from audio_process import Audio2MelSpectrogram
from librosa.core import load
from librosa.output import write_wav

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0001,beta):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def data_process(filename):
    audio,sr=load(filename, sr=16000)
    rnd=np.random.randint(audio.shape[0]-1024)
    audio=audio[rnd:rnd+1024]
    mel=Audio2MelSpectrogram(audio,sr,melsize=80,fftsize=1024,windowsize=1024,windowsshiftsize=256)

    return audio,mels

waveglow=WaveGlow()
waveglow.to_gpu()
wg_opt=set_optimizer(waveglow)

audio_path=''
audio_list=os.listdir(audio_path)

for epoch in range(epochs):
    sum_loss=0
    for batch in range(0,Ntrain,batchsize):
        audio_name=audio_path+np.random.choice(audio_list)
        audio,mels=data_process(audio_name)
        z,gaussian_nll,log_detW,log_z=waveglow(audio,mels)
        nll=gausiaan_nll-log_detW-log_z+xp.log(2**16)
        loss=F.mean(z*z/(2*(var**2)))-log_detW-log_z
        
        waveglow.cleargrads()
        loss.backward()
        wg_opt.update()
        loss.unchain_backward()

        sum_loss+=loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz('waveglow.model',waveglow)

    print('epoch:{}'.format(epoch))
    print('Loss:{}').format(sum_loss/Ntrain)
