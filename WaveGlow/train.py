import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
from model import Glow
from librosa.core import load
from librosa.output import write_wav
import librosa

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0001,beta=0.9):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def data_process(filename, length):
    audio,sr=load(filename)
    rnd=np.random.randint(audio.shape[0]-length)
    audio=audio[rnd:rnd+length]
    
    mel=librosa.feature.melspectrogram(audio, sr, n_fft=1024, hop_length=256, n_mels=80)
    mel=(mel / np.max(mel))**0.6
    mel=mel[:, :int(length/256)]
    audio = np.expand_dims(np.expand_dims(audio,axis=0), axis=0)
    mel = np.expand_dims(mel,axis=0)

    return audio,mel

parser=argparse.ArgumentParser(description="WaveGlow")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",default=16,type=int,help="batch size")
parser.add_argument("--var",default=0.5,type=float,help="variance")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")
parser.add_argument("--length",default=8192*2,type=int,help="the number of iterations")

args=parser.parse_args()
epochs=args.epochs
batchsize=args.batchsize
var=args.var
interval=args.interval
iterations=args.iterations
length=args.length

waveglow=Glow()
waveglow.to_gpu()
wg_opt=set_optimizer(waveglow)

audio_path='/JSUT/basic5000/wav/'
audio_list=os.listdir(audio_path)

for epoch in range(epochs):
    sum_loss=0
    for batch in range(0,iterations,batchsize):
        audio_name=audio_path+np.random.choice(audio_list)
        audio,mels=data_process(audio_name, length)

        audio=chainer.as_variable(xp.array(audio).astype(xp.float32))
        mels=chainer.as_variable(xp.array(mels).astype(xp.float32))

        loss=waveglow(audio,mels)
        
        waveglow.cleargrads()
        loss.backward()
        wg_opt.update()
        loss.unchain_backward()

        sum_loss+=loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz('waveglow.model',waveglow)

    print('epoch:{}'.format(epoch))
    print('Loss:{}'.format(sum_loss/iterations))