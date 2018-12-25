import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import pylab
from model import AudioDec,AudioEnc,TextEnc
from dataset import batch_make
from librosa.display import specshow

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0001,beta=0.9):
    optimizer=optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description='ETTTS')
parser.add_argument('--epochs',default=1000,type=int,help='the number of epochs')
parser.add_argument('--iterations',default=2000,type=int,help='the number of iterations')
parser.add_argument('--batchsize',default=16,type=int,help='batch size')
parser.add_argument('--interval',default=1,type=int,help="the interval of snapshot")

audio_dir='./data/audio/'
text_dir='./data/text/'

args=parser.parse_args()
epochs=args.epochs
iterations=args.iterations
batchsize=args.batchsize
interval=args.interval

outdir='outdir'
if not os.path.exists(outdir):
    os.mkdir(outdir)

audio_encoder=AudioEnc(256,80)
audio_encoder.to_gpu()
ae_opt=set_optimizer(audio_encoder)

audio_decoder=AudioDec(256,80)
audio_decoder.to_gpu()
ad_opt=set_optimizer(audio_decoder)

text_encoder=TextEnc(32,128,256)
text_encoder.to_gpu()
te_opt=set_optimizer(text_encoder)

for epoch in range(epochs):
    sum_loss_bin=0
    sum_loss_l1=0
    sum_loss_attn=0
    for batch in range(0,iterations,batchsize):
        text,x,t,textlens,xlens=batch_make(batchsize)

        text=chainer.as_variable(cuda.to_gpu(text))
        x=chainer.as_variable(cuda.to_gpu(x))
        t=chainer.as_variable(cuda.to_gpu(t))

        text_c=text_encoder(text)
        v=text_c[:,0:256,:]
        k=text_c[:,256:,:]
        q=audio_encoder(x)

        kq=F.matmul(F.transpose(k,(0,2,1)),q)
        a=F.softmax(kq/chainer.as_variable(xp.sqrt(256.0).astype(xp.float32)))
        r=F.matmul(v,a)
        r=F.concat([r,q])
        mel=audio_decoder(r)

        loss_bin=0
        for i in range(batchsize):
            loss_bin+=F.mean(F.bernoulli_nll(t[i,:,:xlens[i]],mel[i,:,:xlens[i]], 'no'))
        loss_bin /= batchsize

        mel=F.sigmoid(mel)

        loss_l1=0
        for i in range(batchsize):
            loss_l1+=F.mean_absolute_error(t[i,:,:xlens[i]],mel[i,:,:xlens[i]])
        loss_l1 /= batchsize

        loss_att=0
        for i in range(batchsize):
            N=textlens[i]
            T=xlens[i]
            def w_fun(n,t):
                return 1-np.exp(-((n/(N-1)-t/(T-1))**2)/(2*(0.2**2)))
            w=np.fromfunction(w_fun,(a.shape[1],T),dtype=np.float32)
            w=xp.array(w)
            loss_att+=F.mean(w*a[i,:,:T])
        loss_att /= batchsize

        loss=loss_bin+loss_l1+loss_att

        audio_encoder.cleargrads()
        audio_decoder.cleargrads()
        text_encoder.cleargrads()

        loss.backward()

        ae_opt.update()
        ad_opt.update()
        te_opt.update()

        loss.unchain_backward()

        sum_loss_bin+=loss_bin.data.get()
        sum_loss_l1+=loss_l1.data.get()
        sum_loss_attn+=loss_att.data.get()

        if epoch % interval==0 and batch==0:
            serializers.save_npz('audio_encoder.model',audio_encoder)
            serializers.save_npz('audio_decoder.model',audio_decoder)
            serializers.save_npz('text_encoder.model',text_encoder)

            pylab.subplot(3,1,1)
            specshow(((mel[0]**(-0.6))*100-100).data.get())
            pylab.savefig('comparison.png')

            pylab.subplot(3,1,2)
            specshow(a[0].data.get())
            pylab.savefig('comparison.png')

            pylab.subplot(3,1,3)
            specshow(((t[0]**(-0.6)*100)-100).data.get())
            pylab.savefig('comparison.png')

    print('epoch : {}'.format(epoch))
    print('Binary Loss : {}'.format(sum_loss_bin/iterations))
    print('L1 Loss : {}'.format(sum_loss_l1/iterations))
    print('Attention Loss : {}'.format(sum_loss_attn/iterations))
