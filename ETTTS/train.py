import chainer
import chainer.funcions as F
import chainer.links as L
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import AudioDec,AudioEnc,TextEnc

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description='ETTTS')
parser.add_argument('--epochs',default=1000,typ=int,help='the number of epochs')
parser.add_argument('--iterations',default=2000,type=int,help='the number of iterations')
parser.add_argument('--batchsize',default=16,type=int,help='batch size')

audio_dir='./data/audio/'
text_dir='./data/text/'

args=parser.parse_args()
epochs=args.epochs
iterations=args.iterations
batchsize=args.batchsize

outdir='outdir'
if not os.path.exists(outdir):
    os.mkdir(outdir)

audio_encoder=AudioEnc(80,256)
audio_encoder.to_gpu()
ae_opt=set_optimizer(audio_encoder)

audio_decoder=AudioDec(256,80)
audio_decoder.to_gpu()
ad_opt=set_optimizer(audio_decoder)

text_encoder=TextEnc(32,128,256)
text_encoder.to_gpu()
te_opt=set_optimizer(text_encoder)

for epoch in range(epochs):
    sum_loss=0
    for batch in range(0,iterations,batchsize):
        texts=[]
        mels=[]
        mellens=[]
        textlens=[]

        rnd=np.random.randint(Ntrain)
        audio_name=audio_dir+audio_list[rnd]
        text_name=text_dir+text_list[rnd]

        audio=np.load(audio_name)
        text=np.load(text_name)

        textlens.append(text.shape[0])
        mellens.append(mel.shape[1])

        ctext=np.pad(text,(0,np.max(textlens)-text.shape[0]),'constant',constant_values=31)
        cx=np.pad(mel,(0,np.max(mellens)-mel.shape[1]),'constant',constant_values=0)

        texts.append(ctext)
        mels.append(cx)

        text=xp.array(texts).astype(xp.int32)
        audio=xp.array(mels).astype(xp.float32)

        text=chainer.as_variable(text)
        audio=chainer.as_variable(audio)

        text_c=text_encoder(text)
        v=text_c[:,0:dims,:]
        k=text_c[:,dims:,:]
        q=audio_encoder(audio)

        kq=F.matmul(F.transpose(k,(0,2,1)),q)
        a=F.softmax(kq/F.sqrt(dims))
        r=F.matmul(v,a)
        r=F.concat([r,q])
        mel=audio_decoder(r)

        loss_bin=F.mean(F.bernoulli_nll(t,mel,'no'))
        loss_l1=F.mean_absolute_error(t,mel)
        w_nt=1-np.exp(-np.square(n/N-t/T)/(2*g*g))
        loss_att=F.mean(w_nt*a)

        loss=loss_bin+loss_l1+loss_att

        audio_encoder.cleargrads()
        audio_decoder.cleargrads()
        text_encoder.cleargrads()

        loss.backward()

        ae_opt.update()
        ad_opt.update()
        te_opt.update()

        loss.unchain_backward()

        if epoch % interval==0 and batch==0:
            serializers.save_npz('audio_encoder.model',audio_encoder)
            serializers.save_npz('audio_decoder.model',audio_decoder)
            serializers.save_npz('text_encoder.model',text_encoder)

    print('epoch : {}'.format(epoch))
    print('Loss : {}'.format(loss/iterations))
