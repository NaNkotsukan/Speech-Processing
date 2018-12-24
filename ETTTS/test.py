import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,serializers
import numpy as np
import os
import argparse

xp=cuda.cupy
cuda.get_device(0).use()

rnd=np.random.randint(2400, 2450)
mel_name='BASIC5000_'+str(rnd)+'_mel.npy'
txt_name='BASIC5000_'+str(rnd)+'_txt.npy'

mel=np.load(mel_name).astype(np.float32)
txt=np.load(txt_name).astype(np.int32)

txt=xp.expand_dims(xp.array(txt),0)
x=xp.zeros((1,80,1)).astype(xp.float32)
cnt=100

for i in range(100):
    with chainer.using_config('train',False):
        vk = self.text_enc(txt)
        v = vk[:, :d, :]
        k = vk[:, d:, :]
        q = self.audio_enc(x)

        a = F.matmul(F.transpose(k, (0, 2, 1)), q)
        a = F.softmax(a / self.xp.sqrt(self.d))

        prva = -1
        for i in range(a.shape[2]):
            if (self.xp.argmax(a.data[0, :, i]) < prva - 1
                or prva + 3 < self.xp.argmax(a.data[0, :, i])):
                a = a.data
                a[0, :, i] = np.zeros(a.shape[1], dtype='f')
                pos = min(a.shape[1]-1, prva+1)
                a[0, pos, i] = 1
                a = chainer.Variable(a)
            prva = self.xp.argmax(a.data[0, :, i])

        r = F.matmul(v, a)
        rd = F.concat((r, q))

        y = self.audio_dec(rd)
        y = F.sigmoid(y)

    x=xp.concatenate((xp.zeros((1,80,1)).astype(xp.float32),y),axis=2)

    cnt-=1
    if xp.argmax(a[0,:,-1]) >= text.shape[0]-3:
        cnt=min(cnt,10)

    if cnt <= 0:
        break

    pylab.subplot(2,1,1)
    pylab.imshow(y[0])
    pylab.savefig('eval/comparison.png')

    pylab.subplot(2,1,2)
    pylab.imshow(mel[0])
    pylab.savefig('eval/comparison.png')
