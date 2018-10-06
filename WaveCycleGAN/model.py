import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np
from instance_normalization import InstanceNormalization


xp=cuda.cupy
cuda.get_device(0).use()

class GLU(Chain):
    def __init__(self):
        super(GLU,self).__init__()
        
    def __call__(self,x):
        a,b=F.split_axis(x,2,axis=1)
        h=a*F.sigmoid(b)

        return h

def pixel_shuffler(out_ch, x, r = 2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0,3,4,1,5,2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))
    
    return out_map

class CIGLU(Chain):
    def __init__(self,in_ch,out_ch,up=False):
        super(CIGLU,self).__init__()
        w=initializers.Normal(0.02)
        self.up=up
        with self.init_scope():
            self.c0=L.Convolution1D(in_ch,out_ch,60,2,56,initialW=w)
            self.i0=InstanceNormalization(out_ch)
            self.glu0=GLU()

    def __call__(self,x):
        h=self.c0(x)
        if self.up:
            h=pixel_shuffler(h)
        h=self.glu0(self.i0(h))

        return h

class ResBlock(Chain):
    def __init__(self,in_ch,out_ch):
        super(ResBlock,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.ciglu0=CIGLU(in_ch,out_ch)
            self.ciglu1=CIGLU(out_ch,in_ch)

    def __call__(self,x):
        h=self.ciglu0(x)
        h=self.ciglu1(h)

        return h+x

class Generator(Chain):
    def __init__(self,base=32):
        super(Generator,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=L.Convolution1D(1,base,60,1,56,initialW=w)
            self.glu0=GLU()
            self.ciglu0=CIGLU(base,base*2)
            self.ciglu1=CIGLU(base*2,base)
            self.res0=ResBlock(base,base*2)
            self.res1=ResBlock(base,base*2)
            self.res2=ResBlock(base,base*2)
            self.res3=ResBlock(base,base*2)
            self.res4=ResBlock(base,base*2)
            self.res5=ResBlock(base,base*2)
            self.ciglu2=CIGLU(base,base,up=True)
            self.ciglu3=CIGLU(base,base*2,up=True)
            self.c1=L.Convolution1D(base*2,1,60,1,56,initialW=w)

    def __call__(self,x):
        h=self.glu0(self.c0(x))
        h=self.ciglu0(h)
        h=self.ciglu1(h)
        h=self.res0(h)
        h=self.res1(h)
        h=self.res2(h)
        h=self.res3(h)
        h=self.res4(h)
        h=self.res5(h)
        h=self.ciglu2(h)
        h=self.ciglu3(h)
        h=self.c1(h)

        return h

class Discriminator(Chain):
    def __init__(self,base=32):
        super(Discriminator,self).__init__()
        w=initializers.Normal
        with self.init_scope():
            self.c0=L.Convolution1D(1,base,60,1,56)
            self.glu=GLU()
            self.ciglu0=CIGLU(base,base*2)
            self.ciglu1=CIGLU(base*2,base)
            self.l0=L.Linear(None,1,initialW=w)

    def __call__(self,x):
        h=self.glu(self.c0(x))
        h=self.ciglu0(h)
        h=self.ciglu1(h)
        h=self.l0(h)

        return h
