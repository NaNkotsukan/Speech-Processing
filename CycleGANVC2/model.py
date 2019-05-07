import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers
import numpy as np

xp = cuda.cupy
cuda.get_device(0).use()

def glu(x):
    a, b = F.split_axis(x, 2, axis=1)

    return a * F.sigmoid(b)

class C2BG(Chain):
    """2D convolution -> Batch Normalization -> Gated linear unit

    """
    def __init__(self, in_ch, out_ch, up=False, down=False):
        super(C2BG, self).__init__()
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        with self.init_scope():
            self.cup = L.Convolution2D(in_ch, out_ch,5,1,2,initialW=w)
            self.cpara = L.Convolution2D(in_ch, out_ch,3,1,1,initialW=w)
            self.cdown = L.Convolution2D(in_ch, out_ch, 4,2,1,initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        if self.up:
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = glu(self.bn0(self.cup(h)))

        elif self.down:
            h = glu(self.bn0(self.cdown(x)))

        else:
            h = glu(self.bn0(self.cpara(x)))

        return h

class C1BG(Chain):
    """1D convolution -> Batch Normalization -> Gated linear unit
    
    """
    def __init__(self, in_ch, out_ch, up=False, down=False):
        super(C1BG, self).__init__()
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        with self.init_scope():
            self.cup = L.Convolution1D(in_ch, out_ch,3,1,1,initialW=w)
            self.cpara = L.Convolution1D(in_ch, out_ch,3,1,1,initialW=w)
            self.cdown = L.Convolution1D(in_ch, out_ch, 4,2,1,initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        if self.up:
            h = F.unpooling_1d(x, 2, 2, 0, cover_all=False)
            h = glu(self.bn0(self.cup(h)))

        if self.down:
            h = glu(self.bn0(self.cdown(x)))

        else:
            h = glu(self.bn0(self.cpara(x)))

        return h

class ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.cbg0 = C1BG(in_ch, out_ch)
            self.c0 = L.Convolution1D(in_ch, in_ch, 3,1,1,initialW=w)
            self.bn0 = L.BatchNormalization(in_ch)

    def __call__(self, x):
        h = self.cbg0(x)
        h = self.bn0(self.c0(h))

        return h + x

class Generator(Chain):
    def __init__(self, base=128):
        w = initializers.Normal(0.02)
        super(Generator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(1, base, (15, 5), 1, (7, 2), initialW=w)
            self.cbg0 = C2BG(int(base/2), base*2, down=True)
            self.cbg1 = C2BG(base, base*4, down=True)
            
            self.c1 = L.Convolution1D(2304, base*2, 1, 1, 0, initialW=w)
            self.bn1 = L.BatchNormalization(base*2)

            self.res0 = ResBlock(base*2, base*4)
            self.res1 = ResBlock(base*2, base*4)
            self.res2 = ResBlock(base*2, base*4)
            self.res3 = ResBlock(base*2, base*4)
            self.res4 = ResBlock(base*2, base*4)
            self.res5 = ResBlock(base*2, base*4)

            self.c2 = L.Convolution1D(base*2, 2304, 1, 1, 0, initialW=w)
            self.bn2 = L.BatchNormalization(2304)

            self.cbg2 = C2BG(base*2, base*8, up=True)
            self.cbg3 = C2BG(base*4, 72, up=True)
            
            self.c3 = L.Convolution2D(36, 1, 3, 1, 1, initialW=w)

    def __call__(self, x):
        b = x.shape[0]
        h = glu(self.c0(x))
        h = self.cbg0(h)
        h = self.cbg1(h)
        h = F.transpose(h, (0, 1, 3, 2)).reshape(b, 2304, 32)
        h = self.bn1(self.c1(h))
        h = self.res0(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.bn2(self.c2(h))
        h = F.transpose(F.reshape(h, (b, 256, 9, 32)), (0, 1, 3, 2))
        h = self.cbg2(h)
        h = self.cbg3(h)
        h = self.c3(h)

        return h

class BG(Chain):
    def __init__(self, out_ch):
        super(BG, self).__init__()
        with self.init_scope():
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = glu(self.bn0(x))

        return h

class Discriminator(Chain):
    def __init__(self,base=64):
        w = initializers.Normal(0.02)
        super(Discriminator, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(1, base*2, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(base, base*4, 3, 2, 1, initialW=w)
            self.bg1 = BG(base*4)
            self.c2 = L.Convolution2D(base*2, base*8, 3, 2, 1, initialW=w)
            self.bg2 = BG(base*8)
            self.c3 = L.Convolution2D(base*4, base*16, 3, 2, 1, initialW=w)
            self.bg3 = BG(base*16)
            self.c4 = L.Convolution2D(base*8, base*16, (5, 1), 1, (2, 0), initialW=w)
            self.bg4 = BG(base*16)
            self.c5 = L.Convolution2D(base*8, 1, (3, 1), 1, (1, 0), initialW=w)

    def __call__(self, x):
        h = glu(self.c0(x))
        h = self.bg1(self.c1(h))
        h = self.bg2(self.c2(h))
        h = self.bg3(self.c3(h))
        h = self.bg4(self.c4(h))
        h = self.c5(h)

        return h