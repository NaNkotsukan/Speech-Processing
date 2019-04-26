import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np

xp=cuda.cupy
cuda.get_device(0).use()

class GLU(Chain):
    def __init__(self):
        super(GLU, self).__init__()
        
    def __call__(self, x):
        a, b = F.split_axis(x, 2, axis=1)
        h = a * F.sigmoid(b)

        return h

def pixel_shuffler(out_ch, x, r = 2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0,3,4,1,5,2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))
    
    return out_map

class CIGLU(Chain):
    def __init__(self, in_ch, out_ch, up=False, down=False):
        super(CIGLU, self).__init__()
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        with self.init_scope():
            self.cup = L.Convolution1D(in_ch,out_ch, 59, 1, 29, initialW=w)
            self.cdown = L.Convolution1D(in_ch, out_ch, 60, 2, 29, initialW=w)
            self.cpara = L.Convolution1D(in_ch, out_ch, 59, 1, 29, initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)
            self.glu0 = GLU()

    def __call__(self,x):
        if self.down:
            h = self.cdown(x)
            h = self.glu0(self.bn0(h))

        elif self.up:
            h = F.unpooling_1d(x,2,2,0,cover_all=False)
            h = self.cup(h)
            h = self.glu0(self.bn0(h))

        else:
            h = self.cpara(x)
            h = self.glu0(self.bn0(h))

        return h

class ResBlock(Chain):
    def __init__(self,in_ch,out_ch):
        super(ResBlock,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.ciglu0=CIGLU(in_ch,out_ch)
            self.ciglu1=CIGLU(in_ch,out_ch)

    def __call__(self,x):
        h=self.ciglu0(x)
        h=self.ciglu1(h)

        return h+x

class Generator(Chain):
    def __init__(self,base=32):
        super(Generator,self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution1D(1, base, 59, 1, 29, initialW=w)
            self.glu0 = GLU()
            self.ciglu0 = CIGLU(int(base/2), base*2, down=True)
            self.ciglu1 = CIGLU(base, base*2, down=True)
            self.res0 = ResBlock(base, base*2)
            self.res1 = ResBlock(base, base*2)
            self.res2 = ResBlock(base, base*2)
            self.res3 = ResBlock(base, base*2)
            self.res4 = ResBlock(base, base*2)
            self.res5 = ResBlock(base, base*2)
            self.ciglu2 = CIGLU(base, base*2,up=True)
            self.ciglu3 = CIGLU(base, base*2,up=True)
            self.c1 = L.Convolution1D(base, 1, 59, 1, 29,initialW=w)

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
            self.c0=L.Convolution1D(1, base*2, 59, 1, 29)
            self.glu=GLU()
            self.ciglu0=CIGLU(base, base*2)
            self.ciglu1=CIGLU(base, base*2)
            self.l0=L.Linear(None, 1, initialW=w)

    def __call__(self,x):
        h=self.glu(self.c0(x))
        h=self.ciglu0(h)
        h=self.ciglu1(h)
        h=self.l0(h)

        return h

class STFT(Chain):
    def __init__(self, fftsize=1024, hop_length=160, win_length=400, window=np.hanning):
        super(STFT, self).__init__()
        self.hop_length = hop_length
        self.n_bin = fftsize // 2

        weight_real = xp.cos(-2*xp.pi*xp.arange(fftsize).reshape((fftsize, 1)) * xp.arange(fftsize) / fftsize)[:fftsize//2]
        weight_imag = xp.som(-2*xp.pi*xp.arange(fftsize).reshape((fftsize, 1)) * xp.arange(fftsize) / fftsize)[:fftsize//2]

        window = window(win_length)
        window = xp.array(window.reshape(1, 1, 1, fftsize))

        self.add_persistent(
            'weight_real',
            window * weight_real.reshape((fftsize//2, 1, 1, fftsize))
        )

        self.add_persistent(
            'weight_imag',
            window * weight_imag.reshape((fftsize//2, 1, 1, fftsize))
        )

    def _convolve(self, x):
        x = x.transpose((0, 1, 3, 2))
        real = F.convolution_2d(x, self.weight_real, stride=(1, self.hop_length))
        imag = F.convolution_2d(x, self.weight_imag, stride=(1, self.hop_length))

        return real, imag

    def _power(self, x):
        real, imag = self._convolve(x)
        power = real ** 2 + imag ** 2

        return power

    def __call__(self, x):
        power = self._power(x)

        return F.sqrt(power)