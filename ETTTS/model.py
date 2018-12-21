import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers

class CasualConv(Chain):
    def __init__(self,in_ch,out_ch,ksize,dilate=1,casual=True,dropout=0.1):
        self.casual=casual
        self.dropout=dropout
        if self.casual:
            self.pad=(ksize-1) * dilate
        else:
            self.pad=(ksize-1) * dilate//2
        super(CasualConv,self).__init__(
            conv=L.DilatedConvolution2D(in_ch,out_ch,(1,ksize),1,(0,self.pad),(1,dilate,initialW=initializers.Normal(0.02)))
        )

    def __call__(self,x):
        h=F.expand_dims(x,2)
        h=self.conv(h)
        if self.casual and self.pad > 0:
            h=h[:,0,:-self.pad]
        else:
            h=h[:,0,:]
        h=F.dropout(h,self.dropout)

        return h

class HighwayNet(Chain):
    def __init__(self,d,ksize,dilate,casual):
        self.d=d
        super(HighwaNet,self).__init__(
            conv=CasualConv(d,2*d,ksize,dilate,casual)
              )

    def __call__(self,x):
        h=self.conv(x)
        h1=h[:,0:self.d,...]
        h2=h[:,self.d:,...]
        h3=F.sigmoid(h1)

        return h3*h2+(1-h3)*x

class TextEnc(Chain):
    def __init__(self,s,e,d):
        super(TextEnc,self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(s,e,initialW=initializers.Normal(0.02))
            self.c0=CasualConv(e,2*d,1,1,casual=False)
            self.c1=CasualConv(2*d,2*d,1,1,casual=False)
            self.h0=HighwayNet(2*d,3,1,casual=False)
            self.h1=HighwayNet(2*d,3,3,casual=False)
            self.h2=HighwayNet(2*d,3,9,casual=False)
            self.h3=HighwayNet(2*d,3,27,casual=False)
            self.h4=HighwayNet(2*d,3,1,casual=False)
            self.h5=HighwayNet(2*d,3,3,casual=False)
            self.h6=HighwayNet(2*d,3,9,casual=False)
            self.h7=HighwayNet(2*d,3,27,casual=False)
            self.h8=HighwayNet(2*d,3,1,casual=False)
            self.h9=HighwayNet(2*d,3,3,casual=False)
            self.h10=HighwayNet(2*d,3,9,casual=False)
            self.h11=HighwayNet(2*d,3,27,casual=False)

    def __call__(self,x):
        h=self.embed(x)
        h=F.transpose(h,(0,2,1))
        h=F.relu(self.c0(h))
        h=self.c1(h)
        h=self.h0(h)
        h=self.h1(h)
        h=self.h2(h)
        h=self.h3(h)
        h=self.h4(h)
        h=self.h5(h)
        h=self.h6(h)
        h=self.h7(h)
        h=self.h8(h)
        h=self.h9(h)
        h=self.h10(h)
        h=self.h11(h)

        return h

class AudioEnc(Chain):
    def __init__(self,f,d):
        super(AudioEnc,self).__init__()
        with self.init_scope():
            self.c0=CasualConv(f,d,1,1)
            self.c1=CasualConv(d,d,1,1)
            self.c2=CasualConv(d,d,1,1)
            self.h0=HighwayNet(d,3,1)
            self.h1=HighwayNet(d,3,3)
            self.h2=HighwayNet(d,3,9)
            self.h3=HighwayNet(d,3,27)
            self.h4=HighwayNet(d,3,1)
            self.h5=HighwayNet(d,3,3)
            self.h6=HighwayNet(d,3,9)
            self.h7=HighwayNet(d,3,27)
            self.h8=HighwayNet(d,3,3)
            self.h9=HighwayNet(d,3,3)

    def __call__(self,x):
        h=F.relu(self.c0(x))
        h=F.relu(self.c1(h))
        h=self.c2(h)
        h=self.h0(h)
        h=self.h1(h)
        h=self.h2(h)
        h=self.h3(h)
        h=self.h4(h)
        h=self.h5(h)
        h=self.h6(h)
        h=self.h7(h)
        h=self.h8(h)
        h=self.h9(h)

        return h

class AudioDec(Chain):
    def __init__(self,d,f):
        super(AudioDec,self).__init__()
        with self.init_scope():
            self.c0=CasualConv(2*d,d,1,1)
            self.c1=HighwayNet(d,3,1)
            self.c2=HighwayNet(d,3,3)
            self.c3=HighwayNet(d,3,9)
            self.c4=HighwayNet(d,3,27)
            self.c5=HighwayNet(d,3,1)
            self.c6=HighwayNet(d,3,1)
            self.c7=CasualConv(d,d,1,1,dropout=0)
            self.c8=CasualConv(d,d,1,1,dropout=0)
            self.c9=CasualConv(d,d,1,1,dropout=0)
            self.c10=CasualConv(d,f,1,1,dropout=0)

    def __call__(self,x):
        h=self.c0(h)
        h=self.c1(h)
        h=self.c2(h)
        h=self.c3(h)
        h=self.c4(h)
        h=self.c5(h)
        h=self.c6(h)
        h=F.relu(self.c7(h))
        h=F.relu(self.c8(h))
        h=F.relu(self.c9(h))
        h=self.c10(h)
        
        return h
