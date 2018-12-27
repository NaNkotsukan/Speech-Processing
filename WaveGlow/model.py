import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np

xp=cuda.cupy
cuda.get_device(0).use()

class InvertibleConvolution(Chain):
    def __init__(self,in_ch):
        W=xp.linalg.qr(xp.random.normal(0,1,(in_ch,in_ch)))[0].astype(xp.float32)
        if xp.linalg.det(W):
            W[:,0]*=-1
        W=W.reshape(W.shape+(1,))
        super(IntertibleConvolution,self).__init__()

        with self.init_scope():
            self.w=chainer.Parameter(W)

    def invW(self):
        return F.expand_dims(F.inv(self.w[:,0]), axis=2)

    def __call__(self,x):
        conv=F.convolution_nd(x,self.w)
        logdet=F.log(F.det(self.w[...,0]))

        return conv, logdet

    def backward(self,x):
        return F.convolution_nd(x,self.invW)

def normalize(x):
    g=xp.sqrt(F.sum(x**2,axis=(1,2),keepdims=True)
    w=x/(g+1e-16)

    return w,g

def weight_norm(link):
    assert hasattr(link, 'W')

    def _W(self):
        return self.v * (self.g + 1e-16)

    def _remove(self):
        W = _W(self)
        del self.g
        del self.v
        del self.W
        with self.init_scope():
            self.W = chainer.Parameter(W)

    def _replace(args):
        W = _W(args.link)
        g, v = _normalize(W.array)
        args.link.g.array[...] = g
        args.link.v.array[...] = v
        args.link.W = _W(args.link)

    g, v = normalize(link.W.array)
    del link.W

    with link.init_scope():
        link.g = chainer.Parameter(g)
        link.v = chainer.Parameter(v)

    link.remove = _remove

    hook = chainer.LinkHook()
    hook.forward_preprocess = _replace
    link.add_hook(hook)

    return link

class WaveNet(Chain):
    def __init__(self,n_channels,out_channels,n_layers,cond_channels):
        dilated_conv=chinaer.ChainList()
        conditional_conv=chainer.ChainList()
        residual_conv=chainer.ChainList()
        skip_conv=chainer.ChainList()

        for layer in range(n_layers):
            dilated_conv.add_link(weight_norm(L.ConvolutionND(
                        1,n_channels,n_channels*2,3,pad=2**layer,dilate=2**layer)))
            conditional_conv.add_link(weight_norm(L.ConvolutionND(
                        1,n_channels,n_channels*2,1)))
            residual_conv.add_link(weight_norm(L.ConvolutionND(
                        1,n_channels,n_channels,1)))
            skip_conv.add_link(weight_rnom(L.ConvolutionND(
                        1,n_channels,n_channels,1)))

        super(WaveNet,self).__init__()
        with self.init_scope():
            self.cin=weight_norm(L.ConvolutionND(
                        1,out_channels//2,n_channels,1))
            self.dilated=dilated_conv
            self.conditional=conditional_conv
            self.residual=residual_conv
            self.skip=skip_conv
            self.cout=L.ConvolutionND(n_channels,out_channels,1)

    def __call__(self,x,condition):
        h=self.cin(x)
        skip_connections=0
        for dilated,cond,residual,skip in zip(self.dilated,self.conditional,self.residual,self.skip):
            h=dilated(h)+cond(condition)
            a,b=F.split(h,2,axis=1)
            z=F.tanh(a)+F.sigmoid(b)
            h=residual(z)
            z=skip(z)
            skip_connections+=z
        out=self.cout(skip_connections)
        logdet,out=F.split(h,2,axis=1)

        return logdet,out

class AffineCouplingLayer(Chain):
    def __init__(self,n_channels,out_channels,n_layers,cond_channels):
        super(AffineCouplingLayer,self).__init__()
        with self.init_scope():
            self.encoder=WaveNet(n_channels,out_channels,n_layers,cond_channels)

    def __call__(self,x,condition):
        a,b=F.split(x,2,axis=1)
        logdet,t=self.encoder(a,condition)
        b=F.exp(logdet)*b+t

        return F.concat([a,b],axis=1), F.sum(logdet)

    def backward(self,x,condtion):
        a,b=F.split(x,2,axis=1)
        logdet,t=self.encoder(a,condition)
        b=(b-t)/(F.exp(logdet))

        return F.concat([a,b],axis=1)

class Flow(Chain):
    def __init__(self,n_channels,out_channels,n_layers,cond_channels):
        super(Flow,self).__init__()
        with self.init_scope():
            self.int_conv=InvertibleConvolution(n_channels)
            self.acl=AffineCouplingLayer(n_channels,out_channels,n_layers,cond_channels)

    def __call__(self,x,condition):
        h,log_detW=self.int_conv(x)
        h,logz=self.acl(h,condition)

        return h,log_detW,logz

    def backward(self,x,condition):
        h=self.int_conv.backward(x)
        h=self.acl.backward(h,condition)

        return h

def squeeze(x,factor):
    b,c,l=x.shape
    h=x.reshape(b,c,int(l/factor),factor).transpose(0,1,3,2)

    return h.reshape(b,c*factor,int(l/factor))

def unsqueeze(x,factor):
    b,c,l=x.shape
    h=x.reshape(b,int(c/factor),factor,l).transpose(0,1,3,2)

    return h.reshape(b,int(c/factor),l*factor)

class WaveGlow(Chain):
    def __init__(self,
                 hop_length=256,
                 n_mels=80,
                 input_channels=1,
                 scale_factor=8,
                 n_flow=12,
                 n_layers=8,
                 n_channels=512,
                 early_layer=4,
                 early_channels=2,
                 var=0.5):
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.input_channels=1
        self.n_flows=12
        self.n_layers=8
        self.factor=scale_factor
        self.n_channels=channels
        self.early_layer=early_layer
        self.early_channels=early_channels
        self.var=var
        self.ln_var=xp.log(var)

        flow=chainer.ChainList()
        for i in range(n_flows):
            flow.add_link(Flow(
                input_channels*scale_factor-early_channels*(i//early_size),
                n_mels,n_layers,n_channels))

        super(WaveGlow,self).__init__()
        with self.init_scope():
            self.encoder=L.DeconvolutionND(1,n_mels,n_mels,
                            hop_length*4,hop_length,pad=hop_length*3//2)
            self.flows=flow

    def __call__(self,x,condition):
        h=self.encoder(x)
        h=squeeze(x, self.factor)
        condition=squeeze(condition, self.factor)
        sum_log_detW=0
        sum_logz=0
        outputs=[]
        for i,flow in enumerate(self.flows.children()):
            h,log_detW,logz=flow(h,condtition)
            if (i+1)%early_layer==0:
                output,h=h[;,:self.early_channels], h[;,self.early_channels:]
                outputs.append(output)
            sum_log_detW+=log_detW
            sum_logz+=logz
        outputs.append(h)
        z=F.concat(outputs,axis=1)
        gaussian_nll=F.gaussian_nll(
                z,
                mean=xp.zeros_like(z).astype(xp.float32),
                ln_var=self.ln_var*(xp.ones_like(z).astype(xp.float32))
        )
        gaussian_nll=gaussian_nll/xp.prod(z.shape)
        sum_log_detW /= xp.prod(z.shape)
        sum_logz /= xp.prod(z.shape)

        return z, gaussiah_nll, sum_log_detW, sum_logz

    def backward(self,x,condition,var=0):
        condition=self.encoder(condition)
        condition=squeeze(x,self.factor)
        b,_,l=condition.shape
        if x is None:
            x=xp.random.normal(0,1,(b,self.input_channels*self.factor,l)).astype(xp.float32)
        _, c, _=x.shape
        s_channels=c-self.early_channels*(self.n_flows//self.early_layer)
        x,z=x[:, -s_channels:],x[:, :-s_channels]
        for i,flow in enumerate(reversed(list(self.flows.children()))):
            if (self.n_flows-i)%self.early_layer:
                x,z=F.concat((z[;,-self.early_channels:],x)),z[;,:-self.early_channels]
            x=flow.reverse(x,condition)
        x=unsqueeze(x,self.factor)

        return x

    def generate(self,condition,var=0.06**2):
        return self.backward(None,condition,var)
