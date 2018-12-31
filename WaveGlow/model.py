import chainer
import chainer.functions as F
import chainer.links as L
import numpy

def _normalize(W):
    xp = chainer.cuda.get_array_module(W)
    g = xp.sqrt(xp.sum(W ** 2, axis=(1, 2), keepdims=True))
    v = W / (g + 1e-16)
    return g, v


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

    g, v = _normalize(link.W.array)
    del link.W
    with link.init_scope():
        link.g = chainer.Parameter(g)
        link.v = chainer.Parameter(v)

    link.remove = _remove

    hook = chainer.LinkHook()
    hook.forward_preprocess = _replace
    link.add_hook(hook)
    return link


class Invertible1x1Convolution(chainer.link.Link):
    def __init__(self, channel):
        super(Invertible1x1Convolution, self).__init__()
        xp = self.xp

        W = xp.linalg.qr(xp.random.normal(
            0, 1, (channel, channel)))[0].astype(xp.float32)

        if xp.linalg.det(W) < 0:
            W[:, 0] *= -1
        W = W.reshape(W.shape + (1,))

        with self.init_scope():
            self.W = chainer.Parameter(W)

    @property
    def invW(self):
        return F.expand_dims(F.inv(self.W[..., 0]), axis=2)

    def __call__(self, x):
        return F.convolution_1d(x, self.W), \
            x.shape[0] * x.shape[-1] * F.log(F.det(self.W[..., 0]))

    def reverse(self, x):
        return F.convolution_1d(x, self.invW)


class WaveNet(chainer.Chain):
    def __init__(self, out_channel, n_condition, n_layers, n_channel):
        super(WaveNet, self).__init__()
        dilated_convs = chainer.ChainList()
        residual_convs = chainer.ChainList()
        skip_convs = chainer.ChainList()
        condition_convs = chainer.ChainList()
        for i in range(n_layers):
            dilated_convs.add_link(weight_norm(
                L.Convolution1D(
                    n_channel, 2 * n_channel, 3, pad=2 ** i, dilate=2 ** i)))
            residual_convs.add_link(weight_norm(
                L.Convolution1D(n_channel, n_channel, 1)))
            skip_convs.add_link(weight_norm(
                L.Convolution1D(n_channel, n_channel, 1)))
            condition_convs.add_link(weight_norm(
                L.Convolution1D(n_condition, 2 * n_channel, 1)))
        with self.init_scope():
            self.input_conv = weight_norm(
                L.Convolution1D(out_channel // 2, n_channel, 1))
            self.dilated_convs = dilated_convs
            self.residual_convs = residual_convs
            self.skip_convs = skip_convs
            self.condition_convs = condition_convs
            self.output_conv = L.Convolution1D(
                n_channel, out_channel, 1,
                initialW=chainer.initializers.Zero())

    def __call__(self, x, condition):
        x = self.input_conv(x)
        skip_connection = 0
        for dilated, residual, skip, condition_conv in zip(
                self.dilated_convs, self.residual_convs, self.skip_convs,
                self.condition_convs):
            z = dilated(x) + condition_conv(condition)
            z_tanh, z_sigmoid = F.split_axis(z, 2, axis=1)
            z = F.tanh(z_tanh) * F.sigmoid(z_sigmoid)
            x = residual(z)
            skip_connection += skip(z)
        y = self.output_conv(skip_connection)
        log_s, t = F.split_axis(y, 2, axis=1)
        return log_s, t


class AffineCouplingLayer(chainer.Chain):
    def __init__(self, channels, n_condition, n_layers, wn_channels):
        super(AffineCouplingLayer, self).__init__()
        with self.init_scope():
            self.encoder = WaveNet(channels, n_condition, n_layers, wn_channels)

    def __call__(self, x, condition):
        x_a, x_b = F.split_axis(x, 2, axis=1)
        log_s, t = self.encoder(x_a, condition)
        x_b = F.exp(log_s) * x_b + t
        return F.concat((x_a, x_b), axis=1), F.sum(log_s)

    def reverse(self, z, condition):
        x_a, x_b = F.split_axis(z, 2, axis=1)
        log_s, t = self.encoder(x_a, condition)
        x_b = (x_b - t) * F.exp(-log_s)
        return F.concat((x_a, x_b), axis=1)


class Flow(chainer.Chain):
    def __init__(self, channel, n_condition, n_layers, wn_channel):
        super(Flow, self).__init__()
        with self.init_scope():
            self.invertible1x1convolution = Invertible1x1Convolution(
                channel)
            self.affinecouplinglayer = AffineCouplingLayer(
                channel, n_condition, n_layers, wn_channel)

    def __call__(self, x, condition):
        x, log_det_W = self.invertible1x1convolution(x)
        z, log_s = self.affinecouplinglayer(x, condition)
        return z, log_s, log_det_W

    def reverse(self, z, condition):
        z = self.affinecouplinglayer.reverse(z, condition)
        x = self.invertible1x1convolution.reverse(z)
        
        return x

def _squeeze(x, squeeze_factor):
    batchsize, channel, length = x.shape
    x = x.reshape(
        (batchsize, channel, length // squeeze_factor, squeeze_factor))
    x = x.transpose((0, 1, 3, 2))
    x = x.reshape(
        (batchsize, channel * squeeze_factor, length // squeeze_factor))
    return x


def _unsqueeze(x, squeeze_factor):
    batchsize, channel, length = x.shape
    x = x.reshape(
        (batchsize, channel // squeeze_factor, squeeze_factor, length))
    x = x.transpose((0, 1, 3, 2))
    x = x.reshape(
        (batchsize, channel // squeeze_factor, length * squeeze_factor))
    return x


class Glow(chainer.Chain):
    def __init__(
            self, hop_length=256, n_mels=80, input_channel=1,
            squeeze_factor=8, n_flows=9, n_layers=4,
            wn_channel=512, early_every=4, early_size=2, var=0.5):
        super(Glow, self).__init__()
        self.input_channel = input_channel
        self.squeeze_factor = squeeze_factor
        self.n_flows = n_flows
        self.early_every = early_every
        self.early_size = early_size
        self.var = float(var)
        self.ln_var = float(numpy.log(var))
        flows = chainer.ChainList()
        for i in range(n_flows):
            flows.add_link(Flow(
                input_channel * squeeze_factor -
                early_size * (i // early_every),
                n_mels * squeeze_factor, n_layers, wn_channel))
        with self.init_scope():
            self.encoder = chainer.links.Deconvolution1D(
                n_mels, n_mels, hop_length * 4, hop_length,
                pad=hop_length * 3 // 2)
            self.flows = flows

    def __call__(self, x, condition):
        z, gaussian_nll, sum_log_s, sum_log_det_W = self._forward(x, condition)
        nll = gaussian_nll - sum_log_s - sum_log_det_W + float(numpy.log(2 ** 16))
        loss = chainer.functions.mean(z * z / (2 * self.var)) - \
            sum_log_s - sum_log_det_W

        return loss

    def _forward(self, x, condition):
        condition = self.encoder(condition)
        x = _squeeze(x, self.squeeze_factor)
        condition = _squeeze(condition, self.squeeze_factor)
        sum_log_s = 0
        sum_log_det_W = 0
        outputs = []
        for i, flow in enumerate(self.flows.children()):
            x, log_s, log_det_W = flow(x, condition)
            if (i + 1) % self.early_every == 0:
                output, x = x[:, :self.early_size], x[:, self.early_size:]
                outputs.append(output)
            sum_log_s += log_s
            sum_log_det_W += log_det_W
        outputs.append(x)
        z = chainer.functions.concat(outputs, axis=1)
        gaussian_nll = chainer.functions.gaussian_nll(
            z,
            mean=self.xp.zeros_like(z, dtype=self.xp.float32),
            ln_var=self.ln_var * self.xp.ones_like(z, dtype=self.xp.float32)
        )
        gaussian_nll /= numpy.prod(z.shape)
        sum_log_s /= numpy.prod(z.shape)
        sum_log_det_W /= numpy.prod(z.shape)
        return z, gaussian_nll, sum_log_s, sum_log_det_W

    def _reverse(self, z, condition, var=0):
        condition = self.encoder(condition)
        condition = _squeeze(condition, self.squeeze_factor)
        batchsize, _, length = condition.shape
        if z is None:
            z = self.xp.random.normal(
                0, var,
                (batchsize, self.input_channel * self.squeeze_factor, length))
            z = z.astype(self.xp.float32)
        _, channel, _ = z.shape
        start_channel = channel - \
            self.early_size * (self.n_flows // self.early_every)
        x, z = z[:, -start_channel:], z[:, :-start_channel]
        for i, flow in enumerate(reversed(list(self.flows.children()))):
            if (self.n_flows - i) % self.early_every == 0:
                x, z = chainer.functions.concat((
                    z[:, -self.early_size:], x)), z[:, :-self.early_size]
            x = flow.reverse(x, condition)
        x = _unsqueeze(x, self.squeeze_factor)
        return x

    def generate(self, condition, var=0.6 ** 2):
        return self._reverse(None, condition, var)