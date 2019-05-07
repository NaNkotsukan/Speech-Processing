import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, serializers
import numpy as np
import os
import argparse
import pylab
from model import Generator, Discriminator
from lain.audio.navi import load, save
from lain.audio.layer.converter import audio2af, encode_sp
from lain.audio.layer.augmentation import random_crop
from librosa.display import specshow

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta1=0.5):
    """Adam optimizer setup
    
    Args:
        model : model paramter
        alpha (float, optional): Defaults to 0.0002. the alpha parameter of Adam
        beta1 (float, optional): Defaults to 0.5. the beta1 parameter of Adam
    """
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)

    return optimizer

def normalize(x):
    """normalization of spectral envelope. mean 0, variance 1
    
    Args:
        x (numpy.float): spectral envelope
        
    Returns:
        numpy.float: normalized spectral envelope
    """
    x_mean, x_std = np.mean(x, axis=1, keepdims=True), np.std(x, axis=1, keepdims=True)

    return (x - x_mean) / x_std

def crop(sp, upper_bound=128):
    """Cropping of spectral envelope
    
    Args:
        sp (numpy.float): spectral envelope
        upper_bound (int, optional): Defaults to 128. The size of cropping
    
    Returns:
        numpy.float: cropped spectral envelope
    """
    if sp.shape[0] < upper_bound + 1:
        sp = np.pad(sp, ((0, upper_bound-sp.shape[0] + 2), (0, 0)), 'constant', constant_values=0)

    start_point = np.random.randint(sp.shape[0] - upper_bound)
    cropped = sp[start_point : start_point + upper_bound, :]

    return cropped

parser = argparse.ArgumentParser(description='CycleGAN-VC2')
parser.add_argument('--epoch', default=1000, type=int, help="the number of epochs")
parser.add_argument('--batchsize', default=4, type=int, help="batch size")
parser.add_argument('--testsize', default=2, type=int, help="test size")
parser.add_argument('--Ntrain', default=2000, type=int, help="data size")
parser.add_argument('--cw', default=10.0, type=float, help="the weight of cycle loss")
parser.add_argument('--iw', default=5.0, type=float, help="the weight of identity loss")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
testsize = args.testsize
Ntrain = args.Ntrain
cycle_weight = args.cw
identity_weight = args.iw

x_path = './jsut_sp/'
x_list = os.listdir(x_path)
x_len = len(x_list)
y_path = './ayanami/'
y_list = os.listdir(y_path)
y_len = len(y_list)

generator_xy = Generator()
generator_xy.to_gpu()
gen_xy_opt = set_optimizer(generator_xy)

generator_yx = Generator()
generator_yx.to_gpu()
gen_yx_opt = set_optimizer(generator_yx)

discriminator_y = Discriminator()
discriminator_y.to_gpu()
dis_y_opt = set_optimizer(discriminator_y, alpha=0.0001)

discriminator_x = Discriminator()
discriminator_x.to_gpu()
dis_x_opt = set_optimizer(discriminator_x, alpha=0.0001)

#discriminator_xyx = Discriminator()
#discriminator_xyx.to_gpu()
#dis_xyx_opt = set_optimizer(discriminator_xyx, alpha=0.0001)

#discriminator_yxy = Discriminator()
#discriminator_yxy.to_gpu()
#dis_yxy_opt = set_optimizer(discriminator_yxy, alpha=0.0001)

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, Ntrain, batchsize):
        x_sp_box = []
        y_sp_box = []
        for _ in range(batchsize):
            # sp loading -> mel conversion -> normalization -> crop
            rnd_x = np.random.randint(x_len)
            sp_x = np.load(x_path + x_list[rnd_x])
            sp_x = normalize(encode_sp(sp_x, mel_bins=36))
            rnd_y = np.random.randint(y_len)
            sp_y = np.load(y_path + y_list[rnd_y])
            sp_y = normalize(encode_sp(sp_y, mel_bins=36))
            sp_x = crop(sp_x, upper_bound=128)
            sp_y = crop(sp_y, upper_bound=128)

            x_sp_box.append(sp_x[np.newaxis,:])
            y_sp_box.append(sp_y[np.newaxis,:])

        x = chainer.as_variable(xp.array(x_sp_box).astype(xp.float32))
        y = chainer.as_variable(xp.array(y_sp_box).astype(xp.float32))

        # Discriminator update
        xy = generator_xy(x)
        xyx = generator_yx(xy)

        yx = generator_yx(y)
        yxy = generator_xy(yx)

        xy.unchain_backward()
        xyx.unchain_backward()
        yx.unchain_backward()
        yxy.unchain_backward()

        dis_y_fake = discriminator_y(xy)
        dis_y_real = discriminator_y(y)
        dis_x_fake = discriminator_x(yx)
        dis_x_real = discriminator_x(x)

        #dis_xx_fake = discriminator_xyx(xyx)
        #dis_xx_real = discriminator_xyx(x)
        #dis_yy_fake = discriminator_yxy(yxy)
        #dis_yy_real = discriminator_yxy(y)

        dis_loss = F.mean(F.softplus(dis_y_fake)) + F.mean(F.softplus(-dis_y_real))
        dis_loss += F.mean(F.softplus(dis_x_fake)) + F.mean(F.softplus(-dis_x_real))
        #dis_loss += F.mean(F.softplus(dis_xx_fake)) + F.mean(F.softplus(-dis_xx_real))
        #dis_loss += F.mean(F.softplus(dis_yy_fake)) + F.mean(F.softplus(-dis_yy_real))

        discriminator_x.cleargrads()
        discriminator_y.cleargrads()
        #discriminator_xyx.cleargrads()
        #discriminator_yxy.cleargrads()
        dis_loss.backward()
        dis_x_opt.update()
        dis_y_opt.update()
        #dis_xyx_opt.update()
        #dis_yxy_opt.update()
        dis_loss.unchain_backward()

        # Generator update
        xy = generator_xy(x)
        xyx = generator_yx(xy)
        id_y = generator_xy(y)

        yx = generator_yx(y)
        yxy = generator_xy(yx)
        id_x = generator_yx(x)

        dis_y_fake = discriminator_y(xy)
        dis_x_fake = discriminator_x(yx)
        #dis_xx_fake = discriminator_xyx(xyx)
        #dis_yy_fake = discriminator_yxy(yxy)

        cycle_loss_x = F.mean_absolute_error(x, xyx)
        cycle_loss_y = F.mean_absolute_error(y, yxy)
        cycle_loss = cycle_loss_x + cycle_loss_y
        
        identity_loss_x = F.mean_absolute_error(id_y, y)
        identity_loss_y = F.mean_absolute_error(id_x, x)
        identity_loss = identity_loss_x + identity_loss_y

        if epoch > 20:
            identity_weight = 0.0
        
        gen_loss = F.mean(F.softplus(-dis_x_fake)) + F.mean(F.softplus(-dis_y_fake))
        #gen_loss += F.mean(F.softplus(-dis_xx_fake)) + F.mean(F.softplus(-dis_yy_fake))
        gen_loss += cycle_weight * cycle_loss + identity_weight * identity_loss

        generator_xy.cleargrads()
        generator_yx.cleargrads()
        gen_loss.backward()
        gen_xy_opt.update()
        gen_yx_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if batch == 0:
            serializers.save_npz('generator_xy.model', generator_xy)
            serializers.save_npz('generator_yx.model', generator_yx)

    print('epoch : {}'.format(epoch))
    print('Generator loss : {}'.format(sum_gen_loss / Ntrain))
    print('Discriminator loss : {}'.format(sum_dis_loss / Ntrain))