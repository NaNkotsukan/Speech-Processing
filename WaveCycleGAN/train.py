import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Discriminator,Generator, STFT
from lain.audio.layer import augmentation
from lain.audio import navi
from arisu.loss import least_square_loss

xp=cuda.cupy
cuda.get_device(0).use()

def normalize(x):
    mean, std =np.mean(x), np.std(x)

    return (x - mean) / (std + 1e-8), mean, std

def set_optimizer(model, alpha, beta1=0.5, beta2=0.99):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="WaveCylceGAN")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",default=16,type=int,help="batchsize")
parser.add_argument("--testsize",default=2,type=int,help="testsize")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--cw",default=10.0,type=float,help="the weight of cycle loss")
parser.add_argument("--iw",default=5.0,type=float,help="the weight of identity loss")
parser.add_argument("--ver", default=1, type=int, help="choose WaveCycleGAN or WaveCycleGAN2")

args = parser.parse_args()
epochs = args.epochs
batchsize = args.batchsize
testsize = args.testsize
interval = args.interval
cycle_weight = args.cw
identity_weight = args.iw
version = args.ver

x_path = './Dataset/Speech/kaede_reconstruct/'
x_list = os.listdir(x_path)
Ntrain = len(x_list) - 10
t_path = './Dataset/Speech/kaede_trim/'

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

test_box1 = []
norm_box1 = []
test_box2 = []
norm_box2 = []
max_list = []
for index in range(testsize):
    rnd = np.random.randint(Ntrain, Ntrain + 10)
    x_name = t_path + x_list[rnd]
    x_tmp = navi.load(x_name, sampling_rate=16000)
    x = augmentation.random_crop(x_tmp, upper_bound=80000)
    x, mean, std = normalize(x)
    x = x.reshape(1, 80000)
    max_list.append(np.max(x_tmp))
    if index == 0:
        for i in range(40):
            test_box1.append(x[:, 2000 * i : 2000 * i + 2000])
            norm_box1.append(mean)
            norm_box1.append(std)
    if index == 1:
        for i in range(40):
            test_box2.append(x[:, 2000 * i : 2000 * i + 2000])
            norm_box2.append(mean)
            norm_box2.append(std)

    navi.save(outdir + 'original_{}.wav'.format(index), x_tmp, sampling_rate=16000)

x_test1 = chainer.as_variable(xp.array(test_box1).astype(xp.float32))
x_test2 = chainer.as_variable(xp.array(test_box2).astype(xp.float32))

generator_xy=Generator()
generator_xy.to_gpu()
gen_opt_xy=set_optimizer(generator_xy,alpha=0.0002)

generator_yx=Generator()
generator_yx.to_gpu()
gen_opt_yx=set_optimizer(generator_yx,alpha=0.0002)

discriminator_xy=Discriminator()
discriminator_xy.to_gpu()
dis_opt_xy=set_optimizer(discriminator_xy,alpha=0.0001)

discriminator_yx=Discriminator()
discriminator_yx.to_gpu()
dis_opt_yx=set_optimizer(discriminator_yx,alpha=0.0001)

#stft = STFT()
#stft.to_gpu()

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,Ntrain,batchsize):
        input_box=[]
        output_box=[]
        for index in range(batchsize):
            rnd = np.random.randint(Ntrain)
            x_name = x_path + x_list[rnd]
            t_name = t_path + x_list[rnd]
            x_tmp = navi.load(x_name, sampling_rate=16000)
            t_tmp = navi.load(t_name, sampling_rate=16000)
            x_crop, t_crop = augmentation.random_crop_double(x_tmp, t_tmp)
            x_crop, _, _ = normalize(x_crop)
            t_crop, _, _ = normalize(t_crop)
            x_crop = x_crop.reshape(1, 2000)
            t_crop = t_crop.reshape(1, 2000)
            input_box.append(x_crop)
            output_box.append(t_crop)

        x=chainer.as_variable(xp.array(input_box).astype(xp.float32))
        t=chainer.as_variable(xp.array(output_box).astype(xp.float32))

        x_y=generator_xy(x)
        y_x=generator_yx(t)

        x_y.unchain_backward()
        y_x.unchain_backward()

        dis_fake_xy=discriminator_xy(x_y)
        dis_real_xy=discriminator_xy(t)
        #dis_loss_xy=F.mean(F.softplus(dis_fake_xy))+F.mean(F.softplus(-dis_real_xy))
        dis_loss_xy = least_square_loss(dis_fake_xy, dis_real_xy)

        dis_fake_yx=discriminator_yx(y_x)
        dis_real_yx=discriminator_yx(x)
        #dis_loss_yx=F.mean(F.softplus(dis_fake_yx))+F.mean(F.softplus(-dis_real_yx))
        dis_loss_yx = least_square_loss(dis_fake_yx, dis_real_yx)

        dis_loss = dis_loss_xy + dis_loss_yx

        discriminator_xy.cleargrads()
        discriminator_yx.cleargrads()
        dis_loss.backward()
        dis_opt_xy.update()
        dis_opt_yx.update()
        dis_loss.unchain_backward()

        x_y = generator_xy(x)
        x_y_x = generator_yx(x_y)

        y_x = generator_yx(t)
        y_x_y = generator_xy(y_x)

        dis_fake_xy = discriminator_xy(x_y)
        dis_fake_yx = discriminator_yx(y_x)

        #gen_loss_xy = F.mean(F.softplus(-dis_fake_xy))
        gen_loss_xy = least_square_loss(dis_fake_xy)
        #gen_loss_yx = F.mean(F.softplus(-dis_fake_yx))
        gen_loss_yx = least_square_loss(dis_fake_yx)
        gen_loss = gen_loss_xy + gen_loss_yx

        cycle_xy = F.mean_absolute_error(x, x_y_x)
        cycle_yx = F.mean_absolute_error(t, y_x_y)
        cycle_loss = cycle_xy + cycle_yx

        id_x = generator_yx(x)
        id_y = generator_xy(t)
        identity_x = F.mean_absolute_error(id_x, x)
        identity_t = F.mean_absolute_error(id_y, t)
        identity_loss = identity_x + identity_t

        if epoch > 10:
            identity_weight = 0.0

        gen_loss += cycle_weight * cycle_loss + identity_weight * identity_loss

        generator_xy.cleargrads()
        generator_yx.cleargrads()
        gen_loss.backward()
        gen_opt_xy.update()
        gen_opt_yx.update()
        gen_loss.unchain_backward()

        sum_gen_loss += dis_loss.data.get()
        sum_dis_loss += gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("generator_xy.model",generator_xy)
            serializers.save_npz("generator_yx.model",generator_yx)

            with chainer.using_config('train', False):
                y1 = generator_xy(x_test1)
            y1.unchain_backward()
            y1 = y1.data.get()

            with chainer.using_config('train', False):
                y2 = generator_xy(x_test2)
            y2.unchain_backward()
            y2 = y2.data.get()

            tmp = y1 * norm_box1[1] + norm_box1[0]
            t1 = np.zeros(80000, dtype=np.float32)
            for i in range(40):
                t1[2000 * i : 2000 * i + 2000] = tmp[i][0]
            navi.save(outdir + 'convert_0.wav', t1, sampling_rate=16000)

            tmp = y2 * norm_box2[1] + norm_box2[0]
            t2 = np.zeros(80000, dtype=np.float32)
            for i in range(40):
                t2[2000 * i : 2000 * i + 2000] = tmp[i][0]
            navi.save(outdir + 'convert_1.wav', t2, sampling_rate=16000)

    print("epoch : {}".format(epoch))
    print("Generator : {}".format(sum_gen_loss / Ntrain))
    print("Discriminator : {}".format(sum_dis_loss / Ntrain))