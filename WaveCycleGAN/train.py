import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Discriminator,Generator

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha,beta1=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta1)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="WaveCylceGAN")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",defaultZ-32,type=int,help="batchsize")
parser.add_argument("--testsize",default=4,type=int,help="testsize")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--cw",default=10.0,type=float,help="the weight of cycle loss")
parser.add_argument("--iw",default=5.0,type=float,help="the weight of identity loss")

args=parser.parse_args()
epochs=args.epoch
batchsize=args.batchsize
testsize=args.testsize
interval=args.interval
cycle_weight=args.cw
identity_weight=args.iw

outdir="./output"
if not os.path.exists(outdir):
    os.mkdir(outdir)

generator_xy=Generator()
generator_xy.to_gpu()
gen_opt_xy=set_optimizer(generator_xy,alpha=0.0002)

generator_yx=Generator()
generator_yx.to_gpu()
gen_opt_yx=set_optimizer(generator_yx,alpha=0.0002)

discriminator_xy=Discriminator()
discriminstor_xy.to_gpu()
dis_opt_xy=set_optimizer(discriminator_xy,alpha=0.0001)

discriminator_yx=Discriminator()
discriminator_yx.to_gpu()
dis_opt_yx=set_optimizer(discriminator_yx,alpha=0.0001)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,Ntrain,batchsize):
        input_box=[]
        output_box=[]
        for index in range(batchsize):


        x=chainer.as_variable(xp.array(input_box).astype(xp.float32))
        y=chainer.as_variable(xp.array(output_box).astype(xp.float32))

        x_y=generator_xy(x)
        x_y_x=generator_yx(x_y)

        y_x=generator_yx(y)
        y_x_y=generator_xy(y_x)

        dis_fake_xy=discriminator_xy(x_y)
        dis_real_xy=discriminator_xy(y)
        dis_loss_xy=F.mean(F.softplus(dis_fake_xy))+F.mean(F.softplus(-dis_real_xy))

        dis_fake_yx=discriminator_yx(y_x)
        dis_real_yx=discriminator_yx(x)
        dis_loss_yx=F.mean(F.softplus(dis_fake_yx))+F.mean(F.softplus(-dis_real_yx))

        discriminator_xy.cleargrads()
        dis_loss_xy.backward()
        dis_opt_xy.update()

        discriminator_yx.cleargrads()
        dis_loss_yx.backward()
        dis_opt_yx.update()

        gen_loss_xy=F.mean(F.softplus(-dis_fake_xy))
        gen_loss_yx=F.mean(F.softplus(-dis_fake_yx))
        cycle_xy=F.mean_absolute_error(x,x_y_x)
        cycle_yx=F.mean_absolute_error(y_x_y,y)
        gen_loss=gen_loss_xy+gen_loss_yx+cycle_weight*(cycle_xy+cycle_yx)

        generator_xy.cleargrads()
        generator_yx.cleargrads()
        gen_loss.backward()
        gen_opt_xy.update()
        gen_opt_yx.update()

        gen_loss.unchain_backward()
        dis_loss_xy.unchain_backward()
        dis_loss_yx.unchain_backward()

        sum_gen_loss+=dis_loss_xy.data.get()+dis_loss_yx.data.get()
        sum_dis_loss+=gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("generator_xy.model",generator_xy)
            serializers.save_npz("generator_yx.model",generator_yx)

    print("epoch : {}".format(epoch))
    print("Generator : {}".format(sum_gen_loss))
    print("Discriminator : {}".format(sum_dis_loss))


