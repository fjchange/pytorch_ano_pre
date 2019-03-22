import torch

import sys

from util import psnr_error
from losses import *

sys.path.append('..')
from Dataset import img_dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from liteFlownet.lite_flownet import Network,batch_estimate
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter


from utils import utils
import os

training_data_folder='your_path'
testing_data_folder='your_path'

writer_path='your_path'
model_generator_save_path='your_path'
model_discriminator_save_path='your_path'

lite_flow_model_path='your_path'

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

#----------- all the param as ano pred said ---------
batch_size=4
epochs=20000
pretrain=False

# color dataset
g_lr=0.0002
d_lr=0.00002

lam_int=1.0
lam_gd=1.0
lam_op=2.0
lam_adv=0.05

#for gradient loss
alpha=1
#for int loss
l_num=2

num_clips=5
num_his=1
num_unet_layers=4

num_channels=3#avenue is 3, UCSD is 1
discriminator_channels=[128,256,512,512]

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train(frame_num,layer_nums,input_channels,output_channels,discriminator_num_filters,bn=False):
    generator=UNet(n_channels=input_channels,layer_nums=layer_nums,output_channel=output_channels,bn=bn)
    discriminator=PixelDiscriminator(output_channels,discriminator_num_filters,use_norm=False)

    generator = generator.cuda()
    discriminator = discriminator.cuda()

    flow_network=Network()
    flow_network.load_state_dict(torch.load(lite_flow_model_path))
    flow_network.cuda().eval()

    adversarial_loss=Adversarial_Loss().cuda()
    discriminate_loss=Discriminate_Loss().cuda()
    gd_loss=Gradient_Loss(alpha,num_channels).cuda()
    op_loss=Flow_Loss().cuda()
    int_loss=Intensity_Loss(l_num).cuda()

    if not pretrain:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    print('initializing the model with Generator-Unet {} layers,'
          'PixelDiscriminator with filters {} '.format(layer_nums,discriminator_num_filters))

    optimizer_G=torch.optim.Adam(generator.parameters(),lr=g_lr)
    optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=d_lr)


    writer=SummaryWriter(writer_path)

    dataset=img_dataset.ano_pred_Dataset(training_data_folder,frame_num)
    dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    test_dataset=img_dataset.ano_pred_Dataset(testing_data_folder,frame_num)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=1,drop_last=True)

    step = 0
    for epoch in range(epochs):
        for (input,_),(test_input,_) in zip(dataset_loader,test_dataloader):
            # generator = generator.train()
            # discriminator = discriminator.train()

            target=input[:,-1,:,:,:].cuda()

            input=input[:,:-1,]
            input_last=input[:,-1,].cuda()
            input=input.view(input.shape[0],-1,input.shape[-2],input.shape[-1]).cuda()

            test_target=test_input[:,-1,].cuda()
            test_input=test_input[:,:-1].view(test_input.shape[0],-1,test_input.shape[-2],test_input.shape[-1]).cuda()

            #------- update optim_G --------------

            G_output=generator(input)

            pred_flow_esti_tensor=torch.cat([input_last,G_output],1)
            gt_flow_esti_tensor=torch.cat([input_last,target],1)

            flow_gt=batch_estimate(gt_flow_esti_tensor,flow_network)
            flow_pred=batch_estimate(pred_flow_esti_tensor,flow_network)

            g_adv_loss=adversarial_loss(discriminator(G_output))
            g_op_loss=op_loss(flow_pred,flow_gt)
            g_int_loss=int_loss(G_output,target)
            g_gd_loss=gd_loss(G_output,target)

            g_loss=lam_adv*g_adv_loss+lam_gd*g_gd_loss+lam_op*g_op_loss+lam_int*g_int_loss

            optimizer_G.zero_grad()

            g_loss.backward()
            optimizer_G.step()

            train_psnr=psnr_error(G_output,target)

            #----------- update optim_D -------
            optimizer_D.zero_grad()

            d_loss=discriminate_loss(discriminator(target),discriminator(G_output.detach()))
            #d_loss.requires_grad=True

            d_loss.backward()
            optimizer_D.step()

            #----------- cal psnr --------------
            test_generator=generator.eval()
            test_output=test_generator(test_input)
            test_psnr=psnr_error(test_output,test_target).cuda()

            if step%10==0:
                print("[{}/{}]: g_loss: {} d_loss {}".format(step,epoch,g_loss,d_loss))
                print('\t gd_loss {}, op_loss {}, int_loss {}'.format(g_gd_loss,g_op_loss,g_int_loss))
                print('\t train psnr{}，test_psnr {}'.format(train_psnr,test_psnr))

                writer.add_scalar('train_psnr', train_psnr, global_step=step)
                writer.add_scalar('test_psnr', test_psnr, global_step=step)

                writer.add_scalar('g_loss', g_loss, global_step=step)
                writer.add_scalar('d_loss', d_loss, global_step=step)
                writer.add_scalar('adv_loss', g_adv_loss, global_step=step)
                writer.add_scalar('op_loss', g_op_loss, global_step=step)
                writer.add_scalar('int_loss', g_int_loss, global_step=step)
                writer.add_scalar('gd_loss', g_gd_loss, global_step=step)

                writer.add_image('train_target', target[0], global_step=step)
                writer.add_image('train_output', G_output[0], global_step=step)
                writer.add_image('test_target', test_target[0], global_step=step)
                writer.add_image('test_output', test_output[0], global_step=step)

            if step%1000==0:
                utils.saver(generator.state_dict(),model_generator_save_path,step)
                utils.saver(discriminator.state_dict(),model_discriminator_save_path,step)
            step+=1

if __name__=='__main__':
    train(num_clips,num_unet_layers,num_channels*(num_clips-num_his),num_channels,discriminator_channels)

