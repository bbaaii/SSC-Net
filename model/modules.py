import torch 
import torch.nn as nn
from util import util
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as init

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, norm_layer=nn.BatchNorm2d, actv=nn.LeakyReLU(0.2),
                 normalize=True, downsample=False):
        super(ResBlk, self).__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = norm_layer(dim_in)
            self.norm2 = norm_layer(dim_in)

        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance



class VGGEncoder(nn.Module):
    def __init__(self,input_nc=1):
        super(VGGEncoder, self).__init__()

        self.enc_1 = nn.Sequential(
                nn.Conv2d(input_nc, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU())  
        self.enc_2 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),  
                nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU())  
        self.enc_3 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),  
                nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU())  
        self.enc_4 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  
                nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 512, (3, 3)),
                nn.ReLU())  
    def encode(self, input):

        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    def forward(self, x):

        return self.encode(x)
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class VGGDecoder(nn.Module):
    def __init__(self, output_nc = 2, style_dim = 64):
        super(VGGDecoder, self).__init__()
        self.dec_1 = nn.Sequential(
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(512, 256, (3, 3)),
                        )
        self.dec_2 = nn.Sequential(
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 128, (3, 3)),
                        )
        self.dec_3 = nn.Sequential(
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 128, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 64, (3, 3)),
                        )
        self.dec_4 = nn.Sequential(
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, 64, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, output_nc, (3, 3)),
                        )
        self.norm1 = AdaIN(style_dim, 512)
        self.norm2 = AdaIN(style_dim, 256)
        self.norm3 = AdaIN(style_dim, 128)
        self.norm4 = AdaIN(style_dim, 64)
    def decode(self, input,style):
        for i in range(4):
            input = getattr(self, 'norm{:d}'.format(i + 1))(input, style)
            input = getattr(self, 'dec_{:d}'.format(i + 1))(input)
            
        return input
    def forward(self, x, s):
        return self.decode(x, s)

class Encoder(nn.Module):
    def __init__(self, input_nc = 3, img_size=256, style_dim=64, max_conv_dim=512):
        super(Encoder, self).__init__()
        dim_in = 2**14 // img_size #if 256  dim_in = 64
        blocks = []
        self.enc_0 = nn.Conv2d(input_nc, dim_in, 3, 1, 1)
        self.enc_1 = ResBlk(dim_in, 128, downsample=True)  
        self.enc_2 = ResBlk(128, 256, downsample=True)  
        self.enc_3 = ResBlk(256, 512, downsample=True)  
        self.enc_4 = ResBlk(512, 512, downsample=True)  
        self.enc_5 = ResBlk(512, 512, downsample=True)  
        self.enc_6 = ResBlk(512, 512, downsample=True)  
    
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(512, 512, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.linear = nn.Linear(512, style_dim)
    

    def forward(self, x):

        output0 = self.enc_0(x)
        output1 = self.enc_1(output0)
        output2 = self.enc_2(output1)
        output3 = self.enc_3(output2)

        output4 = self.enc_4(output3)
        output = self.enc_5(output4)
        output = self.enc_6(output)
        h = self.shared(output)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        return h, [output0, output1, output2, output3] , output4


class Decoder(nn.Module):
    def __init__(self, output_nc=3,norm_layer=nn.BatchNorm2d):
        
        super(Decoder, self).__init__()
        use_bias = True
        block0_1 = [nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias), norm_layer(512), nn.ReLU(True)]
        
        self.resblock0_2 = ResBlk(512, 512)
        self.resblock0_3 = ResBlk(512, 512)

        upmodel0up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        upmodel0short=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        upmodel0=[nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        upmodel0+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(256)]

        block1_1 = [nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias), norm_layer(256), nn.ReLU(True)]
        self.resblock1_2 = ResBlk(256, 256)
        self.resblock1_3 = ResBlk(256, 256)

        upmodel1up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]
        upmodel1short=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        upmodel1=[nn.ReLU(True), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(128)]

        block2_1 = [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), norm_layer(128), nn.ReLU(True)]
        self.resblock2_2 = ResBlk(128, 128)
        self.resblock2_3 = ResBlk(128, 128)
        
        upmodel2up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]
        upmodel2short=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        upmodel2=[nn.ReLU(True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias), nn.LeakyReLU(negative_slope=.2)]
        
        model_class = [nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ]

        model_out = [nn.Conv2d(128, output_nc, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), nn.Tanh()]

        self.block0_1 = nn.Sequential(*block0_1)
        self.block1_1 = nn.Sequential(*block1_1)
        self.block2_1 = nn.Sequential(*block2_1)

        self.model0up = nn.Sequential(*upmodel0up)
        self.model0 = nn.Sequential(*upmodel0)
        self.model1up = nn.Sequential(*upmodel1up)
        self.model1 = nn.Sequential(*upmodel1)
        self.model2up = nn.Sequential(*upmodel2up)
        self.model2 = nn.Sequential(*upmodel2)
        self.modelshort0 = nn.Sequential(*upmodel0short)
        self.modelshort1 = nn.Sequential(*upmodel1short)
        self.modelshort2 = nn.Sequential(*upmodel2short)
        self.model_out = nn.Sequential(*model_out)
        self.model_class = nn.Sequential(*model_class)

    def forward(self,stage1_out,align_feats):

        conv0_block1 = self.block0_1(stage1_out[3]+align_feats[0])
        conv0_resblock2 = self.resblock0_2(conv0_block1)
        conv0_resblock3 = self.resblock0_3(conv0_resblock2)
        conv0_up = self.model0up(conv0_resblock3) + self.modelshort0(stage1_out[2]) # change to stage1_out 256 * 64 *64
        conv_0= self.model0(conv0_up)
        
        out_class = self.model_class(conv_0)
        conv1_block1 = self.block1_1(conv_0 + align_feats[1]) 
        conv1_resblock2 = self.resblock1_2(conv1_block1)
        conv1_resblock3 = self.resblock1_3(conv1_resblock2)
        conv1_up = self.model1up(conv1_resblock3) + self.modelshort1(stage1_out[1]) # change to stage1_out 128 * 128 *128
        conv_1 = self.model1(conv1_up)

        conv2_block1 = self.block2_1(conv_1 + align_feats[2])
        conv2_resblock2 = self.resblock2_2(conv2_block1)
        conv2_resblock3 = self.resblock2_3(conv2_resblock2)
        conv2_up = self.model2up(conv2_resblock3) + self.modelshort2(stage1_out[0]) # change to stage1_out 128 * 256 *256
        conv_2 = self.model10(conv2_up)

        output = self.model_out(conv_2)
        
        return output,out_class


class Classify_network(nn.Module):
    def __init__(self):
        super(Classify_network, self).__init__()
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class Align_network(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        
        super(Align_network, self).__init__()
        self.w_v = nn.Conv1d(960, 512, kernel_size=1, stride=1, padding=0, bias=True)

        model0 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(512)]

        self.model0 = nn.Sequential(*model0)
        model1 = [nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(256)]
        model2 = [nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(128),nn.Upsample(scale_factor=2, mode='nearest')]
        self.model1 = nn.Sequential(*model5)
        self.model2 = nn.Sequential(*model6)
    def forward(self, feats, corr, H=64, W=64):

        feats = self.w_v(feats.permute(0, 2, 1))
        align = torch.bmm(feats, corr)

        align = align.view(align.shape[0], align.shape[1], H, W)
        align_0 = self.model0(align)
        align_1 = self.model1(align_0)
        align_2 = self.model2(align_1)


        return [align_0[:,:,::2,::2], align_1, align_2] 