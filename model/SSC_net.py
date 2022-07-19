import torch 
import torch.nn as nn
from . import modules
from util import util
from .base_net import BaseNet
import itertools
import torch.nn.functional as F
import numpy as np

from util import util
from skimage import color

class SSCNet(BaseNet):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        if is_train:

            parser.add_argument('--lambda_S1', type=float, default=100.0, help='weight for S1 loss')
            parser.add_argument('--lambda_S2', type=float, default=100.0, help='weight for S2 loss')
            parser.add_argument('--lambda_cls', type=float, default=0.1, help='weight for classification loss')
            parser.add_argument('--lambda_tv', type=float, default=10.0, help='weight for tv ')
            parser.add_argument('--lambda_his', type=float, default=1.0, help='weight for histogram loss')

            
        return parser
    def __init__(self,opt):
        BaseNet.__init__(self, opt)
        self.loss_names = ['S1', 'S2','cls','tv','his','total']
        self.visual_names = ['src', 'src_rgb', 'ref','stage1','stage2']
        self.model_names = ['VE', 'E', 'VD', 'C','A','CLS']

        norm_layer = modules.get_norm_layer(norm_type=opt.norm) 
        
        self.netD = modules.Decoder(output_nc=2, norm_layer=norm_layer).to(self.device) 
        
        self.netVE = modules.VGGEncoder(input_nc=1).to(self.device)
        self.netE = modules.Encoder().to(self.device)
        self.netA = modules.Align_network().to(self.device)
        self.netCLS = modules.Classify_network().to(self.device)
        


        self.netVD = modules.VGGDecoder(output_nc = 2, style_dim = 64).to(self.device) 

        if self.isTrain:
            self.criterionSL1 = torch.nn.SmoothL1Loss()
            self.criterionCls  = torch.nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(itertools.chain(self.netVE.parameters(), self.netE.parameters(), self.netVD.parameters(), self.netD.parameters(), self.netA.parameters(),self.netCLS.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = [self.optimizer]


    
    def set_input(self, input):
        
        self.src_l = input['src_l'].to(self.device) 
        self.ref_l = input['ref_l'].to(self.device) 
        self.src_ab = input['src_ab'].to(self.device)
        self.ref_ab = input['ref_ab'].to(self.device)

        self.rand = input['rand'].to(self.device)
        self.target = input['target'].long().to(self.device)
        self.real_B_enc = util.encode_ab_ind(self.src_ab[:, :, ::4, ::4])

    def forward(self, alpha=1.0):

        style_feat, ref_out ,class_input = self.netE(torch.cat((self.ref_l,self.ref_ab),1))
        content_feat = self.netVE(self.src_l, inter = False)

        self.class_output = self.netCLS(class_input)

        params = list(self.netCLS.parameters())
        
        weight_softmax = np.squeeze(params[-2].data)

        h_x = F.softmax(self.class_output, dim=1).data.squeeze().view(-1,1000)
        _, idx = h_x.sort(1, True)
        
        CAM = util.returnCAM(class_input, weight_softmax, idx[:,0])
        self.index  = torch.flatten( CAM,1)

        self.stage1 = self.netVD(content_feat, style_feat)

        _, stage1_out, _ = self.netE(torch.cat((self.src_l,self.stage1),1))

        scale = 1/4

        ref_feats = F.interpolate(ref_out[0], scale_factor=scale, mode='bilinear')
        stage1_feats = F.interpolate(stage1_out[0], scale_factor=scale, mode='bilinear')
        for i in range(3):

            scale *= 2

            ref_feats = torch.cat((ref_feats, F.interpolate(ref_out[i+1], scale_factor=scale, mode='bilinear')), dim=1)
            stage1_feats = torch.cat((stage1_feats, F.interpolate(stage1_out[i+1], scale_factor=scale, mode='bilinear')), dim=1)


        R_feats_flat_o = ref_feats.view(ref_feats.shape[0], ref_feats.shape[1], -1).permute(0, 2, 1)

        R_feats_flat_o = util.top_k_gather(self.index, self.rand , R_feats_flat_o, self.opt.k) 
        
        T_feats_flat_o = stage1_feats.view(stage1_feats.shape[0], stage1_feats.shape[1], -1).permute(0, 2, 1)

        T_feats_flat = T_feats_flat_o / torch.norm(T_feats_flat_o, p=2, dim=-1, keepdim=True) 
        R_feats_flat = R_feats_flat_o / torch.norm(R_feats_flat_o, p=2, dim=-1, keepdim=True)
        corr = torch.bmm(R_feats_flat, T_feats_flat.permute(0, 2, 1))
        corr = F.softmax(corr/0.01, dim=1)
        self.corr = corr

        align_feats = self.netA(R_feats_flat_o, corr) 
        
        self.stage2,self.fake_B_class = self.netD(stage1_out, align_feats)


    def backward(self):

        self.loss_S1 = self.criterionSL1(self.stage1, self.src_ab)
        self.loss_S2 = self.criterionSL1(self.stage2, self.src_ab)
        self.loss_cls =  self.criterionCls(self.class_output, self.target.view(-1))
        self.loss_tv = util.tv_loss( self.stage2 )
        self.loss_his = self.criterionCls(self.fake_B_class.type(torch.cuda.FloatTensor),
                                          self.real_B_enc[:, 0, :, :].type(torch.cuda.LongTensor))

        self.loss_total = self.loss_S1 * self.opt.lambda_S1 + self.loss_S2 * self.opt.lambda_S2 + self.loss_cls * self.opt.lambda_cls + self.loss_tv * self.opt.lambda_tv + self.loss_his * self.opt.lambda_his
        self.loss_total.backward()

    def lab2rgb(self, L, AB):
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def compute_visuals(self):
        if self.isTrain:
            self.src = self.src_l
            self.src_rgb = self.lab2rgb(self.src_l, self.src_ab)
            self.ref = self.lab2rgb(self.ref_l, self.ref_ab)
            self.stage1 = self.lab2rgb(self.src_l,self.stage1)
            self.stage2 = self.lab2rgb(self.src_l,self.stage2)

        else:
            self.src = self.src_l.unsqueeze(dim=0)
            self.src_rgb = torch.from_numpy(self.lab2rgb(self.src_l, self.src_ab)).permute(2, 0, 1 ).unsqueeze(dim=0)
            self.ref = torch.from_numpy(self.lab2rgb(self.ref_l, self.ref_ab)).permute(2, 0, 1 ).unsqueeze(dim=0)
            self.stage1 = torch.from_numpy(self.lab2rgb(self.src_l,self.stage1)).permute(2, 0, 1 ).unsqueeze(dim=0)
            self.stage2 = torch.from_numpy(self.lab2rgb(self.src_l,self.stage2)).permute(2, 0, 1 ).unsqueeze(dim=0)
    def optimize_parameters(self):

        self.forward()               
        self.optimizer.zero_grad()   
        self.backward()              
        self.optimizer.step()        