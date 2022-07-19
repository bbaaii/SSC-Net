import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset_index

from PIL import Image 
import numpy as np
import torchvision.transforms as transforms
from util import util
from skimage import color
import torch
import random
class LabDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):

        return parser

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)

        self.samples = make_dataset_index(self.dir)

        self.trans2rgb = util.get_transform(self.opt, stage=1, grayscale=False, convert=False)
        self.r = opt.r

    def __getitem__(self, index):

        src_path, target = self.samples[index]

        src_lab = self.process_img(src_path,self.trans2rgb)

        ref_lab = util.tps_transform_s(src_lab.unsqueeze(dim=0)).squeeze(dim=0)
        src_l = src_lab[[0], ...] / 50.0 - 1.0
        ref_l = ref_lab[[0], ...] / 50.0 - 1.0
        src_ab = src_lab[[1, 2], ...] / 110.0
        ref_ab = ref_lab[[1, 2], ...] / 110.0

        rand = torch.LongTensor( random.sample([ i for i in range(4096)] , self.r)).view(-1,1)

        rand = rand.repeat(1,960)

        return {'src_l': src_l, 'ref_l': ref_l, 'src_ab': src_ab, 'ref_ab': ref_ab,'target' :target,'rand':rand} 
    def __len__(self):

        return len(self.samples)

    def process_img(self, im_path, transform):
        im = Image.open(im_path).convert('RGB')
        im = transform(im)
        im = np.array(im)
        
        
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        return lab_t
    def reset(self):
        self.samples = make_dataset_index(self.dir)