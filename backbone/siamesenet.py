import torch
import torch.nn as nn
from backbone.unet import *

class SiameseNet(nn.Module):
    def __init__(self, input_channels,out_classes):
        super(SiameseNet, self).__init__()
        self.cnn = Unet(input_channels,out_classes)

    def load_pretrain_weights(self,pretrain_weights_dir):
        weights = torch.load(pretrain_weights_dir)
        own_state_dict = self.cnn.state_dict()
        pretrain_dicts = {k:v for k,v in weights.items() if k in own_state_dict}
        own_state_dict.update(pretrain_dicts)
        self.cnn.load_state_dict(own_state_dict)
        print('pretrain weights has been loaded sucessfully')

    def forward(self, img1,img2):

        feat_t0 = self.cnn(img1)
        feat_t1 = self.cnn(img2)
        return feat_t0,feat_t1