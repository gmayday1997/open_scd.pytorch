import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

### original contrastive loss ###

class ContrastiveLoss(nn.Module):
    def __init__(self, device,margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.margin = margin

    def forward(self, feat_t0,feat_t1,labels):

        batch_sz, nchannels,height,width = feat_t0.shape
        feat_t0_rz,feat_t1_rz = torch.transpose(feat_t0.view(batch_sz,nchannels,height* width),2,0),\
                                torch.transpose(feat_t1.view(batch_sz,nchannels,height* width),2,0)
        gt_tensor = torch.from_numpy(np.array(labels.data.cpu().numpy(),np.float32))
        label_rz = torch.transpose(gt_tensor.view(batch_sz,1,height * width),2,0)
        label_rz = label_rz.to(self.device)
        distance_maps = F.pairwise_distance(feat_t0_rz,feat_t1_rz,keepdim=True)
        constractive_loss = torch.sum((1-label_rz)* torch.pow(distance_maps,2 ) + \
                                       label_rz * torch.pow(torch.clamp(self.margin - distance_maps, min=0.0),2))
        return constractive_loss