import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils.distance_map_util import *
from utils.util import *
from os.path import join

class Trainer(object):

    def __init__(self,device, cfgs,train_loader,val_loader,model,detection_loss,optimizer):

        self.device = device
        self.cfgs = cfgs
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = detection_loss
        self.optimizer = optimizer
        self.valid_interval = self.cfgs['validate_interval']
        self.dislay_interval = self.cfgs['display_interval']
        self.save_weights_interval = self.cfgs['save_weights_interval']
        self.max_epoch = self.cfgs['max_epoch']
        self.visualization = self.cfgs['visualization']
        self.save_visualize_dir = self.cfgs['save_base_dir']
        check_dirs(self.save_visualize_dir)
        self.epoch = 0
        self.iteration = 0
        self.best_metric = 0

    def train_epoch(self):

        self.model.train()
        for batch_idx, (img1,img2,target,_) in enumerate(self.train_loader):
            self.iteration = batch_idx + self.epoch * len(self.train_loader)
            if((self.iteration + 1) % self.valid_interval == 0):
                self.validate_epoch()
            img1, img2, targets = Variable(img1.to(self.device)),Variable(img2.to(self.device)), \
                                 Variable(target.to(self.device))
            feat_t0, feat_t1 = self.model(img1,img2)
            loss_fun = self.loss(feat_t0,feat_t1,targets)
            self.optimizer.zero_grad()
            loss_fun.backward()
            #self.lr_reduce_policy.step()
            self.optimizer.step()
            if((self.iteration + 1) % self.dislay_interval == 0):
                print('---Epoch %d, Iteration %d, Loss %.3f' %(self.epoch,self.iteration,loss_fun.item()))
            if((self.iteration + 1) % self.save_weights_interval ==0):
                save_weights_dir = join(self.cfgs.save_weights_dir,'unet_' + str(self.iteration) + '.weights')
                torch.save(model,save_weights_dir)

    def train(self):

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_epoch()

    def validate_epoch(self):

        self.model.eval()
        for batch_idx,(img1,img2,target,name) in enumerate(tqdm(self.val_loader)):
            img1, img2, targets = Variable(img1.to(self.device)),Variable(img2.to(self.device)),\
                                  Variable(target.to(self.device))
            feat_t0,feat_t1 = self.model(img1,img2)
            distance_maps = generate_distance_maps(feat_t0,feat_t1)
            #pred = distance2pred(distance_maps,self.cfgs['thresh'])
            if self.visualization:
                save_distance_map_dir,save_dist_distribution_map_dir = join(self.cfgs['save_base_dir'],'iteration_' + str(self.iteration), 'distance_maps'),\
                join(self.cfgs['save_base_dir'],'iteration_' + str(self.iteration),'distribution')
                check_dirs(save_distance_map_dir),check_dirs(save_dist_distribution_map_dir)
                visualize_distance_maps(distance_maps,save_dir = join(save_distance_map_dir,name[0]))
                plot_distance_distribution_maps(distance_maps,targets,save_dir = join(save_dist_distribution_map_dir,name[0]))