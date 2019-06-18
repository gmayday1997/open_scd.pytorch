import os
import torch
from os.path import join
from utils.util import *
from utils.data_transforms import *
from core.loss import *
import dataset.tsunami as dates
from torch.utils.data import DataLoader
import backbone.siamesenet as models
from trainer.trainer import *

def main():
    ### step one : configure file ###
    base_dir = os.getcwd()
    #options = [tsunami,gsv]
    cfgs_base_dir,dataset_name = join(base_dir,'config'),'tsunami'
    cfgs = parse_yaml_files(cfgs_base_dir,dataset_name)
    ### transform ###
    if cfgs['data_augment']:
        normalize = Normalize(mean=cfgs['mean_value_t0'],std=[1,1,1])
        transform_med = Compose([
            RandomRotate(cfgs['random_rotate']),
            #RandomScale(cfgs['random_scale']),
            RandomHorizontalFlip(),
            ToTensor(),normalize
        ])
    train_dataset_sets = {
        'data_root': cfgs['dataset_base_dir'],
        'dataset_name' : cfgs['dataset_name'],
        'dataset_txt_root': cfgs['train_txt_dir'],
        'batch_size' : cfgs['batch_size'],
        'splits': 'train',
        'data_augment':cfgs['data_augment'],
        'mean_value_t0': cfgs['mean_value_t0'],
        'mean_value_t1' : cfgs['mean_value_t1']
    }
    val_dataset_sets = {
        'data_root' : cfgs['dataset_base_dir'],
        'dataset_name': cfgs['dataset_name'],
        'dataset_txt_root' : cfgs['val_txt_dir'],
        'batch_size' : 1,
        'splits': 'val',
        'data_augment' : cfgs['data_augment'],
        'mean_value_t0': cfgs['mean_value_t0'],
        'mean_value_t1': cfgs['mean_value_t1']
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### step two : loading datasets ###
    train_dates = dates.Dataset(train_dataset_sets,transform_med=None)
    train_loader = DataLoader(train_dates,batch_size=cfgs['batch_size'],
                              shuffle=True,num_workers=4)
    val_dates = dates.Dataset(val_dataset_sets,transform_med=None)
    val_loader = DataLoader(val_dates,batch_size=cfgs['batch_size'],
                            shuffle=False,num_workers=4)
    ### step three: build different backbones ###
    ### firstly training unet from scratch ###
    model = models.SiameseNet(input_channels=3,out_classes=2)
    model.load_pretrain_weights(join(cfgs['pretrain_weights_dir'],'unet_MODEL.pth'))
    model = model.to(device)
    detection_loss = ContrastiveLoss(device,margin=cfgs['margin'])
    ### step four : build optimizer, learning rate reduce policy ###
    optimizer = torch.optim.SGD(model.parameters(), lr= cfgs['init_learning_rate'],momentum = cfgs['momentum'],weight_decay= cfgs['weight_decay'])
    ### step five : trainer ###
    trainer = Trainer(device, cfgs,train_loader,val_loader,model,detection_loss,optimizer)
    trainer.train()

if __name__ == '__main__':
    main()
