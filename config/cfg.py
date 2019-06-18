import os
from os.path import join

base_dir = '/home/online/open-scd.pytorch'
dataset_base_dir = join(base_dir,'data')
train_data_dir = join(dataset_base_dir,'train')
train_txt_dir = join(dataset_base_dir,'train.txt')
val_data_dir = join(dataset_base_dir,'val')
val_txt_dir = join(dataset_base_dir,'val.txt')
test_data_dir = join(dataset_base_dir,'test')
test_txt_dir = join(dataset_base_dir,'test.txt')
pretrain_weights_dir = join(base_dir,'pretrain_weights')
save_weights_dir = join(base_dir,'backup')
batch_size = 8
init_learning_rate = 1e-4
weight_decay = 5e-5
momentum = 0.95
ignore_value = 255
display_interval = 20
validate_interval = 10000
save_weights_interval = 5000
backbone = 'unet'
training_policy = 'sgd'
data_augment = True