import yaml
import os
from os.path import join

def parse_yaml_files(cfg_base_dir,dataset_name = 'tsunami'):

    cfgs_file_dir = join(cfg_base_dir,dataset_name + '.yaml')
    files = open(cfgs_file_dir)
    cfg_contents = yaml.load(files)
    return cfg_contents

def check_dirs(save_dirs):

    if not os.path.exists(save_dirs):
        os.makedirs(save_dirs)
    #else:
        #print('%s has created'% save_dirs)
