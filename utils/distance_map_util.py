import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import seaborn as sns

def distance2pred(distance_map,thresh):

    distance_map[distance_map >= thresh] = 1
    distance_map[distance_map < thresh] = 0
    return distance_map

def generate_distance_maps(feat_t0,feat_t1):

    batch_sz,nchannels,height,width = feat_t0.shape
    pred = F.pairwise_distance(feat_t0,feat_t1)
    distance_map = pred.view(batch_sz,height, width)
    return distance_map

def visualize_distance_maps(distance_map, save_dir):

    dist_map_colorize = cv2.applyColorMap(np.uint8(255 * np.squeeze(tonumpy(distance_map))), cv2.COLORMAP_JET)
    cv2.imwrite(save_dir, dist_map_colorize)

def tonumpy(cuda_data):
    return cuda_data.data.cpu().numpy()

def plot_distance_distribution_maps(distance_map, label, save_dir):

    plt.cla()
    n, h, w = distance_map.shape
    cm_rz = np.squeeze(np.reshape(tonumpy(distance_map), (n, h * w)), axis=0)
    gt = np.array(tonumpy(label), np.int32)
    gt_rz = np.squeeze(np.reshape(gt, (n, h * w)), axis=0)
    x_t, y_t = [], []
    colors = ['#0000ff', '#009900']
    markers = ['*', 'o']
    for i in range(h * w):
        t = np.random.random() * 2 * np.pi - np.pi
        dist, lbl = cm_rz[i], gt_rz[i]
        x, y = dist * np.cos(t), dist * np.sin(t)
        x_t.append(x), y_t.append(y)
    x_t, y_t = np.array(x_t), np.array(y_t)
    for i in range(2):
        plt.scatter(x_t[gt_rz == i], y_t[gt_rz == i], c=colors[i], s=40, marker=markers[i])
    plt.legend(['unchanged', 'changed'], loc='upper right')
    plt.xlim(min(x_t) * 1.2, max(x_t) * 1.2)
    plt.ylim(min(y_t) * 1.2, max(y_t) * 1.2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('distance distribution')
    plt.grid(True)
    plt.savefig(save_dir)