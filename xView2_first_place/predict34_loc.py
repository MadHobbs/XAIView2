import os

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch
torch.manual_seed(1)

import pandas as pd
from tqdm import tqdm
import timeit
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from zoo.models import Res34_Unet_Loc

from utils import *

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = '../test/images'
pred_folder = 'pred34_loc'
models_folder = 'weights'
TOP_K = 10 # how many pixels to average over to get 

if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(pred_folder, exist_ok=True)
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    snap_to_load = 'res34_loc_0_1_best'
    model = Res34_Unet_Loc() 
    model = nn.DataParallel(model) 
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
    model.eval()


    for f in tqdm(sorted(listdir(test_dir))):
        if '_pre_' in f:
            fn = path.join(test_dir, f)

            img = cv2.imread(fn, cv2.IMREAD_COLOR)
            img = preprocess_inputs(img)

            inp = []
            inp.append(img)
            inp.append(img[::-1, ...])
            inp.append(img[:, ::-1, ...])
            inp.append(img[::-1, ::-1, ...])
            inp = np.asarray(inp, dtype='float')
            inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
            inp = Variable(inp) 

            inp.requires_grad_() # have to do this so we can do .backward() to calculate gradient            
            msk = model(inp)
            msk = torch.sigmoid(msk)

            # create saliency map
            value_of_interest = torch.mean(torch.topk(msk.flatten(),TOP_K).values)
            value_of_interest.backward() # calculate gradient
            grad = inp.grad.data.abs().numpy()
            saliency = []
            saliency.append(grad[0, ...])
            saliency.append(grad[1, :, ::-1, :])
            saliency.append(grad[2, :, :, ::-1])
            saliency.append(grad[3, :, ::-1, ::-1])
            saliency = np.asarray(saliency)
            saliency_max = np.max(saliency, axis=1) # take max over 3 channels
            saliency_full = np.mean(saliency_max, axis=0) # take average over 4 layers
            save_dir = snap_to_load + "_saliency/"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fname = save_dir + fn.split("/")[-1].split(".")[0]+"_loc_saliency_k"+str(TOP_K)+".png"
            plt.imsave(fname, saliency_full, cmap=plt.cm.hot)

            # generate prediction
            msk_np = msk.detach().cpu().numpy()
            pred = [] 
            pred.append(msk_np[0, ...])
            pred.append(msk_np[1, :, ::-1, :])
            pred.append(msk_np[2, :, :, ::-1])
            pred.append(msk_np[3, :, ::-1, ::-1])
            pred_full = np.asarray(pred).mean(axis=0)
            out_msk = pred_full * 255
            out_msk = out_msk.astype('uint8').transpose(1, 2, 0)
            cv2.imwrite(path.join(pred_folder, '{0}.png'.format(f.replace('.png', '_part1.png'))), out_msk[..., 0], [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))