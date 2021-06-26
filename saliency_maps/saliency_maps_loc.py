'''
Adapted for segmentation inspired by this pytorch example for classification:
https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
'''
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch.autograd import Variable
import torch
torch.manual_seed(1)
import matplotlib.pyplot as plt
from os import path, listdir
from tqdm import tqdm
import cv2
from pathlib import Path

from xView2_first_place.zoo.models import Res34_Unet_Loc
from xView2_first_place.utils import *

# set global variables
test_dir='test/images'
models_folder = 'xView2_first_place/weights'
snap_to_load = 'res34_loc_0_1_best'
loc_model = Res34_Unet_Loc()
PRED_THRESHOLD = 0.5 # threshold of interest for saliency maps, not determining actual model prediction
TOP_K = 10

# load checkpoints
print("=> loading checkpoint '{}'".format(snap_to_load))
checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
loaded_dict = checkpoint['state_dict']
sd = loc_model.state_dict()
for k in loc_model.state_dict():
    if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
        sd[k] = loaded_dict[k]
loaded_dict = sd
loc_model.load_state_dict(loaded_dict)
print("loaded checkpoint '{}' (epoch {}, best_score {})"
        .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

# don't need to find gradients with respect to the parameters of the NN
for param in loc_model.parameters():
    param.requires_grad = False

loc_model.eval()
# load and preprocess
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

        inp.requires_grad_()
        msk = loc_model(inp)
        pred = torch.sigmoid(msk) # convert to "probability" scores
        pred = torch.mean(pred, 0) # average over 4 layers
        pred = torch.squeeze(pred, 0) # get rid of unnecessary dimension

        # get value we want to calculate the gradient of
        # we'll select the n pixels predicted building with 
        # probability > PRED_THRESHOLD and take the average
        whether_above_thresh = pred > PRED_THRESHOLD
        indices_above_thresh = whether_above_thresh.nonzero()
        value_of_interest = torch.sum(
                            torch.where(pred > PRED_THRESHOLD, 
                                        pred,                       # if
                                        torch.zeros(pred.shape[0],  # else 
                                                    pred.shape[1]))
                                    )/len(indices_above_thresh)
        #value_of_interest = torch.mean(torch.topk(msk.flatten(),TOP_K).values)
        value_of_interest.backward() # calculate gradient
        # most_building = torch.max(pred)
        # most_building.backward()
        grad = inp.grad.data.abs().numpy()
        saliency = []
        saliency.append(grad[0, ...])
        saliency.append(grad[1, :, ::-1, :])
        saliency.append(grad[2, :, :, ::-1])
        saliency.append(grad[3, :, ::-1, ::-1])
        saliency = np.asarray(saliency)
        saliency_max = np.max(saliency, axis=1) # take max over 3 channels
        saliency_full = np.mean(saliency_max, axis=0) # take average over 4 layers

        # now, rearrange just as done for pred
        # saliency, _ = torch.max(torch.mean(inp.grad.data.abs(), 0),dim=0)
        
        # code to plot and save the saliency map as a heatmap
        save_dir = snap_to_load + "_saliency/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = save_dir + fn.split("/")[-1].split(".")[0]+"_loc_saliency_"+str(PRED_THRESHOLD)+".png"
        plt.imsave(fname, saliency_full, cmap=plt.cm.hot)
        
