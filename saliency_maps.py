'''
Inspired by https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
'''
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from os import path, listdir
from tqdm import tqdm
import cv2
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from xView2_first_place.zoo.models import Res34_Unet_Loc
from xView2_first_place.utils import *

test_dir='test/images'
models_folder = 'xView2_first_place/weights'
snap_to_load = 'res34_loc_0_1_best'
loc_model = Res34_Unet_Loc()

# helpers
def deprocess_inputs(x):
    '''Reverse what preprocess_inputs does for display purposes'''
    x += 1
    x *= 127
    x = np.asarray(np.rint(x), dtype='uint8')
    return x

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

#with torch.no_grad():

# load and preprocess
for f in tqdm(sorted(listdir(test_dir))[:3]):
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
        
        loc_model.eval()
        inp.requires_grad_()
        msk = loc_model(inp)
        pred = torch.mean(msk, 0)
        pred = torch.squeeze(pred, 0)
        most_building = torch.max(pred)
        most_building.backward()
        saliency, _ = torch.max(torch.mean(inp.grad.data.abs(), 0),dim=1)
        
        # code to plot the saliency map as a heatmap
        plt.imshow(saliency[0], cmap=plt.cm.hot)
        plt.axis('off')
        plt.show()
        # target_layer = loc_model.conv5[-1]
        # Construct the CAM object once, and then re-use it on many images:
        # cam = GradCAM(model=loc_model, target_layer=target_layer, use_cuda=False)        

        # CODE for SALIENCY MAP #
        # loc_model.eval() # put model in eval mode
        # get saliency map
        # inp.requires_grad_()
        
        # 

        # score, indices = torch.max(torch.squeeze(msk1, 1), 1)
        # score.backward()
        
        # we don't actually want the scores
        # msk1 = torch.sigmoid(msk0)
        # msk2 = msk1.cpu().numpy()
        # Get the index corresponding to the maximum score and the maximum score itself.
        # score_max_index = scores.argmax()
        # score_max = scores[0,score_max_index]


            # pred = []
            # for model in models:               
            #     msk = model(inp)
            #     msk = torch.sigmoid(msk)
            #     msk = msk.cpu().numpy()
                
            #     pred.appendm =(msk[0, ...])
            #     pred.append(msk[1, :, ::-1, :])
            #     pred.append(msk[2, :, :, ::-1])
            #     pred.append(msk[3, :, ::-1, ::-1])

            # pred_full = np.asarray(pred).mean(axis=0)
            
            # msk = pred_full * 255
            # msk = msk.astype('uint8').transpose(1, 2, 0)
            # cv2.imwrite(path.join(pred_folder, '{0}.png'.format(f.replace('.png', '_part1.png'))), msk[..., 0], [cv2.IMWRITE_PNG_COMPRESSION, 9])
