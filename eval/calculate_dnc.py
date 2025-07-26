#!/usr/bin/env python
# coding: utf-8

# In[9]:


import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from os.path import join as ospj
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import yaml
from dataset.data_loader import CellDataset, CellDatasetFold
from evaluation.eval import evaluate
from evaluation.gan_metrics.inception import InceptionV3
from evaluation.gan_metrics.fid import *
from evaluation.gan_metrics.density_and_coverage import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from pytorch_gan_metrics.utils import get_inception_score, get_fid, calc_and_save_stats, get_inception_score_and_fid_from_directory
# In[2]:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ground_truth",
    type=str,
    nargs="?",
    default="/mnt/petrelfs/fanyimin/26739/test_vae",
    help="Ground truth dataset"
)
parser.add_argument(
    "--generated_output",
    type=str,
    nargs="?",
    default="/mnt/petrelfs/fanyimin/stable-diffusion/outputs/perturb_no_dpm_low_lr/samples",
    help="Generated Output"
)
parser.add_argument(
    "--stats",
    type=str,
    nargs="?",
    default="/mnt/petrelfs/fanyimin/26739/stats.npz",
    help="Generated Output"
)
parser.add_argument(
    "--output",
    type=str,
    nargs="?",
    default="/mnt/petrelfs/fanyimin/morph_metrics/output.json",
    help="Generated Output"
)

args = parser.parse_args()
results = dict()
test_set = CellDatasetFold(args.ground_truth)

if not os.path.exists(args.stats):
    calc_and_save_stats(args.ground_truth, args.stats)
score = get_inception_score_and_fid_from_directory(args.generated_output, args.stats)
# images = ... # [N, 3, H, W] normalized to [0, 1]
# IS, IS_std = get_inception_score(images)        # Inception Score
# FID = get_fid(images, args.stats) # Frechet Inception Distance
results['inception_score'] = score[0][0]
results['inception_score_std'] = score[0][1]
results['fid'] = score[1]
test_loader= torch.utils.data.DataLoader(test_set, batch_size=16,num_workers=8, drop_last=True)

generated_set=CellDatasetFold(args.generated_output)
generated_loader= torch.utils.data.DataLoader(generated_set, batch_size=16,num_workers=8, drop_last=True)


block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx],resize_input=False)
#model.cuda()
model.eval()

# In[6]:


test_features,test_mu,test_sigma=inception_activations(test_loader, model, 2048, [0,1,2], False)
generated_features, generated_mu,generated_sigma=inception_activations(generated_loader, model, 2048, [0,1,2], False)

result=compute_d_c(test_features,generated_features,5)
results['precision'] = result['precision']
results['recall'] = result['recall']
results['density'] = result['density']
results['coverage'] = result['coverage']
with open(args.output, "w") as f:
    json.dump(results, f)