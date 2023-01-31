import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train import train_func, valid_func
from models import TimmModel
from Dataloader import compute_dict, CTScanDataset

import warnings
warnings.filterwarnings('ignore')



dict_config = {'model':'resnet10',
                  'normalization':'global',
                   'n_epoch' : 80,
                  'use_wandb' : True,
               'type_scan':'diastole',
               'Analyse':'2.5D',
                    'oversample':True,
                   'augment':True,
                   'size':100,
                   'bs':6,
                   'name_target':'pm_post_tavi',
                   }

n_epochs = dict_config['n_epoch']
use_wandb = dict_config['use_wandb']
bs = dict_config['bs']

if use_wandb :
    wandb.init(project="Predict-Tavi", entity="brain-imt", config = dict_config,settings=wandb.Settings(start_method='fork'))


list_dls = compute_dict(dict_config, 'newsplits_no_val_df.pkl')

    
for i,dls in enumerate(list_dls) : 
    
    loader_train, loader_valid = dls
    
    
    backbone = 'tf_efficientnetv2_s_in21ft1k'
    model = TimmModel(backbone, bs, in_chans = 3, pretrained=True)
    model = model.to('cuda')
    
    optimizer = optim.AdamW(model.parameters(), lr=23e-5)
    scaler = torch.cuda.amp.GradScaler()
    
    best_acc, best_auc, best_f1 = 0, 0, 0
    loss_min = np.inf
    
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_epochs, eta_min=23e-6)
    
    
    for epoch in range(1, n_epochs+1):
        
        scheduler_cosine.step(epoch-1)
    
        print('Epoch:', epoch, '\n')
    
        train_loss = train_func(model, loader_train, optimizer, scaler)
        print("Loss : {:.5f}".format(train_loss), end = ' ')

        acc, auc, f1, val_loss = valid_func(model, loader_valid)
    
        content =  f'acc : {acc:.5f}, auc: {auc:.5f}, f1: {f1:.5f}, val_loss:{val_loss:.5f}'
        print(content)
        
        if use_wandb :
            wandb.log({'epoch':epoch,
                       'train_loss':train_loss,
                       'test_auc':auc,'test_acc': acc,                    
                   'test_f1':f1,'val_loss' : val_loss })
    
        if auc > best_auc:
            print(f'metric_best ({best_auc:.6f} --> {auc:.6f}). Saving model ...')
            best_auc = auc
            
    if use_wandb:
        wandb.log({'best_auc':best_auc
        })
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
wandb.finish()
