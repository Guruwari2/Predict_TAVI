import numpy as np
import torch
import os
from copy import copy
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split 
import monai
from torch.utils.data import Dataset,DataLoader
import torchio as tio
import pickle
import wandb
import pandas as pd
import gc
import torch.nn.functional as F
import torch.nn as nn
import warnings
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,balanced_accuracy_score,roc_auc_score


from Dataloader import compute_dict, CTScanDataset
from Projector import Projector, LinearClassifier
from losses import SupConLoss, SupConCELoss
from scl import train_scl, linear_scl

######## Data 

dict_config = {'model':'resnet10',
                  'normalization':'global',
                  'patience':15,
                  'n_epoch_proj':270,
                   'n_epoch_class' : 80,
                  'use_wandb' : True,
               'type_scan':'diastole',
               'crop':'cube_irl',
                    'oversample':True,
                   'augment':False,
                'augment_prob':(),
                   'size':200,
                   'bs':3,
                   'name_target':'pm_post_tavi',
                   'model_size': 6,
                   }


list_dls = compute_dict(dict_config, 'newsplits_no_val_df.pkl')
use_wandb = dict_config['use_wandb']

if use_wandb :
    wandb.init(project="Predict-Tavi", entity="brain-imt", config = dict_config,settings=wandb.Settings(start_method='fork'))


for i,dls in enumerate(list_dls) : 
    if i==3:
        
        train_loader, test_loader = dls
        
        ## Model used 
        
        model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=2, inplanes = (64,128,256,512))
        model.fc = torch.nn.Identity()
        model.load_state_dict(torch.load("model/contrastive_split3.pth"))

        model.to('cuda')
        
        projector = Projector(name='resnet_10', out_dim=128, device='cuda')
        classifier = LinearClassifier(name='resnet_10', num_classes=2, device='cuda')
        
        optimizer = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=0.001, weight_decay=5e-4) 
        optimizer2 = torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=5e-4, weight_decay=5e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6) 
    
        criterion_ce = nn.CrossEntropyLoss() 
        
        criterion = SupConLoss(temperature=0.1, device='cuda')
        
        #sl_train_losses, model, last_checkpoint = train_scl(model, projector, train_loader, criterion, optimizer, scheduler, dict_config['n_epoch_proj'], use_wandb)
        
        history = linear_scl(model,classifier, train_loader, test_loader, criterion_ce, optimizer2, dict_config['n_epoch_class'], use_wandb)
        
        del history, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
            