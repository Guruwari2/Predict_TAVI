import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,balanced_accuracy_score,roc_auc_score

bce = nn.BCEWithLogitsLoss(reduction='none')


def criterion(logits, targets, activated=False):
    if activated:
        losses = nn.BCELoss(reduction='none')(logits.view(-1), targets.view(-1))
    else:
        losses = bce(logits.view(-1), targets.view(-1))
    losses[targets.view(-1) > 0] *= 2
    norm = torch.ones(logits.view(-1).shape[0]).to('cuda')
    norm[targets.view(-1) > 0] *= 2
    
    return losses.sum() / norm.sum()

def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    total_loss = 0
    total_bs = 0

    for batch in loader_train:
        images = batch['image'].to('cuda')
        targets = batch['target'].to('cuda')
        
        optimizer.zero_grad()        

        with torch.cuda.amp.autocast():
            logits = model(images.half())
            logits = logits.view(6, 98).contiguous()

            loss = criterion(logits, targets)
            
        #total_loss += loss.item()
        #total_bs += 1

        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    return np.nanmean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    gts = []
    outputs = []
    val_loss = []
    with torch.no_grad():
        for i,batch in enumerate(loader_valid):
            
            images = batch['image'].to('cuda')
            targets = batch['target'].to('cuda')
            
            feat = model(images.float())
            
            out = feat.view(6, 98).contiguous()
            loss = criterion(out, targets)

            feat = torch.argmax(feat, dim = 1)            
            
            targets = targets.view(-1)
                
            gts.append(targets.detach().to('cpu'))
                       
            outputs.append(feat.detach().to('cpu'))
            val_loss.append(loss.item())                        
    
    gts = torch.cat(gts)
    outputs = torch.cat(outputs)
                       
    auc = roc_auc_score(gts,outputs)
    acc = accuracy_score(gts,outputs)
    f1 = f1_score(gts,outputs)

    return acc,auc,f1,np.nanmean(val_loss)