import torch
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,balanced_accuracy_score,roc_auc_score
import pickle
import wandb
import torchio as tio
import gc


def train_epoch_scl(encoder, projector, train_loader, criterion, optimizer, scheduler):

    epoch_loss = 0.0
    
    for i,batch in enumerate(train_loader):
        
        data, target_s = batch['image'],batch['target0'].to('cuda')
        
        
        transforms_dict1 = {tio.RandomNoise(std=75):0.33,tio.RandomGamma(0.3):0.34,tio.RandomSwap():0,tio.RandomAffine(scales=(0.96, 1.04),degrees=20):0.33}
        transform_augment1 = tio.OneOf(transforms_dict1)
        transforms1 = [transform_augment1]
        transform1 = tio.Compose(transforms1)
        
        transforms_dict2 = {tio.RandomNoise(std=75):0.33,tio.RandomGamma(0.3):0.34,tio.RandomSwap():0,tio.RandomAffine(scales=(0.96, 1.04),degrees=20):0.33}
        transform_augment2 = tio.OneOf(transforms_dict2)
        transforms2 = [transform_augment2]
        transform2 = tio.Compose(transforms2)
        list_aug1, list_aug2 = [], []
        with torch.no_grad():
            for image in data :
                image_aug1 = transform1(image)
                image_aug2 = transform2(image)
                list_aug1.append(image_aug1)
                list_aug2.append(image_aug2)
                
            #data_t1 = transform(data)
            #data_t2 = transform(data)
        data_1 = torch.cat(list_aug1).unsqueeze(1)
        data_2 = torch.cat(list_aug2).unsqueeze(1)
        data_1, data_2 = data_1.to('cuda'), data_2.to('cuda')
        feat1, feat2 = encoder(data_1), encoder(data_2)
        proj1, proj2 = projector(feat1), projector(feat2)

        loss = criterion(proj1, proj2, target_s)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)
    scheduler.step()
     
    return epoch_loss

def train_scl(encoder, projector, train_loader, criterion, optimizer, scheduler, epochs, use_wandb):

    best_loss = None
    train_losses = []

    encoder.train()
    projector.train()

    for i in range(1, epochs+1):

        print(f"Epoch {i}")
        train_loss = train_epoch_scl(encoder, projector, train_loader, criterion, optimizer, scheduler)
        print(f"Current Train Loss : {format(train_loss, '.4f')}")
        train_losses.append(train_loss)  

        if best_loss is None:
            best_loss = train_loss
        if best_loss > train_loss:
            best_loss = train_loss
        if use_wandb :
            wandb.log({
                'train_loss' : train_loss,           
            })
    
    torch.save(encoder.state_dict(), 'model/contrastive_split3.pth')
    final_state = {"encoder": encoder.state_dict()}
    print(f"Last Loss : {format(train_loss, '.4f')}\tBest Loss : {format(best_loss, '.4f')}")

    return train_losses, encoder, final_state

def linear_train_epoch(encoder, classifier, train_loader, criterion, optimizer):

    epoch_loss = 0.0  
    
    encoder.train()
    classifier.train()

    for i, batch in enumerate(train_loader):
        
        data, target_pm = batch['image'].to('cuda'),batch['target1'].to('cuda')
        
        #with torch.no_grad():
        
        features = encoder(data)
        
        #optimizer.zero_grad()

        output = classifier(features)
        loss = criterion(output, target_pm)
            
        epoch_loss += loss.item()

        _, labels_predicted = torch.max(output, dim=1)
    
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)
    

    return epoch_loss

def linear_eval_epoch(encoder, classifier, val_loader, criterion):

    epoch_loss = 0.0

    list_targets = []
    list_decisions = []

    classifier.eval()
    encoder.eval()

    with torch.no_grad():

        for i, batch in enumerate(val_loader):
            
            data, target_pm = batch['image'].to('cuda'), batch['target1'].to('cuda')
            output = classifier(encoder(data))
            loss = criterion(output, target_pm)
            epoch_loss += loss.item()

            _, labels_predicted = torch.max(output, dim=1)
            
            list_targets.append(target_pm.detach().to('cpu'))
            list_decisions.append(labels_predicted.detach().to('cpu'))
            
            if i == 0 :
                print(list_targets, labels_predicted)

    epoch_loss = epoch_loss / len(val_loader)
    targets = torch.cat(list_targets).detach().to('cpu')
    decisions_ = torch.cat(list_decisions).detach().to('cpu')
    
    true = targets.detach().to('cpu')
    pred = decisions_.detach().to('cpu')
    
    auc = roc_auc_score(true,pred)
    acc = accuracy_score(true,pred)
    f1 = f1_score(true,pred)
    del data, target_pm,targets,decisions_,true,pred,output,labels_predicted,list_decisions,list_targets
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_loss, auc,f1, acc

    #se = sum(TP[1:])/sum(GT[1:])
    #sp = TP[0]/GT[0]
    #acc = sum(TP)/sum(GT)

    #return epoch_loss, se, sp, acc

def linear_scl(encoder, classifier, train_loader, val_loader, criterion, optimizer, epochs, use_wandb):

    train_losses = []; val_losses = []; val_auc_scores = []; val_f1_scores = []; val_acc_scores = []

    best_val_acc = 0
    best_auc = 0
    best_f1 = 0
    best_epoch_acc = 0

    #state_dict = checkpoint["encoder"]
    #encoder.load_state_dict(state_dict)

    #for param in encoder.parameters():
    #    param.requires_grad = False
    encoder.eval()

    for i in range(1, epochs+1):

        print(f"Epoch {i}")

        train_loss = linear_train_epoch(encoder, classifier, train_loader, criterion, optimizer)
        train_losses.append(train_loss);
        print(f"Train loss : {format(train_loss, '.4f')}")

        val_loss, val_auc, val_f1, val_acc = linear_eval_epoch(encoder, classifier, val_loader, criterion)
        
        val_losses.append(val_loss); val_auc_scores.append(val_auc); val_f1_scores.append(val_f1); val_acc_scores.append(val_acc);
        print(f"Val loss : {format(val_loss, '.4f')}\tVal AUC : {format(val_auc, '.4f')}\tVal F1 : {format(val_f1, '.4f')}\tVal Acc : {format(val_acc, '.4f')}")
        
        wandb.log({
            'train_loss' : train_loss,
            'val_loss' : val_loss,
            'val_auc' :val_auc,
            'val_f1' : val_f1,
            'val_acc' : val_acc,
        })
        
        
        if best_val_acc == 0:
            best_val_acc = val_acc

        if i == 1:
            best_auc = val_auc
            best_f1 = val_f1

        if best_auc < val_auc:
            best_auc = val_auc
            best_f1 = val_f1

        if best_val_acc < val_acc:
            best_epoch_acc = i
            best_val_acc = val_acc
        
    print(f"best auc score is {format(best_auc, '.4f')} (f1:{format(best_f1, '.4f')} acc:{format(best_val_acc, '.4f')})")
    if use_wandb :
        wandb.log({
            'best_auc' : best_auc,
            'best_f1' : best_f1,
            'best_acc' : best_val_acc,
        })
    return train_losses, val_losses, val_auc_scores, val_f1_scores, val_acc_scores