import numpy as np
import torch
import os
from copy import copy
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split 
from torchvision.transforms import Resize
import monai
from torch.utils.data import Dataset,DataLoader
import torchio as tio
import pickle
import time
import torch.nn.functional as F
import wandb
import scipy.stats as st
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,balanced_accuracy_score,roc_auc_score
import gc
import warnings


from medcam import medcam
import random as rd

warnings.filterwarnings('ignore')


path_0 = '/users2/local/data_amine/dicom_final'
with open('Maxcropbound.txt', 'rb') as f:
                maxcropbound = pickle.load(f)
with open('MeanCropTypescan.txt', 'rb') as f:
                meansizes = pickle.load(f)
        
        
def getmaxcrop(path_0):
    
    ids_ = os.listdir(path_0) 
    
    max_calcique,max_ds = 0,0
    
    list_x_c,list_x_ds = [],[]
    list_y_c, list_y_ds = [],[]
    list_z_c, list_z_ds = [],[]
    for ID in tqdm(ids_) :             
        for type_scan in (['calcique','systole','diastole']):
            _3d_images = torch.load('/users2/local/data_amine/tensor_final_cropped/'+ID+'/'+ID+'_'+type_scan+'.pt')
            (x_size,y_size,z_size) = _3d_images.size()
            if type_scan != 'calcique':
                list_x_ds.append(x_size),list_y_ds.append(y_size),list_z_ds.append(z_size)
            else :
                list_x_c.append(x_size),list_y_c.append(y_size),list_z_c.append(z_size)
    return {'calcique': [max(list_x_c), max(list_y_c), max(list_z_c)], 'diastole' : [max(list_x_ds), max(list_y_ds), max(list_z_ds)], 'systole' : [max(list_x_ds), max(list_y_ds), max(list_z_ds)] }


 
def minicube(_3d_image, id_, type_scan ):
    mesures = pd.read_csv('/users2/local/data_amine/mesures.csv')
    mes =mesures[(mesures['id']==id_) & (mesures['typescan']==type_scan)]  
    
    minx,maxx,miny,maxy,minz,maxz,zshape = abs(int(mes.minx)),abs(int(mes.maxx)),abs(int(mes.miny)),abs(int(mes.maxy)),abs(int(mes.minz)),abs(int(mes.maxz)),abs(int(mes.z_shape))
    x_crop, y_crop, z_crop = rd.randint(minx, maxx-100), rd.randint(miny, maxy-100), rd.randint(minz, maxz-100) 
    return _3d_image[x_crop:x_crop+100, y_crop:y_crop+100, zshape-(z_crop+100):zshape-(z_crop)]




class CTScanDataset(Dataset):
    def __init__(self, data,splits_id, name_target='pm_post_tavi',type_scan='diastole',crop='no_crop',augment=False,prob=(0.25,0.25,0.25,0.25),size=200):
        self.name_target = name_target
        self.data = data
        self.splits_id = splits_id
        self.type_scan = type_scan
        self.crop = crop
        self.augment = augment
        
    
            
        with open(r"stats/stats_mean_std_"+self.type_scan+"_"+str(self.splits_id)+".pkl", "rb") as input_file:
            self.mean,self.std = pickle.load(input_file)
        #self.mesures = pd.read_csv('mesures_n_s_se.csv')
        self.size = size
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        ID = row.ID
        target = int(row[self.name_target])
        
        
        #df_pst=pd.read_excel('/users2/local/data_amine/PixelSpacingThickness.xlsx')
        #mesures = pd.read_excel('NewCrop.xlsx')
        #df_mes = pd.read_csv('/users2/local/data_amine/mesures.csv')
            
        if self.crop == 'cube_irl':
            #_3d_images = torch.load('/users/local/tensor_croppedmiddledist/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
            
            
            resizex, resizey, resizez = meansizes[self.type_scan][0], meansizes[self.type_scan][1], meansizes[self.type_scan][2]
            resize = tio.Resize((int(resizex/2), int(resizey/2), int(resizez/4)))
            #
            normalize = tio.ZNormalization()
            #p = rd.random()
            #threshold = (0.8 if target==1 else 0.14)
            #threshold = 0.8
            #if p <threshold and row.augment == True:
            ##########################################################################################################################################################################################
            if row.augment == True and self.augment == True :
                _3d_images = torch.load('/users/local/tensor_croppedmiddledist/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
                
                 
                
                #transforms_dict = {tio.RandomNoise(std=self.noise):0.33,tio.RandomGamma(self.gamma):0.34,tio.RandomSwap():0,tio.RandomAffine(scales=(1-self.affine, 1+self.affine),degrees=self.degreaffine):0.33}              ########## Augmentation
                transforms_dict = {tio.RandomNoise(std=60):0.33,tio.RandomGamma(0.3):0.34,tio.RandomSwap(2):0.34,tio.RandomAffine(scales=(0.96, 1.04),degrees=15):0.33}
                transform_augment = tio.OneOf(transforms_dict)
                transform_augment2 = tio.OneOf(transforms_dict) 
                
                transforms = [resize,transform_augment,transform_augment2,normalize]
                transform = tio.Compose(transforms)
                _3d_images = transform(_3d_images.unsqueeze(0))
                
            else :
                _3d_images = torch.load('/users3/local/tensor_crop_resized2_norm/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
                
                
        
        
            #_3d_images = torch.load('/users3/local/tensor_crop_resized2_norm/'+ID+'/'+ID+'_'+self.type_scan+'.pt')

        
        if self.crop  == 'cube100' :
            _3d_images = torch.load('/users3/local/resize100_norm/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
            
            #_3d_images = torch.load('/users3/local/tensor_crop_resized2_norm/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
        
        elif self.crop == 'random_cube' :
            _3d_images = torch.load('/users3/local/tensor_final/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
            _3d_images = _3d_images.transpose(0,2)
            type_scan = 'ds' if self.type_scan!='calcique' else 'calcique'
            _3d_images = minicube(_3d_images, ID, type_scan)
            if _3d_images.shape[2] != 100 :
                print(_3d_images.shape, ID)
            normalize = tio.ZNormalization()
            transforms = [normalize]
            
            if row.augment == True :
                transforms_dict = {tio.RandomNoise(std=75):0.33,tio.RandomGamma(0.3):0.34,tio.RandomSwap():0,tio.RandomAffine(scales=(0.96, 1.04),degrees=20):0.33}
                transform_augment = tio.OneOf(transforms_dict)
                transforms = [transform_augment, normalize]
                transform = tio.Compose(transforms)
                _3d_images = transform(_3d_images.unsqueeze(0))
                
            else :
                transform = tio.Compose(transforms)
                _3d_images = transform(_3d_images.unsqueeze(0))
        
        elif self.crop == 'sv_randomcube':
            j = rd.randint(0,3)
            _3d_images = torch.load('/users3/local/smallcube100/'+ID+'/'+ID+'_'+self.type_scan+'_'+str(j)+'.pt')
            normalize = tio.ZNormalization()
            transforms = [normalize]
            
            if self.augment == True and row.augment == True :
                ransforms_dict = {tio.RandomNoise(std=75):0.33,tio.RandomGamma(0.3):0.34,tio.RandomSwap():0,tio.RandomAffine(scales=(0.96, 1.04),degrees=20):0.33}
                transform_augment = tio.OneOf(transforms_dict)
                transforms = [transform_augment, normalize]
                transform = tio.Compose(transforms)
            
            else :
                
                transform = tio.Compose(transforms)
                
            _3d_images = transform(_3d_images.unsqueeze(0))
                
            


        #if self.augment and row.augment==True:
        #    
        #    transforms_dict = {tio.RandomNoise(std=75):0,tio.RandomGamma(0.4):0,tio.RandomSwap():1,tio.RandomAffine(scales=(0.96, 1.04),degrees=15):0}
        #    transform_augment = tio.OneOf(transforms_dict)
        #    _3d_images = transform_augment(_3d_images)
        #
        return {"image": _3d_images, "target": target, "ID": ID}

def compute_dict(dict_config, splitname):
    bs = dict_config['bs']
    oversample = dict_config['oversample'] 
    augment = dict_config['augment'] 
    type_scan = dict_config['type_scan']
    name_target = dict_config['name_target']
    crop = dict_config['crop']
    size = dict_config['size']
    
    
    with open(splitname, "rb") as input_file:
        splits = pickle.load(input_file)
    splits_id = 0
    
    data_loaders = []
    for index in range(len(splits)): 
        df_train = splits[index][0]
        dataset = CTScanDataset(df_train.reset_index(drop=True),splits_id,name_target=name_target,type_scan=type_scan,crop=crop,augment=augment,size=size)
        train_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        #df_val = splits[index][2]
        #dataset = CTScanDataset(df_val.reset_index(drop=True),splits_id,name_target=name_target,type_scan=type_scan,crop=crop,augment=augment,size=size)
        #val_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=bs,drop_last=True,persistent_workers=True)
        df_test = splits[index][1]
        dataset = CTScanDataset(df_test.reset_index(drop=True),splits_id,name_target=name_target,type_scan=type_scan,crop=crop,augment=False,size=size)
        test_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        data_loaders.append((train_data_loader,test_data_loader))
    return data_loaders
    
def train_epoch(model,dl, optimizer):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_bs = 0
    x,y = 0,0
    for i, batch in enumerate(dl):
        #x = batch['t1'][tio.DATA].to('cuda')
        #y = batch['diagnosis'].to('cuda')
        x,y = batch['image'].to('cuda'),batch['target'].to('cuda')
        
        output = model(x)
        
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
    
        
        
        
        total_loss += loss.item()
        total_bs += 1
        optimizer.step()
    return (total_loss / total_bs)

def test(model, dl):
    model.eval()
    with torch.no_grad():
        score, total = 0, 0
        list_y = []
        list_decisions = []
        for i, batch in enumerate(dl):
            #x = batch['t1'][tio.DATA].to('cuda')
            #y = batch['diagnosis'].to('cuda')
            x,y = batch['image'].to('cuda'),batch['target'].to('cuda')
            output = model(x)
            decisions = torch.argmax(output, dim = 1)            
            #score += (decisions == y).int().sum()
            #total += decisions.shape[0]
            list_y.append(y.detach().to('cpu'))
            list_decisions.append(decisions.detach().to('cpu'))
    y_ = torch.cat(list_y).detach().to('cpu')
    decisions_ = torch.cat(list_decisions).detach().to('cpu')
    print(decisions_)
    #precision = precision_score(y_.detach().to('cpu'),decisions_.detach().to('cpu'),zero_division=0)
    #recall = recall_score(y_.detach().to('cpu'),decisions_.detach().to('cpu'))
    true = y_.detach().to('cpu')
    pred = decisions_.detach().to('cpu')
    auc = roc_auc_score(true,pred)
    acc = accuracy_score(true,pred)
    f1 = f1_score(true,pred)
    del x, y,y_,decisions_,true,pred,output,decisions,list_decisions,list_y
    gc.collect()
    torch.cuda.empty_cache()
    return auc,acc,f1
    #return (score / total).item()


def train(model,dls,use_wandb, patience=5,total_n_epoch = 20, silent = False ):
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, nesterov = True, weight_decay = 5e-4)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [30,60,90], gamma = 0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=patience/2)

    optimizer = torch.optim.Adam(model.parameters(),lr = 5e-5, weight_decay = 5e-4)
    scheduler = None
    best_test_auc, best_test_acc, best_test_f1,best_loss = 0,0,0, 1e3 
    patience_counter= 0
    dl_train,dl_test = dls
    i = 0 
    while os.path.exists('model/Model'+'pretrainedcube'+str(i)+'.pth'):
        i+=1
    for epoch in range(total_n_epoch):

        #loss = train_epoch(model,x_train,y_train, optimizer, mixup = True)
        #if epoch >= 10 :
        #    for param in model.parameters():
        #            param.requin_grad = True
        loss = train_epoch(model,dl_train, optimizer)
        if not silent:
            print("\rEpoch {:3d} with loss {:.5f}".format(epoch, loss), end = ' ')
        
        t_auc,t_acc,t_f1  = test(model,dl_test)
        if t_auc > best_test_auc:
            torch.save(model.state_dict(), 'model/Model'+'pretrainedcube'+str(i)+'.pth')
            best_test_auc = t_auc
        if t_acc > best_test_acc:
            best_test_acc = t_acc
        if t_f1 > best_test_f1:
            best_test_f1 = t_f1
        
        best_test_metrics = (best_test_auc,best_test_acc,best_test_f1)
        if not silent:
            print("test {:.3f} best test {:.3f}".format(t_auc, best_test_auc), end = '')
        if scheduler:
            #scheduler.step(loss)
            scheduler.step()
        if patience :
            if round(loss,3) < round(best_loss,3):
                best_loss=loss
                patience_counter = 0
            else :
                patience_counter+=1
            if patience_counter==patience:
                break        
        if use_wandb :
            wandb.log({'epoch':epoch,'train_loss':loss,
                       'test_auc':t_auc,'test_acc': t_acc,                    
                   'test_f1':t_f1 })
    
    
    
    
    
    return best_loss,epoch,best_test_metrics

def avg_score(dict_config, resnet_inplanes, list_dls, n_runs = 1,use_wandb=False):
    if use_wandb:
        wandb.init(project="Predict-Tavi", entity="brain-imt", config = dict_config,settings=wandb.Settings(start_method='fork'))
        #wandb.init(project="Tavi_projet", entity="tavi_team", config = dict_config,settings=wandb.Settings(start_method='fork'))
    use_wandb = dict_config['use_wandb']
    

    avg_test_score = 0
    avg_train_loss = 0
    auc_test_scores = []
    f1_test_scores = []
    acc_test_scores = []


    epochs_max = []
    indice_split = 0
    for dls in list_dls:
        print("\n Split ", indice_split+1)
        indice_split+=1
        
        for i in range(n_runs):
            print("     Run", i+1)
            if dict_config['model']=='resnet10':
                
                model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=2, inplanes = resnet_inplanes).to('cuda')
                #pretrain = torch.load("resnet_101.pth")
                #pretrain['state_dict'] = {k.replace("module.", ""):v for k, v in pretrain['state_dict'].items()}
                #model.load_state_dict(pretrain['state_dict'], strict = False)
                #for param in model.parameters():
                #    param.requires_grad = False
                #for param in model.layer4.parameters():
                #    param.requires_grad = True
                #model.fc = torch.nn.Identity()
                #model.fc = torch.nn.Linear(2048,2)
                
                #model = medcam.inject(model, output_dir="attention_maps",label=1, layer='layer4')
            elif dict_config['model']=='resnet50':
                model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, n_classes=2).to('cuda')
                model.to('cuda')
            elif dict_config['model']=='resnet50pretrained':
                
                model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, n_classes=2).to('cuda')
                pretrain = torch.load("resnet_50_23dataset.pth")
                pretrain['state_dict'] = {k.replace("module.", ""):v for k, v in pretrain['state_dict'].items()}
                model.load_state_dict(pretrain['state_dict'], strict = False)
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.layer3.parameters():
                    param.requires_grad = True
                for param in model.layer4.parameters():
                    param.requires_grad = True
                model.fc = torch.nn.Identity()
                model.fc = torch.nn.Linear(2048,2)
                model.to('cuda')
                #model = medcam.inject(model, output_dir="attention_maps",label=1, layer='layer4')
                
                
            else : 
                print('no other model yet')
            train_loss,epoch_,best_test_metrics = train(model,dls, dict_config['use_wandb'], dict_config['patience'],dict_config['n_epoch'])
            torch.cuda.empty_cache()
            t_auc,t_acc,t_f1 = best_test_metrics


            auc_test_scores.append(t_auc)
            f1_test_scores.append(t_f1)
            acc_test_scores.append(t_acc)
                
            
            epochs_max.append(epoch_)
            
    epochs_mean = np.mean(epochs_max)
    
    auc_test_mean = np.mean(auc_test_scores)
    
    auc_test_scores = stats(auc_test_scores)
    

    f1_test_mean  =np.mean(f1_test_scores  )
    acc_test_mean  =np.mean(acc_test_scores  )
    
    
    
    denom = n_runs*len(list_dls)
    #if use_wandb:
     #   wandb.log({'test_auc':auc_test_scores[0],'t_std':auc_test_scores[1],'t_low':auc_test_scores[2], 't_up':auc_test_scores[3], 't_min':auc_test_scores[4],'t_max':auc_test_scores[5],
    #                'val_auc':auc_val_scores[0],'v_std':auc_val_scores[1],'v_low':auc_val_scores[2], 'v_up':auc_val_scores[3], 'v_min':auc_val_scores[4],'v_max':auc_val_scores[5],
    #                'train_loss':avg_train_loss/denom,'epochs_mean':epochs_mean,
    #               'test_f1':f1_test_mean,'val_f1':f1_val_mean,
    #               'acc_test':acc_test_mean,'acc_val':acc_val_mean
    #    })
        #wandb.finish()
    #dict_results[str(params)]=stats_test[0]
    
    if use_wandb :
        wandb.log({'Mean_best_f1':f1_test_mean,
                   'Mean_best_auc': auc_test_mean,
                   'Mean_best_acc' : acc_test_mean 
                     })
        wandb.finish()
    return auc_test_scores



def stats(scores, name=" "):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    #if name == "":
    #    return np.mean(scores), up - np.mean(scores)
    mean_ = np.mean(scores)
    std_ = np.std(scores)
    min_ = np.min(scores)
    max_ = np.max(scores)
    print("{:s} {:.2f} (± {:.2f}) (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * mean_ , 100 * std_, 100 * low, 100 * up, 100 * min_, 100 * max_))
    return mean_,std_,low,up,min_,max_
        
    
    
    




    



#class CTScanDataset(Dataset):
#    def __init__(self, data, name_target='pm_post_tavi',type_scan='diastole',bound_crop = (0.25,0.4),test=False): #340 200
#        self.name_target = name_target
#        self.data = data
#        self.type_scan = type_scan
#        self.bound_crop = bound_crop
#        self.test = test
#    def __len__(self):
#        return len(self.data)
#
#    def __getitem__(self, index):
#        start = time.time()
#        row = self.data.loc[index]
#        ID = row.ID
#        target = int(row[self.name_target])
#        _3d_images = torch.load('/users/local/data_amine/tensor_final/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
#        _3d_images = _3d_images.transpose(0,2)
#        transform = custom_transform(_3d_images,self.bound_crop,self.test)
#        _3d_images = transform(_3d_images.unsqueeze(0))
#        return {"image": _3d_images, "target": target, "ID": ID}
#    
#    
#
##def basic_rus_ros(df_,type_sampling='rus'):
##    if type_sampling=='rus':
##        rus = RandomUnderSampler(random_state=10)
##        rus_df,rus_y = rus.fit_resample(df_[['ID']],df_.pm_post_tavi)
##        rus_df['pm_post_tavi'] = rus_y
##        return rus_df
##    elif type_sampling == 'ros' :
##        ros = RandomOverSampler(random_state=10)
##        ros_df,ros_y = ros.fit_resample(df_[['ID']],df_.pm_post_tavi)
##        ros_df['pm_post_tavi'] = ros_y
##        return ros_df
##    else :
##        return df_
##def split_val_test(df_):
##    test,val = train_test_split(df_, stratify = df_.pm_post_tavi,test_size=0.5,random_state=10)
##    return test,val
##
##
##def ret_dl(dataset,bs=5,type_scan='calcique',bound_crop = (0.25,0.4),test=False):
##    return torch.utils.data.DataLoader(CTScanDataset(data=dataset.reset_index(drop=True),type_scan=type_scan,bound_crop=bound_crop,test=test),batch_size=bs,                        shuffle=True,num_workers=bs,drop_last=True,persistent_workers=True,pin_memory=True)
##
#
#    
#    
##def compute_config(dict_config):
##    n_fold =dict_config['n_fold']
##    bs = dict_config['bs']
##    bound_crop = dict_config['bound_crop']
##    type_scan = dict_config['type_scan']
##    type_sampling = dict_config['resampling'] 
##    type_scan = dict_config['type_scan']
##    n_fold = dict_config['n_fold']
##    
##    df = pd.read_csv('/users/local/data_amine/label_pm_correct.csv')
##    df = df[df['marque_scanner']==dict_config['marque_scanner']].drop(columns='marque_scanner').reset_index(drop=True)
##    kf = StratifiedKFold(n_splits=n_fold,shuffle = True,random_state=42)
##    splits_df = [(basic_rus_ros(df.loc[list(tr)],type_sampling),split_val_test(df.loc[list(te)])[0],split_val_test(df.loc[list(te)])[1]) for tr,te in kf.split(df.ID,df.pm_post_tavi)]
##    splits_loaders = [(ret_dl(train,bs,type_scan,bound_crop,False),ret_dl(test,bs,type_scan,bound_crop,True),ret_dl(val,bs,type_scan,bound_crop,True))  for train,test,val in splits_df]
##    
##    return splits_df,splits_loaders
#
#
##def crop_border(t):
##    if -2048 in t[0]:
##        r = 512/2
##        s = int(np.ceil((512 - r*np.sqrt(2))/2))
##        v,_ = t[:,s:512-s,s:512-s].flatten(1,2).min(dim=1)
##        int_v = (v!=-2048).int()
##        start = int_v.argmax()
##        return t[:,s:512-s,s:512-s][start:-start]
##    else:
##        return t
##    
#
#

#def compute_config(dict_config):
#    n_fold =dict_config['n_fold']
#    bs = dict_config['bs']
#    x_y = dict_config['x_y']
#    z = dict_config['z']
#    type_scan = dict_config['type_scan']
#    type_sampling = dict_config['resampling'] 
#    df = pd.read_csv('/users/local/data_amine/label_pm_correct.csv')
#    df['path'] = df.ID.apply(lambda x : '/users/local/data_amine/dicom_final/'+x+'/'+type_scan+'/')
#    df['subject'] = df.apply(lambda x : tio.Subject(t1=tio.ScalarImage(x.path),diagnosis=x.pm_post_tavi),axis=1)
#    
#    normalize = tio.ZNormalization()
#    resize  = tio.Resize((x_y, x_y, z))
#    transforms = [resize,normalize]
#    transform = tio.Compose(transforms)
#    
#    
#    kf = StratifiedKFold(n_splits=n_fold,shuffle = True,random_state=42)
#    #type_sampling = dict['config'] resampling
#    transform = tio.Compose(transforms)
#    splits_df = [(basic_rus_ros(df.loc[list(tr)],transform,type_sampling),split_val_test(df.loc[list(te)],transform)[0],split_val_test(df.loc[list(te)],transform)[1]) for tr,te in kf.split(df.ID,df.pm_post_tavi)]
#    splits_loaders = [(ret_dl(train,bs),ret_dl(test,bs),ret_dl(val,bs) )  for train,test,val in splits_df]
#    
#    return splits_loaders



### last dataset

#class CTScanDataset(Dataset):
#    def __init__(self, data,splits_id, name_target='pm_post_tavi',type_scan='diastole',crop='square_even',augment=False,prob=(0.25,0.25,0.25,0.25),size=300):
#        self.name_target = name_target
#        self.data = data
#        self.splits_id = splits_id
#        self.type_scan = type_scan
#        self.crop = crop
#        self.augment = augment
#        self.prob = prob
#        with open(r"stats/stats_mean_std_"+self.type_scan+"_"+str(self.splits_id)+".pkl", "rb") as input_file:
#            self.mean,self.std = pickle.load(input_file)
#        self.mesures = pd.read_csv('mesures_n_s_se.csv')
#        self.size = size
#    def __len__(self):
#        return len(self.data)
#
#    def __getitem__(self, index):
#        row = self.data.loc[index]
#        if self.augment and row.augment==True:
#            transforms_dict = {tio.RandomNoise():self.prob[0],tio.RandomSwap(20,120): self.prob[1],
#                           tio.RandomGamma():self.prob[2],tio.RandomAffine(scales=(0.8, 1.2),degrees=10):self.prob[3]} 
#            transform_augment = tio.OneOf(transforms_dict)
#        ID = row.ID
#        target = int(row[self.name_target])
#        mes_type_scan = 'ds' if self.type_scan!='calcique' else 'calcique'
#        mes = self.mesures[(self.mesures['id']==ID) & (self.mesures['typescan']==mes_type_scan)]
#        if self.crop =='square':
#            minx,maxx,miny,maxy,minz,maxz,zshape = int(mes.s_minx),int(mes.s_maxx),int(mes.s_miny),int(mes.s_maxy),int(mes.s_minz),int(mes.s_maxz),int(mes.z_shape)
#        if self.crop =='small' : 
#            minx,maxx,miny,maxy,minz,maxz,zshape = int(mes.minx),int(mes.maxx),int(mes.miny),int(mes.maxy),int(mes.minz),int(mes.maxz),int(mes.z_shape)
#        if self.crop == 'square_even':
#            minx,maxx,miny,maxy,minz,maxz,zshape = int(mes.se_minx),int(mes.se_maxx),int(mes.se_miny),int(mes.se_maxy),int(mes.se_minz),int(mes.se_maxz),int(mes.z_shape)
#
#        
#        _3d_images = torch.load('/users/local/data_amine/tensor_final/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
#        _3d_images = _3d_images.transpose(0,2)
#        if self.crop:
#            _3d_images = _3d_images[minx:maxx,miny:maxy,zshape -maxz :zshape - minz]
#        
#        _3d_images = (_3d_images - self.mean)/self.std
#        _3d_images = _3d_images.unsqueeze(0)
#        if self.crop :
#            crop = tio.CropOrPad((40,40,40))
#            resize = tio.Resize((self.size, self.size, self.size))
#            #transforms = [crop,resize]
#            transforms = [crop]
#            transform = tio.Compose(transforms)
#            _3d_images = transform(_3d_images)
#        else : 
#            resize = tio.Resize((self.size,self.size, self.size))
#            _3d_images = resize(_3d_images)
#        if self.augment and row.augment==True:
#            _3d_images = transform_augment(_3d_images)
#        return {"image": _3d_images, "target": target, "ID": ID}
#    