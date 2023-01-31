import numpy as np
import torch
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
warnings.filterwarnings('ignore')

with open('MeanCropTypescan.txt', 'rb') as f:
                meansizes = pickle.load(f)

class CTScanDataset(Dataset):
    def __init__(self, data,splits_id, name_target='pm_post_tavi',type_scan='diastole',crop='no_crop',augment=False,prob=(0.25,0.25,0.25,0.25),size=200):
        self.name_target = name_target
        self.data = data
        self.splits_id = splits_id
        self.type_scan = type_scan
        self.crop = crop
        self.augment = augment
        self.prob = prob
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
            
            if row.augment == True :
                _3d_images = torch.load('/users/local/tensor_croppedmiddledist/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
                
                #wandb.log({'gamma':self.gamma, 'noise':self.noise, 'affine':self.affine, 'degre_affine':self.degreaffine
                 #})
                
                #transforms_dict = {tio.RandomNoise(std=self.noise):0.33,tio.RandomGamma(self.gamma):0.34,tio.RandomSwap():0,tio.RandomAffine(scales=(1-self.affine, 1+self.affine),degrees=self.degreaffine):0.33}              ########## Augmentation
                transforms_dict = {tio.RandomNoise(std=75):0.33,tio.RandomGamma(0.4):0.34,tio.RandomSwap():0,tio.RandomAffine(scales=(0.95, 1.05),degrees=15):0.33}
                transform_augment = tio.OneOf(transforms_dict)
                transforms = [resize,transform_augment, normalize]
                transform = tio.Compose(transforms)
                _3d_images = transform(_3d_images.unsqueeze(0))
                
            else :
                _3d_images = torch.load('/users3/local/tensor_crop_resized2_norm/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
        
        
        #if self.size == 200 and self.crop == 'no_crop' : 
        #    if self.augment and row.augment==True:
        #        index_random = str(int(np.random.choice(np.arange(0,5,1))))
        #        aug_name = self.augment 
        #        _3d_images =  torch.load('/users2/local/data_amine/tensor_final/ready/normalized_resized_200/augment/'+aug_name+'/'+ID+'/'+ID+'_'+self.type_scan+'_'+index_random+'.pt')       
        #    else : 
        #        _3d_images = torch.load('/users/local/data_amine/ready/normalized_resized_200/base/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
#
        #if self.size == 200 and self.crop =='square_even':
        #    if self.augment and row.augment==True:
        #        index_random = str(int(np.random.choice(np.arange(0,5,1))))
        #        aug_name = self.augment 
  #àacha#nger              _#3d_images =  torch.load('/users2/local/data_amine/tensor_final/ready/normalized_resized_200/augment/'+aug_name+'/'+ID+'/'+ID+'_'+self.type_scan+'_'+index_random+'.pt')       
        #    else : 
        #        _3d_images = torch.load('/users/local/data_amine/ready/crop_square_normalized_resized_200/base/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
        #    
        #if self.augment and row.augment==True:
        #    _3d_images = transform_augment(_3d_images)
        return {"image": _3d_images, "target": target, "ID": ID}
    

def compute_dict(dict_config):
    bs = dict_config['bs']
    oversample = dict_config['oversample'] 
    augment = dict_config['augment'] 
    type_scan = dict_config['type_scan']
    name_target = dict_config['name_target']
    crop = dict_config['crop']
    size = dict_config['size']
    
    
    with open(r"splits_df.pkl", "rb") as input_file:
        splits = pickle.load(input_file)
    splits_id = 0
    
    data_loaders = []
    for index in range(len(splits)): j
        df_train = splits[index][0]
        dataset = CTScanDataset(df_train.reset_index(drop=True),splits_id,name_target=name_target,type_scan=type_scan,crop=crop,augment=augment,size=size)
        train_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        df_val = splits[index][2]
        dataset = CTScanDataset(df_val.reset_index(drop=True),splits_id,name_target=name_target,type_scan=type_scan,crop=crop,augment=augment,size=size)
        val_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        df_test = splits[index][1]
        dataset = CTScanDataset(df_test.reset_index(drop=True),splits_id,name_target=name_target,type_scan=type_scan,crop=crop,augment=augment,size=size)
        test_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        data_loaders.append((train_data_loader,test_data_loader,val_data_loader))
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


def train(model,use_wandb, dls,patience=5,total_n_epoch = 20, silent = False):
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, nesterov = True, weight_decay = 5e-4)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [30,60,90], gamma = 0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=patience/2)

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    scheduler = None
    best_val, best_test,best_loss = 0, 0, 1e3 
    patience_counter= 0
    dl_train,dl_test,dl_val = dls
    for epoch in range(total_n_epoch):

        #loss = train_epoch(model,x_train,y_train, optimizer, mixup = True)
        loss = train_epoch(model,dl_train, optimizer)
        if not silent:
            print("\rEpoch {:3d} with loss {:.5f}".format(epoch, loss), end = ' ')
        
        v_auc,v_acc,v_f1 = test(model,dl_val)
        t_auc,t_acc,t_f1  = test(model,dl_test)
        if v_auc > best_val:
            best_val = v_auc
            best_test = t_auc
            best_val_metrics = (v_auc,v_acc,v_f1)
            best_test_metrics = (t_auc,t_acc,t_f1)
        if not silent:
            print("val {:.3f} test {:.3f} best test {:.3f}".format(v_auc, t_auc, best_test), end = '')
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
                       'val_auc' : v_auc, 'val_acc' : v_acc,'val_f1' : v_f1,
                       'test_auc':t_auc,'test_acc': t_acc, 'test_f1':t_f1                       
                    })
    del loss, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return best_loss,epoch,best_val_metrics,best_test_metrics

def avg_score(dict_config,list_dls, n_runs = 1,use_wandb=False):
    use_wandb = dict_config['use_wandb']
    if use_wandb:
        wandb.init(project="predict_tavi", entity='brain-imt',config = dict_config,settings=wandb.Settings(start_method='fork'))

    avg_test_score = 0
    avg_val_score = 0
    avg_train_loss = 0
    auc_test_scores = []
    auc_val_scores = []
    f1_test_scores = []
    f1_val_scores = []
    acc_test_scores = []
    acc_val_scores = []


    epochs_max = []
    indice_split = 0
    for dls in list_dls:
        print("\n Split ", indice_split+1)
        indice_split+=1
        for i in range(n_runs):
            print("     Run", i+1)
            if dict_config['model']=='resnet10':
                model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=2).to('cuda')
            else : 
                print('no other model yet')
            train_loss,epoch_, best_val_metrics,best_test_metrics = train(model, use_wandb, dls,dict_config['patience'],dict_config['n_epoch'])
            torch.cuda.empty_cache()
            t_auc,t_acc,t_f1 = best_test_metrics
            v_auc,v_acc,v_f1 = best_val_metrics


            auc_test_scores.append(t_auc)
            auc_val_scores.append(v_auc)
            f1_test_scores.append(t_f1)
            f1_val_scores.append(v_f1)
            acc_test_scores.append(t_acc)
            acc_val_scores.append(v_acc)
            
            
            epochs_max.append(epoch_)
        break
    epochs_mean = np.mean(epochs_max)
    
    auc_test_scores = stats(auc_test_scores)
    auc_val_scores = stats(auc_val_scores)
    

    f1_test_mean  =np.mean(f1_test_scores  )
    f1_val_mean   =np.mean(f1_val_scores   )
    acc_test_mean  =np.mean(acc_test_scores  )
    acc_val_mean   =np.mean(acc_val_scores   )
    
    denom = n_runs*len(list_dls)
    if use_wandb:
        wandb.log({'test_auc':auc_test_scores[0],'t_std':auc_test_scores[1],'t_low':auc_test_scores[2], 't_up':auc_test_scores[3], 't_min':auc_test_scores[4],'t_max':auc_test_scores[5],
                    'val_auc':auc_val_scores[0],'v_std':auc_val_scores[1],'v_low':auc_val_scores[2], 'v_up':auc_val_scores[3], 'v_min':auc_val_scores[4],'v_max':auc_val_scores[5],
                    #'train_loss':avg_train_loss/denom,'epochs_mean':epochs_mean,
                   'test_f1_mean':f1_test_mean,'val_f1_mean':f1_val_mean,
                   'acc_test_mean':acc_test_mean,'acc_val_mean':acc_val_mean
        })
        wandb.finish()
    #dict_results[str(params)]=stats_test[0]
    return auc_test_scores,auc_val_scores



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
        
    
    
    


#def custom_transform(_3d_images,bound_crop,test):
#    ratio_crop = np.random.choice(np.arange(bound_crop[0],bound_crop[1],0.001))
#    ratio_resize = np.random.choice(np.arange(ratio_crop-0.05,ratio_crop+0.05,0.001))
#    
#    init_x_y, init_z = _3d_images.shape[1],_3d_images.shape[0]
#    z_crop_size = int(ratio_crop * init_z)
#    x_y_crop_size = int(ratio_crop *init_x_y)
#    r_x = np.random.random()
#    r_y = np.random.random()
#    r_z = np.random.random()
#    
#    x_0 = int(r_x*(init_x_y - x_y_crop_size))
#    x_1 = init_x_y - x_0 - x_y_crop_size
#    y_0 = int(r_y*(init_x_y - x_y_crop_size))
#    y_1 = init_x_y - y_0 -x_y_crop_size
#    z_0 = int(r_z*(init_z - z_crop_size))
#    z_1 = init_z - z_0 -z_crop_size
#    
#    x_y_resize = int(ratio_resize*init_x_y)
#    z_resize  = int(ratio_resize*init_z)
#    
#    #x_y_0 = (_3d_images.shape[0] - 448)/2
#    #x_y_1 = _3d_images.shape[0] - x_y_0
#    #z0 = (_3d_images.shape[2] - 68)//2
#    #z1 = _3d_images.shape[0] - z0 
#    #crop = tio.Crop((z_0,z_1,x_0,x_1,y_0,y_1))
#    #crop = tio.Crop((x_0,x_1,y_0,y_1,z_0,z_1))
#    #crop = tio.Crop((x_y_0,x_y_1,x_y_0,x_y_1,z0,z_1))
#    crop = tio.CropOrPad((380,380,68))
#    normalize = tio.ZNormalization()
#    #resize = tio.Resize((x_y_resize, x_y_resize, z_resize))
#    if test : 
#        resize = tio.Resize((50, 376, 376))
#        transforms = [crop,normalize]
#    else : 
#        resize = tio.Resize((50, 376, 376))
#        transforms = [crop,normalize]
#        #transforms = [normalize,crop,resize]  
#    
#    transform = tio.Compose(transforms)
#    return transform
#
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