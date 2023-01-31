from torch.utils.data import Dataset,DataLoader
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
from transform import transforms_train, transforms_valid
import torchio as tio
import numpy as np
import random as rd


def minicube(_3d_image, id_, type_scan ):
    mesures = pd.read_csv('/users2/local/data_amine/mesures.csv')
    mes =mesures[(mesures['id']==id_) & (mesures['typescan']==type_scan)]  
    
    minx,maxx,miny,maxy,minz,maxz,zshape = abs(int(mes.minx)),abs(int(mes.maxx)),abs(int(mes.miny)),abs(int(mes.maxy)),abs(int(mes.minz)),abs(int(mes.maxz)),abs(int(mes.z_shape))
    x_crop, y_crop, z_crop = rd.randint(minx, maxx-100), rd.randint(miny, maxy-100), rd.randint(minz, maxz-100) 
    return _3d_image[x_crop:x_crop+100, y_crop:y_crop+100, zshape-(z_crop+100):zshape-(z_crop)]

class CTScanDataset(Dataset):
    def __init__(self, data,splits_id,transform, name_target='pm_post_tavi',type_scan='diastole',augment=False,prob=(0.25,0.25,0.25,0.25),size=200):
        self.name_target = name_target
        self.data = data
        self.splits_id = splits_id
        self.type_scan = type_scan
        self.augment = augment 
        self.transform = transform
        #self.mesures = pd.read_csv('mesures_n_s_se.csv')
        self.size = size
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        ID = row.ID

        target = int(row[self.name_target])

        _3d_images = torch.load('/users3/local/tensor_final/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
        _3d_images = _3d_images.transpose(0,2)
        type_scan = 'ds' if self.type_scan!='calcique' else 'calcique'
        _3d_images = minicube(_3d_images, ID, type_scan)
        if row.augment :
            transforms_dict = {tio.RandomNoise(std=50):0.25,tio.RandomGamma(0.3):0.25,tio.RandomSwap(2):0.25,tio.RandomAffine(scales=(0.96, 1.04),degrees=15):0.25}
            transform_augment = tio.OneOf(transforms_dict)            
            transforms = [transform_augment]
            transform = tio.Compose(transforms)
            _3d_images = transform(_3d_images.unsqueeze(0))
            _3d_images = _3d_images[0].short()
        _2d_images = []
        
        for i in range (1,99):
            image = _3d_images[:,:,i-1:i+2]
            #if row.augment :
            #    image = image.numpy()
            #    image = self.transform(image=image)['image']
            #    image = image.astype(np.float32)
            #    image = torch.tensor(image).short()
            image = image.transpose(0,2)

            _2d_images.append(image)
                         
        _2d_images = torch.stack(_2d_images,0)
        target = torch.tensor([target]*98).float()
                
        return {"image": _2d_images, "target" : target, "ID": ID}
    

    
def compute_dict(dict_config, splitname):
    bs = dict_config['bs']
    oversample = dict_config['oversample'] 
    augment = dict_config['augment'] 
    type_scan = dict_config['type_scan']
    name_target = dict_config['name_target']
    size = dict_config['size']
    
    
    with open(splitname, "rb") as input_file:
        splits = pickle.load(input_file)
    splits_id = 0
    
    data_loaders = []
    for index in range(len(splits)): 
        df_train = splits[index][0]
        dataset = CTScanDataset(df_train.reset_index(drop=True),splits_id,transforms_train, name_target=name_target,type_scan=type_scan,augment=augment,size=size)
        train_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        df_test = splits[index][1]
        dataset = CTScanDataset(df_test.reset_index(drop=True),splits_id,transforms_valid, name_target=name_target,type_scan=type_scan,augment=False,size=size)
        test_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        data_loaders.append((train_data_loader,test_data_loader))
    return data_loaders
    