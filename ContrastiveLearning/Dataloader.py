from torch.utils.data import Dataset,DataLoader
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
class CTScanDataset(Dataset):
    def __init__(self, data,splits_id, name_target='pm_post_tavi',type_scan='diastole',crop='no_crop',augment=False,prob=(0.25,0.25,0.25,0.25),size=200):
        self.name_target = name_target
        self.data = data
        self.splits_id = splits_id
        self.type_scan = type_scan
        self.crop = crop
        self.augment = augment    
        #self.mesures = pd.read_csv('mesures_n_s_se.csv')
        self.size = size
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        ID = row.ID

        target1 = int(row[self.name_target])
        
        hsm_cluster = pd.read_csv('clusters_hauteur.csv')
        hsm_patient = hsm_cluster[hsm_cluster['n_inclusion']==ID] 
        
        target0 = int(hsm_patient.hsm_cluster)

        
        _3d_images = torch.load('/users3/local/tensor_crop_resized2_norm/'+ID+'/'+ID+'_'+self.type_scan+'.pt')
                
                
        return {"image": _3d_images, "target0": target0, "target1" : target1, "ID": ID}
    

    
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
        df_test = splits[index][1]
        dataset = CTScanDataset(df_test.reset_index(drop=True),splits_id,name_target=name_target,type_scan=type_scan,crop=crop,augment=False,size=size)
        test_data_loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
        data_loaders.append((train_data_loader,test_data_loader))
    return data_loaders
    