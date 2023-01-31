from utils import compute_dict, avg_score

dict_config = {'model':'resnet10',
              'normalization':'global',
              'patience':15,
              'n_epoch':3,
             'earlystopping':True,
              'use_wandb' : False,
               'type_scan':'calcique',
               'crop':'crop_padd',
                'oversample':True,
               'augment':False,
            'augment_prob':(),
               'size':199,
               'comment':'',
               'bs':1,
               'name_target':'pm_post_tavi'
               }


list_dls = compute_dict(dict_config)
stats_test,stats_val = avg_score(dict_config,list_dls, n_runs = 1,use_wandb=dict_config['use_wandb'])

stats/stats_mean_std_diastole_0.pkl
