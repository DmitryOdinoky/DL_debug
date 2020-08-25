from __future__ import print_function, division

from tools import radam

import datetime

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

# import os

# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os



import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchnet as tnt

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.utils

import warnings

from data_class import fsd_dataset

warnings.filterwarnings("ignore")

plt.ion()  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#%%

parser = argparse.ArgumentParser(description='Audio classification example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


#parser.add_argument("--model", type=str, default='model_v7', help="model:model_v")

parser.add_argument("--model", type=str, default='DIY2_conv', help="model:model_v")

parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')

parser.add_argument('--batchsize_train', type=int, default=40,
                    help='input batch size for training')

parser.add_argument(
    '--batchsize_test', type=int, default=40,
    help='Steps per epoch during validation')

parser.add_argument('--epochs', type=int, default=7,
                    help='number of epochs to train')

parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate for a single GPU')

parser.add_argument('--kernel_size', type=int, default=7,
                    help='kernel size for convolution')

parser.add_argument('--padding', type=int, default=1,
                    help='padding')

parser.add_argument('--stride', type=int, default=2,
                    help='stride')

parser.add_argument('--classes_amount', type=int, default=41,
                    help='wtf')


# parser.add_argument(
#     '--decay', type=float, default=0.8,
#     help='LR decay every 10 epochs')
# parser.add_argument('--warmup-epochs', type=float, default=5,
#                     help='number of warmup epochs')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='SGD momentum')
# parser.add_argument('--wd', type=float, default=0.000005,
#                     help='weight decay')

args = parser.parse_args()

# Checkpoints will be written in the log directory.
args.checkpoint_format = os.path.join(args.log_dir, 'checkpoint-{epoch}.h5')


#%%

# train_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/train_4_classes_vs_other.csv',
#                                 path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/',
#                                 train = True)


# test_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/test_4_classes_vs_other.csv',
#                                 path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_test/',
#                                 train = False)

# train_loader = torch.utils.data.DataLoader(train_dataset,
#                             shuffle=True,
#                             batch_size = args.batchsize_train)

# test_loader = torch.utils.data.DataLoader(test_dataset,
#                             shuffle=True,
#                             batch_size = args.batchsize_test)


#%%



    
    
train_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/train_mini_dataset.csv',
                                path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/wavfiles/',
                                train = True)


test_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/test_mini_dataset.csv',
                                path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_test/',
                                train = False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                            shuffle=True,
                            batch_size = args.batchsize_train)

test_loader = torch.utils.data.DataLoader(test_dataset,
                            shuffle=False,
                            batch_size = args.batchsize_test)





#%%

# train_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/train_post_competition.csv',
#                                 path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/',
#                                 train = True)


# test_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/test_post_competition.csv',
#                                 path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_test/',
#                                 train = False)

# train_loader = torch.utils.data.DataLoader(train_dataset,
#                             shuffle=True,
#                             batch_size = args.batchsize_train)

# test_loader = torch.utils.data.DataLoader(test_dataset,
#                             shuffle=True,
#                             batch_size = args.batchsize_test)




#%%

#torch.manual_seed(1234)



Model = getattr(__import__(f'models_4_audio.{args.model}', fromlist=['Model']), 'Model')
net = Model(args)
net.to(device)




#optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
optimizer = radam.RAdam(net.parameters(), lr=args.learning_rate)





    
    
def write_report(var_dict, string):
    
    path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/ConvNets_audioClassification/reports_out/'
    
    postfix = string
    this_time = str(datetime.datetime.now().time()).replace(':','-').replace('.','-')
    this_date = str(datetime.datetime.now().date())
    todays_date = this_date + '_time_'  + this_time[:-7] + '_' + str(args.model)
    
    attributes = list(var_dict.keys())
    index = var_dict['iterationz']
    
    lst = []
    
    for key in var_dict:
    
        lst.append(var_dict[key])
      
    
    transposed = list(map(list, zip(*lst)))
    
    df = pd.DataFrame(transposed ,index=index, columns=attributes)  
    df = df.drop(columns=['iterationz'],axis=0)
      
    df.to_excel(f'output_ver_{todays_date}_{postfix}.xlsx')
    
    
    
    


number_of_epochs = args.epochs



var_dict = {'iterationz': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'train_f1_scores': [],
            'test_f1_scores': [],
            'train_tnt_loss': [],
            'test_tnt_loss': [],
            #'train_mAPMeter': [],
            #'test_mAPMeter': []
            
            }

meters = {
            'train_tnt_loss': tnt.meter.AverageValueMeter(),
            'test_tnt_loss': tnt.meter.AverageValueMeter(),
            #'train_mAPMeter': tnt.meter.mAPMeter(),
            #'test_mAPMeter': tnt.meter.mAPMeter(),
            'train_confusion': tnt.meter.ConfusionMeter(len(train_dataset.labels)),
            'test_confusion': tnt.meter.ConfusionMeter(len(train_dataset.labels))


            }


counter = 0
stage = ''




for epoch in range(number_of_epochs):
    
    for key in meters.keys():
        meters[key].reset()
        
              
    counter += 1
    print(f'Epoch #{counter} started')
    
    
    iter_epoch = 0
    
    for loader in [train_loader, test_loader]:
        
        if loader == train_loader:
            stage = 'train'
        else:
            stage = 'test'
            
        var_dict_epoch = {
            
            'iterationz': [] ,
            'tnt_loss_epoch': [] ,
            'loss_epoch': [] ,
            'accuracy_epoch': [] ,
            'f1_epoch': []
            
            }
        

        helper = 0
        
        for batch in loader:
            
            helper += 1
             
            iter_epoch += 1
            
            # print(helper)
            
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            train_X = images
            train_y = labels
            
            optimizer.zero_grad()
            y_prim = net.forward(train_X)
            
            class_count = y_prim.size(1)          
            tmp = torch.arange(class_count)          
            
            
            y = (train_y.unsqueeze(dim=1) == tmp).float()
            
            
            
            loss = torch.mean(-y*torch.log(y_prim))
            
            meters[f'{stage}_tnt_loss'].add(loss.item())
            
           
            #confusion_meter: tnt.meter.ConfusionMeter = meters[f'{stage}_confusion']
            
            meters[f'{stage}_confusion'].add(y_prim.detach(), labels.detach())
            
 

            _, predict_y = torch.max(y_prim, 1)
            
            correct = 0
            total = 0
            
            for i in range(len(images)):
                act_label = torch.argmax(y_prim[i]) # act_label = 1 (index)
                pred_label = torch.argmax(y[i]) # pred_label = 1 (index)
            
                if(act_label == pred_label):
                    correct += 1
                total += 1
            
            accuracy = correct/total
            
            #accuracy = accuracy_score(train_y.data, predict_y.data)
            
            
            f1 = sklearn.metrics.f1_score(train_y.data, predict_y.data, average='macro')
            
            
            var_dict_epoch['tnt_loss_epoch'].append(meters[f'{stage}_tnt_loss'].value()[0])
            
            var_dict_epoch['loss_epoch'].append(loss.item())
            var_dict_epoch['accuracy_epoch'].append(accuracy)
            var_dict_epoch['f1_epoch'].append(f1.item())
            var_dict_epoch['iterationz'].append(iter_epoch)
            
            
           
            
            
            if not (helper % 1):
                print(f"Iteration: {helper}, Loss: {loss.item()}, Accuracy: {accuracy}")
            
            
            
            if loader == train_loader:
                
                loss.backward()
                optimizer.step()
                
        
                
        if stage == 'train':
   
           var_dict[f'{stage}_loss'].append(np.average(var_dict_epoch['loss_epoch']))
           var_dict[f'{stage}_accuracies'].append(np.average(var_dict_epoch['accuracy_epoch']))
           var_dict[f'{stage}_f1_scores'].append(np.average(var_dict_epoch['f1_epoch']))
           
           var_dict[f'{stage}_tnt_loss'].append(meters[f'{stage}_tnt_loss'].value()[0])
           #meters[f'{stage}_mAPMeter'].add(meters[f'{stage}_mAPMeter'].value())
             

                                       
        else:
           

            var_dict[f'{stage}_loss'].append(np.average(var_dict_epoch['loss_epoch']))
            var_dict[f'{stage}_accuracies'].append(np.average(var_dict_epoch['accuracy_epoch']))
            var_dict[f'{stage}_f1_scores'].append(np.average(var_dict_epoch['f1_epoch']))
 
            var_dict[f'{stage}_tnt_loss'].append(meters[f'{stage}_tnt_loss'].value()[0])
            #meters[f'{stage}_mAPMeter'].add(meters[f'{stage}_mAPMeter'].value())
            
            
        confusion_matrix = meters[f'{stage}_confusion'].value()
        
        #confusion_matrix = confusion_meter.value() 
        
        
        
        
        write_report(var_dict_epoch, f'per_epoch_{stage}')
            
        
   
    
    var_dict['iterationz'].append(counter)
    


  
    
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False)

ax1.set_title('1: Loss.  2: Accuracy. 3: F1. ')

sc1 = ax1.plot(var_dict['iterationz'], var_dict['train_loss'], label='train')
sc2 = ax1.plot(var_dict['iterationz'],  var_dict['test_loss'], label='test')


sc3 = ax2.plot(var_dict['iterationz'],  var_dict['train_accuracies'])
sc4 = ax2.plot(var_dict['iterationz'],  var_dict['test_accuracies'])

sc5 = ax3.plot(var_dict['iterationz'], var_dict['train_f1_scores'])
sc6 = ax3.plot(var_dict['iterationz'], var_dict['test_f1_scores'])


ax1.set_xticks([])
ax2.set_xticks([])

leg = ax1.legend(loc='upper right');







write_report(var_dict, 'general')





#%%