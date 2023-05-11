# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:25:48 2022

@author: Marwa
"""

from models_db_seg import  U_Net,  AttU_Net, NestedUNet, AttNested_U_Net,AttDC_Nested_U_Net
from dataload_db2 import DataBone
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
img_size = 224
batch_size = 8
epochs = 60
ALPHA = 0.8
GAMMA = 2
model_Inputs = [AttDC_Nested_U_Net]
model_Inputs_str = ['AttDC_Nested_U_Net']
######################


############################
# Part 2
train_transform = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

############################
# Part 2

for foldd in range(1,6):
    print(foldd)
    tr_path = './train_db_'+str(foldd)
    test_path = './test_db_'+str(foldd)

    tr_indxs = torch.load('train_db_idx_'+str(foldd)+'.pt')
    test_indxs = torch.load('test_db_idx_'+str(foldd)+'.pt')
    
    train_set = DataBone(
            list_IDs = tr_indxs, 
            path = tr_path, 
            transform=train_transform
    )
    valid_set = DataBone(
            list_IDs = test_indxs, 
            path = test_path, 
            transform=val_transforms
    )
    
    ############################
    
    ############################
    val_path = "./Models"
    if not os.path.exists(val_path):
        os.makedirs(val_path) 
    ############################
    for Lili in range (1):
        lil=1
        modl_n = model_Inputs_str[lil]
        obj_n = '_inf'
        data_n = '_data2'
        epoch_n = '_60epochs'
        aug_n = '_aug'
        loss_name = '_dice'
        
        name_trR ='./Models/Results' +modl_n + obj_n+ data_n  +  epoch_n + aug_n +loss_name +'_tr'+str(foldd)+'.pt'
        name_valR = './Models/Results' +modl_n + obj_n+ data_n  +  epoch_n + aug_n +loss_name +'_val'+str(foldd)+'.pt'
        name_model_final = './Models/Model' +modl_n + obj_n+ data_n  +  epoch_n + aug_n +loss_name +'_fi'+str(foldd)+'.pt'
        name_model_bestF1 = './Models/Model' +modl_n + obj_n+ data_n  + epoch_n + aug_n +loss_name +'_bt'+str(foldd)+'.pt'
        
    
    
    ############################
    
    criterion0 =nn.BCEWithLogitsLoss() 
    ############################
    
    
    def model_unet(model_input, in_channel=3, out_channel=1):
        model_test = model_input(in_channel, out_channel)
        return model_test
    
    #passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary
    
    
    model = model_unet(model_Inputs[1], 3, 1)
    
    torch.set_grad_enabled(True)
    ############################
    # Part 5  
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    validate_loader = torch.utils.data.DataLoader(valid_set, batch_size = 1, shuffle = True)  
    
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    start = time.time()
    # device = "cpu"
    model.to(device)
    
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_dise, valid_dise = [], []
    
    
    train_IoU, valid_IoU = [], []
    
    train_F1score, valid_F1score = [], []
    
    train_Spec, valid_Spec = [], []
    train_Sens, valid_Sens = [], []
    train_Prec, valid_Prec = [], []
    
    
    
    epoch_count = []
    
    best_F1score = -1
    
    segm= nn.Sigmoid()
    
    
    for epoch in range(epochs):
        epoch_count.append(epoch)
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        
        lr = 0.01
        if epoch>20:
            lr = 0.005  
        if epoch>40:
            lr = 0.001         
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_loader
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = validate_loader
    
            running_loss = 0.0
            
            num_correct = 0
            num_pixels = 0            
    
            step = 0
    
            # iterate over data
            dice_scores = 0
    
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for batch in tqdm(dataloader):
                x,y,z = batch
                x = x.to(device)
                y = y.float().to(device)
                z = z.float().to(device)
                step += 1
    
                # forward pass
                if phase == 'train':
                    # zero the gradient
                    output_0,output_1 = model(x)
                    
                    
                    # calculate the loss 
                    loss_0 = criterion0(output_0.squeeze(dim=1), y)
                    loss_1 = criterion0(output_1.squeeze(dim=1), z)

                    
                    # print(loss)
                    loss=0.3*loss_0+ 0.7*loss_1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                
                else:
                    with torch.no_grad():
                        output_0,output_1 = model(x)
    
                        # calculate the loss 
                        loss_0 = criterion0(output_0.squeeze(dim=1), y)
                        loss_1 = criterion0(output_1.squeeze(dim=1), z)
                        # loss_2 = criterion1(output_1.squeeze(dim=1), z)
    
                        # print(loss)
                        loss=0.3*loss_0+ 0.7*loss_1
                        
                running_loss += loss*dataloader.batch_size
    
                
    
                pred_1 = segm(output_1)>0.5
                pred_1= pred_1.squeeze(dim = 1).cpu().numpy().astype(int)
                zz = z >0.5
                zz = zz.squeeze(dim = 1).cpu().numpy().astype(int)
                num_correct += np.sum(pred_1== zz)
                TP += np.sum(( (pred_1 == 1).astype(int) + (zz == 1).astype(int)) == 2)
                TN += np.sum(( (pred_1 == 0).astype(int) + (zz == 0).astype(int)) == 2)
                FP += np.sum(( (pred_1 == 1).astype(int) + (zz == 0).astype(int)) == 2)
                FN += np.sum(( (pred_1 == 0).astype(int) + (zz == 1).astype(int)) == 2)
                num_pixels += pred_1.size
               
               
                 
                for idice in range(pred_1.shape[0]):
                        dice_scores += (2 * (pred_1[idice] * zz[idice]).sum()) / (
                            (pred_1[idice]  + zz[idice]).sum() + 1e-8
                            
                        ) 
                        
                    
               
                
                del x; del y; del z
    
            epoch_loss = running_loss / len(dataloader.dataset)
          
            
            epoch_acc2 = (num_correct/num_pixels)*100
            epoch_dise = dice_scores/len(dataloader.dataset)
    
    
            Spec = 1 - (FP/(FP+TN))
            Sens = TP/(TP+FN) # Recall
            Prec =  TP/(TP+FP+ 1e-8) 
            F1score = TP / (TP + ((1/2)*(FP+FN))+ 1e-8)
            IoU = TP / (TP+FP+FN)
            
            if phase == 'valid':
                if F1score > best_F1score:
                    best_F1score = F1score
                    torch.save(model.state_dict(), name_model_bestF1)
                      
            print('{} Loss: {:.4f} Acc: {:.8f} Dice: {:.8f}  IoU: {:.8f} F1: {:.8f} Spec: {:.8f} Sens: {:.8f} Prec: {:.8f}' \
                  .format(phase, epoch_loss, epoch_acc2, epoch_dise,  IoU, F1score, Spec, Sens, Prec))
            
            train_loss.append(np.array(epoch_loss.detach().cpu())) if phase=='train' \
                else valid_loss.append(np.array(epoch_loss.detach().cpu()))
            train_acc.append(np.array(epoch_acc2)) if phase=='train' \
                else valid_acc.append((np.array(epoch_acc2)))
            train_dise.append(np.array(epoch_dise)) if phase=='train' \
                else valid_dise.append((np.array(epoch_dise)))
     
                
            train_IoU.append(np.array(IoU)) if phase=='train' \
                else valid_IoU.append((np.array(IoU)))            
                
            train_F1score.append(np.array(F1score)) if phase=='train' \
                else valid_F1score.append((np.array(F1score)))
    
                
            train_Spec.append(np.array(Spec)) if phase=='train' \
                else valid_Spec.append((np.array(Spec))) 
            train_Sens.append(np.array(Sens)) if phase=='train' \
                else valid_Sens.append((np.array(Sens)))
            train_Prec.append(np.array(Prec)) if phase=='train' \
                else valid_Prec.append((np.array(Prec)))            
    
    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    
    ll = (epoch_count, train_loss, train_acc, train_dise,train_IoU, train_F1score, train_Sens, train_Spec, train_Prec) 
    lll =   (epoch_count, valid_loss, valid_acc, valid_dise, valid_IoU, valid_F1score, valid_Spec, \
              valid_Sens, valid_Prec)
        
    torch.save(ll, name_trR)
    torch.save(lll, name_valR)
    torch.save(model.state_dict(), name_model_final)
    

    
    
    


