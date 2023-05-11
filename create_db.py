# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:22:09 2022

@author: Marwa
"""

import torch
import cv2
import os
from sklearn.model_selection import KFold
Kf=KFold(n_splits=5,shuffle=True,random_state=42)
from sklearn.model_selection import train_test_split


database_path = r'E:\marwa\BMSeg'
images_path = 'image'
mask_b_path = 'bone mask'
mask_l_path = 'lesion mask'

indicies = list(range(len(os.listdir(os.path.join(database_path, images_path)))))
for fold,(train_idx,test_idx)in enumerate(Kf.split(indicies)):
    fold=fold+1
    print(fold)
 

################
    Training_img = []
    Validation_img = []
    
    Training_mask_b = []
    Validation_mask_b = []
    
    Training_mask_l = []
    Validation_mask_l = []
    
    model_results_path = "./model_path"
    if not os.path.exists(model_results_path):
        os.makedirs(model_results_path)
    
    ############################
    train_path = "./train_db_"+str(fold)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    
    test_path = "./test_db_"+str(fold)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    ############################
    
        
    ## split the data into train and val
    
    
    # Part 4
    images = os.listdir((os.path.join(database_path, images_path)))
    ind_tr=-1
    ind_val=-1
    ind_test=-1

    im=-1          
    for img in images:
        im+=1
        if im in train_idx:
            ind_tr+= 1
            im_path = os.path.join(os.path.join(database_path,'image',img))
            img1 = cv2.imread(im_path)
            
            im_path2 = os.path.join(database_path,'bone mask',img)
            bone_mask = cv2.imread(im_path2, cv2.IMREAD_GRAYSCALE) 
            
            im_path3 = os.path.join(database_path,'lesion mask',img)
            lesion_mask = cv2.imread(im_path3, cv2.IMREAD_GRAYSCALE) 
    
            img1=cv2.resize(img1, (224,224), interpolation = cv2.INTER_AREA)
            bone_mask=cv2.resize(bone_mask, (224,224), interpolation = cv2.INTER_AREA)
            lesion_mask=cv2.resize(lesion_mask, (224,224), interpolation = cv2.INTER_AREA)
    
            name =  os.path.join(train_path, str(ind_tr) + '.pt') 
            pt_data = (img1, bone_mask, lesion_mask)
            torch.save(pt_data, name)

        if im in test_idx:
            ind_test+= 1
            im_path = os.path.join(os.path.join(database_path,'image',img))
            img1 = cv2.imread(im_path)
            
            im_path2 = os.path.join(database_path,'bone mask',img)
            bone_mask = cv2.imread(im_path2, cv2.IMREAD_GRAYSCALE) 
            
            im_path3 = os.path.join(database_path,'lesion mask',img)
            lesion_mask = cv2.imread(im_path3, cv2.IMREAD_GRAYSCALE) 
    
            img1=cv2.resize(img1, (224,224), interpolation = cv2.INTER_AREA)
            bone_mask=cv2.resize(bone_mask, (224,224), interpolation = cv2.INTER_AREA)
            lesion_mask=cv2.resize(lesion_mask, (224,224), interpolation = cv2.INTER_AREA)
    
            name =  os.path.join(test_path, str(ind_test) + '.pt') 
            pt_data = (img1, bone_mask, lesion_mask)
            torch.save(pt_data, name)
    ############################
    # Part 5           
    x = []
    for i in range(ind_tr+1):            
        x.append(str(i))      
    torch.save(x,'train_db_idx_'+str(fold)+'.pt') 

    z= []

    for i in range(ind_test+1):            
        z.append(str(i))      
    torch.save(z,'test_db_idx_'+str(fold)+'.pt') 
    fold=fold+1