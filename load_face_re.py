# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import scipy.io as sio

#读取训练数据
images1 = []
labels1 = []
images2 = []
labels2 = []

def read_train_re(path_name): 
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if "train" in full_path:
            if os.path.isdir(full_path):
                read_train_re(full_path)
            else:
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path,1) 
                    images1.append(image)
                    dirname=path_name.index("train\\")+6
                    labels1.append(int(path_name[dirname])-1)  
    return images1,labels1

def read_test_re(path_name): 
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if "test" in full_path:
            if os.path.isdir(full_path):
                read_test_re(full_path)
            else:
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path,1)   # 0为加载灰度图像，1为加载彩色图像  
                    images2.append(image)
                    dirname=path_name.index("test\\")+5
                    labels2.append(int(path_name[dirname])-1)              
    return images2,labels2
    
def load_train_re(path_name):
    images,labels = read_train_re(path_name)    
    images = np.array(images)
    print("train: ")
    print(images.shape)
    labels = np.array(labels)    
    return images,labels

def load_test_re(path_name):
    images,labels = read_test_re(path_name)    
    images = np.array(images)
    print("test: ")
    print(images.shape)
    labels = np.array(labels)  
    return images,labels
