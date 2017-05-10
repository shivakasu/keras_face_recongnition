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

def read_train_pie(path_name): 
    for dir_item in os.listdir(path_name):
        if dir_item.endswith('.mat'):
            full_path = os.path.abspath(os.path.join(path_name, dir_item))
            data = sio.loadmat(full_path)
            for index in range(len(data['fea'])):
                if data['isTest'][index][0]==0:
                    images1.append(data['fea'][index].reshape(64,64))
                    labels1.append(int(data['gnd'][index][0])-1)                
    return images1,labels1

def read_test_pie(path_name): 
    for dir_item in os.listdir(path_name):
        if dir_item.endswith('.mat'):
            full_path = os.path.abspath(os.path.join(path_name, dir_item))
            data = sio.loadmat(full_path)
            for index in range(len(data['fea'])):
                if data['isTest'][index][0]==1:
                    images2.append(data['fea'][index].reshape(64,64))
                    labels2.append(int(data['gnd'][index][0])-1)                
    return images2,labels2

def read_train_orl(path_name):    
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            read_train_orl(full_path)
        else:
            if dir_item.endswith('.BMP') and not dir_item.startswith('4') and not dir_item.startswith('8'):
                image = cv2.imread(full_path,0) 
                images1.append(image)
                dirname=path_name.index("bmp2\\")+6
                labels1.append(int(path_name[dirname:])-1)                  
    return images1,labels1

def read_test_orl(path_name):    
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            read_test_orl(full_path)
        else:
            if dir_item.startswith('4') or dir_item.startswith('8'):
                image = cv2.imread(full_path,0)   # 0为加载灰度图像，1为加载彩色图像  
                images2.append(image)
                dirname=path_name.index("bmp2\\")+6
                labels2.append(int(path_name[dirname:])-1)              
    return images2,labels2
    
def load_train_pie(path_name):
    images,labels = read_train_pie(path_name)    
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    images = np.array(images)
    print("train: ")
    print(images.shape)
    labels = np.array(labels)    
    return images,labels

def load_test_pie(path_name):
    images,labels = read_test_pie(path_name)
    images = np.array(images)
    print("test: ")
    print(images.shape)
    labels = np.array(labels)  
    return images,labels


def load_train_orl(path_name):
    images,labels = read_train_orl(path_name)    
    images = np.array(images)
    print("train: ")
    print(images.shape)
    labels = np.array(labels)    
    return images,labels

def load_test_orl(path_name):
    images,labels = read_test_orl(path_name)    
    images = np.array(images)
    print("test: ")
    print(images.shape)
    labels = np.array(labels)  
    return images,labels

if __name__ == '__main__':
        print(12)