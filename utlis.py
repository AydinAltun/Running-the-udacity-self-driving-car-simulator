# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 23:40:48 2021

@author: Aydin ALTUN
"""

import os
import matplotlib.image as mpimage#To use the images are bgr
from imgaug import augmenters as iaa
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

#Getting the file path
def getName(filePath):
    return filePath.split('\\')[-1]
 
def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = columns)
    #### REMOVE FILE PATH AND GET ONLY FILE NAME
    #print(getName(data['center'][0]))
    data['Center']=data['Center'].apply(getName)
    #print(data.head())
    print('Total Images Imported',data.shape[0])
    
    return data

'''
we have about 200 values between -1 and 1 .
 if we plot each one of them it will get very crowded. so we make categories for example 
-1 to -0.8  is one bin then -0.8 to -0.6 is another bin. 
so instead of 200 bins we will have 31 This way it is easier to visualize.

def balanceData(data,display=True):
    nBin = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data['Steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
        
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)
 
    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    

    if display:
         hist, _ = np.histogram(data['Steering'], (nBin))
         plt.bar(center, hist, width=0.06)
         plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
         plt.show()
    '''
             
def loadData(path, data):
  imagesPath = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append(f'{path}/IMG/{indexed_data[0]}')
    steering.append(float(indexed_data[3]))
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  
  return imagesPath, steering 

def augmentersImage(imgPath, steering):
    img = mpimage.imread(imgPath)
    
    if np.random.rand() < 0.5:
        
        pan = iaa.Affine(translate_percent = {"x": (-0.1, 0.1), "y":(-0.1, 0.1)})
        img = pan.augment_image(img)
        
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
        
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4 , 1.2))
        img = brightness.augment_image(img)
        
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = - steering

    return img, steering

def preProcessing(img):
    img = img [60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)#to make lane lines more visiable.
    img = cv2.GaussianBlur(img,(3,3), 0)
    img = cv2.resize(img,(200,66))
    
    img = img / 255 #Normalizing the img
    
    return img
    
def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    #trainFlag is used  if we set it true it will augment the data otherwise it will not 
    while True:
        
        imgBatch = []
        steeringBatch = []
        
        for i in range( batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            
            if trainFlag:
                img , steering = augmentersImage(imagesPath[index], steeringList[index])
            else :
                img = mpimage.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))#To make our list in form of np.array again
        

def createModel():
    #Based on Nvidia's models architecture for this simulator.
    
    model = Sequential()
    model.add(Convolution2D(24, (5,5),(2,2),input_shape = (66,200,3),activation = "elu"))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1))
    
    model.compile(Adam(lr=0.0001),loss='mse')
    
    return model
