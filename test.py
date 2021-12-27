# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 23:39:52 2021

@author: Aydin ALTUN
"""
from sklearn.model_selection import train_test_split
import time
from utlis import *

#step 1
path = 'C:/Users/DARK/Desktop/myData'
data = importDataInfo(path)

'''
#step 2
data = balanceData(data,display=False)

'''
#step 3
imagesPath, steerings = loadData(path,data)

#step 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))


#step 5
model = createModel()   
model.summary()

start = time.time()

# define our training paramters
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch = 300, epochs= 5,
          validation_data=batchGen(xVal, yVal,100,0),validation_steps=200)
#step 6
stop = time.time()
print(f"Training time: {stop - start}s")

model.save("model.h5")
print("Model is saved..")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.legend(["Traning","Validation"])
plt.ylim([0,0.1])
plt.title("Loss")
plt.xlabel("Epoch")
plt.show()


