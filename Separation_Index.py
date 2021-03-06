# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:38:34 2021

@author: mostafa
"""



import glob
import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.applications import VGG19
from keras.engine import Model
from keras.layers import  Flatten, Dense, Input
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
import datetime
from keras.utils import np_utils

# Loading train images
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      image = image + gauss
      return image
# ============================================================================= 
# Loading training and validation sets
# =============================================================================

images_path = "/content/gdrive/My Drive/CT_COVID/"
images = glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpg")
images.sort()

X_train = []
width = 224
height = 224
for img in images:
    image = cv2.imread(img)
    image = cv2.resize(image, (width, height))
    image = image / np.max(image)
    image = noisy("gauss",image)
    image = image.astype(np.float32)
    X_train.append(image)

#--------------------------------------------------

images_path = "/content/gdrive/My Drive/CT_NonCOVIDD/"
images = glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpg")
images.sort()

for img in images:
    image = cv2.imread(img)
    image = cv2.resize(image, (width, height))
    image = image / np.max(image)
    image = noisy("gauss",image)
    image = image.astype(np.float32)
    X_train.append(image)

X_train=np.array(X_train)

Y_train=np.zeros((1,346))
Y_train1=np.ones((1,346))
Y_train=np.hstack([Y_train,Y_train1])
Y_train=Y_train.reshape(692,1)
X_train, Y_train=shuffle(X_train,Y_train)

print(X_train.shape)
print(Y_train.shape)


# ============================================================================= 
# Transfer Learning (Fitting)
# =============================================================================


base_model = VGG19(weights='imagenet', include_top=False)
Y_train=np_utils.to_categorical(Y_train)
#base_model.summary()
for layer in base_model.layers:
    layer.trainable = False

myinput = Input(shape=(224,224,3))
#x = base_model.output
base_model = base_model(myinput)
x = GlobalAveragePooling2D()(base_model)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
pred = Dense(2, activation='softmax')(x)
model = Model(inputs=myinput, outputs=pred)

for i, layer in enumerate(model.layers):
   print(i, layer.name)
#model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

start=datetime.datetime.now()
trained_model=model.fit(X_train,Y_train , batch_size=128 , epochs=100 , validation_split=0.2)
end=datetime.datetime.now()
Total_time_training=end-start

print ('Total_time_training:',Total_time_training )

history=trained_model.history


losses=history['loss']
val_losses=history['val_loss']
ac=history['accuracy']
val_ac=history['val_accuracy']

import matplotlib.pyplot as plt
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
plt.plot(val_losses)
plt.legend(['loss','val_loss'])

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(ac)
plt.plot(val_ac)
plt.legend(['acc','val_acc'])


# ============================================================================= 
# Computing of Separation Index at the Last layer of selected Pre-Trained Deep Neural Network
# =============================================================================


Q=692

X_train1=X_train[0:Q,:]

base_model = VGG19(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
myinput = Input(shape=(224,224,3))
base_model = base_model(myinput)
activation_model = Model(inputs = myinput, outputs = base_model) 
activations = activation_model.predict(X_train1) 
print(activations.shape)

Y_train5=Y_train[0:Q,:]            
M=np.zeros((Q,Q))
M=M.astype('float32')
si=0
p=np.eye(Q)*200000
X_train2=activations.reshape(Q,268203)
C=X_train2.dot(X_train2.T)
print(C)

for i in range(Q):
    for j in range(Q):
        M[i,j]=C[i,i]
D1=np.add(M,M.T)
D=np.subtract(D1,2*C)
D=np.add(D,p)                             
for i in range(Q):
    
    minp = np.argmin(D[i,:])
    if np.subtract(Y_train5[i],Y_train5[minp])==0:
        si +=1
SI=si/Q
print(SI) 
