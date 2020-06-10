import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import fnmatch

from sklearn.model_selection import train_test_split
from glob import glob



imagePatches = glob('./IDC_regular_ps50_idx5/**/*.png',recursive = True)

def multiplot():
  plt.rcParams['figure.figsize'] = (10.0, 10.0)
  plt.subplots_adjust(wspace=0, hspace=0)
  count = 0
  for i in imagePatches[0:20]:
    im = cv2.imread(i)
    im = cv2.resize(im,(50,50))
    plt.subplot(5,4,count+1)
    plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));plt.axis('off')
    count += 1
    
images_zero = '*class0.png'
images_one = '*class1.png'
class_zero = fnmatch.filter(imagePatches,images_zero)
class_one = fnmatch.filter(imagePatches,images_one)

def process_images(lower,upper):
  X = []
  Y = []

  WIDTH = 224
  HEIGHT = 224
  for img in imagePatches[lower:upper]:
    fullim = cv2.imread(img)
    X.append(cv2.resize((fullim),(WIDTH,HEIGHT),interpolation = cv2.INTER_CUBIC))

    if img in class_zero:
      Y.append(0)
    elif img in class_one:
      Y.append(1)
    else:
      return
  
  return X,Y
  
X,Y = process_images(0,5000)
df = pd.DataFrame()
df['images'] = X
df['labels'] = Y
X2=df["images"]
Y2=df["labels"]
X2=np.array(X2)
imgs0=[]
imgs1=[]
imgs0 = X2[Y2==0]
imgs1 = X2[Y2==1]

def Datainfo(a,b):
  print('Total number of images: {}'.format(len(a)))
  print('Number of IDC(-) Images: {}'.format(np.sum(b==0)))
  print('Number of IDC(+) Images: {}'.format(np.sum(b==1)))
  print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))
  print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))
  
X = np.array(X)
X = X/255.0

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
Y_trainHot = tf.keras.utils.to_categorical(Y_train,num_classes=2)
Y_testHot = tf.keras.utils.to_categorical(Y_test,num_classes=2)
