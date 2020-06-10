from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from VGG16 import *
from data_preprocess import *

if __name__ == "__main__":
  vgg16_arch()
  opt = Adam(lr=0.001)
  model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
  checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
  
