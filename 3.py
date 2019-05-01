

import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import layers
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy, categorical_crossentropy,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import os
import cv2
import sys
import argparse


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--phase', help='Description for foo argument', required=True)
parser.add_argument('--epochs', help='Description for foo argument')
args = vars(parser.parse_args())
phase=args['phase']
epochs=args['epochs']


input_layer = layers.Input(shape=(560,352,3))
x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
features = Flatten()(x)
x = Dense(512,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dense(128,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dense(64,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dense(32,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dense(32,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
output = Dense(2)(x)
network = Model(input_layer,output)

# network.summary()

network.compile(optimizer = 'RMSprop', loss = mean_squared_error, metrics = ['mape'])

if phase == 'train':
	def load_data(data_path):
	    print("Loading the data........")
	    X=[]
	    Y=[]
	    gt=[]
	    re_size=(352,560)
	    images=os.listdir(data_path)
	    for image in images:
	        img = cv2.imread(data_path+image)
	        height,width = img.shape[:2]
	        # cv2.imwrite("img1.jpg",img)
	        img = cv2.copyMakeBorder(img,0,560-height,0,352-width,cv2.BORDER_CONSTANT,value = [0,0,0])
	        X.append(img)
	        with open(data_path+'Ground_truth/'+image[:-5]+'_gt.txt') as f:
	            a=np.array([float(x) for x in f.read().split()])
	            a[0]=float(a[0])/height
	            a[1]=float(a[1])/width
	            Y.append(a)
	    print("Data Loaded successfully")
	    return X,Y


	path=input("Enter the training folder:")
	data_path = path+'/'
	X,Y = load_data(data_path)

	X=np.array(X)
	X=X.astype('float32')/255.0
	Y=np.array(Y)
	print("Total images loaded =",len(X))

	print("Splitting data into test and train set")
	split = train_test_split(X,Y,test_size=0.2, random_state=42)
	(X_train,X_test,Y_train,Y_test) = split
	print("Train Set =", len(X_train), "Test Set =",len(X_test) )

	print(X_train.shape, Y_train.shape)

	print("\n\nStarting the training")
	stats = network.fit(X_train,Y_train, epochs = int(epochs),validation_data = (X_test,Y_test), batch_size=16,verbose = 1,shuffle = True)
	print("\n\n Training completed successfully, saving the weights\n")

	network.save_weights('3_weights.h5')


if phase == 'test':
	network.load_weights('3_final_weights.h5')

	path=input("Enter the testing images folder:")
	data_path = path+'/'
	print('Predicting the test data\n')


	images=os.listdir(data_path)
	for image in images:
	    X=[]
	    img = cv2.imread(data_path+image)
	    height,width = img.shape[:2]
	    img = cv2.copyMakeBorder(img,0,560-height,0,352-width,cv2.BORDER_CONSTANT,value = [0,0,0])
	    X.append(img)
	    X=np.array(X)
	    X=X.astype('float32')/255.0
	    pred=network.predict(X, batch_size=None, verbose=0)
	    pred[0][0]=int(pred[0][0]*height)
	    pred[0][1]=int(pred[0][1]*width)
	    pred=pred.astype('int32')
	    file1 = open('3_results/'+image[:-4]+'txt',"w")
	    z=str(pred[0][0])+' '+str(pred[0][1])
	    file1.write(z) 
	    file1.close()
	print('Testing completed\n\n')
	



