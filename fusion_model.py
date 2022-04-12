
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print(tf.__version__)

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import ndimage
import time
import random
import sys
import rasterio

# !pip install pycuda

# Wriite to new raster
# requires a reference raster file 'rasterfn'

def array2raster(refereceRasterPath, outPath, inArray, nBands=6):
  with rasterio.open(refereceRasterPath) as src:
    T0 = src.transform
    CRS  = src.crs
  
  print(CRS)
  if((nBands == 1) & (len(inArray.shape) == 2)):
    inArray = inArray[np.newaxis,:,:]
    
  if(inArray.shape[0] != nBands):
    inArray = inArray.transpose(2, 0, 1)

  out_dataset = rasterio.open(outPath, 'w', driver = 'GTiff',
                              height = inArray.shape[1], width = inArray.shape[2], 
                              count = nBands, dtype = str(inArray.dtype),
                              crs = CRS,
                              transform = T0)
  out_dataset.write(inArray)
  out_dataset.close()

# read raster data with gdal
def readRaster(rasterfn, factor = 1):
    rasGdal = rasterio.open(rasterfn)
    ras = rasGdal.read()
    rasfloat = ras.astype('float32') / factor
    return rasfloat

# pairs = np.genfromtxt('./dates.csv', dtype=np.str, delimiter=',')
# pairs

pairs = np.array([['2017073', '2017153', '2017265']], dtype='<U7')


# parameters for SRCNN
# batch_size = 128
# num_classes = 10
# epochs = 2

# input image dimensions
sub_dim = 33
img_rows, img_cols = sub_dim, sub_dim
###############################################
# may need to make sub_step larger, or the RAM might be used up
sub_step = 66
# filter dimensions
f1_dim = 9
f2_dim = 5
f3_dim = 5
# number of hidden layers of filters
n_filter = 3

## y input dimension - discard these since we now use padding 
# yInputDim = sub_dim - (f1_dim + f2_dim + f3_dim) + n_filter
# yDiff = int((sub_dim - yInputDim) / 2)

def readImages(pairs = pairs, iPairs = 0):
    print(pairs[iPairs])
    days = pairs[iPairs]
    # xArray - MODIS; yArray - Landsat
    for i in range(len(days)):
        print(days[i])
        if i==0:
            ###############################################
            # change directoties
            xArray = readRaster('/datafusion/MODIS_composite_resample/mod_' + days[i]+ '_resample.tif')#[:,0:3000,0:3000]
            yArray = readRaster('/datafusion/Landsat_clip/ls_' + days[i]+ '_clip.tif')#[:,0:3000,0:3000]
            xArray = xArray[np.newaxis,:,:,:]
            yArray = yArray[np.newaxis,:,:,:]
        else:
            xTmp = readRaster('/datafusion/MODIS_composite_resample/mod_' + days[i]+ '_resample.tif')#[:,0:3000,0:3000]
            xArray = np.concatenate((xArray,xTmp[np.newaxis,:,:,:]), axis = 0)
            ###############################################
            # we do not have LS images at time 2 for most of the time
            if i==2:
                yTmp = readRaster('/datafusion/Landsat_clip/ls_' + days[i]+ '_clip.tif')#[:,0:3000,0:3000]
                yArray = np.concatenate((yArray,yTmp[np.newaxis,:,:,:]), axis = 0)

    xArray = xArray.astype('float32')
    xArray = xArray/10000
    yArray = yArray.astype('float32')
    yArray = yArray/10000
    nLayer = xArray.shape[0]

    return xArray, yArray, nLayer

def SRCNN_processing(xArray, yArray):
    # we use the 1st and 3rd images, and concatenate them together (on x-axis)
    xArray = np.concatenate((xArray[0],xArray[2]), axis = 1)
    ###############################################
    # Change yArray[2] to yArray[1]
    yArray = np.concatenate((yArray[0],yArray[1]), axis = 1)

    # How many sub-images can the original image have
    xDim1Sub = int((xArray.shape[1] - sub_dim) / sub_step) + 1
    xDim2Sub = int((xArray.shape[2] - sub_dim) / sub_step) + 1
    xNSub = xDim1Sub * xDim2Sub

    # initialize x (n of sub images, n of bands, ncols, nrows)
    x = np.zeros((xNSub, xArray.shape[0], img_cols, img_rows), dtype=np.float32)
    # fill in x
    for i in list(range(0,xDim1Sub)):
        for j in list(range(0,xDim2Sub)):
            n = i*xDim2Sub + j
            x[n] = xArray[:, i*sub_step:(i*sub_step + sub_dim), j*sub_step:(j*sub_step + sub_dim)]

    # How many sub-images can the original image have
    yDim1Sub = int((yArray.shape[1] - sub_dim) / sub_step) + 1
    yDim2Sub = int((yArray.shape[2] - sub_dim) / sub_step) + 1
    yNSub = yDim1Sub * yDim2Sub
    # initialize y 
    y = np.zeros((yNSub, yArray.shape[0], sub_dim, sub_dim), dtype=np.float32)
    # fill in y
    for i in list(range(0,yDim1Sub)):
        for j in list(range(0,yDim2Sub)):
            n = i*yDim2Sub + j
            y[n] = yArray[:, i*sub_step:(i*sub_step + sub_dim), j*sub_step:(j*sub_step + sub_dim)]

    # remove extreme values 
    x_ext_id = np.unique(np.where(x<=0)[0])
    y_ext_id = np.unique(np.where(y<=0)[0])
    ext_id = np.unique(np.concatenate((x_ext_id, y_ext_id)))

    y = np.delete(y, ext_id, 0)
    x = np.delete(x, ext_id, 0)

    trainIndex = int(0.75*x.shape[0]) # x.shape[0]: number of sub-images

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x[0:trainIndex]
        x_test = x[trainIndex:x.shape[0]]
        y_train = y[0:trainIndex]
        y_test = y[trainIndex:y.shape[0]]
        input_shape = (nLayer, img_rows, img_cols)
    else:
        x_last = np.zeros((x.shape[0], img_cols, img_rows, x.shape[1]), dtype=np.float32) # x.shape[1] : nLayer
        y_last = np.zeros((y.shape[0], sub_dim, sub_dim, y.shape[1]), dtype=np.float32) # y.shape[1] : nLayer
        for i in list(range(0, x.shape[0])):
            for j in list(range(0, x.shape[1])):
                x_last[i,:,:,j] = x[i,j,:,:]
                if j <= 1:
                    y_last[i,:,:,j] = y[i,j,:,:]        
        x_train = x_last[0:trainIndex]
        x_test = x_last[trainIndex:x.shape[0]]
        y_train = y_last[0:trainIndex]
        y_test = y_last[trainIndex:y.shape[0]]
        input_shape = (img_rows, img_cols, nLayer)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    rang = y.ptp(axis=0).ptp(axis=1).ptp(axis=1)
    mini = y.min(axis=0).min(axis=1).min(axis=1)
    y_train = (y_train - mini) / rang
    y_test = (y_test - mini) / rang
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train, x_test, y_train, y_test, rang, mini

def SRCNN():
  X_input = tf.keras.layers.Input((None, None, 6))
  X_shortcut = X_input
  X = tf.keras.layers.Conv2D(64, kernel_size=(f1_dim, f1_dim), 
                                    padding="same",
                                    activation='relu',
                                    input_shape=(None, None, 6))(X_input) # input_shape = input_shape
  X = tf.keras.layers.Dropout(0.3)(X)
  X = tf.keras.layers.Conv2D(32, (f2_dim, f2_dim), padding="same", 
                                    activation='relu')(X)
  X = tf.keras.layers.Dropout(0.3)(X)
  X = tf.keras.layers.Conv2D(6, (f3_dim, f3_dim), padding="same")(X)
  X = tf.keras.layers.Add()([X, X_shortcut])
  # Create model
  model = tf.keras.models.Model(inputs = X_input, outputs = X, name='SRCNN')
  return model

def LSTM_processing(xArraySR, yArray):
    # reshape to [samples, timestamps, variables]
    # for 2500 dimension
    d1 = xArraySR.shape[2]
    d2 = xArraySR.shape[3]
    xt1 = xArraySR[0][:,0:d1,0:d2].reshape(6, 1, d1*d2).transpose(2,1,0)
    xt2 = xArraySR[1][:,0:d1,0:d2].reshape(6, 1, d1*d2).transpose(2,1,0)
    xt3 = xArraySR[2][:,0:d1,0:d2].reshape(6, 1, d1*d2).transpose(2,1,0)

    yt1 = yArray[0][:,0:d1,0:d2].reshape(6, 1, d1*d2).transpose(2,1,0)
    ###############################################
    # we do not have yt2
    # yt2 = yArray[1][:,0:d1,0:d2].reshape(6, 1, d1*d2).transpose(2,1,0)
    yt3 = yArray[1][:,0:d1,0:d2].reshape(6, 1, d1*d2).transpose(2,1,0)

    x = np.concatenate((xt1,xt3), axis = 1)
    y = xt2.reshape(d1*d2,6)

    # remove extreme values 
    x_ext_id = np.unique(np.where(x<=-0.1)[0])
    y_ext_id = np.unique(np.where(y<=-0.1)[0])
    x_ext_id2 = np.unique(np.where(x>1)[0])
    y_ext_id2 = np.unique(np.where(y>1)[0])
    ext_id = np.unique(np.concatenate((x_ext_id, y_ext_id, x_ext_id2, y_ext_id2)))
    y = np.delete(y, ext_id, 0)
    x = np.delete(x, ext_id, 0)

    index = np.random.choice(x.shape[0], 250000, replace=False)
    trainInd = index[0:150000]
    testInd = index[150000:200000]
    valInd = index[200000:249999]

    x_train = x[trainInd]
    y_train = y[trainInd]
    x_test = x[testInd]
    y_test = y[testInd]
    # x_val = x[valInd]
    # y_val = y[valInd]
    return x_train, x_test, y_train, y_test

def LSTMModel():
  X_input = tf.keras.layers.Input((2, 6))
  X = tf.keras.layers.LSTM(200, activation='relu', return_sequences=True, input_shape=(2, 6))(X_input)
  X = tf.keras.layers.LSTM(100, activation='relu', input_shape=(2, 6))(X)
  X = tf.keras.layers.Dropout(0.5)(X)
  X = tf.keras.layers.Dense(6)(X)
  
  # Create model
  model = tf.keras.models.Model(inputs = X_input, outputs = X, name='LSTMModel')
  return model

for iPairs in range(len(pairs)):
  xArray, yArray, nLayer = readImages(pairs, iPairs)
  x_train, x_test, y_train, y_test, rang, mini = SRCNN_processing(xArray, yArray)
  SRmodel = SRCNN()
  SRmodel.compile(loss=tf.keras.losses.mean_squared_error,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['acc', 'mae'])
  print('SR model fitting for:', pairs[iPairs])
  SRmodel.fit(x_train, y_train,
            batch_size=128,
            epochs=50,
            verbose=2,
            validation_data=(x_test, y_test))

  SRmodelName = './output/' + pairs[iPairs][0] + '_' + pairs[iPairs][1] + '_' + pairs[iPairs][2]  + '_0316_SRCNN.h5'
  SRmodel.save(SRmodelName)

  # Predict SR images
  xArraySR = xArray
  for iDates in range(3):
    ###############################################
    # may need to split to two halves, or might be out of memory
    x4Pred = xArray[iDates][:,0:4000,:]
    x4Pred = x4Pred.transpose(1,2,0)
    x4Pred = x4Pred.reshape((1, x4Pred.shape[0], x4Pred.shape[1], x4Pred.shape[2]))
    pred = SRmodel.predict(x4Pred)
    pred = pred.reshape((pred.shape[1], pred.shape[2], pred.shape[3]))
    # predFianl = ((pred * rang + mini))
    predFianl = ((pred * rang + mini) + x4Pred[0])
    xArraySR[iDates][:,0:4000,:] = predFianl.transpose(2,0,1)

    x4Pred = xArray[iDates][:,4000:,:]
    x4Pred = x4Pred.transpose(1,2,0)
    x4Pred = x4Pred.reshape((1, x4Pred.shape[0], x4Pred.shape[1], x4Pred.shape[2]))
    pred = SRmodel.predict(x4Pred)
    pred = pred.reshape((pred.shape[1], pred.shape[2], pred.shape[3]))
    # predFianl = ((pred * rang + mini))
    predFianl = ((pred * rang + mini) + x4Pred[0])
    xArraySR[iDates][:,4000:,:] = predFianl.transpose(2,0,1)

  # train LSTM model
  x_train, x_test, y_train, y_test = LSTM_processing(xArraySR, yArray)
  LSTMmodel = LSTMModel()
  LSTMmodel.compile(optimizer='adam', loss='mse', metrics=['acc', 'mae'])
  print('LSTM model fitting for:', pairs[iPairs])
  LSTMmodel.fit(x_train, y_train,
                batch_size=128,
                epochs=100,
                verbose=2,
                validation_data=(x_test, y_test))
  
  LSTMmodelName = sys.os["result_folder"]+ '/' + pairs[iPairs][0] + '_' + pairs[iPairs][1] + '_' + pairs[iPairs][2] + '_0316_LSTM.h5'
  LSTMmodel.save(LSTMmodelName)
  # save the output
  PredDim = [yArray.shape[2], yArray.shape[3]]
  yt1 = yArray[0][:,0:PredDim[0],0:PredDim[1]].reshape(6, 1, PredDim[0]*PredDim[1]).transpose(2,1,0)
  ###############################################
  # change yArray[2] to yArray[1]
  yt3 = yArray[1][:,0:PredDim[0],0:PredDim[1]].reshape(6, 1, PredDim[0]*PredDim[1]).transpose(2,1,0)
  x4Pred = np.concatenate((yt1,yt3), axis = 1)

  yhat = LSTMmodel.predict(x4Pred, verbose=1)
  yPred = yhat.reshape(PredDim[0], PredDim[1], 6)

  array2raster(os.environ["result_folder"]+'/ls_2017073_clip.tif' ,
               os.environ["result_folder"]+ '/LSTM_2pairs_6bands_' + pairs[iPairs][0] + '_' + pairs[iPairs][1] + '_' + pairs[iPairs][2] + '.tif',
               (yPred*10000).astype(int))
