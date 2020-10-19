
import pickle
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from gurobipy import *
import math
import numpy as np
import xlrd #excel
import sys
import datetime
from random import sample

from sympy import *
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import time
import matplotlib.pyplot as plt
import warnings
from sklearn import linear_model
warnings.filterwarnings("ignore")
from numpy import *
import datetime
from read_data import read_mnist_onehot, read_cifar_10, imagetoarray



#mnist
'''
pixel = 28
(N, A, D, x_train, y, num_classes, x_test, vay) = read_mnist_onehot() 
N = 5000
x_train = x_train[range(N), :]
y = y[range(N), :]
x_train = x_train.reshape(-1, pixel, pixel, 1)
x_test = x_test.reshape(-1, pixel, pixel, 1)

filenameresult=r"...\result(CNN)_mnist.txt"
filenameCV=r"...\CV(CNN)_mnist.txt"
filenametime=r"...\time(CNN)_mnist.txt"
'''

#cifar
'''
pixel = 32
file_test = r'.../data/cifar-10-python/test_batch'
file_train1 = r'.../data/cifar-10-python/data_batch_1'
file_train2 = r'.../data/cifar-10-python/data_batch_2'
file_train3 = r'.../data/cifar-10-python/data_batch_3'
file_train4 = r'.../data/cifar-10-python/data_batch_4'
file_train5 = r'.../data/cifar-10-python/data_batch_5'

(N, D, num_classes, A, x_train, y, x_test, vay) = read_cifar_10(file_test, file_train1, file_train2, file_train3, file_train4, file_train5)
x_train = x_train/255
x_test = x_test/255

N = 5000
x_train = x_train[range(N), :]
y = y[range(N)]
y = y - 1
vay = vay - 1
y = to_categorical(y)    
vay = to_categorical(vay) 
x_train = x_train.reshape(-1, pixel, pixel, 1)
x_test = x_test.reshape(-1, pixel, pixel, 1)

filenameresult=r"...\result(CNN)_cifar.txt"
filenameCV=r"...\CV(CNN)_cifar.txt"
filenametime=r"...\time(CNN)_cifar.txt"
'''

#MT
filename1 = '.../data/Magnetic-tile-defect-datasets.-master/MT_Blowhole/black_2500'
filename2 = '.../data/Magnetic-tile-defect-datasets.-master/MT_Break/black_2500'
filename3 = '.../data/Magnetic-tile-defect-datasets.-master/MT_Crack/black_2500'
filename4 = '.../data/Magnetic-tile-defect-datasets.-master/MT_Fray/black_2500'
filename5 = '.../data/Magnetic-tile-defect-datasets.-master/MT_Free/black_2500'
filename6 = '.../data/Magnetic-tile-defect-datasets.-master/MT_Uneven/black_2500'

(x1, vax1) = imagetoarray(filename1, 115, 80)
y1 = np.ones(80)
vay1 = np.ones(35)
(x2, vax2) = imagetoarray(filename2, 85, 60)
y2 = np.ones(60) + 1
vay2 = np.ones(25) + 1
(x3, vax3) = imagetoarray(filename3, 57, 40)
y3 = np.ones(40) + 2
vay3 = np.ones(17) + 2
(x4, vax4) = imagetoarray(filename4, 32, 23)
y4 = np.ones(23) + 3
vay4 = np.ones(9) + 3
(x5, vax5) = imagetoarray(filename5, 100, 70)
y5 = np.ones(70) + 4
vay5 = np.ones(30) + 4
(x6, vax6) = imagetoarray(filename6, 103, 72)
y6 = np.ones(72) + 5
vay6 = np.ones(31) + 5


pixel = 50
N = 345
A = 147
D = 2500
num_classes = 6
x = np.r_[x1, x2, x3, x4, x5, x6]
vax = np.r_[vax1, vax2, vax3, vax4, vax5, vax6]
y = np.r_[y1, y2, y3, y4, y5, y6]
vay = np.r_[vay1, vay2, vay3, vay4, vay5, vay6]

y = y.astype(np.int16)
vay = vay.astype(np.int16)
x = x/255
vax = vax/255
y = y - 1
vay = vay - 1
y = to_categorical(y)    
vay = to_categorical(vay) 

x_train = x.reshape(-1, pixel, pixel, 1)
x_test = vax.reshape(-1, pixel, pixel, 1)

filenameresult=r"...\result(CNN)_MT.txt"
filenameCV=r"...\CV(CNN)_MT.txt"
filenametime=r"...\time(CNN)_MT.txt"


KS=3
NEP=3
TS=5
acc_train=[[[0]*TS for ep in range(NEP)] for size in range(KS)]
acc_test=[[[0]*TS for ep in range(NEP)] for size in range(KS)]
time_train=[[[0]*TS for ep in range(NEP)] for size in range(KS)]

acc_train_max=[[0]*NEP for size in range(KS)] 
acc_train_avg=[[0]*NEP for size in range(KS)] 
acc_test_max=[[0]*NEP for size in range(KS)] 
acc_test_avg=[[0]*NEP for size in range(KS)] 
time_avg=[[0]*NEP for size in range(KS)] 

for size in range(KS):
    for ep in range(NEP):
        for counttt in range(TS):
            ks = 4 *(size + 1)
            epochs = 20 + 10 * ep
            
            start = datetime.datetime.now()
            model = Sequential()
            model.add(Conv2D(filters = 20, kernel_size=(int(ks),int(ks)),
                             activation='relu',
                             input_shape=(pixel, pixel, 1)))
            model.add(MaxPooling2D(pool_size=(2,2), strides = 2, padding = 'same'))
    #        model.add(AveragePooling2D(pool_size=(2,2)))
            model.add(Conv2D(filters = 50, kernel_size=(int(ks),int(ks)), activation='relu', padding = 'same'))
            model.add(MaxPooling2D(pool_size=(2,2), strides = 2, padding = 'same'))
    #        model.add(AveragePooling2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(500, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))
            sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov=True)
            model.compile(loss = keras.losses.categorical_crossentropy,
                          optimizer='adagrad',
                          metrics=['accuracy'])
            
            hist = model.fit(np.array(np.atleast_3d(x_train)), y,
                      batch_size = 32,
                      epochs = epochs,
                      verbose = 1)
            end = datetime.datetime.now()
            ftime= (end - start).total_seconds()
            time_train[size][ep][counttt]=ftime

            acc_train[size][ep][counttt]=float(hist.history['acc'][epochs-1])
            
            score = model.evaluate(np.array(np.atleast_3d(x_test)), vay, verbose = 0)
                
            acc_test[size][ep][counttt]=score[1]
            
            print('Test accuracy:', score[1])
            
           
            File = open(filenameresult, "a")
            File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,float(hist.history['acc'][epochs-1])))
            File.close()
            File = open(filenameCV, "a")
            File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,score[1]))
            File.close()
            File = open(filenametime, "a")
            File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,ftime))
            File.close()


for size in range(KS):
    ks= 4 *(size + 1)
    for ep in range(NEP):
        epochs = 20 + 10 * ep
        acc_train_max[size][ep]=max(acc_train[size][ep])
        acc_train_avg[size][ep]=np.mean(acc_train[size][ep])
        acc_test_max[size][ep]=max(acc_test[size][ep])
        acc_test_avg[size][ep]=np.mean(acc_test[size][ep])
        time_avg[size][ep]=np.mean(time_train[size][ep])
        File = open(filenameresult, "a")
        File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,acc_train_avg= %f, acc_train_max= %f*****\n'% (ks,epochs,acc_train_avg[size][ep], acc_train_max[size][ep]))
        File.close()
        File = open(filenameCV, "a")
        File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,acc_test_avg= %f, acc_test_max= %f*****\n'% (ks,epochs,acc_test_avg[size][ep],acc_test_max[size][ep]))
        File.close()
        File = open(filenametime, "a")
        File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d, time_avg= %f*****\n'% (ks,epochs,time_avg[size][ep]))
        File.close()