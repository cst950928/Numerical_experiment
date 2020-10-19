
import pickle
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.utils import to_categorical

from gurobipy import *
import math
import numpy as np
import xlrd #excel
import sys
#quatratic 
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

y=[]
x=[]
vay=[]
vax=[]

#digital
#N=1115
#TN1=161
#TN2=162
#TN3=159
#TN4=159
#TN5=161
#TN6=159
#TN7=160
#TN8=159
#TN9=155
#TN10=159
#N1=113
#N2=113
#N3=111
#N4=111
#N5=113
#N6=111
#N7=112
#N8=111
#N9=109
#N10=111
#D=256
#pixel=16
#J=10
#num_classes=10
#A=479
#TN=1594
#
#readfile1=r"...\data\digital.xlsx"
#book1 = xlrd.open_workbook(readfile1)
#sh1= book1.sheet_by_name("0")
#sh2= book1.sheet_by_name("1")
#sh3= book1.sheet_by_name("2")
#sh4= book1.sheet_by_name("3")
#sh5= book1.sheet_by_name("4")
#sh6= book1.sheet_by_name("5")
#sh7= book1.sheet_by_name("6")
#sh8= book1.sheet_by_name("7")
#sh9= book1.sheet_by_name("8")
#sh10= book1.sheet_by_name("9")
#batch_size = 250


#number=0
#while number<=N1-1:
#    y.append(sh1.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh1.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N1 and number<=TN1-1:
#    vay.append(sh1.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh1.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1
#
##    
#number=0
#while number<=N2-1:
#    y.append(sh2.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh2.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N2 and number<=TN2-1:
#    vay.append(sh2.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh2.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1
##    
#number=0
#while number<=N3-1:
#    y.append(sh3.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh3.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N3 and number<=TN3-1:
#    vay.append(sh3.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh3.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1  
#    
#number=0
#while number<=N4-1:
#    y.append(sh4.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh4.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N4 and number<=TN4-1:
#    vay.append(sh4.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh4.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1   
#    
#    
#number=0
#while number<=N5-1:
#    y.append(sh5.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh5.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N5 and number<=TN5-1:
#    vay.append(sh5.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh5.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1 
#    
#number=0
#while number<=N6-1:
#    y.append(sh6.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh6.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N6 and number<=TN6-1:
#    vay.append(sh6.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh6.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1    
#    
#    
#number=0
#while number<=N7-1:
#    y.append(sh7.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh7.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N7 and number<=TN7-1:
#    vay.append(sh7.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh7.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1
#    
#    
#number=0
#while number<=N8-1:
#    y.append(sh8.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh8.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N8 and number<=TN8-1:
#    vay.append(sh8.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh8.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1   
#    
#    
#number=0
#while number<=N9-1:
#    y.append(sh9.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh9.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N9 and number<=TN9-1:
#    vay.append(sh9.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh9.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1    
#    
#number=0
#while number<=N10-1:
#    y.append(sh10.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh10.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    x.append(dx1)
#    number=number+1
#    
#while number>=N10 and number<=TN10-1:
#    vay.append(sh10.cell_value(number, D))
#    dx1=[]
#    for j in range(pixel):
#        dx2=[]
#        for k in range(pixel):
#            dx3=[]
#            for p in range(1):
#                dx3.append(sh10.cell_value(number, j*pixel+k))
#            dx2.append(dx3)
#        dx1.append(dx2)
#    vax.append(dx1)
#    number=number+1    
    
# optdigital
N=3823
A1=178
A2=182
A3=177
A4=183
A5=181
A6=182
A7=181
A8=179
A9=174
A10=180
N1=376
N2=389
N3=380
N4=389
N5=387
N6=376
N7=377
N8=387
N9=380
N10=382
D=64
pixel=8
J=10
num_classes=10
A=1797


readfile1=r"...\data\training.xlsx"
book1 = xlrd.open_workbook(readfile1)
sh1_training= book1.sheet_by_name("0")
sh2_training= book1.sheet_by_name("1")
sh3_training= book1.sheet_by_name("2")
sh4_training= book1.sheet_by_name("3")
sh5_training= book1.sheet_by_name("4")
sh6_training= book1.sheet_by_name("5")
sh7_training= book1.sheet_by_name("6")
sh8_training= book1.sheet_by_name("7")
sh9_training= book1.sheet_by_name("8")
sh10_training= book1.sheet_by_name("9")

readfile2=r"...\data\testing.xlsx"
book2 = xlrd.open_workbook(readfile2)
sh1_testing= book2.sheet_by_name("0")
sh2_testing= book2.sheet_by_name("1")
sh3_testing= book2.sheet_by_name("2")
sh4_testing= book2.sheet_by_name("3")
sh5_testing= book2.sheet_by_name("4")
sh6_testing= book2.sheet_by_name("5")
sh7_testing= book2.sheet_by_name("6")
sh8_testing= book2.sheet_by_name("7")
sh9_testing= book2.sheet_by_name("8")
sh10_testing= book2.sheet_by_name("9")


batch_size = 250

number=0
while number<=N1-1:
    y.append(sh1_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh1_training.cell_value(number, j))
    x.append(dx)
    number=number+1
    
    
number=0
while number<=N2-1:
    y.append(sh2_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh2_training.cell_value(number, j))
    x.append(dx)
    number=number+1  

    
number=0
while number<=N3-1:
    y.append(sh3_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh3_training.cell_value(number, j))
    x.append(dx)
    number=number+1

    
number=0
while number<=N4-1:
    y.append(sh4_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh4_training.cell_value(number, j))
    x.append(dx)
    number=number+1  

    
    
number=0
while number<=N5-1:
    y.append(sh5_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh5_training.cell_value(number, j))
    x.append(dx)
    number=number+1

    
number=0
while number<=N6-1:
    y.append(sh6_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh6_training.cell_value(number, j))
    x.append(dx)
    number=number+1    

    
    
number=0
while number<=N7-1:
    y.append(sh7_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh7_training.cell_value(number, j))
    x.append(dx)
    number=number+1

number=0
while number<=N8-1:
    y.append(sh8_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh8_training.cell_value(number, j))
    x.append(dx)
    number=number+1    

    
    
number=0
while number<=N9-1:
    y.append(sh9_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh9_training.cell_value(number, j))
    x.append(dx)
    number=number+1
 
    
number=0
while number<=N10-1:
    y.append(sh10_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh10_training.cell_value(number, j))
    x.append(dx)
    number=number+1
    
    

number=0
while number<=A1-1:
    vay.append(sh1_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh1_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1
    
    
number=0
while number<=A2-1:
    vay.append(sh2_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh2_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1

number=0
while number<=A3-1:
    vay.append(sh3_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh3_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1

number=0
while number<=A4-1:
    vay.append(sh4_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh4_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1
    
number=0
while number<=A5-1:
    vay.append(sh5_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh5_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1   
    
number=0
while number<=A6-1:
    vay.append(sh6_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh6_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1  
    
number=0
while number<=A7-1:
    vay.append(sh7_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh7_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1   
    
    
number=0
while number<=A8-1:
    vay.append(sh8_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh8_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1

number=0
while number<=A9-1:
    vay.append(sh9_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh9_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1
number=0
while number<=A10-1:
    vay.append(sh10_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh10_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1  
    
x_train=[[[[0]*1 for k in range(pixel)] for j in range(pixel)] for i in range(N)]
x_test=[[[[0]*1 for k in range(pixel)] for j in range(pixel)] for i in range(A)]
for i in range(N):
    for j in range(pixel):
        for k in range(pixel):
            x_train[i][j][k][0]=x[i][j*pixel+k]
    
for i in range(A):
    for j in range(pixel):
        for k in range(pixel):
            x_test[i][j][k][0]=vax[i][j*pixel+k]    


    
##The path should be revised
filenameresult=r"...\optdigital\result(CNN).txt"
filenameCV=r"...\optdigital\CV(CNN).txt"
filenametime=r"...\optdigital\time(CNN).txt"

#filenameresult=r"...\digital\result(CNN).txt"
#filenameCV=r"...\digital\CV(CNN).txt"
#filenametime=r"...\digital\time(CNN).txt"

    
y=np.array(y)  
x_train=np.array(x_train)
vay=np.array(vay)
x_test=np.array(x_test)  
 
y = to_categorical(y)    
vay = to_categorical(vay) 

KS=2
NEP=5
TS=10
acc_train=[[[0]*TS for ep in range(NEP)] for size in range(KS)]
acc_test=[[[0]*TS for ep in range(NEP)] for size in range(KS)]
time_train=[[[0]*TS for ep in range(NEP)] for size in range(KS)]

acc_train_max=[[0]*NEP for size in range(KS)] 
acc_train_avg=[[0]*NEP for size in range(KS)] 
acc_test_max=[[0]*NEP for size in range(KS)] 
acc_test_avg=[[0]*NEP for size in range(KS)] 
time_avg=[[0]*NEP for size in range(KS)] 

for size in range(KS):
#    ks=2*(size+1)+1 # for digital
    ks=size+1 # for optdigital
    for ep in range(NEP):
        for counttt in range(TS):
            start = datetime.datetime.now()
            epochs=1+2*ep
            model = Sequential()
            model.add(Conv2D(6, kernel_size=(int(ks),int(ks)),
                             activation='relu',
                             input_shape=(pixel,pixel,1)))
            model.add(MaxPooling2D(pool_size=(2,2)))
#            model.add(AveragePooling2D(pool_size=(2,2)))
            model.add(Conv2D(16, kernel_size=(int(ks),int(ks)), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
#            model.add(AveragePooling2D(pool_size=(2,2)))
            model.add(Flatten())
#            model.add(Dense(16, activation='relu', input_dim=D))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))
            
            model.compile(loss=keras.losses.categorical_crossentropy,
#                          optimizer=keras.optimizers.Adadelta(),
#                          optimizer='rmsprop',
                          optimizer='adagrad',
                          metrics=['accuracy'])
            
            hist=model.fit(np.array(np.atleast_3d(x_train)), y,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)
            end = datetime.datetime.now()
            ftime= (end - start).total_seconds()
            time_train[size][ep][counttt]=ftime
#            print('Training loss', hist.history['loss'][epochs-1])
#            print('Training acc', hist.history['acc'][epochs-1])
            acc_train[size][ep][counttt]=float(hist.history['acc'][epochs-1])
            
            score = model.evaluate(np.array(np.atleast_3d(x_test)), vay, verbose=0)
                
#            print('Test loss:', score[0])
#            print('Test accuracy:', score[1])
            acc_test[size][ep][counttt]=score[1]
            
           
            File = open(filenameresult, "a")
#            File.write('*****optimizer=rmsprop,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,float(hist.history['acc'][epochs-1])))
            File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,float(hist.history['acc'][epochs-1])))
            File.close()
            File = open(filenameCV, "a")
#            File.write('*****optimizer=rmsprop,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,score[1]))
            File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,score[1]))
            File.close()
            File = open(filenametime, "a")
#            File.write('*****optimizer=rmsprop,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,ftime))
            File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,iteration:%d = %f*****\n'% (ks,epochs,counttt+1,ftime))
            File.close()


for size in range(KS):
#    ks=2*(size+1)+1 # for digital
    ks=size+1 # for optdigital
    for ep in range(NEP):
        epochs=1+2*ep
        acc_train_max[size][ep]=max(acc_train[size][ep])
        acc_train_avg[size][ep]=np.mean(acc_train[size][ep])
        acc_test_max[size][ep]=max(acc_test[size][ep])
        acc_test_avg[size][ep]=np.mean(acc_test[size][ep])
        time_avg[size][ep]=np.mean(time_train[size][ep])
        File = open(filenameresult, "a")
#        File.write('*****optimizer=rmsprop,kernel size:%d,epoch:%d,acc_train_avg= %f, acc_train_max= %f*****\n'% (ks,epochs,acc_train_avg[size][ep], acc_train_max[size][ep]))
        File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,acc_train_avg= %f, acc_train_max= %f*****\n'% (ks,epochs,acc_train_avg[size][ep], acc_train_max[size][ep]))
        File.close()
        File = open(filenameCV, "a")
#        File.write('*****optimizer=rmsprop,kernel size:%d,epoch:%d,acc_test_avg= %f,acc_test_max= %f*****\n'% (ks,epochs,acc_test_avg[size][ep],acc_test_max[size][ep]))
        File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d,acc_test_avg= %f, acc_test_max= %f*****\n'% (ks,epochs,acc_test_avg[size][ep],acc_test_max[size][ep]))
        File.close()
        File = open(filenametime, "a")
#        File.write('*****optimizer=rmsprop,kernel size:%d,epoch:%d,time_avg= %f*****\n'% (ks,epochs,time_avg[size][ep]))
        File.write('*****optimizer=adagrad,kernel size:%d,epoch:%d, time_avg= %f*****\n'% (ks,epochs,time_avg[size][ep]))
        File.close()
        


       
    