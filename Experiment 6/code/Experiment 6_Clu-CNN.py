
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

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

from keras.utils import to_categorical
from keras.optimizers import SGD
from read_data import read_mnist, read_cifar_10, imagetoarray
import pylab as pl
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from  sklearn.svm  import LinearSVC
from sklearn import neighbors 
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score  
from sklearn.metrics import hinge_loss
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

MM=sys.float_info.max
para=0.01
tolerance=0.01
gamma=0.001
    

    
#mnist
'''
pixel = 28
(N, A, D, x, y, num_classes, vax, vay) = read_mnist()
y = y - 1 # y = [0, 1, ..., 9]
vay = vay - 1
N = 5000
x = x[range(N), :]
y = y[range(N)] # y = [0, 1, ..., 9]
y = y.astype(np.int16)
vay = vay.astype(np.int16)
y_one_hot = to_categorical(y) #one-hot y for all samples
vay_one_hot = to_categorical(vay)
img_x = x.reshape(-1, pixel, pixel, 1)# used to compute loss in each cluster k
img_vax = vax.reshape(-1, pixel, pixel, 1)

filenameresult = r"...\Source code and data\Experiment 6\result_mnist.txt"
filenameCV = r"...\Source code and data\Experiment 6\CV_mnist.txt"
filenametime_training = r"...\Source code and data\Experiment 6\time(training)_mnist.txt"
filenametime_testing = r"...\Source code and data\Experiment 6\time(testing)_mnist.txt"
'''

#cifar
'''
file_test = r'.../data/cifar-10-python/test_batch'
file_train1 = r'.../data/cifar-10-python/data_batch_1'
file_train2 = r'.../data/cifar-10-python/data_batch_2'
file_train3 = r'.../data/cifar-10-python/data_batch_3'
file_train4 = r'.../data/cifar-10-python/data_batch_4'
file_train5 = r'.../data/cifar-10-python/data_batch_5'

pixel = 32
(N, D, num_classes, A, x, y, vax, vay) = read_cifar_10(file_test, file_train1, file_train2, file_train3, file_train4, file_train5)
y = y - 1
vay = vay - 1
x = x/255
vax = vax/255
N = 5000
x = x[range(N), :]
y = y[range(N)]

y_one_hot = to_categorical(y) #one-hot y for all samples
vay_one_hot = to_categorical(vay)
img_x = x.reshape(-1, pixel, pixel, 1)# used to compute loss in each cluster k
img_vax = vax.reshape(-1, pixel, pixel, 1)

filenameresult = r"...\result_cifar.txt"
filenameCV = r"...\CV_cifar.txt"
filenametime_training = r"...\time(training)_cifar.txt"
filenametime_testing = r"...\time(testing)_cifar.txt"

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
J = 6
num_classes = 6
D = 2500
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

y_one_hot = to_categorical(y) #one-hot y for all samples
vay_one_hot = to_categorical(vay)
img_x = x.reshape(-1, pixel, pixel, 1)# used to compute loss in each cluster k
img_vax = vax.reshape(-1, pixel, pixel, 1)

filenameresult = r"...\result_MT.txt"
filenameCV = r"...\CV_MT.txt"
filenametime_training = r"...\time(training)_MT.txt"
filenametime_testing = r"...\time(testing)_MT.txt"


def L2Distance(vector1, vector2): 
    
    t = np.sum(np.square(vector1 - vector2))
    return t

def optimize_others(sigma, knn):#CNN
    
    sum_sample = np.sum(sigma, axis = 0) #knn columns
    compute_loss = np.zeros((N, knn))# return the loss of each image in each cluster 
    ks = 4
    epochs = 20
#    epochs = 30
#    epochs = 40
    CNN_loss = 0
    m = np.zeros((knn, D))
    
    for k in range(knn):
        x_train = np.multiply(np.transpose(x + 1), sigma[:, k])#exist all zeros
        y_train = np.multiply(np.transpose(y_one_hot), sigma[:, k])
        x_train = np.transpose(x_train)
        y_train = np.transpose(y_train)
        df1 = DataFrame(x_train)
        df2 = DataFrame(y_train) 
    
        x_train = df1.ix[~(df1==0).all(axis=1), :].values  # delete the 0s row
        y_train = df2.ix[~(df2==0).all(axis=1), :].values  # delete the 0s row
        
        x_train = x_train - 1
        m[k, :] = np.sum(x_train, axis = 0)/sum_sample[k]
    
        x_train = x_train.reshape(-1, pixel, pixel, 1)
    
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
        
        hist = model.fit(np.array(np.atleast_3d(x_train)), y_train,
                  batch_size = 32,
                  epochs = epochs,
                  verbose = 1)

        CNN_loss = CNN_loss + hist.history['loss'][epochs - 1]
        
        predictions = model.predict_classes(img_x) # predict the class for all images in cluster k

        diff = 1 - (y == predictions) #true or not

        compute_loss[:, k] = diff

    return compute_loss, CNN_loss, m

    
def optimizesigma(compute_loss, ce, initialsigma, rho, knn):#update clustering
       
    s1 = datetime.datetime.now()    
    m = Model('optimizex')
    sigma = m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
    
    x1 = np.zeros((N, knn))
    objective=0
   
    m.update()    

    m.setObjective(quicksum(sigma[i, k] * compute_loss[i, k] for i in range(N) for k in range(knn))\
                   +rho * quicksum(sigma[i, k] * L2Distance(x[i, :], ce[k, :]) for i in range(N) for k in range(knn))\
                   +gamma * quicksum((1 - initialsigma[i, k])*sigma[i, k]+initialsigma[i, k]*(1 - sigma[i, k]) for k in range(knn) for i in range(N)), GRB.MINIMIZE)

    m.addConstrs(
               (quicksum(sigma[i, k] for k in range(knn)) == 1 for i in range(N)),"c2")
    
#    m.addConstrs(
#                  (quicksum(sigma[i, k] for i in range(N)) >= 1 for k in range(knn)),"c15")

    m.optimize()

    m.write('optimizex.lp')

    if m.status == GRB.Status.OPTIMAL:
        for i in range(N):
            for k in range(knn):
                x1[i, k] = sigma[i, k].x
                
    e1= datetime.datetime.now()   
                    
    return x1

def initialassignment(dataSet, knn):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples, 1)))
    not_find = False
    countt=[0]*knn


    for i in range(N):
        index = int(random.uniform(0, knn))
        clusterAssment[i] = index
        countt[index] = 1
        
    for j in range(knn):  
        if countt[j]<=0.5:
            not_find=True
            break;
        
    return clusterAssment, not_find

dataSet1 = mat(x)
TOL = 1
NP = 1
TW = 5
TS = 5
TK = 3


averageacc_testing = np.zeros((TK, TW))
averageacc_training = np.zeros((TK, TW))
averagetime_training = np.zeros((TK, TW))
averagetime_testing = np.zeros((TK, TW))

maximumacc_testing = np.zeros((TK, TW))
record_training_acc = np.zeros((TK, TW))
record_training_time = np.zeros((TK, TW))
record_testing_time = np.zeros((TK, TW))
record_training_num = np.zeros((TK, TW, num_classes))
record_testing_num = np.zeros((TK, TW, num_classes))


rho_list = [0.1, 0.3, 0.5, 0.7, 0.9]

for counttt in range(TS):
    i_str=str(counttt+1)
       
    for countk in range(TK):
        nk = (countk + 2) 
        f1 = True
        while f1:
            (clusterAssment ,f1) = initialassignment(dataSet1, nk)   
            
        ww = -1
        for rho in rho_list:
            ww = ww + 1
                    
            temp_sigma = np.zeros((N, nk))
        
            for i in range(N):
                temp_sigma[i, int(clusterAssment[i])] = 1

            start2 = datetime.datetime.now()
            itr = 1
            loss2=[]
            loss2.append(MM)       
              
            while 1:
                
                if itr <= 3:
                    (compute_loss, total_CNN_loss, temp_ce) = optimize_others(temp_sigma, nk)
                    
                    cost = total_CNN_loss + sum(temp_sigma[i, k] * L2Distance(x[i, :], temp_ce[k, :]) for i in range(N) for k in range(nk))
                
                    x_sigma = temp_sigma
                    loss2.append(cost)
                    x_ce = temp_ce
                    temp_sigma = optimizesigma(compute_loss, x_ce, x_sigma, rho, nk)
                    temp_sigma = temp_sigma.astype(int)
                    
                else:
                    break
                
                itr=itr+1
                
            end2 = datetime.datetime.now()
            ftime= (end2 - start2).total_seconds()
                      
                      
            
 #====================Assign test images into different clusters according to x=========================           
            ftime_testing = 0
            
            vam = Model("validation")  
            vasigma = np.zeros((A, nk))
            pstart = datetime.datetime.now()
            
            assign = vam.addVars(A, nk, vtype=GRB.BINARY, name='assign')
            vam.update()
                    
            vam.setObjective(quicksum(assign[i, k] * L2Distance(vax[i, :], x_ce[k, :]) for i in range(A) for k in range(nk)), GRB.MINIMIZE)
                        
            vam.addConstrs(
                    (quicksum(assign[i, k] for k in range(nk)) == 1 for i in range(A)),"c21")
            
            vam.optimize()
            pend = datetime.datetime.now()
            ptotaltime = (pend - pstart).total_seconds()
            
            ftime_testing = ftime_testing + ptotaltime
            
            status = vam.status
            if status == GRB.Status.UNBOUNDED:
                print('The model cannot be solved because it is unbounded')
                #exit(0)
            if status == GRB.Status.OPTIMAL:
                print('The optimal objective is %g' % vam.objVal)
                #exit(0)
            if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % status)
                #exit(0)
            
            vam.write('validation.lp')            
            
            if vam.status == GRB.Status.OPTIMAL:
            
                for i in range(A):
                    for k in range(nk):
                        vasigma[i, k] = assign[i, k].x                    
                    
            
            correct_num = 0
            acc_train_record1 = 0
            acc_test_record = 0
            count_testing_samples = np.sum(vasigma, axis = 0)
            count_training_samples = np.sum(x_sigma, axis = 0)
            ks = 4
            epochs = 20
#            epochs = 30
#            epochs = 40
            countacc = [0] * num_classes 
            countacc_test = [0] * num_classes
            
#===================================CNN training under the optimal sigma=============================          
            for k in range(nk):
                
                start_testing = datetime.datetime.now()
                x_train = np.multiply(np.transpose(x + 1), x_sigma[:, k])
                y_train = np.multiply(np.transpose(y_one_hot), x_sigma[:, k])
                x_test = np.multiply(np.transpose(vax + 1), vasigma[:, k])
                y_test = np.multiply(np.transpose(vay_one_hot), vasigma[:, k])
                origin_vay = np.multiply(vay + 1, vasigma[:, k]) #add 1
                origin_y = np.multiply(y + 1, x_sigma[:, k]) #add 1
                
                x_train = np.transpose(x_train)
                y_train = np.transpose(y_train)
                x_test = np.transpose(x_test)
                y_test = np.transpose(y_test)
                 
                df1 = DataFrame(x_train)
                df2 = DataFrame(y_train) 
                df3 = DataFrame(x_test)
                df4 = DataFrame(y_test) 
                df5 = DataFrame(origin_vay) 
                df6 = DataFrame(origin_y) 
                
                x_train = df1.ix[~(df1==0).all(axis=1), :].values  # delete the 0s row
                y_train = df2.ix[~(df2==0).all(axis=1), :].values  # delete the 0s row
                x_test  = df3.ix[~(df3==0).all(axis=1), :].values  # delete the 0s row
                y_test  = df4.ix[~(df4==0).all(axis=1), :].values  # delete the 0s row
                origin_vay = df5.ix[~(df5==0).all(axis=1), :].values  # delete the 0s row
                origin_y = df6.ix[~(df6==0).all(axis=1), :].values  # delete the 0s row
                
                origin_vay = origin_vay - 1 # minus 1, start from 1
                origin_y = origin_y - 1 # minus 1, start from 1
                
                x_train = x_train - 1
                x_test = x_test - 1
                x_train = x_train.reshape(-1, pixel, pixel, 1)
                x_test = x_test.reshape(-1, pixel, pixel, 1)
        
                end_testing = datetime.datetime.now() 
                
                ftime_testing = ftime_testing + (end_testing - start_testing).total_seconds()
                
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
                
                hist = model.fit(np.array(np.atleast_3d(x_train)), y_train,
                          batch_size = 32,
                          epochs = epochs,
                          verbose = 1) 

                
                predictions_y = model.predict_classes(x_train) # predict the class for images in cluster k
                
                predictions_vay = model.predict_classes(x_test) # predict the class for images in cluster k
                correct_num = correct_num + np.sum(predictions_vay == np.transpose(origin_vay))
                
                start_testing = datetime.datetime.now()
                score = model.evaluate(np.array(np.atleast_3d(x_test)), y_test, verbose = 0)
                end_testing = datetime.datetime.now() 
                
                ftime_testing = ftime_testing + (end_testing - start_testing).total_seconds()

            
                acc_test_record = acc_test_record + score[1] * count_testing_samples[k] 
                
                
                          
                sign = (predictions_y == np.transpose(origin_y))
                acc = np.multiply(sign, np.transpose(origin_y) + 1)
                for i in range(num_classes):
                    countacc[i] = countacc[i] + np.sum(acc == (i + 1))
                
                           
                sign = (predictions_vay == np.transpose(origin_vay))
                acc = np.multiply(sign, np.transpose(origin_vay) + 1)
                for i in range(num_classes):
                    countacc_test[i] = countacc_test[i] + np.sum(acc == (i + 1))
                
           
            acc_train_record = np.sum(countacc)/N
               
           
            
            File = open(filenametime_training, "a")
            File.write('iteration = %d, training time when k = %d, rho = %f: %f\n' % (counttt + 1, nk, rho, ftime))
            File.close() 
            
            File = open(filenametime_testing, "a")
            File.write('iteration = %d, testing time when k = %d, rho = %f: %f\n' % (counttt + 1, nk, rho, ftime_testing))
            File.close() 
            
            File = open(filenameresult, "a")
            File.write('iteration = %d, k = %d, rho = %f, obj = %s, training accuracy = %f\n' % (counttt + 1, nk, rho, loss2[-1], acc_train_record))
            File.close()
                                           
            File = open(filenameresult, "a")
            File.write('iteration = %d, k = %d, rho = %f\n' %(counttt + 1, nk, rho))
            for j in range(num_classes):
                File.write('j = %d, correct training_num = %d \n' %(j + 1, int(countacc[j])))
            File.write('\n')
            File.close()
            
            File = open(filenameCV, "a")
            File.write('iteration = %d, k = %d, rho = %f, testing accuracy = %f\n' % (counttt + 1, nk, rho, acc_test_record/A))
            File.close()
            
            File = open(filenameCV, "a")
            File.write('iteration = %d, k = %d, rho = %f\n' %(counttt+1, nk, rho))
            for j in range(num_classes):
                File.write('j = %d, correct testing_num = %d \n' %(j + 1, int(countacc_test[j])))
            File.write('\n')
            File.close()
            


            averageacc_training[countk, ww] = averageacc_training[countk, ww] + acc_train_record
            averageacc_testing[countk, ww] = averageacc_testing[countk, ww] + acc_test_record/A
            averagetime_training[countk, ww] = averagetime_training[countk, ww] + float(ftime)
            averagetime_testing[countk, ww] = averagetime_testing[countk, ww] + float(ftime_testing)
            
            
            if maximumacc_testing[countk, ww] <= acc_test_record/A:
                maximumacc_testing[countk, ww] = acc_test_record/A
                record_training_acc[countk, ww] = acc_train_record
                record_training_time[countk, ww] = float(ftime)
                record_testing_time[countk, ww] = float(ftime_testing)
                record_training_num[countk, ww, :] = countacc
                record_testing_num[countk, ww, :] = countacc_test



File = open(filenameresult, "a")

for countk in range(TK):

    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, maximum_training_acc = %s\n' % (nk, rho, record_training_acc[countk, ww]))

for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_training_acc = %s\n' % (nk, rho, float(averageacc_training[countk, ww]/TS)))

for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        for j in range(num_classes):
            File.write('k = %d, rho = %f, class = %d, maximum correct number: %f\n' % (nk, rho, j+1, record_training_num[countk, ww, j]))

File.close()


File = open(filenametime_training, "a")

for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_training_time : %f \n' % (nk, rho, float(averagetime_training[countk, ww]/TS)))
        
for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, training_time(maxacc): %f \n' % (nk, rho, record_training_time[countk, ww]))

File.close()        
        
File = open(filenametime_testing, "a")    

for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_testing_time : %f \n' % (nk, rho, float(averagetime_testing[countk, ww]/TS)))
        
for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, testing_time(maxacc): %f \n' % (nk, rho, record_testing_time[countk, ww]))

File.close() 


File = open(filenameCV, "a")

for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, maximum_testing_acc: %f\n' % (nk, rho, maximumacc_testing[countk, ww]))
        
        
for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_testing_acc: %f\n' % (nk, rho,float(averageacc_testing[countk, ww]/TS)))
        
for countk in range(TK):
    
    nk = (countk + 2)
    ww = -1
    for rho in rho_list: 
        ww = ww + 1
        for j in range(num_classes):
            File.write('k = %d, rho = %f,class=%d, maximum correct number: %f\n' % (nk, rho, j+1, record_testing_num[countk, ww, j]))        
File.close() 