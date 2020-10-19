# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:48:43 2020

@author: DHU
"""

from PIL import Image, ImageDraw
from gurobipy import *
import math
import numpy as np
import xlrd #excel
import sys
#quatratic 
import datetime
from random import sample
import random as rd
from numpy.linalg import det, inv, matrix_rank

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
from  sklearn.svm  import LinearSVC
from sklearn import neighbors 
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score  
from read_data import read_mnist, read_cifar_10, imagetoarray


#mnist
'''
(N, A, D, x, y, J, vax, vay) = read_mnist()
N = 5000
x = x[range(N), :]
y = y[range(N)]
y = y.astype(np.int16)
vay = vay.astype(np.int16)

filenameresult = r"...\result_mnist.txt"
filenameCV = r"...\CV_mnist.txt"
filenametime_training = r"...\time(training)_mnist.txt"
filenametime_testing = r"...\time(testing)_mnist.txt"
'''

#cifar
'''
file_test = r'.../data/cifar-10-python/test_batch'
file_train1 = r'.../data/cifar-10-python/data_batch_1'
file_train2 = r'.../data/cifar-10-python/data_batch_2'
file_train3 = r'.../data/cifar-10-python/data_batch_3'
file_train4 = r'.../data/cifar-10-python/data_batch_4'
file_train5 = r'.../data/cifar-10-python/data_batch_5'

(N, D, J, A, x, y, vax, vay) = read_cifar_10(file_test, file_train1, file_train2, file_train3, file_train4, file_train5)

x = x/255
vax = vax/255
N = 5000
x = x[range(N), :]
y = y[range(N)]
y = y.astype(np.int16)
vay = vay.astype(np.int16)

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

N = 345
A = 147
J = 6
D = 2500
x = np.r_[x1, x2, x3, x4, x5, x6]
vax = np.r_[vax1, vax2, vax3, vax4, vax5, vax6]
y = np.r_[y1, y2, y3, y4, y5, y6]
vay = np.r_[vay1, vay2, vay3, vay4, vay5, vay6]
y = y.astype(np.int16)
vay = vay.astype(np.int16)
x = x/255
vax = vax/255

filenameresult = r"...\result_MT.txt"
filenameCV = r"...\CV_MT.txt"
filenametime_training = r"...\time(training)_MT.txt"
filenametime_testing = r"...\time(testing)_MT.txt"



gamma = 0.001
alpha = 0.01
tol = 0.01
TK = 3 #changing number of K
TS = 5
TW = 5
iterations = 10
MM = sys.float_info.max


averageacc_testing = np.zeros((TK, TW))
averageacc_training = np.zeros((TK, TW))
averagetime_training = np.zeros((TK, TW))
averagetime_testing = np.zeros((TK, TW))

maximumacc_testing = np.zeros((TK, TW))
record_training_acc = np.zeros((TK, TW))
record_training_time = np.zeros((TK, TW))
record_testing_time = np.zeros((TK, TW))
record_training_num = np.zeros((TK, TW, J))
record_testing_num = np.zeros((TK, TW, J))

def initialassignment(dataSet, knn):
    
    clusterAssment = rd.sample(range(0, D), knn)
  
    if len(set(clusterAssment))==len(clusterAssment):
        not_find = False
    else:
        not_find = True
    clusterAssment = np.array(clusterAssment)  
    return clusterAssment, not_find


def L2Distance(vector1, vector2): 
    
    t = np.sum(np.square(vector1 - vector2))
    return t


def optimizeclassification(x_extraction, knn):# input x' to optimize classification using similarity function
    x1 = np.zeros((knn, D))
    m = Model('optimizeclassification')
    sigma = m.addVars(knn, D, vtype=GRB.BINARY, name='sigma')
    m.update()
    
#    m.setObjective(quicksum(sigma[k,p]*np.linalg.norm([temp1[p] for temp1 in x]-[temp2[k] for temp2 in x_extraction],ord=2)**2  for k in range(knn) for p in range(D)), GRB.MINIMIZE)
    m.setObjective(quicksum(sigma[k, p] * L2Distance(x[:, p], x_extraction[:, k]) for k in range(knn) for p in range(D)), GRB.MINIMIZE)
    m.addConstrs(
               (quicksum(sigma[k,p] for k in range(knn)) == 1 for p in range(D)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[k,p] for p in range(D)) >= 1 for k in range(knn)),"c15")
   
    m.optimize()
        
    status = m.status

    m.write('clustering.lp') 
    
    if m.status == GRB.Status.OPTIMAL:
        objective=m.objVal
        #print(' optimal objective1 is %g\n' % objective)
        
        for k in range(knn):
            for p in range(D):
                x1[k, p] = sigma[k,p].x  
    
    return x1

def optimizeclassification_kmeans(x_extraction, knn):
    
    sigma = np.zeros((knn, D))
    for p in range(D):
        minst = MM
        for k in range(knn):
            if L2Distance(x[:, p], x_extraction[:, k]) < minst:
                minst = L2Distance(x[:, p], x_extraction[:, k])
                temp = k
        sigma[temp, p] = 1

    
    return sigma    
    
    
def optimizelinearsvc(x_training_extraction, knn):#using the package
    

    lin_clf = LinearSVC()
    start_training = datetime.datetime.now()
    lin_clf.fit(x_training_extraction, y) # N*D without 1s column
    end_training = datetime.datetime.now()
    ftime_training=(end_training - start_training).total_seconds()
    
    return lin_clf.intercept_, lin_clf.coef_, ftime_training
                

def compute_margin(Y, m, theta):
    # m needs to plus one column with all one elements
    m = np.c_[np.ones(N), m]

    scores = np.dot(m, theta)
    x_correct_score = scores[range(N), Y - 1].reshape(-1, 1) #the correct score for each data sample (one column)
    
    margin = scores - x_correct_score + 1
    margin[range(N), Y - 1] = margin[range(N), Y - 1] - 1 #remove the 1s of correct class
    loss = np.sum(np.maximum(0, margin)) #loss function of MSVM

    return margin, loss

def compute_similarity(x_slice, sigma, m_slice, knn):
    
    Ms = np.ones((D, knn))
    '''
    Xs = np.transpose(x[i])
    for k in range(knn - 1):
        Xs = np.c_[Xs, np.transpose(x[i])]
    '''  
    Xs = np.ones((knn, D))
    Xs = Xs * x_slice
    Xs = np.transpose(Xs)
    Ms = Ms * m_slice
    
    similarity = np.multiply(np.transpose(sigma), Xs - Ms)
    
    return similarity

def total_similarity(X, sigma, m, knn, rho):
    
    totalsimilarity = 0

    for k in range(knn): 
        temp = np.ones((D, N)) * m[:, k]
        totalsimilarity = totalsimilarity + rho * np.sum(np.square((X - np.transpose(temp)) * sigma[k, :]))
       
    return totalsimilarity
        
def derivative(X, Y, m, theta, sigma, m0, knn):
    # derivative of loss function w.r.t. m
        
    (margin, loss) = compute_margin(Y, m, theta)
    
    sign = margin > 0
    mark = np.sum(sign, axis = 1).reshape(-1,1) # the positive margin of each sample
    
    dM = np.dot(sign, np.transpose(theta[range(1, knn + 1), :]))
    
    dM = dM - mark * np.transpose(theta[range(1, knn + 1), :])[Y - 1]
       
    # derivative of dissimilarity function w.r.t. m
   
#    for i in range(N):
#        
#        similarity = compute_similarity(X[i], sigma, m[i], knn)
#        
#        dM[i] = dM[i] + (-2) * rho * np.sum(similarity, axis = 0)         

    
    M2 = np.sum(np.transpose(sigma), axis = 0) * m
    X2 = np.dot(X, np.transpose(sigma))
    
    dM = dM + (-2) * rho * (X2 - M2)
#    dM[:, range(knn)] = dM[:, range(knn)] + temp[:, range(knn)]
    
    # derivative of regularization function w.r.t. m
    sign_pos = m - m0 > 0
    sign_neg = m - m0 < 0
    sign_equal = m - m0 == 0
    
    rand = rd.uniform(-1, 1)
    dM = dM + sign_pos - sign_neg + rand * sign_equal

    return dM, loss


def computeCost(loss, m, totalsimilarity, m0):
    
    
    cost = loss + totalsimilarity + gamma * np.sum(np.abs(m - m0))
    
    return cost
    
def gradient_descent(X, Y, theta, sigma, iterations, m0, knn, rho):
    
    m = m0 #the initial value of m is set as m0
    (dM, loss_SVM) = derivative(X, Y, m, theta, sigma, m0, knn)
    
    for i in range(iterations):
        
        m = m - alpha * dM
        
        (dM, loss_SVM) = derivative(X, Y, m, theta, sigma, m0, knn)
      
    return m



nk_list = [800, 1000, 1200] #MT
#nk_list = [200, 400, 600] #mnist
#nk_list = [100, 200, 300] #cifar
rho_list = [0.1, 0.3, 0.5, 0.7, 0.9]

for counttt in range(TS):
    
    i_str=str(counttt+1)
    
    countk = -1;
    for nk in nk_list:
        
        countk = countk + 1
        tempsigma = np.zeros((nk, D))
        
        f1=True
        while f1:
            (clusterAssment ,f1) = initialassignment(x, nk)   
          
        ww = -1
        for rho in rho_list:
            ww = ww + 1
            temp_x_extraction = np.zeros((N, nk))
    
            for k in range(nk):
                temp_x_extraction[:, k] = x[:, int(clusterAssment[k])]
      
            start2 = datetime.datetime.now()
            
            itr = 1
            loss2 = []
            x_extraction = np.zeros((N, nk))
            loss2.append(MM)
            
            while 1:
                
                if itr <= 3:
                    #given m, optimize delta and theta
                    temp_sigma = optimizeclassification(temp_x_extraction, nk)
        
                    (temp_intercept, temp_coef, time_train) = optimizelinearsvc(temp_x_extraction, nk)
                    
                    temp_theta = np.c_[temp_intercept, temp_coef] #(K+1)*J
                    
                    temp_theta = np.transpose(temp_theta)
                    
                    (margin, loss) = compute_margin(y, temp_x_extraction, temp_theta) #loss_SVM, (K+1) 
                    
                    loss_similarity = total_similarity(x, temp_sigma, temp_x_extraction, nk, rho)
                        
                    cost = computeCost(loss, temp_x_extraction, loss_similarity, temp_x_extraction)
               
                    
                    x_sigma = temp_sigma
                    x_theta = temp_theta
                    
                    loss2.append(cost)                
   
                    (temp_x_extraction) = gradient_descent(x, y, x_theta, x_sigma, iterations, temp_x_extraction, nk, rho)
                    
                    x_extraction = temp_x_extraction
    
                else:                    
                   
                    break
                
                itr=itr+1

            end2 = datetime.datetime.now()
                    
            ftime1 = (end2 - start2).total_seconds()
            
                       
            
            lin_clf = LinearSVC()

            lin_clf.fit(x_extraction, y) # N*D without 1s column
            
            a_training = accuracy_score(y, lin_clf.predict(x_extraction))
            
#============================Prediction=============================
            start_testing = datetime.datetime.now()
            vasigma = np.zeros((A, nk))
            vasumsigma = np.array(x_sigma).sum(axis = 1) #k dimension
            vasigma = np.dot(vax, np.transpose(x_sigma)) / vasumsigma  
            a_testing = accuracy_score(vay, lin_clf.predict(vasigma))
            end_testing = datetime.datetime.now()
            
            ftime_testing = (end_testing - start_testing).total_seconds()
            
            countacc = [0]*J
            
            sign = (lin_clf.predict(x_extraction) == y )
            acc = np.multiply(sign, y)
            for i in range(J):
                countacc[i] = np.sum(acc == (i + 1))
            
            countacc_test = [0]*J
            
            sign = (lin_clf.predict(vasigma) == vay )
            acc = np.multiply(sign, vay)
            for i in range(J):
                countacc_test[i] = np.sum(acc == (i + 1))
                
                
            File = open(filenametime_training, "a")
            File.write('iteration = %d, training time when k = %d, rho = %f: %f\n' % (counttt + 1, nk, rho, ftime1))
            File.close() 
            
            File = open(filenametime_testing, "a")
            File.write('iteration = %d, testing time when k = %d, rho = %f: %f\n' % (counttt + 1, nk, rho, ftime_testing))
            File.close() 
            
            File = open(filenameresult, "a")
            File.write('iteration = %d, k = %d, rho = %f, obj = %s, training accuracy = %f\n' % (counttt+1, nk, rho, loss2[-1], float(a_training)))
            File.close()
                                           
            File = open(filenameresult, "a")
            File.write('iteration = %d, k = %d, rho = %f\n' %(counttt+1, nk, rho))
            for j in range(J):
                File.write('j = %d, correct training_num = %d \n' %(j+1, int(countacc[j])))
            File.write('\n')
            File.close()
            
            File = open(filenameCV, "a")
            File.write('iteration = %d, k = %d, rho = %f, testing accuracy = %f\n' % (counttt+1, nk, rho, float(a_testing)))
            File.close()
            
            File = open(filenameCV, "a")
            File.write('iteration = %d, k = %d, rho = %f\n' %(counttt+1, nk, rho))
            for j in range(J):
                File.write('j = %d, correct testing_num = %d \n' %(j+1, int(countacc_test[j])))
            File.write('\n')
            File.close()
            


            averageacc_training[countk, ww] = averageacc_training[countk, ww] + float(a_training)
            averageacc_testing[countk, ww] = averageacc_testing[countk, ww] + float(a_testing)
            averagetime_training[countk, ww] = averagetime_training[countk, ww] + float(ftime1)
            averagetime_testing[countk, ww] = averagetime_testing[countk, ww] + float(ftime_testing)
            
            
            if maximumacc_testing[countk, ww] <= float(a_testing):
                maximumacc_testing[countk, ww] = float(a_testing)
                record_training_acc[countk, ww] = float(a_training)
                record_training_time[countk, ww] = float(ftime1)
                record_testing_time[countk, ww] = float(ftime_testing)
                record_training_num[countk, ww, :] = countacc
                record_testing_num[countk, ww, :] = countacc_test
                
                
File = open(filenameresult, "a")

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, maximum_training_acc = %s\n' % (nk, rho, record_training_acc[countk, ww]))


countk = -1;
for nk in nk_list:
    countk = countk + 1 
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_training_acc = %s\n' % (nk, rho, float(averageacc_training[countk, ww]/TS)))

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        for j in range(J):
            File.write('k = %d, rho = %f, class = %d, maximum correct number: %f\n' % (nk, rho, j+1, record_training_num[countk, ww, j]))

File.close()


File = open(filenametime_training, "a")

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_training_time : %f \n' % (nk, rho, float(averagetime_training[countk, ww]/TS)))
        
countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, training_time(maxacc): %f \n' % (nk, rho, record_training_time[countk, ww]))

File.close()        
        
File = open(filenametime_testing, "a")    

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_testing_time : %f \n' % (nk, rho, float(averagetime_testing[countk, ww]/TS)))
        

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, testing_time(maxacc): %f \n' % (nk, rho, record_testing_time[countk, ww]))

File.close() 


File = open(filenameCV, "a")

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, maximum_testing_acc: %f\n' % (nk, rho, maximumacc_testing[countk, ww]))
        

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        File.write('k = %d, rho = %f, avg_testing_acc: %f\n' % (nk, rho,float(averageacc_testing[countk, ww]/TS)))
        

countk = -1;
for nk in nk_list:
    countk = countk + 1
    ww = -1
    for rho in rho_list:
        ww = ww + 1
        for j in range(J):
            File.write('k = %d, rho = %f,class=%d, maximum correct number: %f\n' % (nk, rho, j+1, record_testing_num[countk, ww, j]))        
File.close()         
           

               
        