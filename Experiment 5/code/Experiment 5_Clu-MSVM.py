
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
import pylab as pl
import pandas as pd
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

y=[]
x=[]
vay=[]
vax=[]

##SPF
#N=1358
#N1=110
#N2=133
#N3=274
#N4=50
#N5=39
#N6=281
#N7=471
#TN1=158
#TN2=190
#TN3=391
#TN4=72
#TN5=55
#TN6=402
#TN7=673
#D=27
#J=7
#num_classes=7
#A=583
#TN=1941
#K=5
#TW=5
#TS=10
#readfile1=r"...\data\faults.xlsx"
#book1 = xlrd.open_workbook(readfile1)
#sh1= book1.sheet_by_name("0")
#sh2= book1.sheet_by_name("1")
#sh3= book1.sheet_by_name("2")
#sh4= book1.sheet_by_name("3")
#sh5= book1.sheet_by_name("4")
#sh6= book1.sheet_by_name("5")
#sh7= book1.sheet_by_name("6")

#number=0
#while number<=N1-1:
#    y.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#    
#while number>=N1 and number<=TN1-1:
#    vay.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#
#    
#number=0
#while number<=N2-1:
#    y.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N2 and number<=TN2-1:
#    vay.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N3-1:
#    y.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N3 and number<=TN3-1:
#    vay.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N4-1:
#    y.append(sh4.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh4.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N4 and number<=TN4-1:
#    vay.append(sh4.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh4.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N5-1:
#    y.append(sh5.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh5.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N5 and number<=TN5-1:
#    vay.append(sh5.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh5.cell_value(number, j))
#    vax.append(dx)
#    number=number+1  
#    
#    
#number=0
#while number<=N6-1:
#    y.append(sh6.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh6.cell_value(number, j))
#    x.append(dx)
#    number=number+1    
#while number>=N6 and number<=TN6-1:
#    vay.append(sh6.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh6.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#    
#number=0
#while number<=N7-1:
#    y.append(sh7.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh7.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N7 and number<=TN7-1:
#    vay.append(sh7.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh7.cell_value(number, j))
#    vax.append(dx)
#    number=number+1


##WIL
#
#N=1400
#N1=350
#N2=350
#N3=350
#N4=350
#TN1=500
#TN2=500
#TN3=500
#TN4=500
#D=7
#J=4
#num_classes=4
#A=600
#TN=2000
#K=5
#TW=5
#TS=10
#readfile1=r"...\data\wifi.xlsx"
#book1 = xlrd.open_workbook(readfile1)
#sh1= book1.sheet_by_name("0")
#sh2= book1.sheet_by_name("1")
#sh3= book1.sheet_by_name("2")
#sh4= book1.sheet_by_name("3")

#number=0
#while number<=N1-1:
#    y.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#    
#while number>=N1 and number<=TN1-1:
#    vay.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#
#    
#number=0
#while number<=N2-1:
#    y.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N2 and number<=TN2-1:
#    vay.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N3-1:
#    y.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N3 and number<=TN3-1:
#    vay.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N4-1:
#    y.append(sh4.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh4.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N4 and number<=TN4-1:
#    vay.append(sh4.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh4.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
    
    
##Yeast
#N=1035
#N1=170
#N2=300
#N3=324
#N4=31
#N5=36
#N6=25
#N7=114
#N8=21
#N9=14
#TN1=243
#TN2=429
#TN3=463
#TN4=44
#TN5=51
#TN6=35
#TN7=163
#TN8=30
#TN9=20
#D=8
#J=9
#num_classes=9
#A=443
#TN=1478
#K=5
#TW=5
#TS=10
#readfile1=r"...\data\Yeast.xlsx"
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

#number=0
#while number<=N1-1:
#    y.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#    
#while number>=N1 and number<=TN1-1:
#    vay.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#
#    
#number=0
#while number<=N2-1:
#    y.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N2 and number<=TN2-1:
#    vay.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N3-1:
#    y.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N3 and number<=TN3-1:
#    vay.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N4-1:
#    y.append(sh4.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh4.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N4 and number<=TN4-1:
#    vay.append(sh4.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh4.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N5-1:
#    y.append(sh5.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh5.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N5 and number<=TN5-1:
#    vay.append(sh5.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh5.cell_value(number, j))
#    vax.append(dx)
#    number=number+1  
#    
#    
#number=0
#while number<=N6-1:
#    y.append(sh6.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh6.cell_value(number, j))
#    x.append(dx)
#    number=number+1    
#while number>=N6 and number<=TN6-1:
#    vay.append(sh6.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh6.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#    
#number=0
#while number<=N7-1:
#    y.append(sh7.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh7.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N7 and number<=TN7-1:
#    vay.append(sh7.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh7.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#
#number=0
#while number<=N8-1:
#    y.append(sh8.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh8.cell_value(number, j))
#    x.append(dx)
#    number=number+1    
#while number>=N8 and number<=TN8-1:
#    vay.append(sh8.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh8.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#    
#number=0
#while number<=N9-1:
#    y.append(sh9.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh9.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N9 and number<=TN9-1:
#    vay.append(sh9.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh9.cell_value(number, j))
#    vax.append(dx)
#    number=number+1


##CMC
#N=1031
#N1=440
#N2=233
#N3=358
#TN1=628
#TN2=333
#TN3=511
#D=9
#J=3
#num_classes=3
#A=441
#TN=1472
#K=5
#TW=5
#TS=10
#readfile1=r"...\data\CMC.xlsx"
#book1 = xlrd.open_workbook(readfile1)
#sh1= book1.sheet_by_name("0")
#sh2= book1.sheet_by_name("1")
#sh3= book1.sheet_by_name("2")

#number=0
#while number<=N1-1:
#    y.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#    
#while number>=N1 and number<=TN1-1:
#    vay.append(sh1.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh1.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#
#    
#number=0
#while number<=N2-1:
#    y.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N2 and number<=TN2-1:
#    vay.append(sh2.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh2.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
#    
#number=0
#while number<=N3-1:
#    y.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    x.append(dx)
#    number=number+1
#while number>=N3 and number<=TN3-1:
#    vay.append(sh3.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh3.cell_value(number, j))
#    vax.append(dx)
#    number=number+1
    
    

#white wine
#N=3429
N=3412
#N1=14
N2=114
N3=1020
N4=1539
N5=616
N6=123
#N7=3
#TN1=20
TN2=163
TN3=1457
TN4=2198
TN5=880
TN6=175
#TN7=5
D=11
#J=7
J=5
num_classes=7
#
#A=1469
A=1461
#TN=4898
TN = 4873
K=5
TW=5
TS=10
readfile1=r"...\data\wine_white_quality.xlsx"
book1 = xlrd.open_workbook(readfile1)
#sh1= book1.sheet_by_name("3")
sh2= book1.sheet_by_name("4")
sh3= book1.sheet_by_name("5")
sh4= book1.sheet_by_name("6")
sh5= book1.sheet_by_name("7")
sh6= book1.sheet_by_name("8")
#sh7= book1.sheet_by_name("9")


    
number=0
while number<=N2-1:
    y.append(sh2.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh2.cell_value(number, j))
    x.append(dx)
    number=number+1  
while number>=N2 and number<=TN2-1:
    vay.append(sh2.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh2.cell_value(number, j))
    vax.append(dx)
    number=number+1
    
number=0
#vaTT = 0
while number<=N3-1:
    y.append(sh3.cell_value(number, D))
#    y[N1+N2+number] = 0
    dx=[]
    for j in range(D):
        dx.append(sh3.cell_value(number, j))
    x.append(dx)
    number=number+1
while number>=N3 and number<=TN3-1:
    vay.append(sh3.cell_value(number, D))
#    vay[A1+A2+vaTT] = 0
#    vaTT=vaTT+1
    dx=[]
    for j in range(D):
        dx.append(sh3.cell_value(number, j))
    vax.append(dx)
    number=number+1

number=0
while number<=N4-1:
    y.append(sh4.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh4.cell_value(number, j))
    x.append(dx)
    number=number+1  
while number>=N4 and number<=TN4-1:
    vay.append(sh4.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh4.cell_value(number, j))
    vax.append(dx)
    number=number+1
    
number=0
while number<=N5-1:
    y.append(sh5.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh5.cell_value(number, j))
    x.append(dx)
    number=number+1
while number>=N5 and number<=TN5-1:
    vay.append(sh5.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh5.cell_value(number, j))
    vax.append(dx)
    number=number+1  
    
    
number=0
while number<=N6-1:
    y.append(sh6.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh6.cell_value(number, j))
    x.append(dx)
    number=number+1    
while number>=N6 and number<=TN6-1:
    vay.append(sh6.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh6.cell_value(number, j))
    vax.append(dx)
    number=number+1


#filenameresult_sum=r"...\WIL\result(algorithm).txt"
#filenameCV_sum=r"...\WIL\CV(algorithm).txt"
#filenametime_sum=r"...\WIL\time(algorithm).txt"
#filenamefeature_sum=r"...\WIL\feature(algorithm).txt"
#filenameacc_sum=r"...\WIL\accuracy(algorithm).txt"
    
#filenameresult_sum=r"...\Yeast\result(algorithm).txt"
#filenameCV_sum=r"...\Yeast\CV(algorithm).txt"
#filenametime_sum=r"...\Yeast\time(algorithm).txt"
#filenamefeature_sum=r"...\Yeast\feature(algorithm).txt"
#filenameacc_sum=r"...\Yeast\accuracy(algorithm).txt"
    
#filenameresult_sum=r"...\CMC\result(algorithm).txt"
#filenameCV_sum=r"...\CMC\CV(algorithm).txt"
#filenametime_sum=r"...\CMC\time(algorithm).txt"
#filenamefeature_sum=r"...\CMC\feature(algorithm).txt"
#filenameacc_sum=r"...\CMC\accuracy(algorithm).txt"
#
#filenameresult_sum=r"...\faults\result(algorithm).txt"
#filenameCV_sum=r"...\faults\CV(algorithm).txt"
#filenametime_sum=r"...\faults\time(algorithm).txt"
#filenamefeature_sum=r"...\faults\feature(algorithm).txt"
#filenameacc_sum=r"...\faults\accuracy(algorithm).txt"
    
filenameresult_sum=r"...\white wine\result(algorithm).txt"
filenameCV_sum=r"...\white wine\CV(algorithm).txt"
filenametime_sum=r"...\white wine\time(algorithm).txt"
filenamefeature_sum=r"...\white wine\feature(algorithm).txt"
filenameacc_sum=r"...\white wine\accuracy(algorithm).txt"
    
# for datasets except for white wine

#for i in range(N):
#    y[i]=int(y[i])
#    
#for i in range(A):
#    vay[i]=int(vay[i])
    
# for datasets white wine

for i in range(N):
    y[i]=int(y[i])-4 
    
for i in range(A):
    vay[i]=int(vay[i])-4
    
MM=sys.float_info.max
para=0.01
tolerance=0.01
gamma=0.001

def optimizelinearsvc(sigma, knn, penalty, tolerance):#using the package
    x1=[[0]*D for k in range(knn)]
    lossfunction = [0] * knn
    totalloss = 0
    beta = [0] * knn
    beta0 = [0] * knn
    x_training = [[] for k in range(knn) ]
    y_training = [[] for k in range(knn) ]
    a_training = [0] * knn
#    print(np.array(x_training).shape)
#
#    print(np.array(y_training).shape)
    for k in range(knn):
        for i in range(N):
            if sigma[i][k]>=0.5:
                x_training[k].append(x[i])
                y_training[k].append(y[i])
    

#    index = [[0]*J for k in range(knn)]
#    indicator = [0]* knn
#    for k in range(knn):
#        for j in range(J):
#            for i in range(N):
#                if y_training[k][i]==j:
#                    index[k][j] = 1
#                    break
    
    
    for k in range (knn):
          
        lin_clf = LinearSVC( C = penalty, tol = tolerance)
        lin_clf.fit(x_training[k],y_training[k])
        beta [k] = lin_clf.coef_
        beta0 [k] = lin_clf.intercept_
        a_training [k] = accuracy_score(y_training[k], lin_clf.predict(x_training[k]))
        b_training = lin_clf.decision_function(x_training[k])
        lossfunction [k] = hinge_loss(y_training[k], b_training)

   
    for k in range(knn):
        totalloss = totalloss + a_training [k]*sum(sigma[i][k] for i in range(N))
#    print ('accuracy', accuracy_score(vay, lin_clf.predict(x_testing_extraction)))
    
    for k in range(knn):
        for d in range(D):
            x1[k][d]=sum(sigma[i][k]*x[i][d] for i in range(N))/sum(sigma[i][k] for i in range(N))
    

    
            
    return a_training, beta, beta0, x1, lossfunction, totalloss/N


def optimizeothers(sigma,knn):
    x1=[[0]*D for k in range(knn)]
    beta=[]
    beta0=[]
    s1 = datetime.datetime.now()
    m=Model('optimizeothers')
    xbeta = m.addVars(knn, J, D,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")
    xbeta0 = m.addVars(knn, J,lb=-MM, vtype=GRB.CONTINUOUS, name="beta0")
    temp = m.addVars(N, knn, J,lb=0, vtype=GRB.CONTINUOUS, name="temp")
    m.update()
    
    m.setObjective(quicksum(sigma[i][k]*(quicksum(temp[i,k,j] for j in range(J))) for i in range(N)  for k in range(knn)), GRB.MINIMIZE)
     
    m.addConstrs(
                 (temp[i,k,j]>=sum(xbeta[k,j,d]*x[i][d] for d in range(D))+xbeta0[k,j]-(sum(xbeta[k,int(y[i]),d]*x[i][d] for d in range(D))+xbeta0[k,int(y[i])])+1   for i in range(N) for k in range(knn) for j in range(J)),"c1")

    m.optimize()
        
    status = m.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)

    m.write('clustering.lp')        

    for k in range(knn):
        for d in range(D):
            x1[k][d]=sum(sigma[i][k]*x[i][d] for i in range(N))/sum(sigma[i][k] for i in range(N)) 

    
    if m.status == GRB.Status.OPTIMAL:
        objective=m.objVal
        #print(' optimal objective1 is %g\n' % objective)
#        for i in range(N):
#            for k in range(knn):
#                for j in range(J):
#                    print (temp[i,k,j].x)
                   
        for k in range(knn):
            temp1=[]
            for j in range(J):
                temp2=[]
                for d in range(D):
                    temp2.append(xbeta[k,j,d].x)
                temp1.append(temp2)
            beta.append(temp1)
            
        for k in range(knn):
            temp1=[]
            for j in range(J):
                temp1.append(xbeta0[k,j].x)
            beta0.append(temp1)

    e1 = datetime.datetime.now()   
     
    return x1,beta,beta0,(e1-s1).total_seconds() 

    
def optimizesigma(ce, beta, beta0, initialsigma, weight, knn):#update clustering
       
    s1= datetime.datetime.now()    
    m=Model('optimizex')
    sigma=m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
    x1=[]
    objective=0
   
    m.update()
    

    m.setObjective(quicksum(sigma[i,k]*(sum(max(0,sum(beta[k][j][d]*x[i][d] for d in range(D))+beta0[k][j]-(sum(beta[k][y[i]][d]*x[i][d] for d in range(D))+beta0[k][y[i]])+1) for j in range(J))-1) for i in range(N) for k in range(knn))\
                   +weight*quicksum(sigma[i,k]*sum(pow(x[i][d]-ce[k][d],2) for d in range(D)) for i in range(N) for k in range(knn))\
                   +gamma*quicksum((1-initialsigma[i][k])*sigma[i,k]+initialsigma[i][k]*(1-sigma[i,k]) for k in range(knn) for i in range(N)), GRB.MINIMIZE)
    #m.setObjective(quicksum(0.25*(sigma[i,k]+a[i,k])-0.25*(sigma[i,k]-a[i,k]) for k in range(K) for i in range(N)) , GRB.MINIMIZE)
     # x5 = abs(x1)
             
    m.addConstrs(
               (quicksum(sigma[i,k] for k in range(knn)) == 1 for i in range(N)),"c2")
        
# other datasets           
#    m.addConstrs(
#                  (quicksum(sigma[i,k] for i in range(N1)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1,k] for i in range(N2)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1+N2,k] for i in range(N3)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1+N2+N3,k] for i in range(N4)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1+N2+N3+N4,k] for i in range(N5)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1+N2+N3+N4+N5,k] for i in range(N6)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1+N2+N3+N4+N5+N6,k] for i in range(N7)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1+N2+N3+N4+N5+N6+N7,k] for i in range(N8)) >= 1 for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (quicksum(sigma[i+N1+N2+N3+N4+N5+N6+N7+N8,k] for i in range(N9)) >= 1 for k in range(knn)),"c15")
# 
# white wine dataset
    m.addConstrs(
                  (quicksum(sigma[i,k] for i in range(N2)) >= 1 for k in range(knn)),"c15")
    
    m.addConstrs(
                  (quicksum(sigma[i+N2,k] for i in range(N3)) >= 1 for k in range(knn)),"c15")
    
    m.addConstrs(
                  (quicksum(sigma[i+N3+N2,k] for i in range(N4)) >= 1 for k in range(knn)),"c15")
    
    m.addConstrs(
                  (quicksum(sigma[i+N4+N2+N3,k] for i in range(N5)) >= 1 for k in range(knn)),"c15")
    
    m.addConstrs(
                  (quicksum(sigma[i+N5+N2+N3+N4,k] for i in range(N6)) >= 1 for k in range(knn)),"c15")
    m.optimize()
    status = m.status
#    if status == GRB.Status.UNBOUNDED:
#        print('The model cannot be solved because it is unbounded')
#        #exit(0)
#    if status == GRB.Status.OPTIMAL:
#        print('The optimal objective is %g' % m.objVal)
#        #exit(0)
#    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
#        print('Optimization was stopped with status %d' % status)
#        #exit(0)

    m.write('optimizex.lp')
    
      
    if m.status == GRB.Status.OPTIMAL:
        objective=m.objVal
        for i in range(N):
            temp1=[]
            for k in range(knn):
                temp1.append(sigma[i,k].x)
            x1.append(temp1)
    e1= datetime.datetime.now()                       
    return x1,(e1-s1).total_seconds()

def initialassignment(dataSet, knn):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples, 1)))
    not_find = False
    countt=[0]*knn


    for i in range(N):
        index = int(random.uniform(0, knn))
        clusterAssment[i] = index
        countt[index]=1
        
    for j in range(knn):  
        if countt[j]<=0.5:
            not_find=True
            break;
        
    return clusterAssment, not_find

dataSet1 = mat(x)
TOL = 3
NP = 5


#maximumtime=[[0]*TW for k in range(K)]
#testtime=[[0]*TW for k in range(K)]
#maximumacc=[[0]*TW for k in range(K)] 
#recordcounttt=[[0]*TW for k in range(K)]
#recordx_sigma=[[0]*TW for k in range(K)]
#recordx_beta=[[0]*TW for k in range(K)]
#recordx_beta0=[[0]*TW for k in range(K)]
#recordx_ce=[[0]*TW for k in range(K)]
#record_training_error=[[0]*TW for k in range(K)]
#record_training_error_by_cluster=[[0]*TW for k in range(K)]
#record_training_error_by_class=[[0]*TW for k in range(K)]
#record_testing_error_by_class=[[0]*TW for k in range(K)]

maximumtime = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
testtime = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
maximumacc = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
recordcounttt = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
recordx_sigma = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
recordx_beta = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
recordx_beta0 = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
recordx_ce = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
record_training_error = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
record_training_error_by_cluster = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
record_training_error_by_class = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]
record_testing_error_by_class = [[[[0]*NP for ntol in range(TOL)] for ww in range(TW)] for k in range(K)]


for counttt in range(TS):
    i_str=str(counttt+1)
    
#    filenameresult=r"...\WIL\iterations\result(algorithm)_"+i_str+'.txt'    
#    filenameCV=r"...\WIL\iterations\CV(algorithm)_"+i_str+'.txt'
#    filenametime=r"...\WIL\iterations\time(algorithm)_"+i_str+'.txt'
#    filenamefeature=r"...\WIL\iterations\feature(algorithm)_"+i_str+'.txt'
#    filenameacc=r"...\WIL\iterations\accuracy(algorithm)_"+i_str+'.txt'
#    
#    filenameresult=r"...\Yeast\iterations\result(algorithm)_"+i_str+'.txt'    
#    filenameCV=r"...\Yeast\iterations\CV(algorithm)_"+i_str+'.txt'
#    filenametime=r"...\Yeast\iterations\time(algorithm)_"+i_str+'.txt'
#    filenamefeature=r"...\Yeast\iterations\feature(algorithm)_"+i_str+'.txt'
#    filenameacc=r"...\Yeast\iterations\accuracy(algorithm)_"+i_str+'.txt'
#    
#    filenameresult=r"...\CMC\iterations\result(algorithm)_"+i_str+'.txt'    
#    filenameCV=r"...\CMC\iterations\CV(algorithm)_"+i_str+'.txt'
#    filenametime=r"...\CMC\iterations\time(algorithm)_"+i_str+'.txt'
#    filenamefeature=r"...\CMC\iterations\feature(algorithm)_"+i_str+'.txt'
#    filenameacc=r"...\CMC\iterations\accuracy(algorithm)_"+i_str+'.txt'
#    
#    filenameresult=r"...\faults\iterations\result(algorithm)_"+i_str+'.txt'    
#    filenameCV=r"...\faults\iterations\CV(algorithm)_"+i_str+'.txt'
#    filenametime=r"...\faults\iterations\time(algorithm)_"+i_str+'.txt'
#    filenamefeature=r"...\faults\iterations\feature(algorithm)_"+i_str+'.txt'
#    filenameacc=r"...\faults\iterations\accuracy(algorithm)_"+i_str+'.txt'
    
    filenameresult=r"...\white wine\iterations\result(algorithm)_"+i_str+'.txt'    
    filenameCV=r"...\white wine\iterations\CV(algorithm)_"+i_str+'.txt'
    filenametime=r"...\white wine\iterations\time(algorithm)_"+i_str+'.txt'
    filenamefeature=r"...\white wine\iterations\feature(algorithm)_"+i_str+'.txt'
    filenameacc=r"...\white wine\iterations\accuracy(algorithm)_"+i_str+'.txt'
    
    
    File = open(filenameresult, "a")
    File.write('*****iteration=%d*****\n'% (counttt+1))
    File.close()
    File = open(filenameCV, "a")
    File.write('*****iteration=%d*****\n'% (counttt+1))
    File.close()
    File = open(filenametime, "a")
    File.write('*****iteration=%d*****\n'% (counttt+1))
    File.close()
    File = open(filenamefeature, "a")
    File.write('*****iteration=%d*****\n'% (counttt+1))
    File.close()
    File = open(filenameacc, "a")
    File.write('*****iteration=%d*****\n'% (counttt+1))
    File.close()
    
    for countk in range(K):
        knn = countk
        f1=True
        while f1:
            (clusterAssment ,f1) = initialassignment(dataSet1, knn+1)   
            
        for ww in range(TW):
            weight=2*ww
            
            for ntol in range(TOL):
                para_tol = pow(0.1, 2*ntol + 1)
                
                for t in range(NP):
                    if t<=0.5:
                        para_penal=1
                    else:
                        para_penal=10*t
  
                        
                    File = open(filenamefeature, "a")
                    File.write('***** total clusters:%d,weight=%f, tolerance=%f, penalty=%d*****\n'% (knn+1,weight,para_tol,para_penal))
                    File.close()
                    
                    temp_sigma = [[0]*(knn+1) for i in range(N)]
                
                    for i in range(N):
                        temp_sigma[i][int(clusterAssment[i])]=1
        
                    
                    start2 = datetime.datetime.now()
                    itr = 1
                    loss2=[]
                    loss2.append(MM)
                    actualobj=0        
                      
                    while 1:
                        (temp_train_acc, temp_beta, temp_beta0, temp_ce, objloss, temp_train_acc_total) = optimizelinearsvc(temp_sigma, knn+1,para_penal,para_tol)
#                        (temp_ce,temp_beta,temp_beta0,time1)=optimizeothers(temp_sigma,knn+1)

     
                        calloss = sum(temp_sigma[i][k]*(sum(max(0,sum(temp_beta[k][j][d]*x[i][d] for d in range(D))+temp_beta0[k][j]-(sum(temp_beta[k][y[i]][d]*x[i][d] for d in range(D))+temp_beta0[k][y[i]])+1) for j in range(J))-1) for i in range(N) for k in range(knn+1))

                        obj2=calloss+weight*sum(temp_sigma[i][k]*sum(pow(x[i][d]-temp_ce[k][d],2) for d in range(D)) for i in range(N) for k in range(knn+1))
                      
                        if (loss2[itr-1]-obj2)/obj2>=tolerance:
                            x_sigma = temp_sigma
                            loss2.append(obj2)
                            x_ce = temp_ce
                            x_beta = temp_beta
                            x_beta0 = temp_beta0 
                            final_train_acc = temp_train_acc
                            final_train_acc_total = temp_train_acc_total
                            (temp_sigma,time2)=optimizesigma(x_ce, x_beta, x_beta0, x_sigma, weight, knn+1)
                        else:
                            break
                        itr=itr+1
                        
                    end2 = datetime.datetime.now()
                    ftime= (end2 - start2).total_seconds()
                    
                    File = open(filenametime, "a")
                    File.write('iteration:%d, computational time when k=%d,weight=%f, tolerance=%f, penalty=%d : %f\n' % (counttt+1,knn+1,weight,para_tol,para_penal,ftime))
                    File.close() 
                    
                    train_correct=0
                    score=[[0]*J for i in range(N)]
                    for i in range(N):
                        for j in range(J):
                            score[i][j]=sum(x_sigma[i][k]*(sum(x_beta[k][j][d]*x[i][d] for d in range(D))+x_beta0[k][j]) for k in range(knn+1))
                    
                    
                    predictclass=np.argmax(score,axis=1)
                    for i in range(N):
                        if predictclass[i]==y[i]:
                            train_correct=train_correct+1
                            
                    countacc=[0]*J
                    for j in range(J):
                        for i in range(N):
                            if y[i]==j:
                                if predictclass[i]==y[i]:
                                    countacc[j]=countacc[j]+1
                    
                    
                    
                    File = open(filenameacc, "a")
                    File.write('iteration=%d, K=%d,weight=%f,tolerance=%f, penalty=%d\n' % (counttt+1,knn+1,weight,para_tol,para_penal))
                    for k in range(knn+1):
                        File.write('cluster %d, training accuracy=%f\n' % (k+1,float(final_train_acc[k])))
                    
                    for j in range(J):
                        File.write('class %d, true number = %d\n' % (j+1,int(countacc[j])))
                    File.write('\n')
                    File.close()
                    
                    File = open(filenameresult, "a")
                    File.write('iteration=%d, K=%d,weight=%f,obj=%f,training accuracy=%f\n' % (counttt+1,knn+1,weight,float(loss2[-1]),float(train_correct/N)))
                    File.close()
                    
                    File = open(filenamefeature, "a")
                    for k in range(knn+1):
                        for j in range(J):
                            for d in range(D):
                                File.write('k=%d, class=%d, feature=%d :\t' % (k+1,1+j,d+1))
                                File.write('%f\n' % x_beta[k][j][d])   
                            File.write('k=%d, class=%d\t' % (k+1,1+j))
                            File.write('intercept: %f\n' % x_beta0[k][j])
                            File.write('\n')
                    File.write('\n')
                    File.close()
                            
                            
                    vam = Model("validation")  
                    perror2=0   
                    vasigma=[]  
                    pstart = datetime.datetime.now()
                    
                    assign=vam.addVars(A, int(knn+1), vtype=GRB.BINARY, name='assign')
                    vam.update()
                            
                    vam.setObjective(quicksum(assign[i,k]*sum(pow(vax[i][d]-x_ce[k][d],2) for d in range(D)) for i in range(A) for k in range(knn+1)), GRB.MINIMIZE)
                    
                    
                    vam.addConstrs(
                            (quicksum(assign[i,k] for k in range(int(knn+1))) == 1 for i in range(A)),"c21")
                    
                    vam.optimize()
                    pend = datetime.datetime.now()
                    ptotaltime= (pend - pstart).total_seconds()
                    
                    
                    status = vam.status
                    if status == GRB.Status.UNBOUNDED:
                        print('The model cannot be solved because it is unbounded')
                    if status == GRB.Status.OPTIMAL:
                        print('The optimal objective is %g' % vam.objVal)
                    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
                        print('Optimization was stopped with status %d' % status)
                    
                    vam.write('validation.lp')
                    
                    
                    if vam.status == GRB.Status.OPTIMAL:
                    
                        for i in range(A):
                            temp=[]
                            for k in range(int(knn+1)):
                                temp.append(assign[i,k].x)
                            vasigma.append(temp)
                     
                        
                        
                    score_test=[[0]*J for i in range(A)]
                    for i in range(A):
                        for j in range(J):
                            score_test[i][j]=sum(vasigma[i][k]*(sum(x_beta[k][j][d]*vax[i][d] for d in range(D))+x_beta0[k][j]) for k in range(knn+1))
                    
                    

                    
                    class_test=np.argmax(score_test,axis=1)
                        
                    correctcount=0
                    for i in range(A):
                        if vay[i]==class_test[i]:
                            correctcount=correctcount+1
                            
                    countacc_test=[0]*J
                    for j in range(J):
                        for i in range(A):
                            if vay[i]==j:
                                if class_test[i]==vay[i]:
                                    countacc_test[j]=countacc_test[j]+1       

                    File = open(filenameCV, "a")
                    File.write('iteration=%d, K=%d, weight=%f, tolerance=%f, penalty=%d, testing accuracy=%f\n' % (counttt+1,knn+1,weight,para_tol,para_penal,float(correctcount/A)))
                    File.close()
                    
                    File = open(filenameacc, "a")
                    File.write('iteration=%d, K=%d,weight=%f,tolerance=%f, penalty=%d _testing results\n' % (counttt+1,knn+1,weight,para_tol,para_penal))
        
                    for j in range(J):
                        File.write('class %d, true number = %d\n' % (j+1,int(countacc_test[j])))
                    File.write('\n')
                    File.close()
                    
                    
                    if maximumacc[countk][ww][ntol][t]<=float(correctcount/A):
                        maximumacc[countk][ww][ntol][t]=float(correctcount/A)
                        maximumtime[countk][ww][ntol][t]=ftime      
                        recordx_sigma[countk][ww][ntol][t]=x_sigma
                        recordx_beta[countk][ww][ntol][t]=x_beta
                        recordx_beta0[countk][ww][ntol][t]=x_beta0
                        recordx_ce[countk][ww][ntol][t]=x_ce
                        record_training_error[countk][ww][ntol][t]=float(train_correct/N)
                        testtime[countk][ww][ntol][t]=ptotaltime
                        record_training_error_by_cluster[countk][ww][ntol][t]=final_train_acc
                        record_training_error_by_class[countk][ww][ntol][t]=countacc
                        record_testing_error_by_class[countk][ww][ntol][t]=countacc_test

File = open(filenameresult_sum, "a")
for countk in range(K):
    knn = countk
    
    for ww in range(TW):
        weight=2*ww

        for ntol in range(TOL):
            para_tol = pow(0.1, 2*ntol + 1)
            
            for t in range(NP):
                if t<=0.5:
                        para_penal=1
                else:
                    para_penal=10*t
                File.write('K=%d,weight=%f,tolerance=%f, penalty=%d, maximum training acc=%s\n' % (knn+1,weight,para_tol,para_penal,record_training_error[countk][ww][ntol][t]))

File.close()


File = open(filenametime_sum, "a")
for countk in range(K):
    knn = countk
    
    for ww in range(TW):
        weight=2*ww
       
        for ntol in range(TOL):
            para_tol = pow(0.1, 2*ntol + 1)
            
            for t in range(NP):
                if t<=0.5:
                        para_penal=1
                else:
                    para_penal=10*t
                File.write('K=%d,weight=%f,tolerance=%f, penalty=%d,training time : %f, testing time: %f \n' % (knn+1,weight,para_tol,para_penal,maximumtime[countk][ww][ntol][t],testtime[countk][ww][ntol][t]))

File.close() 


File = open(filenameCV_sum, "a")
for countk in range(K):
    knn = countk
    
    for ww in range(TW):
        weight=2*ww
        
        for ntol in range(TOL):
            para_tol = pow(0.1, 2*ntol + 1)
            
            for t in range(NP):
                if t<=0.5:
                        para_penal=1
                else:
                    para_penal=10*t
                File.write('K=%d,weight=%f,tolerance=%f, penalty=%d,maximum test acc: %f\n' % (knn+1,weight,para_tol,para_penal,maximumacc[countk][ww][ntol][t]))
File.close()        
  
            
File = open(filenamefeature_sum, "a")
for countk in range(K):
    knn = countk
    
    for ww in range(TW):
        weight=2*ww
        
        for ntol in range(TOL):
            para_tol = pow(0.1, 2*ntol + 1)

            for t in range(NP):
                if t<=0.5:
                        para_penal=1
                else:
                    para_penal=10*t
                File.write('K=%d,weight=%f,tolerance=%f, penalty=%d \n' % (knn+1,weight,para_tol,para_penal))
                for k in range(knn+1):
                    for j in range(J):
                        for d in range(D):
                            File.write('k=%d, class=%d, feature=%d :\t' % (k+1,1+j,d+1))
                            File.write('%f\n' % recordx_beta[countk][ww][ntol][t][k][j][d])   
                        File.write('k=%d, class=%d\t' % (k+1,1+j))
                        File.write('intercept: %f\n' % recordx_beta0[countk][ww][ntol][t][k][j])
                        File.write('\n')
                    File.write('\n')
File.close() 
       

File = open(filenameacc_sum, "a")
for countk in range(K):
    knn = countk
    
    for ww in range(TW):
        weight=2*ww
      
        for ntol in range(TOL):
            para_tol = pow(0.1, 2*ntol + 1)

            for t in range(NP):
                if t<=0.5:
                        para_penal=1
                else:
                    para_penal=10*t
                File.write('K=%d,weight=%f,tolerance=%f, penalty=%d\n' % (knn+1,weight,para_tol,para_penal))
#                for k in range(knn+1):
#                    File.write('k=%d, training_acc_by_cluster = %f :\n' % (k+1,float(record_training_error_by_cluster[countk][ww][ntol][t][k])))
                for j in range(J):
                    File.write('class = %d, training_truenumber_by_class = %f :\n' % (j+1,float(record_training_error_by_class[countk][ww][ntol][t][j])))
                for j in range(J):
                    File.write('class = %d, testing_truenumber_by_class = %f :\n' % (j+1,float(record_testing_error_by_class[countk][ww][ntol][t][j]))) 
                File.write('\n')
File.close() 