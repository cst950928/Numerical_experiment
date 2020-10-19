
from gurobipy import *
import math
import numpy as np
import xlrd #excel
import sys
#quatratic 
import datetime
from random import sample
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
from sklearn.metrics import accuracy_score
from  sklearn.svm  import LinearSVC
from sklearn import neighbors 
from sklearn.datasets import make_classification

y=[]
x=[]
vay=[]
vax=[]

#SPF
N=1358
N1=110
N2=133
N3=274
N4=50
N5=39
N6=281
N7=471
TN1=158
TN2=190
TN3=391
TN4=72
TN5=55
TN6=402
TN7=673
D=27
J=7
num_classes=7
A=583
TN=1941

readfile1=r"...\data\faults.xlsx"
book1 = xlrd.open_workbook(readfile1)
sh1= book1.sheet_by_name("0")
sh2= book1.sheet_by_name("1")
sh3= book1.sheet_by_name("2")
sh4= book1.sheet_by_name("3")
sh5= book1.sheet_by_name("4")
sh6= book1.sheet_by_name("5")
sh7= book1.sheet_by_name("6")

number=0
while number<=N1-1:
    y.append(sh1.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh1.cell_value(number, j))
    x.append(dx)
    number=number+1
    
while number>=N1 and number<=TN1-1:
    vay.append(sh1.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh1.cell_value(number, j))
    vax.append(dx)
    number=number+1

    
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
while number<=N3-1:
    y.append(sh3.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh3.cell_value(number, j))
    x.append(dx)
    number=number+1
while number>=N3 and number<=TN3-1:
    vay.append(sh3.cell_value(number, D))
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
    
    
number=0
while number<=N7-1:
    y.append(sh7.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh7.cell_value(number, j))
    x.append(dx)
    number=number+1
while number>=N7 and number<=TN7-1:
    vay.append(sh7.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh7.cell_value(number, j))
    vax.append(dx)
    number=number+1
    
##WIL
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
#readfile1=r"...\data\wifi.xlsx"
#book1 = xlrd.open_workbook(readfile1)
#sh1= book1.sheet_by_name("0")
#sh2= book1.sheet_by_name("1")
#sh3= book1.sheet_by_name("2")
#sh4= book1.sheet_by_name("3")
#
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
#    
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
#
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
#
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
#    
#    
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
#
#readfile1=r"...\data\CMC.xlsx"
#book1 = xlrd.open_workbook(readfile1)
#sh1= book1.sheet_by_name("0")
#sh2= book1.sheet_by_name("1")
#sh3= book1.sheet_by_name("2")
#
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


##white wine
##N=3429
#N=3412
##N1=14
#N2=114
#N3=1020
#N4=1539
#N5=616
#N6=123
##N7=3
##TN1=20
#TN2=163
#TN3=1457
#TN4=2198
#TN5=880
#TN6=175
##TN7=5
#D=11
##J=7
#J=5
#num_classes=5
##A=1469
#A=1461
##TN=4898
#TN=4873
#
#readfile1=r"...\data\wine_white_quality.xlsx"
#book1 = xlrd.open_workbook(readfile1)
##sh1= book1.sheet_by_name("3")
#sh2= book1.sheet_by_name("4")
#sh3= book1.sheet_by_name("5")
#sh4= book1.sheet_by_name("6")
#sh5= book1.sheet_by_name("7")
#sh6= book1.sheet_by_name("8")
##sh7= book1.sheet_by_name("9")
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
    

#filenameresult=r"...\WIL\result(SVC).txt"
#filenameCV=r"...\WIL\CV(SVC).txt"
#filenametime=r"...\WIL\time(SVC).txt"

#filenameresult=r"...\Yeast\result(SVC).txt"
#filenameCV=r"...\Yeast\CV(SVC).txt"
#filenametime=r"...\Yeast\time(SVC).txt"

#filenameresult=r"...\CMC\result(SVC).txt"
#filenameCV=r"...\CMC\CV(SVC).txt"
#filenametime=r"...\CMC\time(SVC).txt"

#filenameresult=r"...\faults\result(SVC).txt"
#filenameCV=r"...\faults\CV(SVC).txt"
#filenametime=r"...\faults\time(SVC).txt"

filenameresult=r"...\white wine\result(SVC)_5.txt"
filenameCV=r"...\white wine\CV(SVC)_5.txt"
filenametime=r"...\white wine\time(SVC)_5.txt"

# for datasets except for white wine

#for i in range(N):
#    y[i]=int(y[i])
#    
#for i in range(A):
#    vay[i]=int(vay[i])
    
# for datasets white wine
#
for i in range(N):
    y[i]=int(y[i])-4 
    
for i in range(A):
    vay[i]=int(vay[i])-4




tolerance=3
Npenalty=5
recordloss=[[[] for t in range(Npenalty)] for nf in range(tolerance)]

for ntol in range(tolerance):
#    print('tolerance is %f' % pow(0.1,2*ntol+1))
    for t in range(Npenalty):
        start1 = datetime.datetime.now()
        if t<=0.5:
            penalty=1
        else:
            penalty=10*t
#        print('penalty is %d' % penalty)
    

        File = open(filenameresult, "a")

#        File.write('*****hinge: tolerance:%f, penalty=%d*****\n'% (pow(0.1,2*ntol+1), penalty))
        File.write('*****squared_hinge: tolerance:%f, penalty=%d*****\n'% (pow(0.1,2*ntol+1), penalty))
          
        File.close()
        
#        lin_clf = LinearSVC(loss = 'hinge', tol = pow(0.1,2*ntol+1), C = penalty)
        lin_clf = LinearSVC(loss = 'squared_hinge', tol = pow(0.1,2*ntol+1), C = penalty)
        lin_clf.fit(x,y)
        y_svc_train = lin_clf.predict(x)
        end1 = datetime.datetime.now()
        ftime= (end1 - start1).total_seconds()
        File = open(filenameresult, "a")

#        File.write('hinge: tolerance:%f, penalty=%d, train_accuracy = %f \n'% (pow(0.1,2*ntol+1), penalty, accuracy_score(y, y_svc_train)))
        File.write('squared_hinge: tolerance:%f, penalty=%d, train_accuracy = %f\n'% (pow(0.1,2*ntol+1), penalty, accuracy_score(y, y_svc_train)))
#          
        File.close()
        
        File = open(filenametime, "a")
#        File.write('hinge: tolerance:%f, penalty=%d, training_time = %f \n'% (pow(0.1,2*ntol+1), penalty, ftime))
        File.write('squared_hinge: tolerance:%f, penalty=%d, training_time = %f \n'% (pow(0.1,2*ntol+1), penalty, ftime))
    
        File.close()
       
        # Predict on new data
        #y_multirf = regr_multirf.predict(X_test)
        pstart2 = datetime.datetime.now()
        y_svc = lin_clf.predict(vax)
        
        
#        print ('accuracy：', accuracy_score(vay, y_svc))

        pend2 = datetime.datetime.now()
        pftime= (pend2 - pstart2).total_seconds()
        
        File = open(filenametime, "a")
#        File.write('hinge: tolerance:%f, penalty=%d, testing_time = %f \n'% (pow(0.1,2*ntol+1), penalty, pftime))
        File.write('squared_hinge: tolerance:%f, penalty=%d, testing_time = %f \n'% (pow(0.1,2*ntol+1), penalty, pftime))
        File.close()
        
        
        File = open(filenameCV, "a")
#        File.write('hinge: tolerance:%f, penalty=%d, test_accuracy = %f \n'% (pow(0.1,2*ntol+1), penalty, float(accuracy_score(vay, y_svc))))
        File.write('squared_hinge: tolerance:%f, penalty=%d, test_accuracy = %f \n'% (pow(0.1,2*ntol+1), penalty, float(accuracy_score(vay, y_svc))))
        File.close()



