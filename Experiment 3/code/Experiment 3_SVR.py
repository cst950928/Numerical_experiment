
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime
import xlrd #excel
import warnings
warnings.filterwarnings("ignore")
from numpy import *
import datetime



tolerance=3
Npenalty=5
x=[]
y=[]
vax=[]
vay=[]
recordloss=[[[] for t in range(Npenalty)] for nf in range(tolerance)]
#REV
readfile=r"...\data\Real estate valuation data set.xlsx"
book = xlrd.open_workbook(readfile)

sh= book.sheet_by_name("Sheet1")

N=290
K=5
D=6
TN=414
A=TN-N
filenameresult=r"...\REV\result(SVR).txt"
filenameCV=r"...\REV\CV(SVR).txt"
filenametime=r"...\REV\time(SVR).txt"


#WQ
#readfile=r"...\data\winequality_white.xlsx"
#book = xlrd.open_workbook(readfile)
#
#sh= book.sheet_by_name("Sheet1")
#N=3429
#K=5
#D=11
#TN=4898
#A=TN-N
#
#filenameresult=r"...\WQ\result(SVR).txt"
#filenameCV=r"...\WQ\CV(SVR).txt"
#filenametime=r"...\WQ\time(SVR).txt"

#RQ
#readfile=r"...\data\winequality-red.xlsx"
#book = xlrd.open_workbook(readfile)
#
#sh= book.sheet_by_name("winequality-red")
#N=1120
#D=11
#TN=1599
#A=TN-N
#
#filenameresult=r"...\RQ\result(SVR).txt"
#filenameCV=r"...\RQ\CV(SVR).txt"
#filenametime=r"...\RQ\time(SVR).txt"
number=0
while number<=N-1:
    y.append(sh.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh.cell_value(number, j))
    x.append(dx)
    number=number+1
    
while number>=N and number<=TN-1:
    vay.append(sh.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh.cell_value(number, j))
    vax.append(dx)
    number=number+1 


for counttt in range(1):
    
    start = datetime.datetime.now()
    File = open(filenameresult, "a")
    File.write('*****iteration:%d*****\n'% (counttt+1))   
    File.close()
    File = open(filenameCV, "a")
    File.write('*****iteration:%d*****\n'% (counttt+1))   
    File.close()
    File = open(filenametime, "a")
    File.write('*****iteration:%d*****\n'% (counttt+1))   
    File.close()
    for ntol in range(tolerance):
        print('tolerance is %f' % pow(0.1,2*ntol+1))
        for t in range(Npenalty):
            start1 = datetime.datetime.now()
            if t<=0.5:
                penalty=1
            else:
                penalty=10*t
            print('penalty is %d' % penalty)
            File = open(filenameresult, "a")
#            File.write('*****kernal=rfb,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
#            File.write('*****kernal=linear,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
            File.write('*****kernal=poly,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
              
            File.close()
            
#            clf = SVR(kernel='rbf',tol=pow(0.1,2*ntol+1), C=penalty)
#            clf = SVR(kernel='linear',tol=pow(0.1,2*ntol+1), C=penalty)
            clf = SVR(kernel='poly',tol=pow(0.1,2*ntol+1), C=penalty)
            clf.fit(x, y) 
            end1 = datetime.datetime.now()
            ftime= (end1 - start1).total_seconds()
            File = open(filenametime, "a")
#            File.write('*****kernal=rfb,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
#            File.write('*****kernal=linear,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
            File.write('*****kernal=poly,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
            File.write('time=%f\n' % ftime)
            File.close()
           
            # Predict on new data
            #y_multirf = regr_multirf.predict(X_test)
            pstart2 = datetime.datetime.now()
            y_rf = clf.predict(vax)
            #print(y_multirf)
            print(y_rf)
#            perror=mean_squared_error(vay,y_rf)#total loss of all data points
            perror=sum(pow(vay[i]-y_rf[i] , 2)for i in range(A))
#            print(mean_squared_error(vay,y_rf))
            recordloss[ntol][t].append(perror/A)
            pend2 = datetime.datetime.now()
            pftime= (pend2 - pstart2).total_seconds()
            File = open(filenametime, "a")
            File.write('prediction time: %f\n' % pftime)
            File.close()
            
            
            File = open(filenameCV, "a")
#            File.write('*****kernal=rfb,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
#            File.write('*****kernal=linear,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
            File.write('*****kernal=poly,tolerance:%f,penalty=%d*****\n'% (pow(0.1,2*ntol+1),penalty))
            File.write('total error=%s\n' % str(perror/A))
            File.close()
            
    end = datetime.datetime.now()
    print(end-start)
    totaltime= (end - start).total_seconds()
    File = open(filenametime, "a")
    #File.write('computational time when kernal=rfb,tolerance:%f,penalty=%d: %s\n'% (pow(0.1,2*ntol+1),penalty,end-start))
#    File.write('totaltime when kernal=rfb in iteration %d is %f\n'% (counttt+1,totaltime))
#    File.write('totaltime when kernal=linear in iteration %d is %f\n'% (counttt+1,totaltime))
    File.write('totaltime when kernal=poly in iteration %d is %f\n'% (counttt+1,totaltime))
    File.close()
