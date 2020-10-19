
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import datetime
import xlrd #excel
import warnings
warnings.filterwarnings("ignore")
from numpy import *
import datetime


N=290
TN=414
D=6
P=1#parts of cv

readfile=r"...\data\Real estate valuation data set.xlsx"
book = xlrd.open_workbook(readfile)

sh= book.sheet_by_name("Sheet1")
A=TN-N

filenameresult=r"...\REV\result(RF).txt"
filenameCV=r"...\REV\CV(RF).txt"
filenametime=r"...\REV\time(RF).txt"
filenametime2=r"...\REV\time2(RF).txt"


#readfile=r"...\data\winequality_white.xlsx"
#book = xlrd.open_workbook(readfile)
#
#sh= book.sheet_by_name("Sheet1")
#N=3429
#K=5
#D=11
#TN=4898
#A=TN-N
#P=1
#filenameresult=r"...\WQ\result(RF).txt"
#filenameCV=r"...\WQ\CV(RF).txt"
#filenametime=r"...\WQ\time(RF).txt"
#filenametime2=r"...\WQ\time2(RF).txt"


#readfile=r"...\data\winequality-red.xlsx"
#book = xlrd.open_workbook(readfile)
#
#sh= book.sheet_by_name("winequality-red")
#N=1120
#K=5
#D=11
#TN=1599
#A=TN-N
#P=1
#filenameresult=r"...\RQ\result(RF).txt"
#filenameCV=r"...\RQ\CV(RF).txt"
#filenametime=r"...\RQ\time(RF).txt"
#filenametime2=r"...\RQ\time2(RF).txt"
initialNE=100
NEcount=16
max_depth = 30
datay=[[] for i in range(P)]#1st dimension=parts of cv
datax=[[] for i in range(P)]
#averageloss=[[0]*NEcount for nf in range(D)]
#recordloss=[[[] for t in range(NEcount)] for nf in range(D)]
datay=[[] for i in range(P)]#1st dimension=parts of cv
datax=[[] for i in range(P)]

#recordloss=[[[] for w in range(TW)] for k in range(K)]

y=[]
x=[]
vax=[]
vay=[]

number=0
while number<=TN-1:
    while number<=N-1:
        y.append(sh.cell_value(number, D))
        dx=[]
        for j in range(D):
            dx.append(sh.cell_value(number, j))
        x.append(dx)
        number=number+1
    
    vay.append(sh.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh.cell_value(number, j))
    vax.append(dx)
    number=number+1
       
for testcount in range(1):# for each CV
    recordloss=[[0]*NEcount for nf in range(D)]
    start = datetime.datetime.now()
#    File = open(filenameresult, "a")
#    File.write('*****iteration:%d*****\n'% (testcount+1))   
#    File.close()
#    File = open(filenameCV, "a")
#    File.write('*****iteration:%d*****\n'% (testcount+1))   
#    File.close()
#    File = open(filenametime, "a")
#    File.write('*****iteration:%d*****\n'% (testcount+1))   
#    File.close()
    
    
    for nf in range(D):#change the numbers of features
        print('numbers of feature is %d' % (nf+1))
        for t in range(NEcount):
            start1 = datetime.datetime.now()
            estimator=initialNE+10*t
            print('numbers of estimator is %d' % estimator)
            File = open(filenameresult, "a")
            #File.write('*****CV=%d,total features:%d,nestimator=%d*****\n'% (testcount+1,nf+1,estimator))
            File.write('*****total features:%d,nestimator=%d*****\n'% (nf+1,estimator))
            File.close()

            regr_rf = RandomForestRegressor(n_estimators=estimator, max_depth=max_depth,max_features=nf+1,
                                            random_state=2)
            regr_rf.fit(x, y)
            
            end1 = datetime.datetime.now()
            ftime= (end1 - start1).total_seconds()
            File = open(filenametime, "a")
            File.write('*** total features=%d, nestimator=%f***\n'% (nf+1,estimator))
            File.write('time=%f\n' % ftime)
            File.close()
            
            
            
            pstart2 = datetime.datetime.now()
            y_rf = regr_rf.predict(vax)

            print(y_rf)
#            perror=mean_squared_error(vay,y_rf)#total loss of all data points
            
            perror=sum(pow(vay[i]-y_rf[i] , 2)for i in range(A)) #L1 distance
#            print(mean_squared_error(vay,y_rf))
            recordloss[nf][t]=perror/A
            pend2 = datetime.datetime.now()
            pftime= (pend2 - pstart2).total_seconds()
            File = open(filenametime, "a")
            File.write('prediction time: %f\n' % pftime)
            File.close()
            File = open(filenameCV, "a")
            #File.write('***CV=%d, total features=%d, nestimator=%f***\n'% (testcount+1,nf+1,estimator))
            File.write('*** total features=%d, nestimator=%f***\n'% (nf+1,estimator))
            File.write('total error=%s\n' % str(perror/A))
            File.close()
            
            
            File = open(filenametime2, "a")
            if t>=NEcount-1:
                File.write('%f\n' % ftime)
            else:
                File.write('%f\t' % ftime)
            File.close()
    
    File = open(filenametime2, "a")
    File.write('\n')
    File.close()    
    
    end = datetime.datetime.now()
    print(end-start)
    totaltime= (end - start).total_seconds()
    File = open(filenameCV, "a")               
    for nf in range(D):
        for t in range(NEcount):
            #averageloss[nf][t]=np.mean(recordloss[nf][t])
            #print('average loss when features=%d,nestimator=%d is %.6f'% (nf+1,initialNE+10*t,averageloss[nf][t]))
            File.write('loss when features=%d,nestimator=%d is %.6f\n'% (nf+1,initialNE+10*t,recordloss[nf][t]))
    File.close()
    
    File = open(filenametime, "a")
    #File.write('computational time when kernal=rfb,tolerance:%f,penalty=%d: %s\n'% (pow(0.1,2*ntol+1),penalty,end-start))
    #File.write('totaltime when kernal=rfb in iteration %d is %f\n'% (counttt+1,totaltime))
    #File.write('totaltime when kernal=linear in iteration %d is %f\n'% (counttt+1,totaltime))
    File.write('totaltime in iteration %d is %f\n'% (testcount+1,totaltime))
    File.close()


