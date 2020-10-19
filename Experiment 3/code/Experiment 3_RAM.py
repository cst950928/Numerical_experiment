

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


##white wine
#N=3429
#K=5
#D=11
#TN=4898
#A=TN-N
#TW=3
#readfile=r"...\data\winequality_white.xlsx"
#book = xlrd.open_workbook(readfile)
#sh= book.sheet_by_name("Sheet1")

##red wine
#N=1120
#K=5
#D=11
#TN=1599
#A=TN-N
#TW=3
#readfile=r"...\data\winequality-red.xlsx"
#book = xlrd.open_workbook(readfile)
#sh= book.sheet_by_name("winequality-red")

#real estate
N=290
K=5
D=6
TN=414
A=TN-N
TW=3
readfile=r"...\data\Real estate valuation data set.xlsx"
book = xlrd.open_workbook(readfile)
sh= book.sheet_by_name("Sheet1")


y=[]
x=[]
vay=[]
vax=[]



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




MM=sys.float_info.max
para=0.01
tolerance=0.01
gamma=0.001


absdimax = []
dimax = []
dimin = []
extrax= []
MM1 = math.sqrt(sum(y[i]**2 for i in range(N))/para)
for j in range(D):
    for i in range(N):
        extrax.append(x[i][j])
        #print(extrax)
        abslist=map(abs, extrax)
        #print(abslist)
    absdimax.append(max(abslist))
    dimax.append(max(extrax))
    dimin.append(min(extrax))
    extrax=[]


filenameCVprint=r"...\REV\CV loss.txt"
filenameresultprint=r"...\REV\result(excel).txt" 

#filenameCVprint=r"...\RQ\CV loss.txt"
#filenameresultprint=r"...\RQ\result(excel).txt" 
#
#filenameCVprint=r"...\WQ\CV loss.txt"
#filenameresultprint=r"...\WQ\result(excel).txt" 


def assignment(ce,beta,beta0,weight,var,x0,y0):
    
    totalobj=math.log(var)+pow(y0-sum(beta[j]*x0[j] for j in range(D))-beta0,2)/(2*pow(var,2))+weight*L2Distance(x0, np.mat(ce))
    return totalobj
def optimizeothers(sigma,weight,knn):
#    s1 = datetime.datetime.now()    
    X=[[] for k in range(knn)]
    XT=[[] for k in range(knn)]
    Y=[[] for k in range(knn)]
    x1=[[0] for k in range(knn)]
    x2=[[0]*D for k in range(knn)]
    x3=[[0] for k in range(knn)]
    
    for k in range(knn):
        for i in range(N):
            if sigma[i][k]>=0.5:
                X[k].append(x[i])
                Y[k].append(y[i])
                
    for k in range(knn):
        XT[k]=np.mat(X[k]).T 
        X[k]=np.mat(X[k])
        Y[k]=np.mat(Y[k]).reshape((len(Y[k]),1))

#        B=np.dot(np.dot(np.dot(XT[k],X[k]).I,XT[k]),Y[k])
        B=np.dot(np.dot(np.linalg.inv(np.dot(XT[k],X[k])),XT[k]),Y[k])
        
        x1[k]=np.array(B)
        x3[k]=(sum(Y[k])-sum(np.dot(X[k],B)))/len(Y[k])
        
        for k in range(knn):
            for j in range(D):
                x2[k][j]=sum(sigma[i][k]*x[i][j] for i in range(N))/sum(sigma[i][k] for i in range(N))    
    
#    e1 = datetime.datetime.now()   
     
#    return x1,x2,x3,(e1-s1).total_seconds() 
    return x1,x2,x3

def optimizesigmanew(ce,beta,beta0,weight,knn,variance):
    
    sigma=[[0]*knn for i in range(A)]
    distance=[[0]*knn for i in range(A)]
    for i in range(A):
        minDist  = 100000.0
        minIndex = 0
        for k in range(knn):
            distance[i][k] = assignment(ce[k],beta[k],beta0[k],weight,variance[k],vax[i],vay[i])
            if distance[i][k] < minDist:
                minDist  = distance[i][k]
                minIndex = k
        sigma[i][minIndex]=1
            
    return sigma
#function 2 fix other variables and calculate sigma
#def optimizeothers2(sigma,weight,knn):
#    x1=[]
#    x2=[]
#    s1 = datetime.datetime.now()
#    #x4=[]
#    objective=0
#    m=Model('optimizeothers')
#    beta = m.addVars(knn, D+1,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")
#    #w = m.addVars(N, K, D, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#    ce = m.addVars(knn, D, lb=-MM,vtype=GRB.CONTINUOUS, name="ce")
#    #L = m.addVars(N, K, D,lb=-MM, ub=MM,vtype=GRB.CONTINUOUS, name='L')
#    m.update()
#    
#    m.setObjective(quicksum(sigma[i][k]*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D])*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D]) for i in range(N) for k in range(knn))\
#                      +weight*quicksum(sigma[i][k]*sum((x[i][j]-ce[k,j])*(x[i][j]-ce[k,j]) for j in range(D)) for i in range(N) for k in range(knn)), GRB.MINIMIZE)
#    
#    m.addConstrs(
#                 (quicksum(ce[k,j]*sigma[i][k] for i in range(N)) == quicksum(sigma[i][k]*x[i][j] for i in range(N)) for k in range(knn) for j in range(D)),"c1")
#   
#    m.optimize()
#        
#    status = m.status
##    if status == GRB.Status.UNBOUNDED:
##        print('The model cannot be solved because it is unbounded')
##        #exit(0)
##    if status == GRB.Status.OPTIMAL:
##        print('The optimal objective is %g' % m.objVal)
##        #exit(0)
##    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
##        print('Optimization was stopped with status %d' % status)
##        #exit(0)
#
#    m.write('clustering.lp')        
#
#    
#
#    
#    if m.status == GRB.Status.OPTIMAL:
#        objective=m.objVal
#        #print(' optimal objective1 is %g\n' % objective)
#        
#        #print ('m')
#        for k in range(knn):
#            temp2=[]
#            for j in range(D):
#                temp2.append(ce[k,j].x)
#                #print ('%d th feature of cluster %d is %.4f' % (j+1,k+1,ce[k,j].x))
#            x1.append(temp2)
#                #print (ce[k,j])
#    
#        #print ('beta')
#        for k in range(knn):
#            temp3=[]
#            for j in range(D+1):
#                temp3.append(beta[k,j].x)
#                #print ('%d th regression of cluster %d is %.4f' % (j+1,k+1,beta[k,j].x))
#            x2.append(temp3)    
#    e1 = datetime.datetime.now()   
#     
#    return x1,x2,(e1-s1).total_seconds()     

    
def optimizesigma2(ce,beta,beta0,weight,knn,initialsigma,variance):#update clustering
    
#    s1= datetime.datetime.now()
    m=Model('optimizex')
  
    sigma = m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
#    temp3= m.addVars(N, knn, lb=0,vtype=GRB.CONTINUOUS, name="temp3")
    x1=[]
    objective=0
    
    m.update()
    
    #quicksum(sigma[i,k]*pow(y[i]-sum(beta[k][j]*x[i][j] for j in range(D))-beta[k][D],2) for i in range(N) for k in range(K))\
    #+para1*quicksum(pow(beta[k][j],2) for k in range(K) for j in range(D+1))\
    
    m.setObjective(quicksum(math.log(variance[k])*quicksum(sigma[i,k] for i in range(N)) for k in range(knn))\
                   +quicksum(sigma[i,k]*pow(y[i]-sum(beta[k][j]*x[i][j] for j in range(D))-beta0[k],2)/(2*pow(variance[k],2)) for i in range(N) for k in range(knn))\
                   +weight*quicksum(sigma[i,k]*sum(pow(x[i][j]-ce[k][j],2) for j in range(D)) for i in range(N) for k in range(knn))\
                   +gamma*quicksum((1-initialsigma[i][k])*sigma[i,k]+initialsigma[i][k]*(1-sigma[i,k]) for k in range(knn) for i in range(N)), GRB.MINIMIZE)
                   #+gamma*quicksum((sigma[i,k]-initialsigma[i][k])*(sigma[i,k]-initialsigma[i][k]) for k in range(knn) for i in range(N)), GRB.MINIMIZE)
   
#    m.addConstrs(
#                  (temp3[i,k]>=sigma[i,k]-initialsigma[i][k] for i in range(N) for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (temp3[i,k]>=initialsigma[i][k]-sigma[i,k] for i in range(N) for k in range(knn)),"c15")
         
    m.addConstrs(
               (quicksum(sigma[i,k] for k in range(knn)) == 1 for i in range(N)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[i,k] for i in range(N)) >= 1 for k in range(knn)),"c15")
        
    #m.Params.TimeLimit = 600
    m.optimize()
    status = m.status
    

    m.write('optimizex.lp')
    
    for i in range(N):
        temp2=[]
        for k in range(knn):
            temp2.append(sigma[i,k].x)

        x1.append(temp2)
#    e1= datetime.datetime.now()    
    #mipgap=m.MIPGap  
#    return x1,(e1-s1).total_seconds()
    return x1
    
  
def L1Distance(vector1, vector2): # L1 distance
    t = sum(abs(vector2 - vector1))
    return t


def L2Distance(vector1, vector2): 
    t=np.sum(np.square(vector1 - vector2))
    return t


def initialassignment(dataSet, knn):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples, 1)))
    not_find = False
    countt=[0]*knn

    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
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
# REV
filenameresult=r"...\REV\result(AP).txt"
filenameCV=r"...\REV\CV(AP).txt"
filenametime=r"...\REV\time(AP).txt"
filenamepredicttime=r"...\REV\predicttime(AP).txt"

filenametimeexcel=r"...\REV\algorithmtime(excel closed form).txt"
filenameexcel=r"...\REV\algorithmresult(excel closed form).txt"
filenameiteration=r"...\REV\algorithmiteration(excel closed form).txt"

##RQ
#filenameresult=r"...\RQ\result(AP).txt"
#filenameCV=r"...\RQ\CV(AP).txt"
#filenametime=r"...\RQ\time(AP).txt"
#filenamepredicttime=r"...\RQ\predicttime(AP).txt"
#
#filenametimeexcel=r"...\RQ\algorithmtime(excel closed form).txt"
#filenameexcel=r"...\RQ\algorithmresult(excel closed form).txt"
#filenameiteration=r"...\RQ\algorithmiteration(excel closed form).txt"
#
##WQ
#filenameresult=r"...\WQ\result(AP).txt"
#filenameCV=r"...\WQ\CV(AP).txt"
#filenametime=r"...\WQ\time(AP).txt"
#filenamepredicttime=r"...\WQ\predicttime(AP).txt"
#
#filenametimeexcel=r"...\WQ\algorithmtime(excel closed form).txt"
#filenameexcel=r"...\WQ\algorithmresult(excel closed form).txt"
#filenameiteration=r"...\WQ\algorithmiteration(excel closed form).txt"

for counttt in range(10):

    start = datetime.datetime.now()
    recordloss1=[[0]*TW for k in range(K)]
    recordloss2=[[0]*TW for k in range(K)]
    i_str=str(counttt+1)

   
    
    for countk in range(K):#run the algorithm for Runtime times
        
        if countk<=0.5:
            knn=countk*5
        else:
            knn=countk*5-1
        

        centroids=[[0]*(D) for k in range(knn+1)]
        tempsigma=[[0]*(knn+1) for i in range(N)]
        f1=True
        #f2=True
        while f1:
            (clusterAssment ,f1) = initialassignment(dataSet1, knn+1)   
        
        for i in range(N):
                tempsigma[i][int(clusterAssment[i])]=1
        
        for k in range(knn+1):
            for j in range(D):
                    centroids[k][j]=sum(tempsigma[i][k]*x[i][j] for i in range(N))/sum(tempsigma[i][k] for i in range(N))
         
        centroids=np.array(centroids)
        
             
        #warm start
        
        temp_sigma2=[[0]*(knn+1) for i in range(N)]
        weight=0
        for i in range(N):        
            temp_sigma2[i][int(clusterAssment[i])]=1
        
        File = open(filenameresult, "a")
        File.write('*****total clusters:%d,weight=%f*****\n'% (knn+1,weight))
        File.close()
        File = open(filenameCV, "a")
        File.write('*****total clusters:%d,weight=%f*****\n'% (knn+1,weight))
        File.close()
        
        start2 = datetime.datetime.now()
            
        itr = 1
        loss2=[]
#        x_ce2=[]
#        x_beta2=[]
        temp_variance=[0]*(knn+1)
        loss2.append(MM)
        actualobj=0        
        obj3=0   
        while 1:
            
            (temp_beta2,temp_ce2,temp_beta0)=optimizeothers(temp_sigma2,weight,knn+1)

            for k in range(knn+1):
                temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta0[k],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))
        
            obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
                 +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta0[k],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
                 +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)


#            (temp_ce2,temp_beta2,time1)=optimizeothers2(temp_sigma2,weight,knn+1)

#            for k in range(knn+1):
#                temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))
#        
#            obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
#                 +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
#                 +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)
           
            if (loss2[itr-1]-obj2)/obj2>=tolerance:
                
                x_sigma2=temp_sigma2
                loss2.append(obj2)
                x_ce2=temp_ce2
                x_beta2=temp_beta2
                x_beta0=temp_beta0
                x_variance=temp_variance              
                (temp_sigma2)=optimizesigma2(x_ce2,x_beta2,x_beta0,weight,knn+1,x_sigma2,x_variance)#obj given other variables
#                File = open(filenametime, "a")
#                File.write('iteration:%d, k=%d,weight=%f, time of beta: %f\n' % (counttt+1,knn+1,weight,time1))
#                File.write('iteration:%d, k=%d,weight=%f, time of sigma: %f\n' % (counttt+1,knn+1,weight,time2))
#                File.close()              
            else:
                
               
                break
            
            itr=itr+1
            
        
        end2 = datetime.datetime.now()
                
        ftime= (end2 - start2).total_seconds()
        File = open(filenametime, "a")
        File.write('computational time when k=%d,weight=%f: %f\n' % (knn+1,weight,ftime))
        File.close()
#        File = open(filenameresultprint, "a")
#        File.write('K=%d,weight=%f,total error=%s\n' % (knn+1,weight,loss2[-1]))
#        File.close()
        temp_sigma2=x_sigma2
#        temp_ce2=x_ce2
#        temp_beta2=x_beta2
        for ww in range(TW):
            
#            temp_sigma1=[[0]*(knn+1) for i in range(N)]#1st dimension=parts of cv
#            temp_sigma2=[[0]*(knn+1) for i in range(N)]
            if ww>=2:
                weight=0.5
            else:
                weight=0.1+0.1*ww
#            weight=0.1+0.1*ww
#            for i in range(N):
#                temp_sigma1[i][int(clusterAssment[i])]=1              
#                temp_sigma2[i][int(clusterAssment[i])]=1
           
            File = open(filenameresult, "a")
            File.write('*****total clusters:%d,weight=%f*****\n'% (knn+1,weight))
            File.close()
            File = open(filenameCV, "a")
            File.write('*****total clusters:%d,weight=%f*****\n'% (knn+1,weight))
            File.close()
            
            
            # AP2
     
            start2 = datetime.datetime.now()
            
            itr = 1
            loss2=[]
            temp_variance=[0]*(knn+1)
#            x_ce2=[]
#            x_beta2=[]
            loss2.append(MM)
            actualobj=0        
            obj3=0   
            while 1:
#                
                (temp_beta2,temp_ce2,temp_beta0)=optimizeothers(temp_sigma2,weight,knn+1)

                for k in range(knn+1):
                    temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta0[k],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))
            
                obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
                     +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta0[k],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
                     +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)
                
                
#                (temp_ce2,temp_beta2,time1)=optimizeothers2(temp_sigma2,weight,knn+1)

#                
#                for k in range(knn+1):
#                    temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))
#            
#                obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
#                     +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
#                     +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)
               
  
#                if loss2[itr-1]-obj2>=tolerance:
#                if (loss2[itr-1]-obj2)/obj2>=tolerance:
                if itr<=3:
                    
                    x_sigma2=temp_sigma2
                    loss2.append(obj2)
                    x_ce2=temp_ce2
                    x_beta2=temp_beta2
                    x_beta0=temp_beta0
                    x_variance=temp_variance              
                    (temp_sigma2)=optimizesigma2(x_ce2,x_beta2,x_beta0,weight,knn+1,x_sigma2,x_variance)#obj given other variables
#                    File = open(filenametime, "a")
#                    File.write('iteration:%d, k=%d,weight=%f, time of beta: %f\n' % (counttt+1,knn+1,weight,time1))
#                    File.write('iteration:%d, k=%d,weight=%f, time of sigma: %f\n' % (counttt+1,knn+1,weight,time2))
#                    File.close()  
                else:                    
                   
                    break
                
                itr=itr+1
                
            
            end2 = datetime.datetime.now()
                    
            ftime= (end2 - start2).total_seconds()
            temp_sigma2=x_sigma2

            File = open(filenameexcel, "a")                
#            if ww>=3.5:
            if ww>=2:
                File.write('%.3f\n'% loss2[-1])
            else:
                File.write('%.3f\t'% loss2[-1])
            
            File.close()
            
            File = open(filenametimeexcel, "a")                
#            if ww>=3.5:
            if ww>=2:
                File.write('%.3f\n'% ftime)
            else:
                File.write('%.3f\t'% ftime)
            
            File.close()
            
            File = open(filenameiteration, "a")                
#            if ww>=3.5:
            if ww>=2:
                File.write('%d\n'% itr)
            else:
                File.write('%d\t'% itr)
            
            File.close()
            
            
            File = open(filenameresultprint, "a")
            File.write('iteration=%d, K=%d,weight=%f,total error=%s\n' % (counttt+1,knn+1,weight,loss2[-1]))
            File.close()            
            
            
            #File = open(filenameresultAP2, "a")
            File = open(filenameresult, "a")
            File.write('AP2: K=%d,weight=%f\n' % (knn+1,weight))
            File.write('obj=%g\n'% loss2[-1])
            File.write('sigma\n')
            for i in range(N):
                for k in range(knn+1):
                      if x_sigma2[i][k]>=0.9:
                             File.write('cluster:%d contain: %d\n' % (k+1,i+1))
                             
            File.write('m\n')
            for k in range(knn+1):
                for j in range(D):
                    File.write('%d th center of cluster %d is %.4f\n' % (j+1,k+1,x_ce2[k][j]))
            
           
            File.write('beta\n')
            for k in range(knn+1):
                File.write('%d th regression of cluster %d is %.4f\n' % (D+1,k+1,x_beta0[k]))
                #if x_z2[t]>=0.9:
                for j in range(D):
                    File.write('%d th regression of cluster %d is %.4f\n' % (j+1,k+1,x_beta2[k][j]))
                    
            File.write('variance\n')
            for k in range(knn+1):               
                File.write('The variance of %d th cluster is %.4f\n' % (k+1,x_variance[k]))
            
            File.close() 
            perror2=0
            pstart = datetime.datetime.now()
#            (vsigma)=optimizesigmanew(x_ce2,x_beta2,x_beta0,weight,knn+1,x_variance)
            
           
            vam = Model("validation")  
            vasigma=[]  
            
            assign=vam.addVars(A, knn+1, vtype=GRB.BINARY, name='assign')
            vam.update()
                               
            vam.setObjective(quicksum(math.log(x_variance[k])*quicksum(assign[i,k] for i in range(A)) for k in range(knn+1))\
                             +quicksum((assign[i,k]*pow((vay[i]-sum(x_beta2[k][j]*vax[i][j] for j in range(D))-x_beta0[k]) ,2))/(2*pow(x_variance[k],2)) for k in range(knn+1) for i in range(A))\
                              +weight*quicksum(assign[i,k]*sum((vax[i][j]-x_ce2[k][j])*(vax[i][j]-x_ce2[k][j]) for j in range(D)) for i in range(A) for k in range(knn+1)), GRB.MINIMIZE)
                         
            vam.addConstrs(
                    (quicksum(assign[i,k] for k in range(knn+1)) == 1 for i in range(A)),"c21")
            
            vam.optimize()
            
            pend = datetime.datetime.now()
            predicttime= (pend - pstart).total_seconds()
            
            status = vam.status
            if status == GRB.Status.UNBOUNDED:
                print('The model cannot be solved because it is unbounded')
            if status == GRB.Status.OPTIMAL:
                print('The optimal objective is %g' % vam.objVal)
            if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % status)
        
            vam.write('validation.lp')
        
        
            if vam.status == GRB.Status.OPTIMAL:
                print ('assign')
                
                for i in range(A):
                    temp=[]
                    for k in range(knn+1):
                        temp.append(assign[i,k].x)
                    vasigma.append(temp)
#                
            File = open(filenamepredicttime, "a")
            File.write('iteration=%d, computational time when k=%d,weight=%f: %f\n' % (counttt+1,knn+1,weight,predicttime))
            File.close()   
                    
            perror2=sum((vay[i]-sum(vasigma[i][k]*(sum(x_beta2[k][j]*vax[i][j] for j in range(D))+x_beta0[k]) for k in range(knn+1)))\
                        *(vay[i]-sum(vasigma[i][k]*(sum(x_beta2[k][j]*vax[i][j] for j in range(D))+x_beta0[k]) for k in range(knn+1))) for i in range(A)) #L1 distance

            recordloss2[countk][ww]=perror2/A

            if vam.status == GRB.Status.OPTIMAL:
                #File = open(filenameCVAP2, "a")
                File = open(filenameCV, "a")
                File.write('***K=%d, weight=%f,obj=%g***\n'% (knn+1,weight,vam.objVal))
                File.write('total error=%s\n' % str(perror2/A))
                File.write('assign\n')
                for i in range(A):
                     for k in range(knn+1):
                             if assign[i,k].x>=0.9:
                                File.write('data point:%d belong to cluster %d\n' % (i+1,k+1))
                File.close()   
            
#             if vam.status == GRB.Status.OPTIMAL:
                #File = open(filenameCVAP2, "a")
#            File = open(filenameCV, "a")
#            File.write('***iteration=%d,K=%d, weight=%f,total error=%s***\n'% (counttt+1,knn+1,weight,str(perror2/A)))
#            File.write('assign\n')
#            for i in range(A):
#                 for k in range(knn+1):
#                         if vsigma[i][k]>=0.9:
#                            File.write('data point:%d belong to cluster %d\n' % (i+1,k+1))
#            File.close()   
            
            File = open(filenameCVprint, "a")
            File.write('iteration=%d, K=%d,weight=%f,total error=%s\n' % (counttt+1,knn+1,weight,str(perror2/A)))
            File.close()         
            
            
            #File = open(filenametimeAP2, "a")
            File = open(filenametime, "a")
            File.write('iteration:%d, computational time when k=%d,weight=%f: %f\n' % (counttt+1,knn+1,weight,ftime))
            File.close() 
            
   
    end = datetime.datetime.now()
    print(end-start)
    totaltime= (end - start).total_seconds()
    File = open(filenametime, "a")
    File.write('iteration:%d, computational time : %f\n' % (counttt+1,totaltime))
    File.close() 
    File = open(filenameexcel, "a")
    File.write('\n')
    File.close()
    
    File = open(filenametimeexcel, "a")
    File.write('\n')
    File.close()
    
    File = open(filenameiteration, "a")
    File.write('\n')
    File.close()
       


 
           




    