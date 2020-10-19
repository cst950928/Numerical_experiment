
from __future__ import division
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from gurobipy import *
import math
import xlrd #excel
import sys 
import datetime
from random import sample
from numpy.linalg import det, inv, matrix_rank
from sympy import *
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import time
import warnings
from sklearn import linear_model
warnings.filterwarnings("ignore")
from numpy import *
import datetime


# parameters and path should be revised
N=14303
TN=20433
D=8
P=1
K=1
TW=1
A=6130
y=[]
x=[]
vay=[]
vax=[]


housing_df = pd.read_csv('.../data/housing.csv')
housing_df.head()
housing_df = housing_df.drop('ocean_proximity',axis=1)
housing_df.head()

#print('Shape :',housing_df.shape)
#print('Features Data types : \n',housing_df.dtypes)
#print('checking if any null values')
#print(housing_df.isnull().sum())

# Null values in total_bedrooms so we will drop them, Its best practice to replace any null values with mean/median.
housing_df = housing_df.dropna(axis=0)
housing_df.shape

X = housing_df.drop(['median_house_value'],axis=1)
Y = housing_df['median_house_value']
#print(X.shape,Y.shape)
#print(X.values[14302])
#print(Y.values[14302])
#meanx=[0]*D
#for j in range(D):
meanx=np.mean(X.values, axis=0) 
meany=np.mean(Y.values)
stdy=np.std(Y.values)
stdx=np.std(X.values,axis=0)
X1=[[0]*D for i in range(TN)]
Y1=[0]*TN 
#Y.values=float(Y.values)
#X.values=float(X.values)
for i in range(TN):
    Y1[i]=(1.0*Y.values[i]-1.0*meany)/(1.0*stdy)
    for j in range(D):
        X1[i][j]=(X.values[i][j]-meanx[j])/(stdx[j])
    
number=0
while number<=N-1:
    y.append(Y1[number])
    dx=[]
    for j in range(D):
        dx.append(X1[number][j])
    x.append(dx)
    number=number+1
    
while number<=TN-1:
    vay.append(Y1[number])
    dx=[]
    for j in range(D):
        dx.append(X1[number][j])
    vax.append(dx)
    number=number+1
    
    
#print('length of y',len(y))
#print('length of x',len(x))
#print('length of vay',len(vay))
#print('length of vax',len(vax))
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
    
def L2Distance(vector1, vector2): 
    t=np.sum(np.square(vector1 - vector2))
    return t

def optimizeothersclosedform(sigma,weight,knn):
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
        if len(Y[k])==1:
            Y[k]=np.mat(Y[k]).reshape((len(Y[k]),1))
            B=np.dot(np.linalg.inv(np.dot(XT[k],X[k])),XT[k])*Y[k]
        else:
            Y[k]=np.mat(Y[k]).reshape((len(Y[k]),1))
#        B=np.dot(np.dot(np.dot(XT[k],X[k]).I,XT[k]),Y[k])
            B=np.dot(np.dot(np.linalg.inv(np.dot(XT[k],X[k])),XT[k]),Y[k])
        

        x1[k]=np.array(B)
        x3[k]=(sum(Y[k])-sum(np.dot(X[k],B)))/len(Y[k])
        
   
        for k in range(knn):
            for j in range(D):
                x2[k][j]=sum(sigma[i][k]*x[i][j] for i in range(N))/sum(sigma[i][k] for i in range(N))    
    
#    e1 = datetime.datetime.now()   
     
    return x1,x2,x3


def optimizeothers(sigma,weight,knn):
    x1=[[0]*D for k in range(knn)]
    x2=[]
    
    #x4=[]
    objective=0
    m=Model('optimizeothers')
    beta = m.addVars(knn, D+1,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")

    m.update()
    
    m.setObjective(quicksum(sigma[i][k]*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D])*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D]) for i in range(N) for k in range(knn)), GRB.MINIMIZE)
  
    m.optimize()
        
    status = m.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)

    m.write('clustering.lp')        

    print('')

    
    if m.status == GRB.Status.OPTIMAL:
        objective=m.objVal
        #print(' optimal objective1 is %g\n' % objective)
        
        for k in range(knn):
            for j in range(D):
                x1[k][j]=sum(sigma[i][k]*x[i][j] for i in range(N))/sum(sigma[i][k] for i in range(N))    

        for k in range(knn):
            temp3=[]
            for j in range(D+1):
                temp3.append(beta[k,j].x)
                #print ('%d th regression of cluster %d is %.4f' % (j+1,k+1,beta[k,j].x))
            x2.append(temp3)    
        return x1,x2



def assignment(ce,beta,beta0,weight,var,x0,y0):
    
    totalobj=math.log(var)+pow(y0-sum(beta[j]*x0[j] for j in range(D))-beta0,2)/(2*pow(var,2))+weight*L2Distance(x0, np.mat(ce))
    return totalobj

def optimizesigmaclosedform(ce,beta,beta0,weight,knn,initialsigma,variance):#update clustering
    
    m=Model('optimizex')
  
    sigma = m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
    x1=[]
    objective=0
    
    m.update()
    m.setObjective(quicksum(math.log(variance[k])*quicksum(sigma[i,k] for i in range(N)) for k in range(knn))\
                   +quicksum(sigma[i,k]*pow(y[i]-sum(beta[k][j]*x[i][j] for j in range(D))-beta0[k],2) for i in range(N) for k in range(knn))\
                   +weight*quicksum(sigma[i,k]*sum(pow(x[i][j]-ce[k][j],2) for j in range(D)) for i in range(N) for k in range(knn))\
                   +gamma*quicksum((1-initialsigma[i][k])*sigma[i,k]+initialsigma[i][k]*(1-sigma[i,k]) for k in range(knn) for i in range(N)), GRB.MINIMIZE)
         
    m.addConstrs(
               (quicksum(sigma[i,k] for k in range(knn)) == 1 for i in range(N)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[i,k] for i in range(N)) >= 1 for k in range(knn)),"c15")
        
    m.optimize()
    status = m.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)

    m.write('optimizex.lp')

    for i in range(N):
        temp2=[]
        for k in range(knn):
            temp2.append(sigma[i,k].x)
        x1.append(temp2)

    return x1  

def optimizesigma(ce,beta,weight,knn,initialsigma,variance):#update clustering
    
    m=Model('optimizex')
  
    sigma = m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
    x1=[]
    objective=0
    
    m.update()
    m.setObjective(quicksum(math.log(variance[k])*quicksum(sigma[i,k] for i in range(N)) for k in range(knn))\
                   +quicksum(sigma[i,k]*pow(y[i]-sum(beta[k][j]*x[i][j] for j in range(D))-beta[k][D],2) for i in range(N) for k in range(knn))\
                   +weight*quicksum(sigma[i,k]*sum(pow(x[i][j]-ce[k][j],2) for j in range(D)) for i in range(N) for k in range(knn))\
                   +gamma*quicksum((1-initialsigma[i][k])*sigma[i,k]+initialsigma[i][k]*(1-sigma[i,k]) for k in range(knn) for i in range(N)), GRB.MINIMIZE)
         
    m.addConstrs(
               (quicksum(sigma[i,k] for k in range(knn)) == 1 for i in range(N)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[i,k] for i in range(N)) >= 1 for k in range(knn)),"c15")
        
    m.optimize()
    status = m.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)

    m.write('optimizex.lp')

    for i in range(N):
        temp2=[]
        for k in range(knn):
            temp2.append(sigma[i,k].x)
        x1.append(temp2)

    return x1
def L1Distance(vector1, vector2): # L1 distance
    t = sum(abs(vector2 - vector1))
    return t

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
filenametimeexcel=r"...\algorithmtime(excel).txt"
filenameexcel=r"...\algorithmresult(excel).txt"
filenameiteration=r"...\algorithmiteration(excel).txt"
filenameCVprint=r"...\CV loss.txt"
filenameresultprint=r"...\result.txt"
filenametimeprint=r"...\time.txt"
filenamefeatureprint=r"...\importantfeature.txt"
minimumtime=[[0]*TW for k in range(K)]
minimumobj=[[MM]*TW for k in range(K)] 
recordcounttt=[[0]*TW for k in range(K)]
recordx_sigma=[[0]*TW for k in range(K)]
recordx_beta=[[0]*TW for k in range(K)]
recordx_beta0=[[0]*TW for k in range(K)]
recordx_ce=[[0]*TW for k in range(K)]
recordx_variance=[[0]*TW for k in range(K)]
record_training_error=[[0]*TW for k in range(K)]
recordloss=[[0]*TW for k in range(K)]
for counttt in range(10):

    start = datetime.datetime.now()
    recordloss1=[[0]*TW for k in range(K)]
    i_str=str(counttt+1)
    filenameresult=r"...\result(algorithm)_"+i_str+'.txt'    
    filenameCV=r"...\CV(algorithm)_"+i_str+'.txt'
    filenametime=r"...\time(algorithm)_"+i_str+'.txt'
    filenamefeature=r"...\feature(algorithm)_"+i_str+'.txt'
    newy=[0]*(N) 
    for countk in range(K):#run the algorithm for Runtime times
        
#        if countk<=0.5:
#            knn=countk*5
#        else:
#            knn=countk*5-1
        knn=4
        centroids=[[0]*(D) for k in range(knn+1)]
        tempsigma=[[0]*(knn+1) for i in range(N)]
        f1=True
        while f1:
            (clusterAssment ,f1) = initialassignment(dataSet1, knn+1)   
        
        for i in range(N):
                tempsigma[i][int(clusterAssment[i])]=1
        
        for k in range(knn+1):
            for j in range(D):
                    centroids[k][j]=sum(tempsigma[i][k]*x[i][j] for i in range(N))/sum(tempsigma[i][k] for i in range(N))
         
        centroids=np.array(centroids)
        

        for ww in range(TW):
            
            temp_sigma2=[[0]*(knn+1) for i in range(N)]
            if ww>=2:
                weight=0.5
            else:
                weight=0.1+0.1*ww
#            weight=0.1+0.1*ww
            for i in range(N):             
                temp_sigma2[i][int(clusterAssment[i])]=1
           
            File = open(filenameresult, "a")
            File.write('*****total clusters:%d,weight=%f*****\n'% (knn+1,weight))
            File.close()
            File = open(filenameCV, "a")
            File.write('*****total clusters:%d,weight=%f*****\n'% (knn+1,weight))
            File.close()
            File = open(filenamefeature, "a")
            File.write('*****total clusters:%d,weight=%f*****\n'% (knn+1,weight))
            File.close()
            
            # AP2
     
            start2 = datetime.datetime.now()
            
            itr = 1
            loss2=[]
            temp_variance=[0]*(knn+1)
            loss2.append(MM)
            actualobj=0        
            obj3=0   
            while 1:
#closed form    
                
                (temp_beta2,temp_ce2,temp_beta0)=optimizeothersclosedform(temp_sigma2,weight,knn+1)
                for k in range(knn+1):
                    temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta0[k],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))

                obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
                     +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta0[k],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
                     +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)
                
                
                if (loss2[itr-1]-obj2)/obj2>=tolerance:
                    x_sigma2=temp_sigma2
                    loss2.append(obj2)
                    x_ce2=temp_ce2
                    x_beta2=temp_beta2
                    x_beta0=temp_beta0
                    x_variance=temp_variance
                    (temp_sigma2)=optimizesigmaclosedform(x_ce2,x_beta2,x_beta0,weight,knn+1,x_sigma2,x_variance)
                    
# Gurobi
                
#                (temp_ce2,temp_beta2)=optimizeothers(temp_sigma2,weight,knn+1)
#                for k in range(knn+1):
#                    temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))
#            
#                obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
#                     +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
#                     +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)

#                
#                if (loss2[itr-1]-obj2)/obj2>=tolerance:
#
#                    x_sigma2=temp_sigma2
#                    loss2.append(obj2)
#                    x_ce2=temp_ce2
#                    x_beta2=temp_beta2
#                    x_variance=temp_variance              
#                    (temp_sigma2)=optimizesigma(x_ce2,x_beta2,weight,knn+1,x_sigma2,x_variance)#closed-form

                else: #if the new result does not have huge improvements
                    
                   
                    break
                
                itr=itr+1
                
            
            end2 = datetime.datetime.now()
                    
            ftime= (end2 - start2).total_seconds()
            #File = open(filenametimeAP2, "a")
            
#closed form            
            for i in range(N):
                for k in range(knn+1):
                    if x_sigma2[i][k]>=0.9:
                        for j in range(D):
                            newy[i]+=x[i][j]*x_beta2[k][j]
                        newy[i]=newy[i]+x_beta0[k]
# Gurobi            
#            for i in range(N):
#                for k in range(knn+1):
#                    if x_sigma2[i][k]>=0.9:
#                        for j in range(D):
#                            newy[i]+=x[i][j]*x_beta2[k][j]
#                        newy[i]=newy[i]+x_beta2[k][D]
            
            
            perrorICLR=sum(pow(newy[i]-y[i],2) for i in range(N))/N
#            temp_sigma2=x_sigma2
                       
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
            File.write('iteration=%d, K=%d,weight=%f,obj=%s\n' % (counttt+1,knn+1,weight,loss2[-1]))
            File.write('iteration=%d, K=%d,weight=%f,training error=%s\n' % (counttt+1,knn+1,weight,perrorICLR))
            File.close()
##            
            
            File = open(filenametimeprint, "a")
            File.write('iteration:%d, computational time when k=%d,weight=%f: %f\n' % (counttt+1,knn+1,weight,ftime))
            File.close() 
            
            File = open(filenamefeature, "a")
            for k in range(knn+1):
                for j in range(D):
                    File.write('k=%d: %d\t' % (k+1,1+j))
                    File.write('%f\n' % x_beta2[k][j])   
                File.write('\n')
            File.write('\n')
            File.close()
            
            File = open(filenameresult, "a")
            File.write(' K=%d,weight=%f\n' % (knn+1,weight))
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
            
            
            vam = Model("validation")  
            perror2=0   
            vasigma=[]  
            pstart = datetime.datetime.now()
            
            assign=vam.addVars(A, knn+1, vtype=GRB.BINARY, name='assign')
            vam.update()
            
            vam.setObjective(quicksum(math.log(x_variance[k])*quicksum(assign[i,k] for i in range(A)) for k in range(int(knn+1)))\
                             +quicksum((assign[i,k]*pow((vay[i]-sum(x_beta2[k][j]*vax[i][j] for j in range(D))-x_beta0[k]) ,2))/(2*pow(x_variance[k],2)) for k in range(int(knn+1)) for i in range(A))\
                              +weight*quicksum(assign[i,k]*sum((vax[i][j]-x_ce2[k][j])*(vax[i][j]-x_ce2[k][j]) for j in range(D)) for i in range(A) for k in range(int(knn+1))), GRB.MINIMIZE)
                         
                   
                        
            vam.addConstrs(
                    (quicksum(assign[i,k] for k in range(int(knn+1))) == 1 for i in range(A)),"c21")
            
            vam.optimize()
            pend = datetime.datetime.now()
            ptotaltime= (pend - pstart).total_seconds()
            File = open(filenametime, "a")
            File.write('iteration=%d, prediction time when K=%d,weight=%f: %f\n' % (counttt+1,int(knn+1),weight,ptotaltime))
            File.close()
            
            
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
                
     
            perror2=sum((vay[i]-sum(vasigma[i][k]*(sum(x_beta2[k][j]*vax[i][j] for j in range(D))+x_beta0[k]) for k in range(int(knn+1))))\
                        *(vay[i]-sum(vasigma[i][k]*(sum(x_beta2[k][j]*vax[i][j] for j in range(D))+x_beta0[k]) for k in range(int(knn+1)))) for i in range(A)) #L1 distance

            if minimumobj[countk][ww]>=loss2[-1]:
                minimumobj[countk][ww]=loss2[-1]
                minimumtime[countk][ww]=ftime
                recordx_sigma[countk][ww]=x_sigma2
                recordx_beta[countk][ww]=x_beta2
                recordx_beta0[countk][ww]=x_beta0
                recordx_ce[countk][ww]=x_ce2
                recordx_variance[countk][ww]=x_variance
                record_training_error[countk][ww]=perrorICLR
                recordloss[countk][ww]=perror2/A
                
                
            if vam.status == GRB.Status.OPTIMAL:
                #File = open(filenameCVAP2, "a")
                File = open(filenameCV, "a")
                File.write('***K=%d, weight=%f,obj=%g***\n'% (int(knn+1),weight,vam.objVal))
                File.write('total error=%s\n' % str(perror2/A))
                File.write('assign\n')
                for i in range(A):
                     for k in range(int(knn+1)):
                             if assign[i,k].x>=0.9:
                                File.write('data point:%d belong to cluster %d\n' % (i+1,k+1))
                File.close()       
            
            File = open(filenameCVprint, "a")
            File.write('iteration=%d, K=%d,weight=%f,MSE=%s\n' % (counttt+1, int(knn+1),weight,str(perror2/A)))
            File.close()
            

    
    File = open(filenametimeexcel, "a")
    File.write('\n')
    File.close()
    
    File = open(filenameiteration, "a")
    File.write('\n')
    File.close()

#for countk in range(K): 
#    knn=4
#    for ww in range(TW):
#        for i in range(N):
#            for k in range(knn+1):
#                if recordx_sigma[countk][ww][i][k]>=0.9:
#                    for j in range(D):
#                        newy[i]+=x[i][j]*recordx_beta[countk][ww][k][j]
#                    newy[i]=newy[i]+recordx_beta0[countk][ww][k]

File = open(filenameresultprint, "a")
for countk in range(K):
#    if countk<=0.5:
#        knn=countk*5
#    else:
#        knn=countk*5-1
    knn=4
    for ww in range(TW):
        if ww>=2:
            weight=0.5
        else:
            weight=0.1+0.1*ww
        File.write('K=%d,weight=%f,minimum obj=%s\n' % (knn+1,weight,minimumobj[countk][ww]))
        File.write('K=%d,weight=%f,minimum training error=%s\n' % (knn+1,weight,record_training_error[countk][ww]))

File.close()


File = open(filenametimeprint, "a")
for countk in range(K):
#    if countk<=0.5:
#        knn=countk*5
#    else:
#        knn=countk*5-1
    knn=4
    for ww in range(TW):
        if ww>=2:
            weight=0.5
        else:
            weight=0.1+0.1*ww
        File.write('K=%d,weight=%f,minimum computational time : %f\n' % (knn+1,weight,minimumtime[countk][ww]))

File.close() 


File = open(filenameCVprint, "a")
for countk in range(K):
#    if countk<=0.5:
#        knn=countk*5
#    else:
#        knn=countk*5-1
    knn=4
    for ww in range(TW):
        if ww>=2:
            weight=0.5
        else:
            weight=0.1+0.1*ww
        File.write('K=%d,weight=%f,minimum CV: %f\n' % (knn+1,weight,recordloss[countk][ww]))
File.close()        
  
       
File = open(filenamefeatureprint, "a")
for countk in range(K):
#    if countk<=0.5:
#        knn=countk*5
#    else:
#        knn=countk*5-1
    knn=4
    for ww in range(TW):
        if ww>=2:
            weight=0.5
        else:
            weight=0.1+0.1*ww
        File.write('K=%d,weight=%f\n' % (knn+1,weight))
        for k in range(knn+1):
            for j in range(D):
                File.write('k=%d: %d\t' % (k+1,1+j))
                File.write('%f\n' % recordx_beta[countk][ww][k][j])       
File.close()

#fig=plt.figure()
#ax1=fig.add_subplot(111)
#ax1.set_ylabel('MSE among testing set')
#ax1.set_xlabel('Regression model') 
#for k in range(knn+1):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                ax1.plot(2, y[i], marker='.', mec='#ffffb2', mfc='#ffffb2')           
#            elif k==1:
#                ax1.plot(2+0.2, y[i], marker='.', mec='black', mfc='black') 
#            elif k==2:
#                ax1.plot(2+0.4, y[i], marker='.', mec='#fed98e', mfc='#fed98e') 
#            elif k==3:
#                ax1.plot(2+0.6, y[i], marker='.', mec='#e6550d', mfc='#e6550d')
#            else:
#                ax1.plot(2+0.8, y[i], marker='.', mec='r', mfc='r')
#plt.xticks([2,2.2,2.4,2.6,2.8], ['k=1', 'k=2', 'k=3', 'k=4', 'k=5'])
#plt.savefig(r'...\original'+str(weight)+'_'+str(counttt+1)+'.png')
#plt.show()

#f7 = plt.figure(7)
#
#for k in range(9):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#fff7f3',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#fde0dd',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#fcc5c0',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#fa9fb5',  s = 10)
#            elif k==4:
#                p5 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#f768a1',  s = 10)
#            elif k==5:
#                p6 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#dd3497',  s = 10)
#            elif k==6:   
#                p7 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#ae017e',  s = 10)
#            elif k==7:   
#                p8 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#7a0177',  s = 10)
#            else:
#                p9 = plt.scatter(x[i][1], x[i][0], marker = '.', c='', edgecolors='#49006a',  s = 10)
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=9.png')
#plt.show()
#
#f7 = plt.figure(7)
#




for i in range(N):
    if recordx_sigma[0][0][i][0]>=0.5:
        p1 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#000000',  s = 10)  
plt.xlim((-2.5, 3))
plt.ylim((-2, 3.1))
plt.xlabel("longtitude")
plt.ylabel("lantitude") 
#plt.legend(handles = [p1], labels = ['$C_1$'], loc = 'upper right')
plt.legend(handles = [p1], loc = 'upper right')
plt.savefig(r'...\img_k=1.5_location_1.png')
plt.show()

for i in range(N):
    if recordx_sigma[0][0][i][1]>=0.5:
        p2 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#000000',  s = 10)
plt.xlim((-2.5, 3))
plt.ylim((-2, 3.1))
plt.xlabel("longtitude")
plt.ylabel("lantitude") 
plt.legend(handles = [p2],  loc = 'upper right')
plt.savefig(r'...\img_k=2.5_location_1.png')
plt.show()

for i in range(N):
    if recordx_sigma[0][0][i][2]>=0.5:
        p3 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#000000',  s = 10)
plt.xlim((-2.5, 3))
plt.ylim((-2, 3.1))
plt.xlabel("longtitude")
plt.ylabel("lantitude") 
plt.legend(handles = [p3],  loc = 'upper right')
plt.savefig(r'...\img_k=3.5_location_1.png')
plt.show()

for i in range(N):
    if recordx_sigma[0][0][i][3]>=0.5:
        p4 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#000000',  s = 10)
plt.xlim((-2.5, 3))
plt.ylim((-2, 3.1))
plt.xlabel("longtitude")
plt.ylabel("lantitude") 
plt.legend(handles = [p4],  loc = 'upper right')
plt.savefig(r'...\img_k=4.5_location_1.png')
plt.show()

for i in range(N):
    if recordx_sigma[0][0][i][4]>=0.5:
        p5 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#000000',  s = 10)
plt.xlim((-2.5, 3))
plt.ylim((-2, 3.1))
plt.xlabel("longtitude")
plt.ylabel("lantitude") 
plt.legend(handles = [p5],  loc = 'upper right')
plt.savefig(r'...\img_k=5.5_location_1.png')
plt.show()
                

for k in range(5):
    for i in range(N):
        if recordx_sigma[countk][ww][i][k]>=0.5:
            if k==0:
                p1 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)  
            elif k==1:
                p2 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
            elif k==2:
                p3 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
            elif k==3:
                p4 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
            else:
                p5 = plt.scatter(x[i][0], x[i][1], marker = '.', c='', edgecolors='#d4b9da',  s = 10)

plt.xlim((-2.5, 3))
plt.ylim((-2, 3.1))                              
plt.xlabel("longtitude")
plt.ylabel("lantitude") 
plt.legend(handles = [p1, p2, p3, p4, p5], labels = ['$C_1$', '$C_2$', '$C_3$', '$C_4$', '$C_5$'], loc = 'best')
plt.savefig(r'...\img_k=5_location_1.png')
plt.show()

#for countk in range(K):
#    knn=1
#    for ww in range(TW):
#        for k in range(knn+1):
#            print('recordx_beta',recordx_beta[countk][ww][k])
#            b = np.argsort(-abs(recordx_beta[countk][ww][k]),axis=0)
#            print('index',b) 
#            print('value',recordx_beta[countk][ww][k][b])
#            for j in b:
#                print('value',recordx_beta[countk][ww][k][j])
#                
#b=array(b)                
#newy[0]=0
#print(x[0],y[0])            
#for i in range(1):
#    for j in range(4):
#        newy[i]=newy[i]+x[i][int(b[j])]*recordx_beta[0][0][1][b][j]  
#print('newy',newy[0])            
#print('1')            
            

#            
#            
#f7 = plt.figure(7)
#
#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][0], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][0], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][0], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][0], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][0], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_1_1.png')
#plt.show()

#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][1], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][1], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][1], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][1], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][1], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_2_1.png')
#plt.show()
#
#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][2], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][2], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][2], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][2], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][2], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_3_1.png')
#plt.show()
#
#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][3], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][3], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][3], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][3], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][3], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_4_1.png')
#plt.show()
#
#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][4], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][4], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][4], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][4], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][4], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_5_1.png')
#plt.show()
#
#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][5], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][5], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][5], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][5], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][5], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_6_1.png')
#plt.show()
#
#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][6], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][6], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][6], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][6], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][6], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_7_1.png')
#plt.show()
#
#for k in range(5):
#    for i in range(N):
#        if recordx_sigma[countk][ww][i][k]>=0.5:
#            if k==0:
#                p1 = plt.scatter(x[i][7], y[i], marker = '.', c='', edgecolors='#fd8d3c',  s = 10)       
#            elif k==1:
#                p2 = plt.scatter(x[i][7], y[i], marker = '.', c='', edgecolors='#ffeda0',  s = 10)
#            elif k==2:
#                p3 = plt.scatter(x[i][7], y[i], marker = '.', c='', edgecolors='#9e9ac8',  s = 10)
#            elif k==3:
#                p4 = plt.scatter(x[i][7], y[i], marker = '.', c='', edgecolors='#41ab5d',  s = 10)
#            else:
#                p5 = plt.scatter(x[i][7], y[i], marker = '.', c='', edgecolors='#d4b9da',  s = 10)
#            
#                                 
#plt.legend(loc = 'best')
#plt.savefig(r'...\img_k=5_8_1.png')
#plt.show()
