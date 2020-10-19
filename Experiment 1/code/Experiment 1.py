
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





N=24
TN=30
D=3
P=1#parts of cv
K=4
TW=5
A=TN-N
readfile=r"...\data\experimentdata.xlsx"
book = xlrd.open_workbook(readfile)

sh= book.sheet_by_name("Sheet9")

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




#totaltime1=[0]*5
#totaltime2=[0]*5
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
        abslist=map(abs, extrax)
    absdimax.append(max(abslist))
    dimax.append(max(extrax))
    dimin.append(min(extrax))
    extrax=[]


def optimizeothers(sigma,weight,knn):
    x1=[]
    x2=[]
    
    objective=0
    m=Model('optimizeothers')
    beta = m.addVars(knn, D+1,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")
    ce = m.addVars(knn, D, lb=-MM,vtype=GRB.CONTINUOUS, name="ce")
    m.update()
    
    m.setObjective(quicksum(sigma[i][k]*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D])*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D]) for i in range(N) for k in range(knn))\
                      +weight*quicksum(sigma[i][k]*sum((x[i][j]-ce[k,j])*(x[i][j]-ce[k,j]) for j in range(D)) for i in range(N) for k in range(knn)), GRB.MINIMIZE)
    
    m.addConstrs(
                 (quicksum(ce[k,j]*sigma[i][k] for i in range(N)) == quicksum(sigma[i][k]*x[i][j] for i in range(N)) for k in range(knn) for j in range(D)),"c1")
   
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

    m.write('clustering.lp')        

    

    
    if m.status == GRB.Status.OPTIMAL:
        objective=m.objVal
        #print(' optimal objective1 is %g\n' % objective)
        
        #print ('m')
        for k in range(knn):
            temp2=[]
            for j in range(D):
                temp2.append(ce[k,j].x)
                #print ('%d th feature of cluster %d is %.4f' % (j+1,k+1,ce[k,j].x))
            x1.append(temp2)
                #print (ce[k,j])
    
        #print ('beta')
        for k in range(knn):
            temp3=[]
            for j in range(D+1):
                temp3.append(beta[k,j].x)
                #print ('%d th regression of cluster %d is %.4f' % (j+1,k+1,beta[k,j].x))
            x2.append(temp3)    
            
    return x1,x2
#function 2 fix other variables and calculate sigma
    
def optimizesigma(ce,beta,weight,knn):#update clustering
  
        
    m=Model('optimizex')
    sigma=m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
    x1=[]
    objective=0
    m.update()
  
    m.setObjective(quicksum(sigma[i,k]*pow(y[i]-sum(beta[k][j]*x[i][j] for j in range(D))-beta[k][D],2) for i in range(N) for k in range(knn))\
                   +weight*quicksum(sigma[i,k]*sum(pow(x[i][j]-ce[k][j],2) for j in range(D)) for i in range(N) for k in range(knn)), GRB.MINIMIZE)

             
    m.addConstrs(
               (quicksum(sigma[i,k] for k in range(knn)) == 1 for i in range(N)),"c2")
        
        
    m.addConstrs(
                  (quicksum(sigma[i,k] for i in range(N)) >= 1 for k in range(knn)),"c15")
    
    
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
                           
    return x1

def optimizeothers2(sigma,initialm,initialbeta,weight,knn):
    x1=[]
    x2=[]

    objective=0
    objective2=0
    m=Model('optimizeothers')
    beta = m.addVars(knn, D+1,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")
    ce = m.addVars(knn, D, lb=-MM,vtype=GRB.CONTINUOUS, name="ce")
    temp1= m.addVars(knn, D+1, lb=0,vtype=GRB.CONTINUOUS, name="temp1")
    temp2= m.addVars(knn, D, lb=0,vtype=GRB.CONTINUOUS, name="temp2")
    m.update()
    
        
    m.setObjective(quicksum(sigma[i][k]*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D])*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D]) for i in range(N) for k in range(knn))\
                   +weight*quicksum(sigma[i][k]*sum((x[i][j]-ce[k,j])*(x[i][j]-ce[k,j]) for j in range(D)) for i in range(N) for k in range(knn))\
                   +gamma*(quicksum(temp1[k,j] for k in range(knn) for j in range(D+1))+quicksum(temp2[k,j] for k in range(knn) for j in range(D))), GRB.MINIMIZE)
                   #+gamma*quicksum(sum((beta[k,j]-initialbeta[k][j])*(beta[k,j]-initialbeta[k][j]) for j in range (D+1))+ sum((ce[k,j]-initialm[k][j])*(ce[k,j]-initialm[k][j]) for j in range (D)) for k in range(knn)), GRB.MINIMIZE)
    
    m.addConstrs(
                  (temp1[k,j]>=beta[k,j]-initialbeta[k][j] for k in range(knn) for j in range(D+1)),"c15")
    
    m.addConstrs(
                  (temp1[k,j]>=initialbeta[k][j]-beta[k,j] for k in range(knn) for j in range(D+1)),"c15")
    
    m.addConstrs(
                  (temp2[k,j]>=ce[k,j]-initialm[k][j] for j in range (D) for k in range(knn)),"c15")
    
    m.addConstrs(
                  (temp2[k,j]>=initialm[k][j]-ce[k,j] for j in range (D) for k in range(knn)),"c15")

    
    
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

    m.write('clustering.lp')        

    
    if m.status == GRB.Status.OPTIMAL:
        objective=m.objVal
        #print(' optimal objective1 is %g\n' % objective)
        
        #print ('ce')
        for k in range(knn):
            temp1=[]
            for j in range(D):
                temp1.append(ce[k,j].x)
                #print ('%d th center of cluster %d is %.4f' % (j+1,k+1,ce[k,j].x))
            x1.append(temp1)
        
        #print ('beta')
        for k in range(knn):
            temp3=[]
            for j in range(D+1):
                temp3.append(beta[k,j].x)
                #print ('%d th regression of cluster %d is %.4f' % (j+1,k+1,beta[k,j].x))
            x2.append(temp3)
             
        objective2=objective-gamma*sum(sum((x2[k][j]-initialbeta[k][j])*(x2[k][j]-initialbeta[k][j]) for j in range (D+1)) + sum((x1[k][j]-initialm[k][j])*(x1[k][j]-initialm[k][j]) for j in range (D)) for k in range(knn))     
    return x1, x2, objective, objective2  
#function 2 fix other variables and calculate sigma
    
def optimizesigma2(ce,beta,weight,knn,initialsigma,variance):#update clustering
    
   
    m=Model('optimizex')
  
    sigma = m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
    temp3= m.addVars(N, knn, lb=0,vtype=GRB.CONTINUOUS, name="temp3")
    x1=[]
    objective=0
    
    m.update()

    
    m.setObjective(quicksum(math.log(variance[k])*quicksum(sigma[i,k] for i in range(N)) for k in range(knn))\
                   +quicksum(sigma[i,k]*pow(y[i]-sum(beta[k][j]*x[i][j] for j in range(D))-beta[k][D],2)/(2*pow(variance[k],2)) for i in range(N) for k in range(knn))\
                   +weight*quicksum(sigma[i,k]*sum(pow(x[i][j]-ce[k][j],2) for j in range(D)) for i in range(N) for k in range(knn))\
                   +gamma*quicksum(temp3[i,k] for k in range(knn) for i in range(N)), GRB.MINIMIZE)
                   #+gamma*quicksum((sigma[i,k]-initialsigma[i][k])*(sigma[i,k]-initialsigma[i][k]) for k in range(knn) for i in range(N)), GRB.MINIMIZE)
   
    m.addConstrs(
                  (temp3[i,k]>=sigma[i,k]-initialsigma[i][k] for i in range(N) for k in range(knn)),"c15")
    
    m.addConstrs(
                  (temp3[i,k]>=initialsigma[i][k]-sigma[i,k] for i in range(N) for k in range(knn)),"c15")
         
    m.addConstrs(
               (quicksum(sigma[i,k] for k in range(knn)) == 1 for i in range(N)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[i,k] for i in range(N)) >= 1 for k in range(knn)),"c15")
        
    #m.Params.TimeLimit = 600
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
    
      
    #if m.status == GRB.Status.OPTIMAL:
    objective=m.objVal

    
    for i in range(N):
        temp2=[]
        for k in range(knn):
            temp2.append(sigma[i,k].x)
        x1.append(temp2)
        
    #mipgap=m.MIPGap  
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

    for i in range(N):
        index = int(random.uniform(0, knn))
        clusterAssment[i] = index
        countt[index]=1
    #print(countt)
        
    for j in range(knn):  
        if countt[j]<=0.5:
            not_find=True
            break;
        
    return clusterAssment, not_find



dataSet1 = mat(x)

filenametimeexcel=r"C:\Users\...\algorithmtime(excel).txt"
filenameexcel=r"C:\Users\...\algorithmresult(excel).txt"
filenameiteration=r"C:\Users\...\algorithmiteration(excel).txt"
for counttt in range(10):
#    File = open(filenameresult, "a")
#    File.write('*****iteration:%d*****\n'% (counttt+1))   
#    File.close()
#    File = open(filenameCV, "a")
#    File.write('*****iteration:%d*****\n'% (counttt+1))   
#    File.close()
#    File = open(filenametime, "a")
#    File.write('*****iteration:%d*****\n'% (counttt+1))   
#    File.close()
    start = datetime.datetime.now()
    recordloss1=[[0]*TW for k in range(K)]
    recordloss2=[[0]*TW for k in range(K)]
    i_str=str(counttt+1)
    filenameresult=r"C:\Users\...\result(algorithm)_"+i_str+'.txt'
    
    filenameCV=r"C:\Users\...\CV(algorithm)_"+i_str+'.txt'
    filenametime=r"C:\Users\....\time(algorithm)_"+i_str+'.txt'
    
    
    
    for countk in range(K):#run the algorithm for Runtime times
        
#        if countk<=0.5:
#            knn=countk*5
#        else:
#            knn=countk*5-1
        
        knn=countk
        
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
            
            temp_sigma1=[[0]*(knn+1) for i in range(N)]#1st dimension=parts of cv
            temp_sigma2=[[0]*(knn+1) for i in range(N)]
       
            weight=0.1+0.1*ww
            for i in range(N):
                temp_sigma1[i][int(clusterAssment[i])]=1              
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
            temp_variance=[0]*(knn+1)
            x_ce2=[]
            x_beta2=[]
            loss2.append(MM)
            actualobj=0        
            obj3=0   
            while 1:
                (temp_ce2,temp_beta2)=optimizeothers(temp_sigma2,weight,knn+1)

                for k in range(knn+1):
                    temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))

            
                obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
                     +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
                     +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)

                
  
                if loss2[itr-1]-obj2>=tolerance:
      
                    x_sigma2=temp_sigma2
                    loss2.append(obj2)
                    x_ce2=temp_ce2
                    x_beta2=temp_beta2
                    x_variance=temp_variance

                    (temp_sigma2)=optimizesigma2(x_ce2,x_beta2,weight,knn+1,x_sigma2,x_variance)#obj given other variables
   
                else:
 
                    break
                
                itr=itr+1
                
            
            end2 = datetime.datetime.now()
                    
            ftime= (end2 - start2).total_seconds()

            File = open(filenameexcel, "a")                
            if ww>=3.5:
                File.write('%.3f\n'% loss2[-1])
            else:
                File.write('%.3f\t'% loss2[-1])
            
            File.close()
            
            File = open(filenametimeexcel, "a")                
            if ww>=3.5:
                File.write('%.3f\n'% ftime)
            else:
                File.write('%.3f\t'% ftime)
            
            File.close()
            
            File = open(filenameiteration, "a")                
            if ww>=3.5:
                File.write('%d\n'% itr)
            else:
                File.write('%d\t'% itr)
            
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
                #if x_z2[t]>=0.9:
                for j in range(D+1):
                    File.write('%d th regression of cluster %d is %.4f\n' % (j+1,k+1,x_beta2[k][j]))
                    
            File.write('variance\n')
            for k in range(knn+1):               
                File.write('The variance of %d th cluster is %.4f\n' % (k+1,x_variance[k]))
            
            File.close() 
          
            
            vam = Model("validation")  
            perror2=0   
            vasigma=[]  
            assign=vam.addVars(A, knn+1, vtype=GRB.BINARY, name='assign')
            vam.update()
                    
               
            vam.setObjective(quicksum(math.log(x_variance[k])*quicksum(assign[i,k] for i in range(A)) for k in range(knn+1))\
                             +quicksum((assign[i,k]*pow((vay[i]-sum(x_beta2[k][j]*vax[i][j] for j in range(D))-x_beta2[k][D]) ,2))/(2*pow(x_variance[k],2)) for k in range(knn+1) for i in range(A))\
                              +weight*quicksum(assign[i,k]*sum((vax[i][j]-x_ce2[k][j])*(vax[i][j]-x_ce2[k][j]) for j in range(D)) for i in range(A) for k in range(knn+1)), GRB.MINIMIZE)
                         
            vam.addConstrs(
                    (quicksum(assign[i,k] for k in range(knn+1)) == 1 for i in range(A)),"c21")
            
            vam.optimize()
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
                print ('assign')
                
                for i in range(A):
                    temp=[]
                    for k in range(knn+1):
                        temp.append(assign[i,k].x)

                    vasigma.append(temp)
                
                
                    
            perror2=sum((vay[i]-sum(vasigma[i][k]*(sum(x_beta2[k][j]*vax[i][j] for j in range(D))+x_beta2[k][D]) for k in range(knn+1)))\
                        *(vay[i]-sum(vasigma[i][k]*(sum(x_beta2[k][j]*vax[i][j] for j in range(D))+x_beta2[k][D]) for k in range(knn+1))) for i in range(A)) #L1 distance
            recordloss2[countk][ww]=perror2/A
            if vam.status == GRB.Status.OPTIMAL:

                File = open(filenameCV, "a")
                File.write('***K=%d, weight=%f,obj=%g***\n'% (knn+1,weight,vam.objVal))
                File.write('total error=%s\n' % str(perror2/A))
                File.write('assign\n')
                for i in range(A):
                     for k in range(knn+1):
                             if assign[i,k].x>=0.9:
                                File.write('data point:%d belong to cluster %d\n' % (i+1,k+1))
                File.close()       
            
#            File = open(filenameCVprint, "a")
#            File.write('iteration=%d, K=%d,weight=%f,total error=%s\n' % (counttt+1,knn+1,weight,str(perror2/A)))
#            File.close()
#            
            
            
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
       


 
           




    