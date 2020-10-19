
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

readfile1=r"...\data\data.xlsx"
book1 = xlrd.open_workbook(readfile1)

sh1= book1.sheet_by_name("Sheet2")

book2 = xlrd.open_workbook(readfile1)

sh2= book2.sheet_by_name("testing")

N=500
K=1
D=1
#TN=10
#A=TN-N
A=500
TW=1
TS=9
y=[]
x=[]
vay=[]
vax=[]
testy=[[0]*A for i in range(TS)]
testx=[[0]*A for i in range(TS)]

number=0
while number<=N-1:
    y.append(sh1.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh1.cell_value(number, j))
    x.append(dx)
    number=number+1
    
for i in range(TS):
    number=0
    while number<=A-1:
        testy[i][number]=sh2.cell_value(number, D+i*(D+1))
        dx=[]
        for j in range(D):
            dx.append(sh2.cell_value(number, j+i*(D+1)))
        testx[i][number]=dx
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


filenameresultAP2=r"C:\Users\...\result(AP2 cluster).txt"
filenameCVAP2=r"C:\Users\...\CV(AP2 cluster).txt"
filenametimeAP2=r"C:\Users\...\time(AP2 cluster).txt"
filenameCVprint=r"C:\Users\...\CV loss(cluster).txt"
filenameresultprint=r"C:\Users\...\result(excel)(cluster).txt" 

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
        
        for k in range(knn):
            for j in range(D):
                x1[k][j]=sum(sigma[i][k]*x[i][j] for i in range(N))/sum(sigma[i][k] for i in range(N))    
    
#        print ('m')
#        for k in range(knn):
#            temp2=[]
#            for j in range(D):
#                temp2.append(ce[k,j].x)
#                #print ('%d th feature of cluster %d is %.4f' % (j+1,k+1,ce[k,j].x))
#            x1.append(temp2)
#                #print (ce[k,j])
    
        print ('beta')
        for k in range(knn):
            temp3=[]
            for j in range(D+1):
                temp3.append(beta[k,j].x)
                #print ('%d th regression of cluster %d is %.4f' % (j+1,k+1,beta[k,j].x))
            x2.append(temp3)    
            
    return x1,x2

#def optimizeothers2(sigma,initialm,initialbeta,weight,knn):
#    x1=[]
#    x2=[]
#    #print('z',z)
#    #print('s',s)
#    #print('sigma',sigma)
#    #x4=[]
#    objective=0
#    objective2=0
#    m=Model('optimizeothers')
#    beta = m.addVars(knn, D+1,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")
#    #w = m.addVars(N, K, D, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#    ce = m.addVars(knn, D, lb=-MM,vtype=GRB.CONTINUOUS, name="ce")
#    temp1= m.addVars(knn, D+1, lb=0,vtype=GRB.CONTINUOUS, name="temp1")
#    temp2= m.addVars(knn, D, lb=0,vtype=GRB.CONTINUOUS, name="temp2")
#    #L = m.addVars(N, K, D,lb=-MM, ub=MM,vtype=GRB.CONTINUOUS, name='L')
#    m.update()
#    
#        
#    m.setObjective(quicksum(sigma[i][k]*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D])*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D]) for i in range(N) for k in range(knn))\
#                   +para*quicksum(beta[k,j]*beta[k,j] for j in range(D+1) for k in range(knn))\
#                   +weight*quicksum(sigma[i][k]*sum((x[i][j]-ce[k,j])*(x[i][j]-ce[k,j]) for j in range(D)) for i in range(N) for k in range(knn))\
#                   +gamma*(quicksum(temp1[k,j] for k in range(knn) for j in range(D+1))+quicksum(temp2[k,j] for k in range(knn) for j in range(D))), GRB.MINIMIZE)
#                   #+gamma*quicksum(sum((beta[k,j]-initialbeta[k][j])*(beta[k,j]-initialbeta[k][j]) for j in range (D+1))+ sum((ce[k,j]-initialm[k][j])*(ce[k,j]-initialm[k][j]) for j in range (D)) for k in range(knn)), GRB.MINIMIZE)
#    
#    m.addConstrs(
#                  (temp1[k,j]>=beta[k,j]-initialbeta[k][j] for k in range(knn) for j in range(D+1)),"c15")
#    
#    m.addConstrs(
#                  (temp1[k,j]>=initialbeta[k][j]-beta[k,j] for k in range(knn) for j in range(D+1)),"c15")
#    
#    m.addConstrs(
#                  (temp2[k,j]>=ce[k,j]-initialm[k][j] for j in range (D) for k in range(knn)),"c15")
#    
#    m.addConstrs(
#                  (temp2[k,j]>=initialm[k][j]-ce[k,j] for j in range (D) for k in range(knn)),"c15")
#
#    
#    
#    m.optimize()
#        
#    status = m.status
#    if status == GRB.Status.UNBOUNDED:
#        print('The model cannot be solved because it is unbounded')
#        #exit(0)
#    if status == GRB.Status.OPTIMAL:
#        print('The optimal objective is %g' % m.objVal)
#        #exit(0)
#    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
#        print('Optimization was stopped with status %d' % status)
#        #exit(0)
#
#    m.write('clustering.lp')        
#
#    
#    if m.status == GRB.Status.OPTIMAL:
#        objective=m.objVal
#        print(' optimal objective1 is %g\n' % objective)
#        
#        print ('ce')
#        for k in range(knn):
#            temp1=[]
#            for j in range(D):
#                temp1.append(ce[k,j].x)
#                print ('%d th center of cluster %d is %.4f' % (j+1,k+1,ce[k,j].x))
#            x1.append(temp1)
#        
#        print ('beta')
#        for k in range(knn):
#            temp3=[]
#            for j in range(D+1):
#                temp3.append(beta[k,j].x)
#                print ('%d th regression of cluster %d is %.4f' % (j+1,k+1,beta[k,j].x))
#            x2.append(temp3)
#             
#        objective2=objective-gamma*sum(sum((x2[k][j]-initialbeta[k][j])*(x2[k][j]-initialbeta[k][j]) for j in range (D+1)) + sum((x1[k][j]-initialm[k][j])*(x1[k][j]-initialm[k][j]) for j in range (D)) for k in range(knn))     
#    return x1, x2, objective, objective2  
##function 2 fix other variables and calculate sigma
#  
def assignment(ce,beta,weight,var,x0,y0):
    
    totalobj=math.log(var)+pow(y0-sum(beta[j]*x0[j] for j in range(D))-beta[D],2)/(2*pow(var,2))+weight*L2Distance(x0, np.mat(ce))
    return totalobj

def assignmentCLR(beta,x0,y0):
    
    totalobj=pow(y0-sum(beta[j]*x0[j] for j in range(D))-beta[D],2)
    return totalobj

def optimizesigmanew(ce,beta,weight,knn,variance,XX,YY):
    
    sigma=[[0]*knn for i in range(A)]
    distance=[[0]*knn for i in range(A)]
    for i in range(A):
        minDist  = 100000.0
        minIndex = 0
        for k in range(knn):
            distance[i][k] = assignment(ce[k],beta[k],weight,variance[k],XX[i],YY[i])

            if distance[i][k] < minDist:
                minDist  = distance[i][k]
                minIndex = k
        sigma[i][minIndex]=1
            
    return sigma

def optimizesigmanewCLR(beta,knn,XX,YY):
    
    sigma=[[0]*knn for i in range(A)]
    distance=[[0]*knn for i in range(A)]
    for i in range(A):
        minDist  = 100000.0
        minIndex = 0
        for k in range(knn):
            distance[i][k] = assignmentCLR(beta[k],XX[i],YY[i])

            if distance[i][k] < minDist:
                minDist  = distance[i][k]
                minIndex = k
        sigma[i][minIndex]=1
            
    return sigma    
def optimizesigmaCLR(ce,beta,weight,knn):#update clustering
    
   
    m=Model('optimizex')
  
    sigma = m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
#    temp3= m.addVars(N, knn, lb=0,vtype=GRB.CONTINUOUS, name="temp3")
    x1=[]
    objective=0
    
    m.update()
    
    m.setObjective(quicksum(sigma[i,k]*pow(y[i]-sum(beta[k][j]*x[i][j] for j in range(D))-beta[k][D],2) for i in range(N) for k in range(knn)), GRB.MINIMIZE)

    m.addConstrs(
               (quicksum(sigma[i,k] for k in range(knn)) == 1 for i in range(N)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[i,k] for i in range(N)) >= 1 for k in range(knn)),"c15")
        
    #m.Params.TimeLimit = 600
    m.optimize()
    status = m.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
        #exit(0)
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
        #exit(0)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)
        #exit(0)

    m.write('optimizex.lp')
    print('')

      
    #if m.status == GRB.Status.OPTIMAL:
    objective=m.objVal
    print(' optimal objective2 is %g\n' % objective)
    
    print ('sigma') 
    for i in range(N):
        temp2=[]
        for k in range(knn):
            temp2.append(sigma[i,k].x)

        x1.append(temp2)
        
    #mipgap=m.MIPGap  
    return x1

def optimizesigma(ce,beta,weight,knn,initialsigma,variance):#update clustering
    
   
    m=Model('optimizex')
  
    sigma = m.addVars(N, knn, vtype=GRB.BINARY, name='sigma')
#    temp3= m.addVars(N, knn, lb=0,vtype=GRB.CONTINUOUS, name="temp3")
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
        
    #m.Params.TimeLimit = 600
    m.optimize()
    status = m.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
        #exit(0)
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
        #exit(0)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)
        #exit(0)

    m.write('optimizex.lp')
    print('')

      
    #if m.status == GRB.Status.OPTIMAL:
    objective=m.objVal
    print(' optimal objective2 is %g\n' % objective)
    
    print ('sigma') 
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
        
    for j in range(knn):  
        if countt[j]<=0.5:
            not_find=True
            break;
        
    return clusterAssment, not_find


dataSet1 = mat(x)
Time=[[0]*TW for k in range(K)]
minimumobj=[[10000]*TW for k in range(K)]
minimumerror=[[10000]*TW for k in range(K)]
recordcounttt=[[10000]*TW for k in range(K)]
recordtime=[[[0]*10 for i in range(TW)] for k in range(K)]
recordbeta=[[0]*(D+1) for k in range(4)]
recordce=[[0]*D for k in range(4)]
recordsigma=[[0]*(4) for i in range(N)]


for counttt in range(1):
    groupx=[]
    groupy=[]
    groupnewy=[]
    counting=[]
    newy=[0]*(N)
    recordloss1=[[0]*TW for k in range(K)]
    recordloss2=[[0]*TW for k in range(K)]
    File = open(filenameresultAP2, "a")
    File.write('iteration:%d\n' % (counttt+1))
    File.close()
#    
#    File = open(filenameCVAP2, "a")
#    File.write('iteration:%d\n' % (counttt+1))
#    File.close()
#        
#    File = open(filenametimeAP2, "a")
#    File.write('iteration:%d\n' % (counttt+1))
#    File.close()
#    
#    File = open(filenameCVprint, "a")
#    File.write('iteration:%d\n' % (counttt+1))
#    File.close()
#    
#    File = open(filenameresultprint, "a")
#    File.write('iteration:%d\n' % (counttt+1))
#    File.close()
    for countk in range(K):#run the algorithm for Runtime times
        knn=3
        f1=True
        while f1:
            (clusterAssment ,f1) = initialassignment(dataSet1, knn+1)   
        for ww in range(TW):
            
            weight=0.05
#            temp_sigma1=[[0]*(knn+1) for i in range(N)]#1st dimension=parts of cv
            #temp_sigma2=[[0]*(N) for i in range(N)]
            temp_sigma2=[[0]*(knn+1) for i in range(N)]
            
            for i in range(N):             
                temp_sigma2[i][int(clusterAssment[i])]=1
            
            
            temp_variance=[0]*(knn+1)
            start2 = datetime.datetime.now()
            itr = 1
            loss2=[]
#            x_ce2=[]
#            x_beta2=[]
            loss2.append(MM)
            actualobj=0        
            obj3=0   
            while 1:
#                if itr<=1:
                (temp_ce2,temp_beta2)=optimizeothers(temp_sigma2,weight,knn+1)
#                else:
#                    (temp_ce2,temp_beta2,obj2,obj3)=optimizeothers2(temp_sigma2,x_ce2,x_beta2,weight,knn+1)
                for k in range(knn+1):
                    temp_variance[k]=max(1,np.sqrt(sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2) for i in range(N))/sum(temp_sigma2[i][k] for i in range(N))))
                
                obj2=sum(math.log(temp_variance[k])*sum(temp_sigma2[i][k]  for i in range(N))  for k in range(knn+1))\
                 +sum(temp_sigma2[i][k]*pow(y[i]-sum(temp_beta2[k][j]*x[i][j] for j in range(D))-temp_beta2[k][D],2)/(2*pow(temp_variance[k],2))  for i in range(N) for k in range(knn+1))\
                 +weight*sum(temp_sigma2[i][k]*sum(pow(x[i][j]-temp_ce2[k][j],2)  for j in range(D)) for k in range(knn+1) for i in range(N))+(N/2)*math.log(2*math.pi)
                if (loss2[itr-1]-obj2)/obj2>=tolerance:  
       
                    x_sigma2=temp_sigma2
                    loss2.append(obj2)
                    x_ce2=temp_ce2
                    x_beta2=temp_beta2
                    x_variance=temp_variance  
                    (temp_sigma2)=optimizesigma(x_ce2,x_beta2,weight,knn+1,x_sigma2,x_variance)#obj given other variables
                    
                    
                else:

                    break
                
                itr=itr+1
                
                
            end2 = datetime.datetime.now()
            ftime= (end2 - start2).total_seconds()
            perror=sum((y[i]-sum(x_sigma2[i][k]*(sum(x_beta2[k][j]*x[i][j] for j in range(D))+x_beta2[k][D]) for k in range(knn+1)))\
                *(y[i]-sum(x_sigma2[i][k]*(sum(x_beta2[k][j]*x[i][j] for j in range(D))+x_beta2[k][D]) for k in range(knn+1))) for i in range(N)) #L1距离
            recordtime[countk][ww][counttt]=ftime
            if loss2[-1]<=minimumobj[countk][ww]:
                minimumobj[countk][ww]=loss2[-1]
                minimumerror[countk][ww]=perror/A
                recordcounttt[countk][ww]=counttt
                recordbeta=x_beta2
                recordce=x_ce2
                recordsigma=x_sigma2
                recordvariance=x_variance
#            for i in range(N):
#                for k in range(knn+1):
#                    if x_sigma2[i][k]>=0.9:
#                        for j in range(D):
#                            newy[i]+=x[i][j]*x_beta2[k][j]
#                        newy[i]=newy[i]+x_beta2[k][D]
#            #recordtime2[countk][ww]=end2-start2
#            #totaltime2[counttt]+=end2-start2
#            for k in range(knn+1):
#                number=0
#                for i in range(N):
#                    if x_sigma2[i][k]>=0.9:
#                        number+=1
#                        groupx.append(x[i][0])
#                        groupy.append(y[i])
#                        groupnewy.append(newy[i])
#                counting.append(number)
#            f1 = plt.figure(1)
#            p1 = plt.scatter(groupx[:counting[0]], groupy[:counting[0]], marker = 'o', color='r', label='1', s = 15)
#            p2 = plt.scatter(groupx[counting[0]:counting[0]+counting[1]], groupy[counting[0]:counting[0]+counting[1]], marker = 'o', color='#808080', label='2', s = 15)
#            p3 = plt.scatter(groupx[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], groupy[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], marker = 'o', color='b', label='3', s = 15)
#            p4 = plt.scatter(groupx[counting[0]+counting[1]+counting[2]:500], groupy[counting[0]+counting[1]+counting[2]:500], marker = 'o', color='#008000', label='4', s = 15)
#            plt.legend(loc = 'upper right')
#            plt.savefig(r'C:\Users\...\original'+str(weight)+'_'+str(counttt+1)+'.png')
#            plt.show() 
#            
#            f2 = plt.figure(2)
#            p1 = plt.scatter(groupx[:counting[0]], groupnewy[:counting[0]], marker = 'o', color='r', label='1', s = 15)
#            p2 = plt.scatter(groupx[counting[0]:counting[0]+counting[1]], groupnewy[counting[0]:counting[0]+counting[1]], marker = 'o', color='#808080', label='2', s = 15)
#            p3 = plt.scatter(groupx[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], groupnewy[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], marker = 'o', color='b', label='3', s = 15)
#            p4 = plt.scatter(groupx[counting[0]+counting[1]+counting[2]:500], groupnewy[counting[0]+counting[1]+counting[2]:500], marker = 'o', color='#008000', label='4', s = 15)
#            plt.legend(loc = 'upper right')
#            plt.savefig(r'C:\Users\...\predict'+str(weight)+'_'+str(counttt+1)+'.png')
#            plt.show() 
#            
#            File = open(filenameresultprint, "a")
#            File.write('AP2: K=%d,weight=%f,total error=%s\n' % (knn+1,weight,loss2[-1]))
#            File.write('AP2: K=%d,weight=%f,regression error=%s\n' % (knn+1,weight,perror))
#            File.close()
#            
#            
#            File = open(filenametimeAP2, "a")
#            File.write('iteration:%d, computational time when k=%d,weight=%f: %f\n' % (counttt+1,knn+1,weight,ftime))
#            File.close()
#            
#            File = open(filenameresultAP2, "a")
#            File.write('AP2: K=%d,weight=%f\n' % (knn+1,weight))
#            File.write('obj=%g\n'% loss2[-1])
#            File.write('sigma\n')
#            for i in range(N):
#                for k in range(knn+1):
#                      if x_sigma2[i][k]>=0.9:
#                             File.write('cluster:%d contain: %d\n' % (k+1,i+1))
#                             
#            File.write('m\n')
#            for k in range(knn+1):
#                for j in range(D):
#                    File.write('%d th center of cluster %d is %.4f\n' % (j+1,k+1,x_ce2[k][j]))
#            
#           
#            File.write('beta\n')
#            for k in range(knn+1):
#                #if x_z2[t]>=0.9:
#                for j in range(D+1):
#                    File.write('%d th regression of cluster %d is %.4f\n' % (j+1,k+1,x_beta2[k][j]))
#            
#            File.close() 
          
File = open(filenameresultprint, "a")
for countk in range(K):
    for ww in range(TW):
        File.write('K=%d,weight=%f,minimum error=%s\n' % (knn+1,weight,minimumobj[countk][ww]))
        File.write('K=%d,weight=%f,minimum regression error=%s\n' % (knn+1,weight,minimumerror[countk][ww]))
File.close()

for i in range(N):
    for k in range(knn+1):
        if recordsigma[i][k]>=0.9:
            for j in range(D):
                newy[i]+=x[i][j]*recordbeta[k][j]
            newy[i]=newy[i]+recordbeta[k][D]

for k in range(knn+1):
    number=0
    for i in range(N):
        if recordsigma[i][k]>=0.9:
            number+=1
            groupx.append(x[i][0])
            groupy.append(y[i])
            groupnewy.append(newy[i])
    counting.append(number)
f1 = plt.figure(1)
#p1 = plt.scatter(groupx[:counting[0]], groupy[:counting[0]], marker = 'o', color='r', label='1', s = 15)
#p2 = plt.scatter(groupx[counting[0]:counting[0]+counting[1]], groupy[counting[0]:counting[0]+counting[1]], marker = 'o', color='#808080', label='2', s = 15)
#p3 = plt.scatter(groupx[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], groupy[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], marker = 'o', color='b', label='3', s = 15)
#p4 = plt.scatter(groupx[counting[0]+counting[1]+counting[2]:500], groupy[counting[0]+counting[1]+counting[2]:500], marker = 'o', color='#008000', label='4', s = 15)
p1 = plt.scatter(groupx[:counting[0]], groupy[:counting[0]], marker = 'o', color='r', label='Cluster 1', s = 15)
p2 = plt.scatter(groupx[counting[0]:counting[0]+counting[1]], groupy[counting[0]:counting[0]+counting[1]], marker = 'x', color='#808080', label='Cluster 2', s = 20)
p3 = plt.scatter(groupx[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], groupy[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], marker = '|', color='b', label='Cluster 3', s = 20)
p4 = plt.scatter(groupx[counting[0]+counting[1]+counting[2]:500], groupy[counting[0]+counting[1]+counting[2]:500], marker = '_', color='#008000', label='Cluster 4', s = 20)
plt.legend(loc = 'upper left')
plt.savefig(r'C:\Users\...\original'+'_'+str(weight)+'.png')
plt.show() 

f2 = plt.figure(2)
#p1 = plt.scatter(groupx[:counting[0]], groupnewy[:counting[0]], marker = 'o', color='r', label='1', s = 15)
#p2 = plt.scatter(groupx[counting[0]:counting[0]+counting[1]], groupnewy[counting[0]:counting[0]+counting[1]], marker = 'o', color='#808080', label='2', s = 15)
#p3 = plt.scatter(groupx[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], groupnewy[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], marker = 'o', color='b', label='3', s = 15)
#p4 = plt.scatter(groupx[counting[0]+counting[1]+counting[2]:500], groupnewy[counting[0]+counting[1]+counting[2]:500], marker = 'o', color='#008000', label='4', s = 15)
p1 = plt.scatter(groupx[:counting[0]], groupnewy[:counting[0]], marker = 'o', color='r', label='Cluster 1', s = 15)
p2 = plt.scatter(groupx[counting[0]:counting[0]+counting[1]], groupnewy[counting[0]:counting[0]+counting[1]], marker = 'x', color='#808080', label='Cluster 2', s = 20)
p3 = plt.scatter(groupx[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], groupnewy[counting[0]+counting[1]:counting[0]+counting[1]+counting[2]], marker = '|', color='b', label='Cluster 3', s = 20)
p4 = plt.scatter(groupx[counting[0]+counting[1]+counting[2]:500], groupnewy[counting[0]+counting[1]+counting[2]:500], marker = '_', color='#008000', label='Cluster 4', s = 20)
plt.legend(loc = 'upper left')
plt.savefig(r'C:\Users\...\predict'+'_'+str(weight)+'.png')
plt.show() 
    
            
for ts in range(TS):         
    vam = Model("validation")  
    perror2=0   
    vasigma=[]  
    assign=vam.addVars(A, knn+1, vtype=GRB.BINARY, name='assign')
    vam.update()
    
    vam.setObjective(sum(math.log(recordvariance[k])*sum(assign[i,k]  for i in range(A))  for k in range(knn+1))\
                      +sum((testy[ts][i]*assign[i,k]-assign[i,k]*(sum(recordbeta[k][j]*testx[ts][i][j] for j in range(D))+recordbeta[k][D]))*(testy[ts][i]*assign[i,k]-assign[i,k]*(sum(recordbeta[k][j]*testx[ts][i][j] for j in range(D))+recordbeta[k][D])) for k in range(knn+1) for i in range(A))\
                      +weight*sum(assign[i,k]*sum((testx[ts][i][j]-recordce[k][j])*(testx[ts][i][j]-recordce[k][j]) for j in range(D)) for i in range(A) for k in range(knn+1)), GRB.MINIMIZE)
    
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
        
        
    perror2=sum((testy[ts][i]-sum(vasigma[i][k]*(sum(recordbeta[k][j]*testx[ts][i][j] for j in range(D))+recordbeta[k][D]) for k in range(knn+1)))\
                *(testy[ts][i]-sum(vasigma[i][k]*(sum(recordbeta[k][j]*testx[ts][i][j] for j in range(D))+recordbeta[k][D]) for k in range(knn+1))) for i in range(A)) #L1 distance
    print(perror2)
    #recordloss2[knn][ww]=perror2/A
    recordloss2[0][0]=perror2/A
    if vam.status == GRB.Status.OPTIMAL:
        File = open(filenameCVAP2, "a")
        File.write('***testing set=%d, K=%d, weight=%f,obj=%g***\n'% (ts+1, knn+1,weight,vam.objVal))
        File.write('total error=%s\n' % str(perror2/A))
        File.write('assign\n')
        for i in range(A):
             for k in range(knn+1):
                     if assign[i,k].x>=0.9:
                        File.write('data point:%d belong to cluster %d\n' % (i+1,k+1))
        File.close()       
    
    File = open(filenameCVprint, "a")
    File.write('testing set=%d,K=%d,weight=%f,total error=%s\n' % (ts+1,knn+1,weight,str(perror2/A)))
    File.close()
                         

