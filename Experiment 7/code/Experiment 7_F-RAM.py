
from PIL import Image, ImageDraw
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
from  sklearn.svm  import LinearSVC
from sklearn import neighbors 
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score  
import pickle



y=[]
x=[]
vay=[]
vax=[]

    
N=3823
N1=376
N2=389
N3=380
N4=389
N5=387
N6=376
N7=377
N8=387
N9=380
N10=382
A1=178
A2=182
A3=177
A4=183
A5=181
A6=182
A7=181
A8=179
A9=174
A10=180
A=1797
D=64
J=10
num_classes=10
K=5
TW=1
TS=10
readfile1=r"...\data\training.xlsx"
book1 = xlrd.open_workbook(readfile1)
sh1_training= book1.sheet_by_name("0")
sh2_training= book1.sheet_by_name("1")
sh3_training= book1.sheet_by_name("2")
sh4_training= book1.sheet_by_name("3")
sh5_training= book1.sheet_by_name("4")
sh6_training= book1.sheet_by_name("5")
sh7_training= book1.sheet_by_name("6")
sh8_training= book1.sheet_by_name("7")
sh9_training= book1.sheet_by_name("8")
sh10_training= book1.sheet_by_name("9")


readfile2=r"...\data\testing.xlsx"
book2 = xlrd.open_workbook(readfile2)
sh1_testing= book2.sheet_by_name("0")
sh2_testing= book2.sheet_by_name("1")
sh3_testing= book2.sheet_by_name("2")
sh4_testing= book2.sheet_by_name("3")
sh5_testing= book2.sheet_by_name("4")
sh6_testing= book2.sheet_by_name("5")
sh7_testing= book2.sheet_by_name("6")
sh8_testing= book2.sheet_by_name("7")
sh9_testing= book2.sheet_by_name("8")
sh10_testing= book2.sheet_by_name("9")

number=0
while number<=N1-1:
    y.append(sh1_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh1_training.cell_value(number, j))
    x.append(dx)
    number=number+1
    
    
number=0
while number<=N2-1:
    y.append(sh2_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh2_training.cell_value(number, j))
    x.append(dx)
    number=number+1  

    
number=0
while number<=N3-1:
    y.append(sh3_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh3_training.cell_value(number, j))
    x.append(dx)
    number=number+1

    
number=0
while number<=N4-1:
    y.append(sh4_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh4_training.cell_value(number, j))
    x.append(dx)
    number=number+1  

    
    
number=0
while number<=N5-1:
    y.append(sh5_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh5_training.cell_value(number, j))
    x.append(dx)
    number=number+1

    
number=0
while number<=N6-1:
    y.append(sh6_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh6_training.cell_value(number, j))
    x.append(dx)
    number=number+1    

    
    
number=0
while number<=N7-1:
    y.append(sh7_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh7_training.cell_value(number, j))
    x.append(dx)
    number=number+1

number=0
while number<=N8-1:
    y.append(sh8_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh8_training.cell_value(number, j))
    x.append(dx)
    number=number+1    

    
    
number=0
while number<=N9-1:
    y.append(sh9_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh9_training.cell_value(number, j))
    x.append(dx)
    number=number+1
 
    
number=0
while number<=N10-1:
    y.append(sh10_training.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh10_training.cell_value(number, j))
    x.append(dx)
    number=number+1
    
    

number=0
while number<=A1-1:
    vay.append(sh1_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh1_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1
    
    
number=0
while number<=A2-1:
    vay.append(sh2_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh2_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1

number=0
while number<=A3-1:
    vay.append(sh3_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh3_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1

number=0
while number<=A4-1:
    vay.append(sh4_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh4_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1
    
number=0
while number<=A5-1:
    vay.append(sh5_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh5_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1   
    
number=0
while number<=A6-1:
    vay.append(sh6_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh6_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1  
    
number=0
while number<=A7-1:
    vay.append(sh7_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh7_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1   
    
    
number=0
while number<=A8-1:
    vay.append(sh8_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh8_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1

number=0
while number<=A9-1:
    vay.append(sh9_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh9_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1
number=0
while number<=A10-1:
    vay.append(sh10_testing.cell_value(number, D))
    dx=[]
    for j in range(D):
        dx.append(sh10_testing.cell_value(number, j))
    vax.append(dx)
    number=number+1     




##digital
#N=1115
#TN1=161
#TN2=162
#TN3=159
#TN4=159
#TN5=161
#TN6=159
#TN7=160
#TN8=159
#TN9=155
#TN10=159
#N1=113
#N2=113
#N3=111
#N4=111
#N5=113
#N6=111
#N7=112
#N8=111
#N9=109
#N10=111
#D=256
#PI=16
##pixel=16
#J=10
#num_classes=10
#A=479
#TN=1594
#TW=1
#K=5
#TS=10
#readfile1=r"...\data\digital.xlsx"
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
#sh10= book1.sheet_by_name("9")
#batch_size = 250

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
#number=0
#while number<=N10-1:
#    y.append(sh10.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh10.cell_value(number, j))
#    x.append(dx)
#    number=number+1  
#while number>=N10 and number<=TN10-1:
#    vay.append(sh10.cell_value(number, D))
#    dx=[]
#    for j in range(D):
#        dx.append(sh10.cell_value(number, j))
#    vax.append(dx)
#    number=number+1    

  
MM=sys.float_info.max
para=0.01
tolerance=0.01
gamma=0.001

absdimax = []
dimax = []
dimin = []
extrax= []
MM1 = math.sqrt(sum(y[i]**2 for i in range(N))/para)


def optimizebeta(x_extraction,weight,knn):# input x' to optimize hyper-parameter
#    s1 = datetime.datetime.now()    
    X=[[] for k in range(knn)]
    XT=[[] for k in range(knn)]
    Y=[[] for k in range(knn)]
    x1=[[0] for k in range(knn)]
    x3=[[0] for k in range(knn)]
    

                
    Y=np.mat(y).reshape((len(y)),1)           

    XT=np.mat(x_extraction).T 
    X=np.mat(x_extraction)

    B=np.dot(np.dot(np.linalg.inv(np.dot(XT,X)),XT),Y)

    x1=np.array(B)
    x3=(sum(Y)-sum(np.dot(X,B)))/len(Y)
        
   
    return x1,x3

def optimizebeta2(x_extraction,weight,knn):# input x' to optimize hyper-parameter

    x1=[0]*knn 
    x2=0
    
    #x4=[]
    objective=0
    m=Model('optimizebeta2')
    beta = m.addVars(knn,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")
    beta0 = m.addVar(lb=-MM, vtype=GRB.CONTINUOUS, name="beta0")
    #w = m.addVars(N, K, D, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#    ce = m.addVars(knn, D, lb=-MM,vtype=GRB.CONTINUOUS, name="ce")
    #L = m.addVars(N, K, D,lb=-MM, ub=MM,vtype=GRB.CONTINUOUS, name='L')
    m.update()
    
#    m.setObjective(quicksum(sigma[i][k]*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D])*(y[i]-sum(beta[k,j]*x[i][j] for j in range(D))-beta[k,D]) for i in range(N) for k in range(knn))\
#                   +weight*quicksum(sigma[i][k]*sum((x[i][j]-ce[k,j])*(x[i][j]-ce[k,j]) for j in range(D)) for i in range(N) for k in range(knn)), GRB.MINIMIZE)
    
    m.setObjective(quicksum((y[i]-sum(beta[k]*x_extraction[i][k] for k in range(knn))-beta0)*(y[i]-sum(beta[k]*x_extraction[i][k] for k in range(knn))-beta0) for i in range(N) ), GRB.MINIMIZE)
#    m.addConstrs(
#                 (quicksum(ce[k,j]*sigma[i][k] for i in range(N)) == quicksum(sigma[i][k]*x[i][j] for i in range(N)) for k in range(knn) for j in range(D)),"c1")
#   
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

    m.write('clustering.lp')        

    
    if m.status == GRB.Status.OPTIMAL:
        
        for k in range(knn):
            x1[k]=beta[k].x   

        x2=beta0.x

    return x1,x2


def optimizelinearsvc(x_training_extraction,x_testing_extraction,knn):#using the package
    

    lin_clf = LinearSVC()
    start_training = datetime.datetime.now()
    lin_clf.fit(x_training_extraction,y)
    end_training = datetime.datetime.now()
    ftime_training=(end_training - start_training).total_seconds()

    
#    a=lin_clf.predict(x_testing_extraction)
#    correctcount=0
#    for i in range(A):
#        if vay[i]==a[i]:
#            correctcount=correctcount+1
#    print(correctcount/A)
    a_training=accuracy_score(y, lin_clf.predict(x_training_extraction))
    start_testing = datetime.datetime.now()
    a_testing=accuracy_score(vay, lin_clf.predict(x_testing_extraction))
    end_testing = datetime.datetime.now()
    ftime_testing=(end_testing - start_testing).total_seconds()
    
    countacc=[0]*J
    for j in range(J):
        for i in range(N):
            if y[i]==j:
                if lin_clf.predict(x_training_extraction)[i]==y[i]:
                    countacc[j]=countacc[j]+1
                    
    countacc_test=[0]*J
    for j in range(J):
        for i in range(A):
            if vay[i]==j:
                if lin_clf.predict(x_testing_extraction)[i]==vay[i]:
                    countacc_test[j]=countacc_test[j]+1
#    print ('accuracy', accuracy_score(vay, lin_clf.predict(x_testing_extraction)))
    
    return a_training, a_testing, ftime_training, ftime_testing, countacc, countacc_test

def optimizebetasvm(x_extraction,weight,knn): #MSVM
    x1=[0]*J
    x2=[]
    s1 = datetime.datetime.now()
    m=Model('optimizeothers')
    beta = m.addVars(J, knn,lb=-MM, vtype=GRB.CONTINUOUS, name="beta")
    beta0 = m.addVars(J, lb=-MM, vtype=GRB.CONTINUOUS, name="beta0")
    temp = m.addVars(N, J,lb=0, vtype=GRB.CONTINUOUS, name="temp")
    m.update()
    
    m.setObjective(quicksum(temp[i,j] for j in range(J) for i in range(N)), GRB.MINIMIZE)
     
    m.addConstrs(
                 (temp[i,j]>=sum(beta[j,k]*x_extraction[i][k] for k in range(knn))+beta0[j]-(sum(beta[int(y[i]),k]*x_extraction[i][k] for k in range(knn))+beta0[int(y[i])])+1   for i in range(N) for j in range(J)),"c1")

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

    m.write('clustering.lp')        

    
    if m.status == GRB.Status.OPTIMAL:
        objective=m.objVal
        for j in range(J):
            x1[j]=beta0[j].x

        for j in range(J):
            temp1=[]
            for k in range(knn):
                temp1.append(beta[j,k].x)
                #print ('%d th feature of cluster %d is %.4f' % (j+1,k+1,ce[k,j].x))
            x2.append(temp1)


#    print('x1')
#    for k in range(knn):
#        for d in range(D):
#            print('k=%d,d=%d: %f'%(k+1,d+1,x1[k][d]))
#            
#    print('beta')
#    for k in range(knn):
#        for j in range(J):
#            for d in range(D+1):
#                print('k=%d,j=%d,d=%d: %f'%(k+1,j+1,d+1,beta[k,j,d].x))
#                
#    print('temp')
#    for i in range(N):
#        for k in range(knn):
#            for j in range(J):
#                print('i=%d,k=%d,j=%d: %f'%(i+1,k+1,j+1,temp[i,k,j].x))
    e1 = datetime.datetime.now()   
     
    return x2,x1


def optimizeclassification(x_extraction,weight,knn,initialsigma):# input x' to optimize classification using similarity function
    x1=[]
    m=Model('optimizeclassification')
    sigma = m.addVars(knn,D, vtype=GRB.BINARY, name='sigma')
    m.update()
    
#    m.setObjective(quicksum(sigma[k,p]*np.linalg.norm([temp1[p] for temp1 in x]-[temp2[k] for temp2 in x_extraction],ord=2)**2  for k in range(knn) for p in range(D)), GRB.MINIMIZE)
    m.setObjective(quicksum(sigma[k,p]*sum((x[i][p]-x_extraction[i][k])**2 for i in range(N))  for k in range(knn) for p in range(D))\
                      +gamma*quicksum((1-initialsigma[k][p])*sigma[k,p]+initialsigma[k][p]*(1-sigma[k,p]) for k in range(knn) for p in range(D)), GRB.MINIMIZE)
    m.addConstrs(
               (quicksum(sigma[k,p] for k in range(knn)) == 1 for p in range(D)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[k,p] for p in range(D)) >= 1 for k in range(knn)),"c15")
   
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

        for k in range(knn):
            temp3=[]
            for p in range(D):
                temp3.append(sigma[k,p].x)

            x1.append(temp3)    
    e1 = datetime.datetime.now()   
     
    return x1
    
  
def optimizeclassificationinitial(x_extraction,weight,knn):# input x' to optimize classification using similarity function
    x1=[]
    m=Model('optimizeclassification')
    sigma = m.addVars(knn,D, vtype=GRB.BINARY, name='sigma')
#    sigma = m.addVars(knn,D, lb = 0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name='sigma')
    m.update()
    
#    m.setObjective(quicksum(sigma[k,p]*np.linalg.norm([temp1[p] for temp1 in x]-[temp2[k] for temp2 in x_extraction],ord=2)**2  for k in range(knn) for p in range(D)), GRB.MINIMIZE)
    m.setObjective(quicksum(sigma[k,p]*sum((x[i][p]-x_extraction[i][k])**2 for i in range(N)) for k in range(knn) for p in range(D)), GRB.MINIMIZE)
    m.addConstrs(
               (quicksum(sigma[k,p] for k in range(knn)) == 1 for p in range(D)),"c1")
    
    m.addConstrs(
                  (quicksum(sigma[k,p] for p in range(D)) >= 1 for k in range(knn)),"c15")
   
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

        for k in range(knn):
            temp3=[]
            for p in range(D):
                temp3.append(sigma[k,p].x)

            x1.append(temp3)    
    e1 = datetime.datetime.now()   
     
    return x1
    
def optimizesigma2(weight,knn,sigma):#update clustering
    
    x2=[[0]*knn for i in range(N)]
    sumsigma = np.array(sigma).sum(axis = 1)

    for i in range(N):
        for k in range(knn):
#            x2[i][k]=sum(sigma[k][p]*x[i][p] for p in range(D))/sum(sigma[k][p] for p in range(D)) 
            x2[i][k]=np.dot (np.array(sigma[k]), np.array(x[i]))/ int (sumsigma[k])
#            print('extracted x[%d][%d]=%f'%(i+1, k+1, x2[i][k]))
    return x2
    



  
def L1Distance(vector1, vector2): # L1 distance
    t = sum(abs(vector2 - vector1))
    return t


def L2Distance(vector1, vector2): 
    t=np.sum(np.square(vector1 - vector2))
    return t


def initialassignment(dataSet, knn):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((knn, 1)))
    not_find = False
    countt=[0]*D

    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    for k in range(knn):
        index = int(random.uniform(0, D))
        if countt[index]<=0.5:
            clusterAssment[k] = index
            countt[index]=1
        else:
            not_find=True
            break;

        
#    print('initial clusterAssment', clusterAssment)   
    return clusterAssment, not_find



dataSet1 = mat(x)

filenameresult=r"...\optdigital\result(algorithm_mean_package).txt"
filenameCV=r"...\optdigital\CV(algorithm_mean_package).txt"
filenametime=r"...\optdigital\time(algorithm_mean_package).txt"

#filenameresult=r"...\digital\result(algorithm_mean_package).txt"
#filenameCV=r"...\digital\CV(algorithm_mean_package).txt"
#filenametime=r"...\digital\time(algorithm_mean_package).txt"

maximumtime=[[0]*TW for k in range(K)]
averageacc=[[0]*TW for k in range(K)]
averagetime=[[0]*TW for k in range(K)]
testtime=[[0]*TW for k in range(K)]
maximumacc=[[0]*TW for k in range(K)] 
recordcounttt=[[0]*TW for k in range(K)]
#recordx_sigma=[[0]*TW for k in range(K)]
#recordx_beta=[[0]*TW for k in range(K)]
#recordx_ce=[[0]*TW for k in range(K)]
record_training_error=[[0]*TW for k in range(K)]
averageacc_training=[[0]*TW for k in range(K)]
averagetime_training=[[0]*TW for k in range(K)]
recordx_training_num=[[0]*TW for k in range(K)]
recordx_testing_num=[[0]*TW for k in range(K)]
for counttt in range(TS):

    start = datetime.datetime.now()
    i_str=str(counttt+1)

   
    
    for countk in range(K):#run the algorithm for Runtime times
        
#        if countk<=0.5:
#            knn=countk*5
#        else:
#            knn=countk*5-1
        
#        knn=4*(countk+2)-1 # for digital
        knn=8*(countk+1)-1 # for optdigital
        tempsigma=[[0]*D for k in range(knn+1)]
        f1=True
        #f2=True
        while f1:
            (clusterAssment ,f1) = initialassignment(dataSet1, knn+1)   
        
  
        for ww in range(TW):
            temp_x_extraction=[[0]*(knn+1) for i in range(N)]
            temp_z=[[0]*D for k in range(knn+1)]
#            temp_sigma1=[[0]*(knn+1) for i in range(N)]#1st dimension=parts of cv
#            temp_sigma2=[[0]*(knn+1) for i in range(N)]
            weight=2*ww

            for i in range(N):
                for k in range(knn+1):
                    temp_x_extraction[i][k]=x[i][int(clusterAssment[k])]
       
            for k in range(knn+1):
                temp_z[k][int(clusterAssment[k])]=1

     
            start2 = datetime.datetime.now()
            
            itr = 1
            loss2=[]
            x_sigma=[]
            loss2.append(MM)
            actualobj=0        
            obj3=0   
            obj2=0
            obj1oss=0
            objdistance=0
            while 1:
#                
                if itr<=1:
                    (temp_sigma)=optimizeclassificationinitial(temp_x_extraction,weight,knn+1)
                
                else:
                    
                    (temp_sigma)=optimizeclassification(temp_x_extraction,weight,knn+1,x_sigma)
              
               
                obj2=sum(temp_sigma[k][p]*sum(pow(x[i][p]-temp_x_extraction[i][k],2) for i in range(N)) for k in range(knn+1) for p in range(D))
 
               
  
                if (loss2[itr-1]-obj2)/obj2>=tolerance:
                    
                    x_sigma=temp_sigma
                    loss2.append(obj2)
                    x_extraction=temp_x_extraction
                    (temp_x_extraction)=optimizesigma2(weight,knn+1,x_sigma)#mean

                else:                    
                   
                    break
                
                itr=itr+1

            end2 = datetime.datetime.now()
                    
            ftime1= (end2 - start2).total_seconds()
            
                    
            vasigma=[[0]*(knn+1) for i in range(A)]
            vasumsigma = np.array(x_sigma).sum(axis = 1)
        
            for i in range(A):
                for k in range(knn+1):
                    vasigma[i][k]=np.dot (np.array(x_sigma[k]), np.array(vax[i]))/ int (vasumsigma[k])        
                    
           
            (training_acc, testing_acc, time_train, time_test, training_num, testing_num)=optimizelinearsvc(x_extraction, vasigma, knn+1)
           
            
            averagetime_training[countk][ww]=averagetime_training[countk][ww]+ftime1+time_train
            File = open(filenametime, "a")
            File.write('iteration:%d, computational time when k=%d,weight=%f: %f\n' % (counttt+1,knn+1,weight,ftime1+time_train))
            File.close()    
            
## original calculation of accuracy                 
#            train_correct=0
#            score=[[0]*J for i in range(N)]
#            for i in range(N):
#                for j in range(J):
#                    score[i][j]=sum(x_beta[j][k]*x_extraction[i][k] for k in range(knn+1))+x_beta0[j]
#            
#            
#            predictclass=np.argmax(score,axis=1)
#            for i in range(N):
#                if predictclass[i]==y[i]:
#                    train_correct=train_correct+1
                    
                    
#            File = open(filenameresult, "a")
#            File.write('iteration=%d, K=%d,weight=%f,obj=%s,training accuracy=%f\n' % (counttt+1,knn+1,weight,loss2[-1],float(train_correct/N)))
#            File.close()
#            averageacc_training[countk][ww]=averageacc_training[countk][ww]+float(train_correct/N)
            
#            score_test=[[0]*J for i in range(A)]
#            for i in range(A):
#                for j in range(J):
#                    score_test[i][j]=sum(x_beta[j][k]*vasigma[i][k] for k in range(knn+1))+x_beta0[j]
#            class_test=np.argmax(score_test,axis=1)
#            correctcount=0
#            for i in range(A):
#                if vay[i]==class_test[i]:
#                    correctcount=correctcount+1
            
## original calculation of accuracy              
            
            
            File = open(filenameresult, "a")
            File.write('iteration=%d, K=%d,weight=%f,obj=%s,training accuracy=%f\n' % (counttt+1,knn+1,weight,loss2[-1],float(training_acc)))
            File.close()
            averageacc_training[countk][ww]=averageacc_training[countk][ww]+float(training_acc)
                                           
            File = open(filenameresult, "a")
            File.write('iteration=%d, K=%d,weight=%f\n' %(counttt+1,knn+1,weight))
            for j in range(J):
                File.write('j=%d, correct training_num = %d \n' %(j+1,int(training_num[j])))
            File.write('\n')
            File.close()
            
            File = open(filenameCV, "a")
            File.write('iteration=%d, K=%d, weight=%f, testing accuracy=%f\n' % (counttt+1,knn+1,weight,float(testing_acc)))
            File.close()
            
            File = open(filenameCV, "a")
            File.write('iteration=%d, K=%d,weight=%f\n' %(counttt+1,knn+1,weight))
            for j in range(J):
                File.write('j=%d, correct testing_num = %d \n' %(j+1,int(testing_num[j])))
            File.write('\n')
            File.close()
            
            averageacc[countk][ww]=averageacc[countk][ww]+float(testing_acc)
            averagetime[countk][ww]=averagetime[countk][ww]+float(time_test)
            if maximumacc[countk][ww]<=float(testing_acc):
                maximumacc[countk][ww]=float(testing_acc)
                maximumtime[countk][ww]=ftime1+time_train    
#                recordx_sigma[countk][ww]=x_sigma
#                recordx_beta[countk][ww]=x_beta
#                recordx_beta0[countk][ww]=x_beta0
                recordx_training_num[countk][ww]=training_num
                recordx_testing_num[countk][ww]=testing_num
                record_training_error[countk][ww]=float(training_acc)
                testtime[countk][ww]=time_test
       
   


File = open(filenameresult, "a")
for countk in range(K):

#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f,minimum training error=%s\n' % (knn+1,weight,record_training_error[countk][ww]))

for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital 
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f,average training error=%s\n' % (knn+1,weight,float(averageacc_training[countk][ww]/TS)))

for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital 
    for ww in range(TW):
        weight=2*ww
        for j in range(J):
            File.write('K=%d,weight=%f,class=%d, maximum correct number: %f\n' % (knn+1,weight,j+1,recordx_training_num[countk][ww][j]))



File.close()


File = open(filenametime, "a")
for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f,training time : %f \n' % (knn+1,weight,maximumtime[countk][ww]))
        
for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f, testing time: %f \n' % (knn+1,weight,testtime[countk][ww]))
for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f,average training time : %f \n' % (knn+1,weight,float(averagetime_training[countk][ww]/TS)))
        
for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f,average testing time: %f \n' % (knn+1,weight,float(averagetime[countk][ww]/TS)))
File.close() 


File = open(filenameCV, "a")
for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f,minimum CV: %f\n' % (knn+1,weight,maximumacc[countk][ww]))
        
        
for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital
    for ww in range(TW):
        weight=2*ww
        File.write('K=%d,weight=%f,average CV: %f\n' % (knn+1,weight,float(averageacc[countk][ww]/TS)))
        
for countk in range(K):
#    knn=4*(countk+2)-1 # for digital
    knn=8*(countk+1)-1 # for optdigital 
    for ww in range(TW):
        weight=2*ww
        for j in range(J):
            File.write('K=%d,weight=%f,class=%d, maximum correct number: %f\n' % (knn+1,weight,j+1,recordx_testing_num[countk][ww][j]))        
File.close()         
           




    