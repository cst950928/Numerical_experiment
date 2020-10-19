% This file demonstrates how to solve CluSL problem by non-convex solver
% knitro, all the path, parameters should be revised according to different
% datasets
filenamedata='C:\Users\...\experimentdata.xlsx';
filenameresult='C:\Users\...\result(knitro).txt';
filenameresultrecord='C:\Users\...\resultexcel(knitro).txt';
filenameCV='C:\Users\...\CV(knitro).txt';
filenametime='C:\Users\...\time(knitro).txt';
filenametimerecord='C:\Users\...\timeexcel(knitro).txt';

K=4;
N=24;
TN=30;
D=1;
TW=5;
A=TN-N;
y=zeros(1,N);
x=zeros(N,D);
vay=zeros(1,A);
vax=zeros(A,D);
num = xlsread(filenamedata,3, 'A1:B30');
for i=1:N
    y(i)=num(i,D+1);
    for j=1:D
        x(i,j)=num(i,j);
    end
end

for i=1:A
    vay(i)=num(N+i,D+1);
    for j=1:D
        vax(i,j)=num(N+i,j);
    end
end

for countt=1:1
    for knn=1:K
        for ww=1:TW
            weight=0.1*ww;
            t1=clock;
            delta=binvar(N,knn,'full');
            beta=sdpvar(knn,D+1,'full');
            ce=sdpvar(knn,D,'full');
            variance=sdpvar(1,knn,'full');
         
            obj1=0;
            obj2=0;
            obj3=0;

            for k=1:knn
                for i=1:N
                    obj1=obj1+log(variance(1,k))*delta(i,k);
                end
            end

            for k=1:knn
                for i=1:N
                    temp=0;
                    for j=1:D
                        temp=temp+beta(k,j)*x(i,j);
                    end
                     obj2=obj2+delta(i,k)*(y(i)-temp-beta(k,D+1))^2/(2*variance(1,k)^2);
                end
            end
            for k=1:knn
                for i=1:N
                    temp=0;
                    for j=1:D
                        temp=temp+(x(i,j)-ce(k,j))^2;
                    end
                    obj3=obj3+weight*delta(i,k)*temp;
                end
            end
            f=obj1+obj2+obj3+0.5*N*log(2*pi);     
                
            F=[];
            F=[F,variance>=1];
             
            for i=1:N
                temp=0;
                for k=1:knn
                    temp=temp+delta(i,k);
                end
                F=[F,temp==1];
            end
            for k=1:knn
                temp=0;
                for i=1:N
                    temp=temp+delta(i,k);
                end
                F=[F,temp>=1];
            end
                
            options = sdpsettings('solver','knitro');
            optimize(F,f,options);
            t2=clock;
            t=etime(t2,t1);
            fprintf('computational time %f\n',double(t));
            fprintf('objective=%f\n',double(f));
            fprintf('delta\n');
            for i=1:N
                for k=1:knn
                    if(value(delta(i,k))>=0.5)
                        fprintf('data point %d belongs to cluster %d\n',i,k);
                    end
                end
            end
            fprintf('beta\n');
            for k=1:knn    
                for j=1:D+1
                    fprintf('The %d th regression parameter of cluster %d is %f\n',j,k,value(beta(k,j)));
                end
            end
            fprintf('ce\n');
            for k=1:knn    
                for j=1:D
                    fprintf('The %d th centroid of cluster %d is %f\n',j,k,value(ce(k,j)));
                end
            end
            fprintf('variance\n');
            for k=1:knn    
                fprintf('The variance of %d th cluster is %f\n',k,value(variance(1,k)));
            end  
            fid = fopen(filenameresultrecord,'a');
            if (ww>=4.5)
                fprintf(fid,'%f\n',double(f));
            else
                fprintf(fid,'%f\t',double(f));
            end
            fclose(fid);
            fid = fopen(filenameresult,'a');

            fprintf(fid,'iteration=%d,K=%d, weight=%f, objective=%f\n',countt,knn,weight,double(f));
            fprintf(fid,'delta\n');
            for i=1:N
                for k=1:knn
                    if(value(delta(i,k))>=0.5)
                        fprintf(fid,'data point %d belongs to cluster %d\n',i,k);
                    end
                end
            end
            fprintf(fid,'beta\n');
            for k=1:knn   
                for j=1:D+1
                    fprintf(fid,'The %d th regression parameter of cluster %d is %f\n',j,k,value(beta(k,j)));
                end
            end
            fprintf(fid,'ce\n');
            for k=1:knn   
                for j=1:D
                    fprintf(fid,'The %d th centroid of cluster %d is %f\n',j,k,value(ce(k,j)));
                end
            end
            fprintf(fid,'variance\n');
            for k=1:knn    
                fprintf(fid,'The variance of %d th cluster is %f\n',k,value(variance(1,k)));
            end  
            fclose(fid);
            fid = fopen(filenametimerecord,'a');
            if (ww>=4.5)
                fprintf(fid,'%f\n',double(t));
            else
                fprintf(fid,'%f\t',double(t));
            end
            fclose(fid);
            fid = fopen(filenametime,'a');
            fprintf(fid,'iteration=%d,K=%d, weight=%f, computational time=%f\n',countt,knn,weight,double(t));
            fclose(fid);        

            assign=binvar(A,knn,'full');
            obj4=0;
            obj5=0;
            obj6=0;
            for k=1:knn
                for i=1:A
                    obj4=obj4+log(value(variance(1,k)))*assign(i,k);
                end
            end

            for k=1:knn
                for i=1:A
                    temp=0;
                    for j=1:D
                        temp=temp+value(beta(k,j))*vax(i,j);
                    end
                     obj5=obj5+assign(i,k)*(vay(i)-temp-value(beta(k,D+1)))^2/(2*value(variance(1,k))^2);
                end
            end
            for k=1:knn
                for i=1:A
                    temp=0;
                    for j=1:D
                        temp=temp+(vax(i,j)-value(ce(k,j)))^2;
                    end
                    obj6=obj6+weight*assign(i,k)*temp;
                end
            end
            f=obj4+obj5+obj6;
            F=[];
            for i=1:A
                temp=0;
                for k=1:knn
                    temp=temp+assign(i,k);
                end
                F=[F,temp==1];
            end
            options = sdpsettings('solver','knitro');
            optimize(F,f,options);  
            perror=0;
            for i=1:A
                for k=1:knn
                    temp=0;
                    for j=1:D
                        temp=temp+vax(i,j)*value(beta(k,j));
                    end
                    perror=perror+value(assign(i,k))*(vay(i)-temp-value(beta(k,D+1)))^2;
                end
            end
            fid = fopen(filenameCV,'a');

            fprintf(fid,'iteration=%d,K=%d, weight=%f, MSE=%f\n',countt,knn,weight,double(perror/A));
            fprintf(fid,'assign\n');
            for i=1:A
                for k=1:knn
                    if(value(assign(i,k))>=0.5)
                        fprintf(fid,'data point %d belongs to cluster %d\n',i,k);
                    end
                end
            end
            fclose(fid);          
        end
    end
    fid = fopen(filenameresultrecord,'a');
    fprintf(fid,'\n');
    fclose(fid);
    fid = fopen(filenametimerecord,'a');
    fprintf(fid,'\n');
    fclose(fid);
end

