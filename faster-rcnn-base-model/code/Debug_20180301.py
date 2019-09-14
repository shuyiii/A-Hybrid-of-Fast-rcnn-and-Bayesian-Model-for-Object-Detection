#Bayesian Loop
import numpy as np
import glob
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import poisson
import math
from itertools import permutations
#path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/test/'
path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/realchairtabletest/'
#output_path='../prior/updated_score'
output_path='../debug/debug_1_image_20180301'
stat=np.load('../prior/stat.npy')
cn=np.load('../prior/chair_num.npy')#0-7
#regr = linear_model.LinearRegression()
#regr.fit(range(1,8), cn[1:8])
tn=np.load('../prior/table_num.npy')
a=stat[0]
b=stat[1]
c=stat[2]
stda=stat[3]
stdb=stat[4]
stdc=stat[5]
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
 
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


chair=np.load('../prior/chair.npy')
wrongchair=np.load('../prior/wrongchair.npy')
table=np.load('../prior/table.npy')
wrongtable=np.load('../prior/wrongtable.npy')
chairkde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(chair[:, np.newaxis])
wrongchairkde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(wrongchair[:, np.newaxis])
tablekde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(table[:, np.newaxis])
wrongtablekde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(wrongtable[:, np.newaxis])
table_num=np.load('../prior/table_num.npy')
chair_num=np.load('../prior/chair_num.npy')
bndbox_path=path+'ILSVRC2012_val_00022117.JPEG.txt.npz'
mm=1
if mm==1:
    path,temp=os.path.split(bndbox_path)
    file_name,rest1,rest2,rest3=temp.split(".")
    chairindicator=np.zeros(300);tableindicator=np.zeros(300);
    chairprob=np.zeros(300);tableprob=np.zeros(300);
    Data=np.load(bndbox_path)
    position=Data['arr_0']
    position=position[0]
    prob=Data['arr_1']
    prob=prob[0]
    category=Data['arr_2']
    category=category[0]
    justtest=np.zeros(1)
    np.save(os.path.join(output_path,file_name),justtest)
    np.save(os.path.join(output_path,"position"),position)
    np.save(os.path.join(output_path,"prob"),prob)
    np.save(os.path.join(output_path,"category"),category)
    chairloc=get_indexes(1,category);tableloc=get_indexes(2,category);
    dic={};
    for m in chairloc:
        for n in tableloc:
             a1=(position[m,1]+position[m,3]-position[n,1]-position[n,3])*0.5/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)
             b1=(position[m,0]+position[m,2]-position[n,0]-position[n,2])*0.5/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)
             c1=np.sqrt((position[m,1]-position[m,3])**2+(position[m,0]-position[m,2])**2)*1.0/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)-1
             dic[(m,n)]={'Prob':scipy.stats.norm(a,stda).pdf(a1)*scipy.stats.norm(b,stdb).pdf(b1)*scipy.stats.norm(c,stdc).pdf(c1)}
    flag=0
    #initialize
    chairindicator=np.zeros(len(chairloc))
    tableindicator=np.zeros(len(tableloc))
    chairprob=[prob[i] for i in chairloc]
    tableprob=[prob[i] for i in tableloc]
    counttable=0
    countchair=0
    temptable=[];tempchair=[];
    for j in range(len(chairloc)):
       if countchair==0:
            chairindicator[j]=1
            countchair=countchair+1
            tempchair.append(position[chairloc[j],])
       elif countchair<8:
            truth=1
            for n in tempchair:
                if bb_intersection_over_union(n,position[chairloc[j],])>0.5:
                   truth=0
            if truth==1:
                countchair=countchair+1
                chairindicator[j]=1
                tempchair.append(position[chairloc[j],])         
       else:
           break
    for k in range(len(tableloc)):
       if counttable==0:
            tableindicator[k]=1
            counttable=counttable+1
            temptable.append(position[tableloc[k],])
       elif counttable<8:
            truth=1
            for n in temptable:
                if bb_intersection_over_union(n,position[tableloc[k],])>0.5:
                   truth=0
            if truth==1:    
               counttable=counttable+1
               tableindicator[k]=1
               temptable.append(position[tableloc[k],])
       else:
           break

    #update chair based on table
    for count in range(2):
         if flag==1:
            break
         for m in range(len(chairloc)):
               if sum(tableindicator)==0 or sum(chairindicator)==0:
                  break
               p1=1
               #kde part
               for n in range(len(tableloc)):
                  if tableindicator[n]==1:
                     p1=p1*np.exp(tablekde.score_samples(tableprob[n]))[0]
                  else:
                     p1=p1*np.exp(wrongtablekde.score_samples(tableprob[n]))[0]
               for l in range(len(chairloc)):
                  if l!=m and chairindicator[l]==1:
                     p1=p1*np.exp(chairkde.score_samples(chairprob[l]))[0]
                  elif l!=m and chairindicator[l]==0:
                     p1=p1*np.exp(wrongchairkde.score_samples(chairprob[l]))[0]
               kde1=p1*np.exp(chairkde.score_samples(chairprob[m]))[0]
               kde0=p1*np.exp(wrongchairkde.score_samples(chairprob[m]))[0]
              
               #gaussian part
               maxlist1=[]
               maxlist0=[]
               maxp1=1
               maxp0=1
               for k in range(len(tableloc)):
                   ptemp=0
                   for g in range(len(chairloc)):
                       if chairindicator[g]==1 and g!=m:
                          re=dic[(chairloc[g],tableloc[k])]['Prob']*1.0*np.sqrt((position[chairloc[g],1]-position[chairloc[g],3])**2+(position[chairloc[g],0]-position[chairloc[g],2])**2)/((position[tableloc[k],1]-position[tableloc[k],3])**2+(position[tableloc[k],0]-position[tableloc[k],2])**2)**2
                          ptemp=max(ptemp,re)
                   #get all max(P(t|c))
                   re1=dic[(chairloc[m],tableloc[k])]['Prob']*1.0*np.sqrt((position[chairloc[m],1]-position[chairloc[m],3])**2+(position[chairloc[m],0]-position[chairloc[m],2])**2)/((position[tableloc[k],1]-position[tableloc[k],3])**2+(position[tableloc[k],0]-position[tableloc[k],2])**2)**2
                   maxlist0.append(ptemp)
                   maxlist1.append(max(ptemp,re1))
               maxlist0=[x*1.0/sum(maxlist0) for x in maxlist0]
               maxlist1=[x*1.0/sum(maxlist1) for x in maxlist1]
               perms=list(permutations(range(int(sum(tableindicator))),int(sum(tableindicator))))
               ones=get_indexes(1,tableindicator)
               maxp0,maxp1=0,0
               for x in perms:
                   denom0,denom1=1,1
                   temp0,temp1=1,1
                   for y in range(int(sum(tableindicator))):
                       temp0=temp0*maxlist0[ones[x[y]]]*1.0/denom0
                       temp1=temp1*maxlist1[ones[x[y]]]*1.0/denom1
                       denom0=denom0-maxlist0[ones[x[y]]]
                       denom1=denom1-maxlist1[ones[x[y]]]
                   maxp0=maxp0+temp0
                   maxp1=maxp1+temp1

               #poisson part
               poissonp=poisson.pmf(int(sum(tableindicator)),table_num)

               #second part
               sizelist=[]
               for h in chairloc:
                   sizelist.append(np.sqrt((position[h,1]-position[h,3])**2+(position[h,0]-position[h,2])**2))
               sizelist=[x*1.0/sum(sizelist) for x in sizelist]
               chairtemp0,chairtemp1=chairindicator,chairindicator
               chairtemp0[m]=0
               chairtemp1[m]=1
               chairones0=get_indexes(1,chairtemp0)
               chairones1=get_indexes(1,chairtemp1)
               perm0=list(permutations(range(len(chairones0)),len(chairones0)))
               perm1=list(permutations(range(len(chairones1)),len(chairones1)))
               sizep0,sizep1=0,0
               denom0,denom1=1,1
               for x in perm0:
                   tempsize0=1
                   for y in range(len(chairones0)):
                       tempsize0=tempsize0*sizelist[chairones0[x[y]]]*1.0/denom0
                       denom0=denom0-sizelist[chairones0[x[y]]]
                   sizep0=sizep0+tempsize0
               for x in perm1:
                   tempsize1=1
                   for y in range(len(chairones1)):
                       tempsize1=tempsize1*sizelist[chairones1[x[y]]]*1.0/denom1
                       denom1=denom1-sizelist[chairones1[x[y]]]
                   sizep1=sizep1+tempsize1
                   
               #calculate poisson2
               poissonchair0,poissonchair1=0,0
               for q in range(8):
                   poissonchair0=poissonchair0+poisson.pmf(len(chairones0),chair_num[q])*poisson.pmf(q,table_num)
                   poissonchair1=poissonchair1+poisson.pmf(len(chairones1),chair_num[q])*poisson.pmf(q,table_num)
               
               p=kde1*maxp1*sizep1*poissonchair1*1.0/(kde1*maxp1*sizep1*poissonchair1+kde0*maxp0*sizep0*poissonchair0)
               
               if math.isnan(p):
                  flag=1
                  break
               chairindicator[m]=np.random.binomial(1,p,1)
               chairprob[m]=p


         #update table based on chair
         for n in range(len(tableloc)):
               if int(sum(chairindicator))==0 or int(sum(tableindicator))==0:
                  break
               p1=1
               for m in range(len(chairloc)):
                  if chairindicator[m]==1:
                     p1=p1*np.exp(chairkde.score_samples(chairprob[m]))[0]
                  else:
                     p1=p1*np.exp(wrongchairkde.score_samples(chairprob[m]))[0]
               for l in range(len(tableloc)):
                  if l!=n and tableindicator[l]==1:
                     p1=p1*np.exp(tablekde.score_samples(tableprob[l]))[0]
                  elif l!=n and tableindicator[l]==0:
                     p1=p1*np.exp(wrongtablekde.score_samples(tableprob[l]))[0]
               kde1=p1*np.exp(tablekde.score_samples(tableprob[n]))[0]
               kde0=p1*np.exp(wrongtablekde.score_samples(tableprob[n]))[0]

               #gaussian part
               maxlist1=[]
               maxlist0=[]
               maxp1=1
               maxp0=1
               for k in range(len(chairloc)):
                   ptemp=0
                   for g in range(len(tableloc)):
                       if tableindicator[g]==1 and g!=n:
                          re=dic[(chairloc[k],tableloc[g])]['Prob']*1.0/(np.sqrt((position[tableloc[g],1]-position[tableloc[g],3])**2+(position[tableloc[g],0]-position[tableloc[g],2])**2)**3)
                          ptemp=max(ptemp,re)
                   #get all max(P(t|c))
                   re1=dic[(chairloc[k],tableloc[n])]['Prob']*1.0/(np.sqrt((position[tableloc[n],1]-position[tableloc[n],3])**2+(position[tableloc[n],0]-position[tableloc[n],2])**2)**3)
                   maxlist0.append(ptemp)
                   maxlist1.append(max(ptemp,re1))
               maxlist0=[x*1.0/sum(maxlist0) for x in maxlist0]
               maxlist1=[x*1.0/sum(maxlist1) for x in maxlist1]
               perms=list(permutations(range(int(sum(chairindicator))),int(sum(chairindicator))))
               ones=get_indexes(1,chairindicator)
               maxp0,maxp1=0,0
               for x in perms:
                   denom0,denom1=1,1
                   temp0,temp1=1,1
                   for y in range(int(sum(chairindicator))):
                       temp0=temp0*maxlist0[ones[x[y]]]*1.0/denom0
                       temp1=temp1*maxlist1[ones[x[y]]]*1.0/denom1
                       denom0=denom0-maxlist0[ones[x[y]]]
                       denom1=denom1-maxlist1[ones[x[y]]]
                   maxp0=maxp0+temp0
                   maxp1=maxp1+temp1
 
               #poisson part
               temp1=tableindicator
               temp1[n]=0
               if sum(temp1)<=7:
                  poissonp0=poisson.pmf(int(sum(chairindicator)),chair_num[sum(temp1)])
               else:
                  poissonp0=poisson.pmf(int(sum(chairindicator)),1.0134+0.1856*sum(temp1))
               if (sum(temp1)+1)<=7:
                  poissonp1=poisson.pmf(int(sum(chairindicator)),chair_num[sum(temp1)+1])
               else:
                  poissonp1=poisson.pmf(int(sum(chairindicator)),1.0134+0.1856*(sum(temp1)+1))
              
               #second part
               sizelist=[]
               for h in tableloc:
                   sizelist.append(np.sqrt((position[h,1]-position[h,3])**2+(position[h,0]-position[h,2])**2))
               sizelist=[x*1.0/sum(sizelist) for x in sizelist]
               tabletemp0,tabletemp1=tableindicator,tableindicator
               tabletemp0[n]=0
               tabletemp1[n]=1
               tableones0=get_indexes(1,tabletemp0)
               tableones1=get_indexes(1,tabletemp1)
               perm0=list(permutations(range(len(tableones0)),len(tableones0)))
               perm1=list(permutations(range(len(tableones1)),len(tableones1)))
               sizep0,sizep1=0,0
               denom0,denom1=1,1
               for x in perm0:
                   tempsize0=1
                   for y in range(len(tableones0)):
                       tempsize0=tempsize0*sizelist[tableones0[x[y]]]*1.0/denom0
                       denom0=denom0-sizelist[tableones0[x[y]]]
                   sizep0=sizep0+tempsize0
               for x in perm1:
                   tempsize1=1
                   for y in range(len(tableones1)):
                       tempsize1=tempsize1*sizelist[tableones1[x[y]]]*1.0/denom1
                       denom1=denom1-sizelist[tableones1[x[y]]]
                   sizep1=sizep1+tempsize1

               #calculate poisson2 
               poissontable1=poisson.pmf(len(tableones1),table_num)
               poissontable0=poisson,pmf(len(tableones0),table_num)

               p=kde1*poissonp1*maxp1*sizep1*poissontable1*1.0/(kde1*poissonp1*maxp1*sizep1*poissontable1+kde0*poissonp0*maxp0*sizep0*poissontable0)

               if math.isnan(p):
                  flag=1
                  break
               tableindicator[n]=np.random.binomial(1,p,1)
               tableprob[n]=p
         

         result=np.zeros(300)
         probility=np.zeors(300)
         index=0
         for ii in chairloc:
             probility[ii]=chairprob[index]
             result[ii]=1
             index=index+1
         index=0
         for ii in tableloc:
             probability[ii]=tableprob[index]
             result[ii]=2
             index=index+1
         outcomes=[result,probability]
         np.save(os.path.join(output_path,count),outcomes)
