#Bayesian Loop
import pylab
import numpy as np
import glob
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import gamma
from scipy.stats import poisson
import math
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import xml.etree.ElementTree as ET
import time
import sys
path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/get_bounding_box/chair_table' #testing output bounding box
output_path='../prior/new/updated_score'
stat=np.load('../prior/stat.npy') #gaussian distribution parameters
a=stat[0]
b=stat[1]
c=stat[2]
stda=stat[3]
stdb=stat[4]
stdc=stat[5]
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

#IoU function
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

#truncated gamma distribution
def truncated_gamma(x, min_data,max_data, alpha, beta):
    gammapdf = gamma.pdf(x, alpha, loc=0, scale=beta)
    norm = gamma.cdf(max_data, alpha, loc=0, scale=beta)-gamma.cdf(min_data, alpha, loc=0, scale=beta)
    return np.where(x < max_data, gammapdf / norm, 0)
#get optimal truncated gamma parameter
def fit_truncated_gamma(data,min_data,max_data):
    energy = lambda p: -np.sum(np.log(truncated_gamma(data, min_data, max_data, *p)))
    initial_guess = [np.mean(data), 2.]
    o = minimize(energy, initial_guess, method='SLSQP')
    fit_alpha, fit_beta = o.x
    return [fit_alpha, fit_beta]

chair=np.load('../prior/new/chair.npy')
wrongchair=np.load('../prior/new/wrongchair.npy')
table=np.load('../prior/new/table.npy')
wrongtable=np.load('../prior/new/wrongtable.npy')
ca,cb=fit_truncated_gamma(chair,0.005,1)
ta,tb=fit_truncated_gamma(table,0.005,1)
wca,wcb=fit_truncated_gamma(wrongchair,0.005,1)
wta,wtb=fit_truncated_gamma(wrongtable,0.005,1)
table_num=np.load('../prior/table_num.npy')
chair_num=np.load('../prior/chair_num.npy')
sizechair=np.load('../prior/new/sizechair.npy')
sizetable=np.load('../prior/new/sizetable.npy')
wrongsizechair=np.load('../prior/new/wrongsizechair.npy')
wrongsizetable=np.load('../prior/new/wrongsizetable.npy')
sca,scb=fit_truncated_gamma(sizechair,0,1)
sta,stb=fit_truncated_gamma(sizetable,0,1)
swca,swcb=fit_truncated_gamma(wrongsizechair,0,1)
swta,swtb=fit_truncated_gamma(wrongsizetable,0,1)
                   

for bndbox_path in glob.glob(path+'*.txt.npz'):
    path,temp=os.path.split(bndbox_path)
    file_name,rest1,rest2,rest3=temp.split(".")
    Data=np.load(bndbox_path)
    position=Data['arr_0']
    position=position[0]
    prob=Data['arr_1']
    prob=prob[0]
    category=Data['arr_2']
    category=category[0]
    chairloc=get_indexes(1,category);tableloc=get_indexes(2,category);
    dic={};
    #calculate all pairs of position feature
    for m in chairloc:
        for n in tableloc:
             a1=(position[m,1]+position[m,3]-position[n,1]-position[n,3])*0.5/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)
             b1=(position[m,0]+position[m,2]-position[n,0]-position[n,2])*0.5/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)
             c1=np.sqrt((position[m,1]-position[m,3])**2+(position[m,0]-position[m,2])**2)*1.0/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)-1
             dic[(m,n)]={'Prob':scipy.stats.norm(a,stda).pdf(a1)*scipy.stats.norm(b,stdb).pdf(b1)*scipy.stats.norm(c,stdc).pdf(c1)}#check if correct
    chairgamma_dic={}
    wrongchairgamma_dic={}
    tablegamma_dic={}
    wrongtablegamma_dic={}
    chairsizegamma_dic={}
    wrongchairsizegamma_dic={}
    tablesizegamma_dic={}
    wrongtablesizegamma_dic={}
            
    #initialize
    chairindicator=np.zeros(len(chairloc))
    tableindicator=np.zeros(len(tableloc))
    chairprob=[prob[i] for i in chairloc]
    tableprob=[prob[i] for i in tableloc]
    for c in range(len(chairloc)):
                   #save truncated gamma distribution scores in dictionary
                   ind=chairloc[c]
                   chairgamma_dic[c]=truncated_gamma(chairprob[c],0.005,1,ca,cb)
                   wrongchairgamma_dic[c]=truncated_gamma(chairprob[c],0.005,1,wca,wcb)
                   chairsizegamma_dic[c]=truncated_gamma((position[ind,3]-position[ind,1])*(position[ind,2]-position[ind,0]),0,1,sca,scb)
                   wrongchairsizegamma_dic[c]=truncated_gamma((position[ind,3]-position[ind,1])*(position[ind,2]-position[ind,0]),0,1,swca,swcb)
    for t in range(len(tableloc)):
                   #save truncated gamma distribution scores in dictionary
                   ind=tableloc[t]
                   tablegamma_dic[t]=truncated_gamma(tableprob[t],0.005,1,ta,tb)
                   wrongtablegamma_dic[t]=truncated_gamma(tableprob[t],0.005,1,wta,wtb)
                   tablesizegamma_dic[t]=truncated_gamma((position[ind,3]-position[ind,1])*(position[ind,2]-position[ind,0]),0,1,sta,stb)
                   wrongtablesizegamma_dic[t]=truncated_gamma((position[ind,3]-position[ind,1])*(position[ind,2]-position[ind,0]),0,1,swta,swtb)

    chairindex=np.argsort(chairprob)
    for count in range(20): #20 is the iteration time
         tableindex=np.argsort(tableprob)#updating from smallest probability to largest probability
         #chairloc position is coresponding to: chairindex[::-1][m]
         for mm in range(len(chairloc)):
            m=chairindex[mm]
            if sum(tableindicator)==0:
               p01=chairsizegamma_dic[m]
               p02=wrongchairsizegamma_dic[m]
               p=p01*1.0/(p01+p02)
               chairindicator[m]=np.random.binomial(1,p,1)
               chairprob[m]=p
            else:
                #let table number less than 5
               if sum(tableindicator)>5:
                       Count=0
                       for t in tableindex:
                               if tableindicator[t]==1 and Count<5:
                                       Count=Count+1
                               elif tableindicator[t]==1 and Count==5:
                                       tableindicator[t]=0
                #let number less than 8
               if sum(chairindicator)>8:
                       Count=0
                       for t in chairindex:
                               if chairindicator[t]==1 and Count<8:
                                       Count=Count+1
                               elif chairindicator[t]==1 and Count==8:
                                       chairindicator[t]=0
               #score truncated gamma part
               gamma1=chairgamma_dic[m]
               gamma0=wrongchairgamma_dic[m]
              
               #gaussian part
               maxlist1=[]
               maxlist0=[]
               maxp1=1
               maxp0=1
               tableones=get_indexes(1,tableindicator)
               chairones=get_indexes(1,chairindicator)
               for k in tableones:
                   ptemp=tablesizegamma_dic[k]
                   for g in chairones:
                       if g!=m:
                          re=dic[(chairloc[g],tableloc[k])]['Prob']*1.0*np.sqrt((position[chairloc[g],1]-position[chairloc[g],3])**2+(position[chairloc[g],0]-position[chairloc[g],2])**2)/((position[tableloc[k],1]-position[tableloc[k],3])**2+(position[tableloc[k],0]-position[tableloc[k],2])**2)**2
                          ptemp=max(ptemp,re)
                   #get all max(P(t|c))
                   re1=dic[(chairloc[m],tableloc[k])]['Prob']*1.0*np.sqrt((position[chairloc[m],1]-position[chairloc[m],3])**2+(position[chairloc[m],0]-position[chairloc[m],2])**2)/((position[tableloc[k],1]-position[tableloc[k],3])**2+(position[tableloc[k],0]-position[tableloc[k],2])**2)**2
                   maxlist0.append(ptemp)
                   maxlist1.append(max(ptemp,re1))
               maxlist0=[x*1.0/sum(maxlist0) for x in maxlist0]
               maxlist1=[x*1.0/sum(maxlist1) for x in maxlist1]
               perms=list(permutations(range(int(sum(tableindicator))),int(sum(tableindicator))))
               maxp0,maxp1=0,0
               for x in perms:
                   denom0,denom1=1,1
                   temp0,temp1=1,1
                   for y in range(len(ones)):
                       temp0=temp0*maxlist0[x[y]]*1.0/denom0
                       temp1=temp1*maxlist1[x[y]]*1.0/denom1
                       denom0=denom0-maxlist0[x[y]]
                       denom1=denom1-maxlist1[x[y]]
                   maxp0=maxp0+temp0
                   maxp1=maxp1+temp1
               sizelist=[]
               for h in chairloc:
                   sizelist.append(chairsizegamma_dic[h])
               sizelist=[x*1.0/sum(sizelist) for x in sizelist]#normalize
               chairtemp0,chairtemp1=list(chairindicator),list(chairindicator)#copy indicator
               chairtemp0[m]=0
               chairtemp1[m]=1
               chairones0=get_indexes(1,chairtemp0)
               chairones1=get_indexes(1,chairtemp1)
               start_time = time.time()
               perm0=list(permutations(range(len(chairones0)),len(chairones0)))
               perm1=list(permutations(range(len(chairones1)),len(chairones1)))
                   
               sizep0,sizep1=0,0
               for x in perm0:
                   tempsize0=1
                   denom0=1
                   for y in range(len(chairones0)):
                       tempsize0=tempsize0*sizelist[chairones0[x[y]]]*1.0/denom0
                       denom0=denom0-sizelist[chairones0[x[y]]]
                   sizep0=sizep0+tempsize0
               for x in perm1:
                   tempsize1=1
                   denom1=1
                   for y in range(len(chairones1)):
                       tempsize1=tempsize1*sizelist[chairones1[x[y]]]*1.0/denom1
                       denom1=denom1-sizelist[chairones1[x[y]]]
                   sizep1=sizep1+tempsize1
                   
               #calculate poisson2
               poissonchair0,poissonchair1=0,0
               for q in range(8):
                   poissonchair0=poissonchair0+poisson.pmf(len(chairones0),chair_num[q])*poisson.pmf(q,table_num)
                   poissonchair1=poissonchair1+poisson.pmf(len(chairones1),chair_num[q])*poisson.pmf(q,table_num)
               
               p=1.0/(1+(gamma0*1.0/gamma1)*(maxp0*1.0/maxp1)*(sizep0*1.0/sizep1)*(poissonchair0*1.0/poissonchair1))
               chairindicator[m]=np.random.binomial(1,p,1)
               chairprob[m]=p


         #update table based on chair
         chairindex=np.argsort(chairprob)
         for nn in range(len(tableloc)):
            n=tableindex[nn]
            if sum(chairindicator)==0:
               p01=tablesizegamma_dic[n]
               p02=wrongtablesizegamma_dic[n]
               p=p01*1.0/(p01+p02)
               tableindicator[n]=np.random.binomial(1,p,1)
               tableprob[n]=p
            else:
                 #let number less than 5
               if sum(tableindicator)>5:
                       Count=0
                       for t in tableindex:
                               if tableindicator[t]==1 and Count<5:
                                       Count=Count+1
                               elif tableindicator[i]==1 and Count==5:
                                       tableindicator[t]=0
                #let number less than 8
               if sum(chairindicator)>8:
                       Count=0
                       for t in chairindex:
                               if chairindicator[t]==1 and Count<8:
                                       Count=Count+1
                               elif chairindicator[t]==1 and Count==8:
                                       chairindicator[t]=0

               gamma1=tablegamma_dic[n]
               gamma0=wrongtablegamma_dic[n]

               #gaussian part
               maxlist1=[]
               maxlist0=[]
               maxp1=1
               maxp0=1
               chairones=get_indexes(1,chairindicator)
               tableones=get_indexes(1,tableindicator)
               for k in chairones:
                   ptemp=chairsizegamma_dic[k]
                   for g in tableones:
                       if g!=n:
                          re=dic[(chairloc[k],tableloc[g])]['Prob']*1.0/(np.sqrt((position[tableloc[g],1]-position[tableloc[g],3])**2+(position[tableloc[g],0]-position[tableloc[g],2])**2)**3)
                          ptemp=max(ptemp,re)
                   #get all max(P(t|c))
                   re1=dic[(chairloc[k],tableloc[n])]['Prob']*1.0/(np.sqrt((position[tableloc[n],1]-position[tableloc[n],3])**2+(position[tableloc[n],0]-position[tableloc[n],2])**2)**3)
                   maxlist0.append(ptemp)
                   maxlist1.append(max(ptemp,re1))
               maxlist0=[x*1.0/sum(maxlist0) for x in maxlist0]
               maxlist1=[x*1.0/sum(maxlist1) for x in maxlist1]
               perms=list(permutations(range(int(sum(chairindicator))),int(sum(chairindicator))))
               maxp0,maxp1=0,0
               for x in perms:
                   denom0,denom1=1,1
                   temp0,temp1=1,1
                   for y in range(len(ones)):
                       temp0=temp0*maxlist0[x[y]]*1.0/denom0
                       temp1=temp1*maxlist1[x[y]]*1.0/denom1
                       denom0=denom0-maxlist0[x[y]]
                       denom1=denom1-maxlist1[x[y]]
                   maxp0=maxp0+temp0
                   maxp1=maxp1+temp1
 
               #poisson part
               temp1=list(tableindicator)
               temp1[n]=0
               if sum(temp1)<=7:
                  poissonp0=poisson.pmf(int(sum(chairindicator)),chair_num[int(sum(temp1))])
               else:
                  poissonp0=poisson.pmf(int(sum(chairindicator)),1.0134+0.1856*sum(temp1))
               if (sum(temp1)+1)<=7:
                  poissonp1=poisson.pmf(int(sum(chairindicator)),chair_num[int(sum(temp1))+1])
               else:
                  poissonp1=poisson.pmf(int(sum(chairindicator)),1.0134+0.1856*(sum(temp1)+1))
              
               #second part
               sizelist=[]
               for h in tableloc:
                   sizelist.append(tablesizegamma_dic[h])
               sizelist=[x*1.0/sum(sizelist) for x in sizelist]
               tabletemp0,tabletemp1=list(tableindicator),list(tableindicator)
               tabletemp0[n]=0
               tabletemp1[n]=1
               tableones0=get_indexes(1,tabletemp0)
               tableones1=get_indexes(1,tabletemp1)
               perm0=list(permutations(range(len(tableones0)),len(tableones0)))
               perm1=list(permutations(range(len(tableones1)),len(tableones1)))
               sizep0,sizep1=0,0
               for x in perm0:
                   tempsize0=1
                   denom0=1
                   for y in range(len(tableones0)):
                       tempsize0=tempsize0*sizelist[tableones0[x[y]]]*1.0/denom0
                       denom0=denom0-sizelist[tableones0[x[y]]]
                   sizep0=sizep0+tempsize0
               for x in perm1:
                   tempsize1=1
                   denom1=1
                   for y in range(len(tableones1)):
                       tempsize1=tempsize1*sizelist[tableones1[x[y]]]*1.0/denom1
                       denom1=denom1-sizelist[tableones1[x[y]]]
                   sizep1=sizep1+tempsize1

               #calculate poisson2 
               poissontable1=poisson.pmf(len(tableones1),table_num)
               poissontable0=poisson.pmf(len(tableones0),table_num)

               p=1.0/(1+(gamma0*1.0/gamma1)*(maxp0*1.0/maxp1)*(poissonp0*1.0/poissonp1)*(sizep0*1.0/sizep1)*(poissontable0*1.0/poissontable1))
               tableindicator[n]=np.random.binomial(1,p,1)
               tableprob[n]=p


    Chairprob=np.zeros(300)
    Tableprob=np.zeros(300)
    index=0
    for i in chairloc:
        Chairprob[i]=chairprob[index]
        index=index+1
    index=0
    for i in tableloc:
        Tableprob[i]=tableprob[index]
        index=index+1
    prob=[Chairprob,Tableprob]
    np.save(os.path.join(output_path,file_name),prob)
                                                          
         
