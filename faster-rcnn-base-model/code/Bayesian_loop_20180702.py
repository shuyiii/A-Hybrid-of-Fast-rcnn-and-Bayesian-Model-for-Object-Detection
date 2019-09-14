from itertools import permutations
import pylab
import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
import scipy
from sklearn import datasets, linear_model
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.stats import gamma
from sklearn.neighbors import KernelDensity
from scipy.stats import poisson
import math
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time
import sys
from numpy.random import choice
from scipy.stats import norm

path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/get_bounding_box/Bndbox/chair_table_new'
output_path='../prior/updated_score/new'
stat=np.load('../prior/stat_new.npy')
cn=np.load('../prior/chair_num_new.npy')
tn=np.load('../prior/table_num_new.npy')
a=stat[0]
b=stat[1]
c=stat[2]
stda=stat[3]
stdb=stat[4]
stdc=stat[5]
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
#function of calculating IoU
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

#fit gamma distribution
def truncated_gamma(x, min_data,max_data, alpha, beta):
    gammapdf = gamma.pdf(x, alpha, loc=0, scale=beta)
    normed = gamma.cdf(max_data, alpha, loc=0, scale=beta)-gamma.cdf(min_data, alpha, loc=0, scale=beta)
    return np.where(x < max_data, gammapdf / normed, 0)

def fit_truncated_gamma(data,min_data,max_data,guess):
    energy = lambda p: -np.sum(np.log(truncated_gamma(data, min_data, max_data, *p)+0.00001))
    initial_guess = [np.mean(data), guess]
    o = minimize(energy, initial_guess, method='SLSQP')
    fit_alpha, fit_beta = o.x
    return [fit_alpha, fit_beta]

def get_new_prob(xml_file,bndbox_path):
    xml_list = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
            if member[0].text=='n04379243':
                index=2
            elif member[0].text=='n03001627':
                index=1
            else:
                continue
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     index,
                     int(member.find('bndbox')[0].text),
                     int(member.find('bndbox')[1].text),
                     int(member.find('bndbox')[2].text),
                     int(member.find('bndbox')[3].text))
            xml_list.append(value)
  
          
    Data=np.load(bndbox_path)
    position=Data['arr_0']
    position=position[0]
    new_prob=[]
    new_cate=[]
    for i in range(len(position)):
        temp=0 
        tempind=0
        boxA=[position[i][1],position[i][0],position[i][3],position[i][2]]
        for j in range(len(xml_list)):
            boxB=[xml_list[j][4]*1.0/xml_list[j][1],xml_list[j][6]*1.0/xml_list[j][2],xml_list[j][5]*1.0/xml_list[j][1],xml_list[j][7]*1.0/xml_list[j][2]]
            if bb_intersection_over_union(boxA, boxB)>temp:
               temp=bb_intersection_over_union(boxA, boxB)
               tempind=xml_list[j][3]
        new_prob.append(temp)
        new_cate.append(tempind)
    return new_prob,new_cate 

chair=np.load('../prior/new/chair_new.npy')# The difference between 1 and 2 is just the threshold
wrongchair=np.load('../prior/new/wrongchair_new.npy')
table=np.load('../prior/new/table_new.npy')
wrongtable=np.load('../prior/new/wrongtable_new.npy')
chair_s=choice(chair, size=1000, replace=False)
table_s=choice(table, size=1000, replace=False)
wrongchair_s=choice(wrongchair, size=1000, replace=False)
wrongtable_s=choice(wrongtable, size=1000, replace=False)
ca,cb=fit_truncated_gamma(chair_s,0.005,1,1)
ta,tb=fit_truncated_gamma(table_s,0.005,1,1)
wca,wcb=fit_truncated_gamma(wrongchair_s,0.005,1,1.5)
wta,wtb=fit_truncated_gamma(wrongtable_s,0.005,1,2)
table_num=np.load('../prior/table_num_new.npy')
chair_num=np.load('../prior/chair_num_new.npy')
sizechair=np.load('../prior/new/sizechair_new.npy')
sizetable=np.load('../prior/new/sizetable_new.npy')
wrongsizechair=np.load('../prior/new/wrongsizechair_new.npy')
wrongsizetable=np.load('../prior/new/wrongsizetable_new.npy')
sizechair_s=choice(sizechair, size=1000, replace=False)
sizetable_s=choice(sizetable, size=1000, replace=False)
wrongsizechair_s=choice(wrongsizechair, size=1000, replace=False)
wrongsizetable_s=choice(wrongsizetable, size=1000, replace=False)
sca,scb=fit_truncated_gamma(sizechair_s,0,1,1.5)
sta,stb=fit_truncated_gamma(sizetable_s,0,1,1.5)
swca,swcb=fit_truncated_gamma(wrongsizechair_s,0,1,1.5)
swta,swtb=fit_truncated_gamma(wrongsizetable_s,0,1,1.5)

for bndbox_path in glob.glob(path+'*.txt.npz'):
    path,temp=os.path.split(bndbox_path)
    file_name,rest1,rest2,rest3=temp.split(".")
    chairindicator=np.zeros(300);tableindicator=np.zeros(300);
    chairprob=np.zeros(300);tableprob=np.zeros(300);
    Data=np.load(bndbox_path)
    position=Data['arr_0']
    position=position[0]
    xml_path='/fs/project/PAS1263/data/ILSVRC/Annotations/test/'+file_name+'.xml'
    newprob,new_cate=get_new_prob(xml_file,bndbox_path)
    prob=new_prob
    category=new_cate
    chairloc=get_indexes(1,category);tableloc=get_indexes(2,category);
    dic={};
    #calculate all pairs of position feature
    for m in chairloc:
        for n in tableloc:
             a1=(position[m,1]+position[m,3]-position[n,1]-position[n,3])*0.5/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)
             b1=(position[m,0]+position[m,2]-position[n,0]-position[n,2])*0.5/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)
             c1=np.sqrt((position[m,1]-position[m,3])**2+(position[m,0]-position[m,2])**2)*1.0/np.sqrt((position[n,1]-position[n,3])**2+(position[n,0]-position[n,2])**2)-1
             dic[(m,n)]={'Prob':scipy.stats.norm(a0,stda0).pdf(a1)*scipy.stats.norm(b0,stdb0).pdf(b1)*scipy.stats.norm(c0,stdc0).pdf(c1)}#check if correct
    chairgamma_dic={}
    wrongchairgamma_dic={}
    tablegamma_dic={}
    wrongtablegamma_dic={}
    chairsizegamma_dic={}
    wrongchairsizegamma_dic={}
    tablesizegamma_dic={}
    wrongtablesizegamma_dic={}
            
    #initialize
    chairindicator=np.ones(len(chairloc))
    tableindicator=np.ones(len(tableloc))
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
    tableindex=np.argsort(tableprob)
    if sum(tableindicator)>5:
                       Count=0
                       for t in tableindex[::-1]:
                               if tableindicator[t]==1 and Count<5:
                                       Count=Count+1
                               elif tableindicator[t]==1 and Count==5:
                                       tableindicator[t]=0
                #let number less than 8
    if sum(chairindicator)>8:
                       Count=0
                       for t in chairindex[::-1]:
                               if chairindicator[t]==1 and Count<8:
                                       Count=Count+1
                               elif chairindicator[t]==1 and Count==8:
                                       chairindicator[t]=0

    #update chair based on table
    for count in range(2):
         gammaratio=np.ones(len(chairloc))
         maxpratio=np.ones(len(chairloc))
         tableindex=np.argsort(tableprob)
         #chairloc position is coresponding to: chairindex[::-1][m]
         for mm in range(len(chairloc)):
            m=chairindex[mm]#m is the place in chairprob, while chairprob actually is corresponding with chairloc
            if sum(tableindicator)==0:
               p01=chairsizegamma_dic[m]
               p02=wrongchairsizegamma_dic[m]
               if p01==0 and p02==0:
                  p=0
               else:
                  p=p01*1.0/(p01+p02)
               chairindicator[m]=np.random.binomial(1,p,1)
               chairprob[m]=p
            else:
                #let number less than 5
               if sum(tableindicator)>5:
                       Count=0
                       for t in tableindex[::-1]:
                               if tableindicator[t]==1 and Count<5:
                                       Count=Count+1
                               elif tableindicator[t]==1 and Count==5:
                                       tableindicator[t]=0
                #let number less than 8
               if sum(chairindicator)>8:
                       Count=0
                       for t in chairindex[::-1]:
                               if chairindicator[t]==1 and Count<8:
                                       Count=Count+1
                               elif chairindicator[t]==1 and Count==8:
                                       chairindicator[t]=0
        
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
                   for y in range(len(tableones)):
                       temp0=temp0*maxlist0[x[y]]*1.0/denom0
                       temp1=temp1*maxlist1[x[y]]*1.0/denom1
                       denom0=denom0-maxlist0[x[y]]
                       denom1=denom1-maxlist1[x[y]]
                   maxp0=maxp0+temp0
                   maxp1=maxp1+temp1

               #poisson part
               poissonp=poisson.pmf(int(sum(tableindicator)),table_num)  
               if gamma0==0 and gamma1==0:
                   gamma0=1
                   gamma0=1
               p=1.0/(1+(gamma0*1.0/gamma1)*(maxp0*1.0/maxp1))
               gammaratio[m]=(gamma0*1.0/gamma1)
               maxpratio[m]=(maxp0*1.0/maxp1)
               chairindicator[m]=np.random.binomial(1,p,1)
               chairprob[m]=p

         #update table based on chair
         gammaratio=np.ones(len(tableloc))
         maxpratio=np.ones(len(tableloc))
         chairindex=np.argsort(chairprob)
         for nn in range(len(tableloc)):
             #from minimum to maximum
            n=tableindex[nn]
            if sum(chairindicator)==0:
               p01=tablesizegamma_dic[n]
               p02=wrongtablesizegamma_dic[n]
               if p01==0 and p02==0:
                   p=0
               else:
                   p=p01*1.0/(p01+p02)
               tableindicator[n]=np.random.binomial(1,p,1)
               tableprob[n]=p
            else:
                 #let number less than 5
               if sum(tableindicator)>5:
                       Count=0
                       for t in tableindex[::-1]:
                               if tableindicator[t]==1 and Count<5:
                                       Count=Count+1
                               elif tableindicator[t]==1 and Count==5:
                                       tableindicator[t]=0
                #let number less than 8
               if sum(chairindicator)>8:
                       Count=0
                       for t in chairindex[::-1]:
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
                   for y in range(len(chairones)):
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
              

               #calculate poisson2 
               tabletemp0,tabletemp1=list(tableindicator),list(tableindicator)
               tabletemp0[n]=0
               tabletemp1[n]=1
               tableones0=get_indexes(1,tabletemp0)
               tableones1=get_indexes(1,tabletemp1)
               poissontable1=poisson.pmf(len(tableones1),table_num)
               poissontable0=poisson.pmf(len(tableones0),table_num)
               
               if gamma0==0 and gamma1==0:
                   gamma0=1
                   gamma0=1
               p=1.0/(1+(gamma0*1.0/gamma1)*(maxp0*1.0/maxp1))
               gammaratio[n]=(gamma0*1.0/gamma1)
               maxpratio[n]=(maxp0*1.0/maxp1)
               if math.isnan(p):
                  flag=1
                  break
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
