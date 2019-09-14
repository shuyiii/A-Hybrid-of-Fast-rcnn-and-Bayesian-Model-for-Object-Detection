#tuning Parameter
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import xml.etree.ElementTree as ET
path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/train/'
stat=np.load('../prior/stat.npy')
cn=np.load('../prior/onlychair_num.npy')
tn=np.load('../prior/onlytable_num.npy')
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
basicerror1,basicerror2=1000000000,100000000
for a in range(1,9):
    for b in range(1,11):
        for c in range(3,8):
            error1,error2=0,0
            for bndbox_path in glob.glob(path+'*.txt.npz'):
                path,temp=os.path.split(bndbox_path)
                file_name,rest1,rest2,rest3=temp.split(".")
                chairindicator=np.zeros(300);tableindicator=np.zeros(300);
                Data=np.load(bndbox_path)
                position=Data['arr_0']
                position=position[0]
                prob=Data['arr_1']
                prob=prob[0]
                category=Data['arr_2']
                category=category[0]
                chairloc=get_indexes(1,category);tableloc=get_indexes(2,category);
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
                      elif countchair<a and prob[chairloc[j]]>=(b*0.05):
                          truth=1
                          for n in tempchair:
                              if bb_intersection_over_union(n,position[chairloc[j],])>(c*0.1):
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
                      elif counttable<a and prob[tableloc[k]]>=(b*0.05):
                          truth=1
                          for n in temptable:
                              if bb_intersection_over_union(n,position[tableloc[k],])>(c*0.1):
                                 truth=0
                          if truth==1:    
                              counttable=counttable+1
                              tableindicator[k]=1
                              temptable.append(position[tableloc[k],])
                      else:
                          break
                error1=error1+(countchair-cn)**2
                error2=error2+(counttable-tn)**2
            error1=error1*1.0/9209
            error2=error2*1.0/9209
            if error1<basicerror1:
                basicerror1=error1
                a1,b1,c1=a,(b*0.05),(c*0.1)
            if error2<basicerror2:
                basicerror2=error2
                a2,b2,c2=a,(b*0.05),(c*0.1)
parameter=[a1,b1,c1,a2,b2,c2]
np.save('../prior/parameter',parameter)
