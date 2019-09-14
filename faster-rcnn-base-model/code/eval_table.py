import numpy as np
import glob
import os
import pandas as pd
from operator import itemgetter
#path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/test/'
path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/get_bounding_box/Bndbox/table/'
gtpath='/fs/project/PAS1263/data/ILSVRC/matconvnet_data/test.csv'
tablelist=[];
numbertable=0;
def intersection_over_union(boxA, boxB):
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
df1=pd.read_csv(gtpath)
df2=df1.set_index("filename")
for bndbox_path in glob.glob(path+'/*.npz'):
    path,temp=os.path.split(bndbox_path)
    file_name,rest1,rest2,rest3=temp.split(".")
    Data=np.load(bndbox_path)
    position=Data['arr_0']
    position=position[0]
    prob=Data['arr_1']
    prob=prob[0]
    category=Data['arr_2']
    category=category[0]
    if df1[df1['filename'].str.contains(file_name)==True].empty!=True:
       temp=df2.loc[file_name,]
       data=temp.as_matrix()
       if data.ndim!=1:
         numbertable=numbertable+list(data[:,2]).count('n04379243');
       else:
         numbertable=numbertable+list(data[2]).count('n04379243');
       if data.ndim!=1:
               for i in range(0,300):
                    signtable=0;
                    for j in range(0,data.shape[0]):
                         boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                         boxB=[data[j,3]*1.0/data[j,0],data[j,5]*1.0/data[j,1],data[j,4]*1.0/data[j,0],data[j,6]*1.0/data[j,1]]
                         if intersection_over_union(boxA, boxB)>0.5:
                                if data[j,2]=='n04379243' and category[i]==1:
                                     signtable=1;
                    if category[i]==1 and signtable==1:
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numbertable])
                    elif category[i]==1 and signtable==0:
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numbertable])

       else:
                 for i in range(0,300):
                     signtable=0;
                     boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                     boxB=[data[3]*1.0/data[0],data[5]*1.0/data[1],data[4]*1.0/data[0],data[6]*1.0/data[1]]
                     if intersection_over_union(boxA, boxB)>0.5:
                                if data[2]=='n04379243' and category[i]==1:
                                     signtable=1;
                     if category[i]==1 and signtable==1:
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numbertable])
                     elif category[i]==1 and signtable==0:
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numbertable])
 


tablelist.sort(key=itemgetter(5),reverse=True)
Numtable=0;
mAPtable=np.zeros((len(tablelist),2));
mAPtable=np.matrix(mAPtable)
for j in range(0,len(tablelist)):
        if tablelist[j][6]==1:
                Numtable=Numtable+1;
                mAPtable[j,0]=(Numtable*1.0/(j+1));
                mAPtable[j,1]=Numtable*1.0/numbertable;



MAPtable=0;
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x <= y]
for h in range(0,11):
         x=h*0.1;
         find=get_indexes(x,mAPtable[:,1]);
         if find!=[]:
              MAPtable=MAPtable+1.0/11*(max(mAPtable[find,0]));
re=MAPtable
np.save('../outcome/table_re',re)

