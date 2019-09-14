import numpy as np
import glob
import os
import pandas as pd
from operator import itemgetter
#path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/test/'
path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/get_bounding_box/Bndbox/chair/'
gtpath='/fs/project/PAS1263/data/ILSVRC/matconvnet_data/test.csv'
chairlist=[];
tablelist=[];
debug=0;
numberchair=0;
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
         numberchair=numberchair+list(data[:,2]).count('n03001627');
       else:
         numberchair=numberchair+list(data[2]).count('n03001627');
       if data.ndim!=1:
               for i in range(0,300):
                    signchair=0;
                    signtable=0;
                    for j in range(0,data.shape[0]):
                         boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                         boxB=[data[j,3]*1.0/data[j,0],data[j,5]*1.0/data[j,1],data[j,4]*1.0/data[j,0],data[j,6]*1.0/data[j,1]]
                         if intersection_over_union(boxA, boxB)>0.5:
                                if data[j,2]=='n03001627' and category[i]==1:
                                     signchair=1;
                    if   category[i]==1 and signchair==1:
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numberchair])
                    elif category[i]==1 and signchair==0:
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numberchair])

       else:
                 for i in range(0,300):
                     signchair=0;
                     signtable=0;
                     boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                     boxB=[data[3]*1.0/data[0],data[5]*1.0/data[1],data[4]*1.0/data[0],data[6]*1.0/data[1]]
                     if intersection_over_union(boxA, boxB)>0.5:
                                if data[2]=='n03001627' and category[i]==1:
                                     signchair=1;
                     if category[i]==1 and signchair==1:
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numberchair])
                     elif category[i]==1 and signchair==0:
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numberchair])
 


chairlist.sort(key=itemgetter(5),reverse=True)
Numchair=0;
mAPchair=np.zeros((len(chairlist),2));
mAPchair=np.matrix(mAPchair)
for j in range(0,len(chairlist)):
        if chairlist[j][6]==1:
                Numchair=Numchair+1;
                mAPchair[j,0]=(Numchair*1.0/(j+1));
                mAPchair[j,1]=Numchair*1.0/numberchair;



MAPchair=0;
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x <= y]
for h in range(0,11):
         x=h*0.1;
         find=get_indexes(x,mAPchair[:,1]);
         if find!=[]:
              MAPchair=MAPchair+1.0/11*(max(mAPchair[find,0]));

re=MAPchair
np.save('../outcome/chair_re',re)

