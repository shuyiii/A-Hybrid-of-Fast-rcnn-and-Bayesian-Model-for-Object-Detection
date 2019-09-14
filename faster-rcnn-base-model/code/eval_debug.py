#only consider those probability greater than 0.1, to see what's wrong in the distribution estimation
import numpy as np
import glob
import os
import pandas as pd
from operator import itemgetter
#path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/test/'
path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/realchairtabletest/'
gtpath='/fs/project/PAS1263/data/ILSVRC/matconvnet_data/test.csv'
chairlist=[];
tablelist=[];
chair_updating_list=[];
table_updating_list=[];
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
datapath='../prior/updated_score'
df1=pd.read_csv(gtpath)
df2=df1.set_index("filename")
for file in glob.glob(datapath+'/*.npy'):
    datapath,temp=os.path.split(file)
    file_name,rest1=temp.split(".")
    updatingData=np.load(file)
    bndbox_path=path+file_name+'.JPEG.txt.npz'
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
         numbertable=numbertable+list(data[:,2]).count('n04379243');
       else:
         numberchair=numberchair+list(data[2]).count('n03001627');
         numbertable=numbertable+list(data[2]).count('n04379243');
       if data.ndim!=1:
               for i in range(0,300):
                    signchair=0;
                    signtable=0;
                    for j in range(0,data.shape[0]):
                         boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                         boxB=[data[j,3]*1.0/data[j,0],data[j,5]*1.0/data[j,1],data[j,4]*1.0/data[j,0],data[j,6]*1.0/data[j,1]]
                         if intersection_over_union(boxA, boxB)>0.7:
                                if data[j,2]=='n03001627' and category[i]==1 and updatingData[0][i]>=0.1:
                                     signchair=1;
                                elif data[j,2]=='n04379243' and category[i]==2 and updatingData[1][i]>=0.1:
                                     signtable=1;
                    if   category[i]==1 and signchair==1:
                           debug=debug+1;
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numberchair])
                           chair_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[0][i],1,numberchair])
                    elif category[i]==2 and signtable==1:
                           debug=debug+1;
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numbertable])
                           table_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[1][i],1,numbertable])
                    elif category[i]==1 and signchair==0:
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numberchair])
                           chair_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[0][i],0,numberchair])
                    elif category[i]==2 and signtable==0:
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numbertable])
                           table_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[1][i],0,numbertable])

       else:
                 for i in range(0,300):
                     signchair=0;
                     signtable=0;
                     boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                     boxB=[data[3]*1.0/data[0],data[5]*1.0/data[1],data[4]*1.0/data[0],data[6]*1.0/data[1]]
                     if intersection_over_union(boxA, boxB)>0.7:
                                if data[2]=='n03001627' and category[i]==1 and updatingData[0][i]>=0.1:
                                     signchair=1;
                                elif data[2]=='n04379243' and category[i]==2 and updatingData[1][i]>=0.1:
                                     signtable=1;
                     if category[i]==1 and signchair==1:
                           debug=debug+1;
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numberchair])
                           chair_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[0][i],1,numberchair])
                     elif category[i]==2 and signtable==1:
                           debug=debug+1;
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],1,numbertable])
                           table_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[1][i],1,numbertable])
                     elif category[i]==1 and signchair==0:
                           chairlist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numberchair])
                           chair_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[0][i],0,numberchair])
                     elif category[i]==2 and signtable==0:
                           tablelist.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],prob[i],0,numbertable])
                           table_updating_list.append([file_name,position[i,0],position[i,1],position[i,2],position[i,3],updatingData[1][i],0,numbertable])
 


chairlist.sort(key=itemgetter(5),reverse=True)
chair_updating_list.sort(key=itemgetter(5),reverse=True)
tablelist.sort(key=itemgetter(5),reverse=True)
table_updating_list.sort(key=itemgetter(5),reverse=True)
Numchair=0;Numtable=0;
Numchairupdate=0;Numtableupdate=0;
mAPtable=np.zeros((len(tablelist),2));
mAPchair=np.zeros((len(chairlist),2));
mAPtableupdate=np.zeros((len(table_updating_list),2));
mAPchairupdate=np.zeros((len(chair_updating_list),2));
mAPtable=np.matrix(mAPtable)
mAPchair=np.matrix(mAPchair)
mAPtableupdate=np.matrix(mAPtableupdate)
mAPchairupdate=np.matrix(mAPchairupdate)
for j in range(0,len(chairlist)):
        if chairlist[j][6]==1:
                Numchair=Numchair+1;
                mAPchair[j,0]=(Numchair*1.0/(j+1));
                mAPchair[j,1]=Numchair*1.0/numberchair;
        if chair_updating_list[j][6]==1:
                Numchairupdate=Numchairupdate+1;
                mAPchairupdate[j,0]=(Numchairupdate*1.0/(j+1));
                mAPchairupdate[j,1]=Numchairupdate*1.0/numberchair;

for j in range(0,len(tablelist)):
        if tablelist[j][6]==1:
                Numtable=Numtable+1;
                mAPtable[j,0]=(Numtable*1.0/(j+1));
                mAPtable[j,1]=Numtable*1.0/numbertable;
        if table_updating_list[j][6]==1:
                Numtableupdate=Numtableupdate+1;
                mAPtableupdate[j,0]=(Numtableupdate*1.0/(j+1));
                mAPtableupdate[j,1]=Numtableupdate*1.0/numbertable;




MAPtable=0;MAPtableupdate=0;MAPchair=0;MAPchairupdate=0;
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x <= y]
for h in range(0,11):
         x=h*0.1;
         find1=get_indexes(x,mAPtable[:,1]);
         find2=get_indexes(x,mAPtableupdate[:,1]);
         find3=get_indexes(x,mAPchair[:,1]);
         find4=get_indexes(x,mAPchairupdate[:,1]);
         if find1!=[]:
              MAPtable=MAPtable+1.0/11*(max(mAPtable[find1,0]));
         if find2!=[]:
              MAPtableupdate=MAPtableupdate+1.0/11*(max(mAPtableupdate[find2,0]));
         if find3!=[]:
              MAPchair=MAPchair+1.0/11*(max(mAPchair[find3,0]));
         if find4!=[]:
              MAPchairupdate=MAPchairupdate+1.0/11*(max(mAPchairupdate[find4,0]));

re=[MAPtable,MAPtableupdate,MAPchair,MAPchairupdate]
np.save('../outcome',re)
np.save('../debug',debug)

