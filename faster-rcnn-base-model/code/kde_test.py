import numpy as np
import glob
import os
import pandas as pd
path='/fs/project/PAS1263/src/models/research/object_detection/chairtable/Bndbox/realchairtabletest/'
gtpath='/fs/project/PAS1263/data/ILSVRC/matconvnet_data/test.csv'
wrongchair=[];
chair=[];
wrongtable=[];
table=[];
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
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1]+ 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


df1=pd.read_csv(gtpath)
df2=df1.set_index("filename")    
for bndbox_path in glob.glob(path+'*.txt.npz'):
    Data=np.load(bndbox_path)
    position=Data['arr_0']
    position=position[0]
    prob=Data['arr_1']
    prob=prob[0]
    category=Data['arr_2']
    category=category[0]
    path,temp=os.path.split(bndbox_path)
    file_name,rest1,rest2,rest3=temp.split(".")
    if df1[df1['filename'].str.contains(file_name)==True].empty!=True:
       temp=df2.loc[file_name,]
       data=temp.as_matrix()
       if data.ndim!=1:
                for i in range(0,300):
                    for j in range(0,data.shape[0]):
                         boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                         boxB=[data[j,3]*1.0/data[j,0],data[j,5]*1.0/data[j,1],data[j,4]*1.0/data[j,0],data[j,6]*1.0/data[j,1]]
                         if intersection_over_union(boxA, boxB)>0.5:
                                if data[j,2]=='n03001627' and category[i]==1:
                                     chair.append(prob[i])
                                elif data[j,2]=='n03001627' and category[i]!=1:
                                     wrongchair.append(prob[i])
                                elif data[j,2]=='n04379243' and category[i]==2:
                                     table.append(prob[i])
                                elif  data[j,2]=='n04379243' and category[i]!=2:
                                     wrongtable.append(prob[i])
       else:
                 for i in range(0,300):
                         boxA=[position[i,1],position[i,0],position[i,3],position[i,2]]
                         boxB=[data[3]*1.0/data[0],data[5]*1.0/data[1],data[4]*1.0/data[0],data[6]*1.0/data[1]]
                         if intersection_over_union(boxA, boxB)>0.5:
                                if data[2]=='n03001627' and category[i]==1:
                                     chair.append(prob[i])
                                elif data[2]=='n03001627' and category[i]!=1:
                                     wrongchair.append(prob[i])
                                elif data[2]=='n04379243' and category[i]==2:
                                     table.append(prob[i])
                                elif data[2]=='n04379243' and category[i]!=2:
                                     wrongtable.append(prob[i])



np.save('../prior/realtest_chair',chair)
np.save('../prior/realtest_wrongchair',wrongchair)
np.save('../prior/realtest_table',table)
np.save('../prior/realtest_wrongtable',wrongtable)
