import numpy as np
import glob
import os
import pandas as pd
path='/fs/project/PAS1263/data/ILSVRC/matconvnet_data/train.csv'
df1=pd.read_csv(path)
df2=df1['filename'].values.tolist()
df3=df1.set_index("filename")
dflist=set(df2)
alist=[]
for file_name in list(dflist):
       temp=df3.loc[file_name,]
       data=temp.as_matrix()
       if data.ndim!=1:
           result=0;
           for i in range(0,data.shape[0]):
             if data[i,2]=='n03001627' or data[i,2]=='n04379243':
                result=1;
                break;
           if result==1:
                alist.append(file_name);
       else:
           result=0;
           if data[2]=='n03001627' or data[2]=='n04379243':
                result=1;
           if result==1:
                alist.append(file_name);


np.save('../prior/table_chair_list_train',alist)
