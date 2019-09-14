import numpy as np
import glob
import os
import pandas as pd
path='/fs/project/PAS1263/data/ILSVRC/train.csv'
A=[];B=[];C=[];
df1=pd.read_csv(path)
df2=df1['filename'].values.tolist()
df3=df1.set_index("filename")
dflist=set(df2)
for file_name in list(dflist):
       temp=df3.loc[file_name,]
       data=temp.as_matrix()
       table=[]
       chair=[]
       if data.ndim!=1:
           for i in range(0,data.shape[0]):
             if data[i,2]=='n03001627':
                chair.append(i);
             elif data[i,2]=='n04379243':
                table.append(i);

       if table!=[] and chair!=[]:         
         for i in chair:
            for j in table:
              A.append((data[i,3]+data[i,4]-data[j,3]-data[j,4])*1.0/2.0/np.sqrt((data[j,3]-data[j,4])**2+(data[j,5]-data[j,6])**2+0.0001))
              B.append((data[i,5]+data[i,6]-data[j,5]-data[j,6])*1.0/2.0/np.sqrt((data[j,3]-data[j,4])**2+(data[j,5]-data[j,6])**2+0.0001))
              C.append(np.sqrt((data[i,3]-data[i,4])**2+(data[i,5]-data[i,6])**2)*1.0/np.sqrt((data[j,3]-data[j,4])**2+(data[j,5]-data[j,6])**2+0.0001)-1)
              
              
stat=[np.mean(A),np.mean(B),np.mean(C),np.std(A),np.std(B),np.std(C)]
np.save('../prior/stat_new',stat)
              
