import numpy as np
import glob
import os
import pandas as pd
path='/fs/project/PAS1263/data/ILSVRC/train.csv'
df1=pd.read_csv(path)
df2=df1['filename'].values.tolist()
df3=df1.set_index("filename")
dflist=set(df2)
table=[]
chair=[]
for file_name in list(dflist):
       countchair=0;
       counttable=0;
       temp=df3.loc[file_name,]
       data=temp.as_matrix()
       if data.ndim!=1:
           for i in range(0,data.shape[0]):
             if data[i,2]=='n03001627':
                countchair=countchair+1;
             elif data[i,2]=='n04379243':
                counttable=counttable+1;
       else:
         if data[2]=='n03001627':
            countchair=countchair+1;
         elif data[2]=='n04379243':
            counttable=counttable+1;
       table.append(counttable);
       chair.append(countchair);

lambdatable=np.mean(table);
lambdaallchair=np.mean(chair);
lambdachair=[];
for k in range(20):
   index=[i for i,x in enumerate(table) if x == k]
   if index!=[]:
      lambdachair.append(np.mean([chair[j] for j in index]))
   else:
      break
index1=[i for i,x in enumerate(chair) if x!=0]
index2=[i for i,x in enumerate(table) if x!=0]
a=sum(chair)*1.0/len(index1)
b=sum(table)*1.0/len(index2)

np.save('../prior/table_num_new',lambdatable)
np.save('../prior/chair_num_new',lambdachair)
np.save('../prior/allchair_num_new',lambdaallchair)
np.save('../prior/onlychair_num_new',a)
np.save('../prior/onlytable_num_new',b)
