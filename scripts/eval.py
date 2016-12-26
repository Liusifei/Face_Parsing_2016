from metric_eval import *
import os
import numpy as np
f=open('data2_1vsN_feature.txt','r')
lines = f.readlines()
f.close()

dic={}
for l in lines:
  s=l.split()
  fea = s[1:]
  imname = s[0]
  imid = os.path.split(imname)[0]
  if imid not in dic:
    dic[imid]=[]
  dic[imid].append(fea)
  

feat=[]
label=[]
id=0
for k in dic:
  feas = dic[k]
  for f in feas:
    label.append(id)
    feat.append(f)
  id +=1


label=np.array(label,dtype=np.float)
feat=np.array(feat,dtype=np.float)
metricEvaluate(feat,label)
