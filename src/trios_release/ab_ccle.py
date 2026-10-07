import math
import numpy as np
from .ab_guarded_helper import select_ab_candidate
# Actual CCLE comparison version; not substituted into historical HB/simulation.

def fit_ms(x,ids):
 x=np.asarray(x,float);ids=np.asarray(ids,str);n=len(x);nm=max(3,math.ceil(.15*n));o=np.asarray(sorted(range(n),key=lambda i:(x[i],ids[i])));sx=x[o];cand=[]
 for k1 in range(nm,n-2*nm+1):
  if not sx[k1-1]<sx[k1]:continue
  for k2 in range(k1+nm,n-nm+1):
   if sx[k2-1]<sx[k2]:cand.append((sum(float(np.sum((g-g.mean())**2)) for g in [sx[:k1],sx[k1:k2],sx[k2:]]),k1,k2))
 if not cand:return {'valid':False,'failure_code':'NO_ADMISSIBLE_THREE_STRATUM_PARTITION','n_min':nm}
 _,k1,k2=select_ab_candidate(sx,cand);lab=np.empty(n,object);lab[o[:k1]]='L';lab[o[k1:k2]]='I';lab[o[k2:]]='H';return {'valid':True,'labels':lab.astype(str),'tau1':float((sx[k1-1]+sx[k1])/2),'tau2':float((sx[k2-1]+sx[k2])/2),'group_n_L':k1,'group_n_I':k2-k1,'group_n_H':n-k2,'minimum_group_size':min(k1,k2-k1,n-k2),'n_min':nm}
