import math
import numpy as np
# Hash-bound historical domain/resolution simulation implementation.

def ab(scores, ids):
    n=len(scores);m=max(3,math.ceil(.15*n));o=np.array(sorted(range(n),key=lambda i:(scores[i],ids[i])));s=scores[o];best=None
    for k1 in range(m,n-2*m+1):
        if not s[k1-1]<s[k1]:continue
        for k2 in range(k1+m,n-m+1):
            if not s[k2-1]<s[k2]:continue
            q=sum(float(np.sum((x-x.mean())**2)) for x in (s[:k1],s[k1:k2],s[k2:]));z=(q,k1,k2)
            if best is None or z<best:best=z
    if best is None:return None
    _,k1,k2=best;lab=np.empty(n,object);lab[o[:k1]]="L";lab[o[k1:k2]]="I";lab[o[k2:]]="H"
    return lab.astype(str),(s[k1-1]+s[k1])/2,(s[k2-1]+s[k2])/2
