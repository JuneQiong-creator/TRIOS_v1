import math
import numpy as np
# Caller binds the existing profile candidate table and candidate grid to LOCAL/CSETS.

def stage(r,ids):
    g=LOCAL[(LOCAL.regimen==r)&LOCAL.profile_id.isin(ids)];rows=[];chosen=None;ms=max(10,3*max(3,math.ceil(.15*len(ids))))
    for u in CSETS[r]:
        a=g[(g.candidate_uM==u)&g.informative];m=len(a);ae=float(a.A_i.median()) if m else np.nan;te=float(a.TRG_i.median()) if m else np.nan;ok=m>=ms and ae>=.8 and te<=.1
        if ok and chosen is None:chosen=u
        rows.append(dict(regimen=r,N=len(ids),candidate_uM=u,m=m,m_stage=ms,median_A=ae,median_TRG=te,passed=ok))
    for x in rows:x['selected_endpoint']=chosen;x['selected']=x['candidate_uM']==chosen
    return chosen,rows
