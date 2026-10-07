import math
import numpy as np
# Shared functions actually imported by the current fixed-representation engine.

class IntegrityStop(RuntimeError):pass

def require(ok,msg):
    if not bool(ok):raise IntegrityStop(msg)

def project(x,t1,t2,trios):
    x=np.asarray(x,float)
    return np.where(x<=t1,'L',np.where(x<=t2 if trios else x<t2,'I','H'))

def task_components(x,labels,t1,t2,pfit,folds,V=True):
    if not V:return dict(V=0,S_BW=None,S_tau=None,S_match=None,P_fit=None,I_SF=None,C_conditional=None,C_operational=0.,A_valid=0.)
    x=np.asarray(x,float);labels=np.asarray(labels);n=len(x);Rx=float(np.ptp(x))
    Bv=float(sum(np.sum(labels==g)*(x[labels==g].mean()-x.mean())**2 for g in 'LIH'))
    W=float(sum(np.sum((x[labels==g]-x[labels==g].mean())**2) for g in 'LIH'))
    require(np.isfinite([Rx,Bv,W,t1,t2,pfit]).all() and Rx>0 and Bv+W>0,'PURPORTED_VALID_UNDEFINED_ARITHMETIC')
    require(len(folds)==n and all(f['attempted'] for f in folds),'INTENDED_FOLD_ACCOUNTING')
    good=[f for f in folds if f['success']];v=len(good);A=v/n
    st=A/(1+math.sqrt(sum(f['d_tau_squared'] for f in good)/v)) if v else 0.
    om=sum(f['omitted_match'] for f in good)/v if v else None
    rt=sum(f['retained_match_fraction'] for f in good)/v if v else None
    sm=A*2*om*rt/(om+rt) if v and om+rt>0 else 0.
    sf=int(min(np.sum(labels==g) for g in 'LIH')>=2)
    vals=[Bv/(Bv+W),st,sm,pfit,sf];require(all(math.isfinite(q) and 0<=q<=1 for q in vals),'COMPONENT_RANGE')
    return dict(V=1,S_BW=vals[0],S_tau=st,S_match=sm,P_fit=pfit,I_SF=sf,C_conditional=sum(vals)/5,C_operational=sum(vals)/5,A_valid=1.,R_x=Rx,SS_B=Bv,SS_W=W,n_success=v,n_failed=n-v,A_LOO=A,omitted_success_mean=om,retained_success_mean=rt)
