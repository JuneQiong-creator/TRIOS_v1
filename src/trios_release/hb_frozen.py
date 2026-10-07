import numpy as np
from .reconstruction import ConstrainedPSpline

def pred(obj,d):return np.asarray(obj.predict(np.asarray(d,float)),float)

def scalar(obj,d):return float(pred(obj,[d])[0])

def spline_from_parameters(reg,pid,params,status,dat):
 g=params[(params.regimen==reg)&(params.sample_id==pid)&(params.model=='MONOTONE_SPLINE')].copy();g['idx']=g.parameter.str.extract(r'(\d+)',expand=False).astype(int);g=g.sort_values('idx');coef=g.unconstrained_value.to_numpy(float);st=status[(status.regimen==reg)&(status.sample_id==pid)&(status.model=='MONOTONE_SPLINE')].iloc[0];dmax=float(dat[(dat.regimen==reg)&(dat.PDO_ID==pid)].dose_uM.max())
 return ConstrainedPSpline(float(coef[12]),np.diff(coef[:12]),coef[:12],(0.,float(np.log1p(dmax))),12,3,float(st.selected_lambda),float(st.k_eff),float(st.GCV),float(st.replicate_level_SSE),'RECONSTRUCTED_FROZEN',0)

def extension(obj,dmax):
 zmax=float(np.log1p(dmax));h=max(1e-8,1e-6*max(zmax,1.));f0=float(obj.predict_z([zmax])[0]);f1=float(obj.predict_z([zmax-h])[0]);f2=float(obj.predict_z([zmax-2*h])[0]);s=(3*f0-4*f1+f2)/(2*h)
 material=s < -1e-8;splus=max(0.,s)
 def ext(d):
  d=np.asarray(d,float);out=pred(obj,np.minimum(d,dmax));mask=d>dmax
  if mask.any():out[mask]=f0+splus*(np.log1p(d[mask])-zmax)
  return out
 return ext,s,splus,material
