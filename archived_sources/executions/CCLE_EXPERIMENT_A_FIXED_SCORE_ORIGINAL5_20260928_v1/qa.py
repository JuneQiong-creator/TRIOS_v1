"""Independent validation from persisted scalar and fold records. No classifiers."""
import math,statistics,json,hashlib
import numpy as np
import engine as m
from engine import A,e,rt
COMP=m.COMP
def close(a,b):return a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-14)
def check(ok,label):e.require(ok,'INDEPENDENT_QA:'+label)
def audit_context(ctx):
    dest=m.location(ctx);b=m.binding(ctx['dataset_id']);checks=0;slots=0;valid=0
    for method in m.METHODS:
        d=m.read_checkpoint(dest/(m.key(method)+'.json.gz'));r=d['result'];ff=d['folds'];ids=d['ids'];n=len(ids);trios=method=='TRIOS';model,metric,alg=('SPLINE','CRS','AB') if trios else method.split('|')
        check(ids==ctx['ids'] and len(ff)==n and [q['heldout_profile'] for q in ff]==ids and r['contract_id']==m.CONTRACT,'KEYS')
        x=np.array(r['full_scores'],float);labs=np.array(r['full_labels']) if r['V'] else None;want=np.array([b['scores'][pid]['TRIOS' if trios else model+'|'+metric] for pid in ids],float);check(np.array_equal(x,want,equal_nan=True),'FULL_SCORE_SOURCE')
        h=m.xhash(x,ids);check(r['full_score_hash']==h and r['D']==(b['D2_domain'] if trios else 8.),'DOMAIN_SCORE_IDENTITY')
        if not r['V']:
            check(all(r[c] is None for c in COMP+['C_conditional']) and r['C_operational']==0 and all(not q['attempted'] and not q['success'] and q['status']=='NOT_ATTEMPTED' for q in ff),'INVALID_SEMANTICS')
        else:
            valid+=1;rx=float(np.ptp(x));pred=np.where(x<=r['tau1'],'L',np.where(x<=r['tau2'] if trios else x<r['tau2'],'I','H'))
            check(rx>0 and np.isfinite(x).all() and np.array_equal(pred,labs) and all(np.any(labs==g) for g in 'LIH'),'VALID_FULL_PROJECTION')
            for j,q in enumerate(ff):
                check(q['attempted'] and q['full_score_hash']==h==q['fold_score_hash'] and q['D_fold']==r['D'] and q['method_id']==method,'FIXED_SCORE_EVERY_FOLD')
                if q['success']:
                    pred=np.where(x<=q['tau1'],'L',np.where(x<=q['tau2'] if trios else x<q['tau2'],'I','H'));keep=np.arange(n)!=j
                    dt=.5*sum(((q[k]-r[k])/rx)**2 for k in ['tau1','tau2'])
                    check(np.array_equal(pred[keep],q['retained_labels']) and str(pred[j])==q['heldout_label'] and close(float(pred[j]==labs[j]),q['omitted_match']) and close(float(np.mean(pred[keep]==labs[keep])),q['retained_match_fraction']) and close(dt,q['d_tau_squared']),'FOLD_PROJECTION_AND_SUFFICIENT_STATS')
                else:check(all(q[k] is None for k in ['tau1','tau2','d_tau_squared','omitted_match','retained_match_fraction','retained_labels','heldout_label']),'FAILED_FOLD_UNDEFINED')
            good=[q for q in ff if q['success']];v=len(good);alo=v/n
            st=alo/(1+math.sqrt(statistics.fmean(q['d_tau_squared'] for q in good))) if v else 0.
            om=statistics.fmean(q['omitted_match'] for q in good) if v else 0.;re=statistics.fmean(q['retained_match_fraction'] for q in good) if v else 0.;sm=alo*2*om*re/(om+re) if om+re else 0.
            between=sum(np.sum(labs==g)*(float(x[labs==g].mean())-float(x.mean()))**2 for g in 'LIH');total=float(np.sum((x-x.mean())**2));pf=statistics.fmean(b['weights'][pid][model] for pid in ids);sf=int(min(np.sum(labs==g) for g in 'LIH')>=2)
            values=[between/total,st,sm,pf,sf];check(all(close(r[k],val) for k,val in zip(COMP,values)) and close(r['C_conditional'],statistics.fmean(values)) and close(r['C_operational'],r['C_conditional']),'ORIGINAL5_INDEPENDENT_FORMULA')
            check(r['n_success']==v and r['n_failed']==n-v and close(r['A_LOO'],alo),'FAILURE_DENOMINATORS')
        if trios:check(bool(r['V']),'TRIOS_INVALID_WITHHOLD_DATASET_RANKING')
        slots+=n;checks+=6+3*n
    out=dict(status='PASS',code_sha256=m.CODE,unit=ctx['unit'],tasks=81,valid_tasks=valid,nominal_fold_slots=slots,checks=checks)
    rt.atomic_json(dest/'QA.json',out,immutable=True);return out

def aggregate(results,scope):
    out=[]
    for cid in e.CID:
        rr=[r for r in results if r['dataset_id']==cid];dataset=[]
        for method in m.METHODS:
            group=[r for r in rr if r['method_id']==method];expected=1 if scope=='FULL' else 100;check(len(group)==expected,'AGGREGATE_INTENDED_KEYS');valid=[r for r in group if r['V']];nv=len(valid)
            av=nv/expected;cc=float(np.mean([r['C_conditional'] for r in valid])) if nv else None;co=sum(r['C_conditional'] for r in valid)/expected
            check((cc is None and co==0 and av==0) or close(co,av*cc),'C_OPERATIONAL_IDENTITY')
            r=dict(contract_id=m.CONTRACT,dataset_id=cid,scope=scope,method_id=method,intended_tasks=expected,valid_tasks=nv,A_valid=av,**{k:float(np.mean([r[k] for r in valid])) if nv else None for k in COMP},C_conditional=cc,C_operational=co,nominal_fold_slots=sum(r['n'] for r in group),attempted_fold_slots=sum(r['n'] for r in valid),successful_folds=sum(r['n_success'] for r in valid),failed_folds=sum(r['n_failed'] for r in valid))
            dataset.append(r)
        for r in dataset:
            for field,rank in [('C_conditional','rank_conditional'),('C_operational','rank_operational')]:r[rank]=None if r[field] is None else 1+sum(s[field] is not None and s[field]>r[field] for s in dataset)
        check(len(dataset)==81 and len({r['method_id'] for r in dataset})==81,'RANK_UNIVERSE')
        out+=dataset
    return out
