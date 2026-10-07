from __future__ import annotations
import ast,csv,hashlib,importlib.util,json,math,os,sys,time,types,zipfile,shutil
from pathlib import Path
from dataclasses import dataclass
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import spearmanr,kendalltau,pearsonr
O=Path(__file__).resolve().parent;ROOT=O.parents[1];RES=ROOT/'04_RESULTS'
AUD=ROOT/'audits/HB_Q75_raw_dose_addendum_v1.0'
P=RES/'TRIOS_CRS_LOGSPACE_AMENDMENT_v1.0_20260912_195937'
B3=RES/'TRIOS_PHASEB_BENCHMARK3_FINAL_v1.1.1';AIC=RES/'TRIOS_HB_AICC_FAIRNESS_AUDIT_v1.0'
REG=['5FU','CARBO','CARBO_A','CDDP','CDDP_A','ETO'];MODELS=['SPLINE','EMAX','HILL','LOGISTIC','WEIBULL'];ALGS=['TRIOS_ADMISSIBLE_BREAKS','TERTILES','JENKS','KMEANS','WARD','TIED_GMM']
EXPECTED=dict(zip(REG,[50.,64.,64.,40.,25.,50.]));SOURCES={};QA=[];GEOMS=[];FOLDS=[]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,(int,np.integer)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def js(rel,x):(O/rel).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n',encoding='utf-8')
def wc(rel,x):
    d=x if isinstance(x,pd.DataFrame) else pd.DataFrame(x);d.to_csv(O/rel,index=False,float_format='%.17g',lineterminator='\n');return d
def log(s):
    s=time.strftime('%H:%M:%S')+' '+s;print(s,flush=True)
    with (O/'progress.log').open('a',encoding='utf-8') as f:f.write(s+'\n')
def check(name,ok,detail=''):
    QA.append(dict(gate=name,status='PASS' if ok else 'FAIL',detail=detail));wc('06_QA/gates.csv',QA)
    if not ok:raise RuntimeError(name+': '+detail)
def bind(p,role,expected=None):
    p=Path(p);h=sha(p)
    if expected is not None and h!=str(expected).lower():raise RuntimeError('PROVENANCE_MISMATCH '+str(p))
    SOURCES[str(p)]=dict(path=str(p),role=role,sha256=h,bytes=p.stat().st_size,expected_sha256=expected or '')
    return p
def manifested(root,rel,manifest='manifest.csv'):
    bind(root/manifest,'parent manifest');m=read(root/manifest);z=m[m.relative_path==rel].iloc[0]
    return bind(root/rel,'manifest-verified input',z.sha256)
def pure(path,names,ns):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));ss=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert set(x.name for x in ss)==set(names);exec(compile(ast.Module(body=ss,type_ignores=[]),str(path),'exec'),ns)
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def labels(s):return dict(x.rsplit(':',1) for x in s.split('|'))
def sbw(x,y):
    mu=np.mean(x);be=sum(np.sum(y==g)*(np.mean(x[y==g])-mu)**2 for g in 'LIH');wi=sum(np.sum((x[y==g]-np.mean(x[y==g]))**2) for g in 'LIH');return float(be/(be+wi)) if be+wi>0 else np.nan
def harmonic(a,b):return 2*a*b/(a+b) if a+b else 0.

def setup():
    for d in ['00_BINDING','01_BENCHMARK1','02_BENCHMARK2','03_BENCHMARK3','04_OLD_VS_AMENDED','05_CROSS_BENCHMARK_PARITY','06_QA','07_PROVENANCE']:(O/d).mkdir(exist_ok=True)
    bind(Path('OMITTED_HOST_PATH/TRIOS_PhaseB_Architecture_Amendment_and_Freeze_v2.1.pdf'),'current Phase-B architecture parent')
    (O/'00_BINDING/USER_REQUEST.txt').write_text('Run complete Phase-B Benchmarks 1, 2 and 3 using audited RAW50 transport endpoints and current direct-log-space CRS; no production changes.\n',encoding='utf-8')
    for z in read(AUD/'source_hashes_before.csv').itertuples():bind(z.path,z.role,z.sha256)
    freeze=bind(RES/'TRIOS_PHASEA_v3.2_PRODUCTION_FREEZE/PRODUCTION_FREEZE_MANIFEST.json','implementation authority');f=json.loads(freeze.read_text())
    for z in f['amendment_evidence']['bound_evidence_files']:bind(P/z['relative_path'],'production-bound evidence',z['sha256'])
    bind(f['amendment_evidence']['complete_archive'],'frozen production archive',f['amendment_evidence']['complete_archive_sha256'])
    for p in (RES/'TRIOS_PHASEA_v3.2_PRODUCTION_FREEZE').rglob('*'):
        if p.is_file():bind(p,'production freeze immutable snapshot')
    for z in read(P/'09_REPORT/TRIOS_CRS_LogSpace_Amendment_Manifest.csv').itertuples():bind(P/z.relative_path,'production parent complete manifest',z.sha256)
    present=ROOT/'06_PRESENTATIONS/TRIOS_METHODS_ONLY_v1.28'
    for ext in ['tex','pdf']:bind(present/f'TRIOS_Methods_Only_Supervisor_Presentation_v1.28.{ext}','current method presentation')
    ep=read(manifested(AUD,'TRIOS_HB_candidate_grid_and_Q75_endpoint_decomposition_v1.0.csv'));D=dict(zip(ep.regimen,ep.RAW50_D_transport));check('AUDITED_RAW50_ENDPOINT_BINDING',D==EXPECTED)
    wc('00_BINDING/audited_endpoints.csv',ep)
    lock=json.loads((P/'01_METHOD_LOCK_CANDIDATE/CRS_logspace_amendment_lock.json').read_text());check('DELTA_ASSAY_K_LOCK',lock['d_low_uM']['HB_PDO_and_HB_like_PhaseC']==1. and lock['K_CRS_production']==7)
    js('00_BINDING/method_and_evaluation_binding.json',dict(method_presentation=str(present),implementation_parent=str(P),delta_assay=1.,K_CRS=7,operator_II_branch='RAW50 median A/TRG on actual measured-dose candidates',evaluation_parents=['Benchmark1 final v1.0','Benchmark3 final v1.1.1 denominator-inclusive'],curve_refits=0,production_modified=False))
    # Extract evaluation-only callables. Curve-fitting routines are never imported.
    src=RES/'HB_PDO_LAYER1_MODEL_OPTIMIZATION_REPLICATE_LEVEL_v1.1/08_SOURCE'
    solver=RES/'UPLOAD_BUNDLES_FINAL/TRIOS_FINAL_UPLOAD_HB_PDO_v1.0/02_FINAL_SCRIPTS/final_method'
    for p in [src/'raw_models.py',src/'model_fitting.py',src/'config_frozen.py',solver/'classifiers_v11_frozen.py',solver/'constrained_pspline_solver.py',ROOT/'work/trios_phaseb_v12/run_phaseb_v12.py',ROOT/'work/trios_crs_logspace_amendment_v10/preflight.py']:bind(p,'frozen implementation')
    sys.path.append(str(ROOT/'tmp/layer1_v11_pydeps'));sys.path.insert(0,str(src));from pygam.utils import b_spline_basis
    rawmodels=mod('raw_models',src/'raw_models.py');classifier=mod('v2_classifier',solver/'classifiers_v11_frozen.py')
    m=types.ModuleType('v2_evaluation_only');sys.modules[m.__name__]=m;ns=m.__dict__;ns.update(np=np,math=math,dataclass=dataclass,b_spline_basis=b_spline_basis,brentq=brentq,MODEL_FUNCTIONS=rawmodels.MODEL_FUNCTIONS,positive=rawmodels.positive,INTERNAL={x:'RAW_'+x for x in MODELS[1:]})
    pure(solver/'constrained_pspline_solver.py',{'_basis','ConstrainedPSpline'},ns)
    pure(src/'model_fitting.py',{'FittedObject'},ns)
    pure(ROOT/'work/trios_phaseb_v12/run_phaseb_v12.py',{'pred','scalar','spline_from_parameters','param_from_parameters','extension','fit_ms','analytic_ic50','root_interval','held'},ns)
    pure(ROOT/'work/trios_crs_logspace_amendment_v10/preflight.py',{'log_grid','crs_geometry'},ns)
    globals().update(NS=ns,CLF=classifier)
    paths={r:Path(next(v['path'] for v in SOURCES.values() if v['role']==r)) for r in ['frozen raw','frozen parameters','frozen fit_status','QA-clean113 membership']}
    # Match the audited production coefficient loader's parser exactly.
    raw=pd.read_csv(paths['frozen raw']);params=pd.read_csv(paths['frozen parameters']);status=pd.read_csv(paths['frozen fit_status']);member=read(paths['QA-clean113 membership']);eligible=set(member[member.included_in_QA_clean_source].pdo_id)
    ids={r:np.array(sorted(raw[(raw.regimen==r)&raw.PDO_ID.isin(eligible)].PDO_ID.unique()),str) for r in REG};check('COHORT_113',sum(map(len,ids.values()))==113 and len(eligible)==113)
    cset=json.loads(manifested(AUD,'regimen_candidate_sets.json').read_text());cset={r:list(map(float,v)) for r,v in cset.items()}
    sources=read(bind(paths['frozen raw'].parent/'source_file_manifest.csv','original raw manifest'))
    for r in REG:
        z=sources[(sources.source_class=='AUTHORITATIVE_CORRECTED_REPLICATE_SOURCE')&(sources.candidate_regimen==r)].iloc[0];p=bind(z.path,'original raw workbook',z.sha256)
        book=pd.read_excel(p,sheet_name='Sheet1',header=None);original=sorted({float(v) for v in book.iloc[:,3] if isinstance(v,(int,float)) and np.isfinite(v) and v>0})
        measured=sorted(raw[(raw.regimen==r)&raw.PDO_ID.isin(eligible)&(raw.dose_uM>0)].dose_uM.unique());check('RAW_CANDIDATE_'+r,cset[r]==measured==original)
    obj={};fs={};dm={};local=[]
    for r in REG:
        for pid in ids[r]:
            dm[r,pid]=float(raw[(raw.regimen==r)&(raw.PDO_ID==pid)].dose_uM.max())
            for model in MODELS:
                o=ns['spline_from_parameters'](r,pid,params,status,raw) if model=='SPLINE' else ns['param_from_parameters'](r,pid,model,params,raw);obj[r,pid,model]=o
                if model=='SPLINE':fun,_,_,bad=ns['extension'](o,dm[r,pid]);assert not bad;fs[r,pid]=fun
            de=float(fs[r,pid]([dm[r,pid]])[0]-fs[r,pid]([0.])[0])
            for u in cset[r]:
                a=float((fs[r,pid]([u])[0]-fs[r,pid]([0.])[0])/de) if de>1e-12 else np.nan;t=float(abs(fs[r,pid]([u])[0]-fs[r,pid]([.6*u])[0])/de) if de>1e-12 else np.nan
                local.append(dict(regimen=r,profile_id=pid,candidate_uM=u,A_i=a,TRG_i=t,delta_obs=de,informative=de>1e-12 and np.isfinite([a,t]).all()))
    local=pd.DataFrame(local);old=read(manifested(AUD,'raw_profile_candidate_quantities.csv'));joined=local.merge(old,on=['regimen','profile_id','candidate_uM'],suffixes=('_new','_audit'));check('PROFILE_STAGE_EXACT_PARITY',len(joined)==len(local) and all(np.array_equal(joined[c+'_new'],joined[c+'_audit'],equal_nan=True) for c in ['A_i','TRG_i','delta_obs']))
    wc('01_BENCHMARK1/profile_candidate_quantities.csv',local)
    globals().update(RAW=raw,IDS=ids,OBJ=obj,FS=fs,DM=dm,CSETS=cset,LOCAL=local,D=D)
    gates=[]
    for r in REG:
        d,g=stage(r,ids[r]);gates.extend(g);check('RAW50_ENDPOINT_'+r,d==D[r])
    wc('01_BENCHMARK1/RAW50_candidate_registry.csv',gates)
    # Reconstruct C1 Akaike weights from frozen AICc, then average equally across profiles.
    weights=read(manifested(AIC,'04_PROFILE_WEIGHTS/HB_profile_model_weights_C1.csv'));newweights=[]
    for (r,pid),g in weights.groupby(['unit_id','sample_id'],sort=False):
        a=g.AICc_C1.to_numpy();w=np.exp(-.5*(a-a.min()));w/=w.sum();check('PFIT_WEIGHTS_'+pid,np.allclose(w,g.weight,rtol=0,atol=1e-14))
        for model,v in zip(g.model,w):newweights.append(dict(regimen=r,profile_id=pid,model=model,weight=v))
    wp=wc('00_BINDING/recomputed_profile_C1_weights.csv',newweights);pf=wp.groupby(['regimen','model'],as_index=False).weight.mean().rename(columns={'weight':'P_fit'});ref=read(manifested(AIC,'05_PFIT/HB_regimen_Pfit_C1.csv')).rename(columns={'unit_id':'regimen'});z=pf.merge(ref,on=['regimen','model']);check('PFIT_C1_PARITY',np.allclose(z.P_fit_x,z.P_fit_y,atol=1e-14,rtol=0));wc('00_BINDING/recomputed_Pfit.csv',pf);globals()['PF']=pf.set_index(['regimen','model']).P_fit.to_dict()
    wc('07_PROVENANCE/source_hashes_before.csv',list(SOURCES.values()));log('Bound current method, reproduced six audited endpoints and frozen fit evaluation; refits=0')

def stage(r,ids):
    g=LOCAL[(LOCAL.regimen==r)&LOCAL.profile_id.isin(ids)];rows=[];chosen=None;ms=max(10,3*max(3,math.ceil(.15*len(ids))))
    for u in CSETS[r]:
        a=g[(g.candidate_uM==u)&g.informative];m=len(a);ae=float(a.A_i.median()) if m else np.nan;te=float(a.TRG_i.median()) if m else np.nan;ok=m>=ms and ae>=.8 and te<=.1
        if ok and chosen is None:chosen=u
        rows.append(dict(regimen=r,N=len(ids),candidate_uM=u,m=m,m_stage=ms,median_A=ae,median_TRG=te,passed=ok))
    for x in rows:x['selected_endpoint']=chosen;x['selected']=x['candidate_uM']==chosen
    return chosen,rows
def geometry(r,d,k,context):
    g=NS['crs_geometry'](1.,float(d),int(k));nodes=g['nodes'];w=g['beta'];assert d>1 and nodes[0]==1 and nodes[-1]==d and np.all(w>0) and abs(sum(w)-1)<1e-12
    assert np.allclose(np.diff(np.log(nodes)),np.log(d)/(k-1),rtol=0,atol=2e-15)
    for j in range(k):GEOMS.append(dict(context=context,regimen=r,D=d,K=k,node_index=j+1,dose_uM=nodes[j],weight=w[j],G_T=g['G_d']))
    return g
def score(r,ids,d,k,model='SPLINE',context=''):
    g=geometry(r,d,k,context);x=np.array([float(g['beta']@np.asarray(FS[r,p](g['nodes']) if model=='SPLINE' else NS['pred'](OBJ[r,p,model],g['nodes']),float)) for p in ids]);return x,g
def fit(x,ids,alg,tag):return NS['fit_ms'](x,ids) if alg==ALGS[0] else CLF.fit_classifier(x,ids,alg,tag)[0]
def classify(r,ids,x,alg,tag,foldscores=None):
    n=len(ids);row=dict(regimen=r,n_profiles=n,valid=False,invalid_reason='',LOO_fold_count=n,LOO_failure_count=0)
    if not np.isfinite(x).all():return row|{'invalid_reason':'NONFINITE_REQUIRED_PROFILE_SCORE'}
    fu=fit(x,ids,alg,tag+'|FULL')
    if not fu.get('valid'):return row|{'invalid_reason':fu.get('failure_code','FULL_FAILURE')}
    y=np.array(fu['labels']);R=float(np.ptp(x));d2=[];om=[];rt=[]
    for h,pid in enumerate(ids):
        keep=np.arange(n)!=h;dh=None;xa=x
        if foldscores is not None:xa,dh=foldscores(h,keep)
        fo=fit(xa[keep],ids[keep],alg,tag+'|LOO|'+pid) if xa is not None and np.isfinite(xa).all() else dict(valid=False,failure_code='NO_PASS_OR_NONFINITE_FOLD_SCORE')
        ok=bool(fo.get('valid')) and R>0
        if ok:
            pred=np.array([NS['held'](v,fo['tau1'],fo['tau2'],alg==ALGS[0]) for v in xa]);ds=.5*(((fo['tau1']-fu['tau1'])/R)**2+((fo['tau2']-fu['tau2'])/R)**2);d2.append(ds);om.append(float(pred[h]==y[h]));rt.append(float(np.mean(pred[keep]==y[keep])))
        FOLDS.append(dict(task=tag,regimen=r,left_out_id=pid,fold_valid=ok,fold_failure_code='' if ok else fo.get('failure_code','INVALID_RANGE'),D_transport_minus_h=dh,tau1_fold=fo.get('tau1'),tau2_fold=fo.get('tau2'),d_h_squared=ds if ok else np.nan,omitted_match=om[-1] if ok else np.nan,retained_match=rt[-1] if ok else np.nan,fold_labels='|'.join(f'{p}:{z}' for p,z in zip(ids,pred)) if ok else ''))
    A=len(d2)/n;stc=1/(1+math.sqrt(np.mean(d2))) if d2 else np.nan;smc=harmonic(np.mean(om),np.mean(rt)) if d2 else np.nan
    return row|dict(valid=True,tau1=fu['tau1'],tau2=fu['tau2'],group_n_L=sum(y=='L'),group_n_I=sum(y=='I'),group_n_H=sum(y=='H'),minimum_group_size=min(sum(y==g) for g in 'LIH'),S_BW=sbw(x,y),S_tau=A*stc if d2 else 0.,S_match=A*smc if d2 else 0.,S_tau_conditional_valid_folds=stc,S_match_conditional_valid_folds=smc,A_LOO=A,LOO_failure_count=n-len(d2),I_SF=float(min(sum(y==g) for g in 'LIH')>=2),full_labels='|'.join(f'{p}:{z}' for p,z in zip(ids,y)))
def ranks(rows,key,components,scorecol):
    df=pd.DataFrame(rows);check(scorecol+'_VALIDITY',bool(df.valid.all()),'Matched layer tasks must be fully evaluable; no rescue')
    df[scorecol]=df[components].mean(axis=1);rank=df.groupby(key,as_index=False)[components+[scorecol]].mean();rank['rank']=rank[scorecol].rank(method='min',ascending=False).astype(int);return df,rank.sort_values('rank')

def benchmark1():
    rows1=[];rows2=[];rows3=[];scores=[];support=[]
    for r in REG:
        ids=IDS[r]
        for model in MODELS:
            x,g=score(r,ids,D[r],7,model,'B1_MODEL_'+model);z=classify(r,ids,x,ALGS[0],f'B1FINAL|MATCHED_L1|{r}|{model}');z.update(model=model,D_transport=D[r],P_fit=PF[r,model]);rows1.append(z)
            scores.extend(dict(regimen=r,profile_id=p,model=model,CRS=v) for p,v in zip(ids,x))
        x,_=score(r,ids,D[r],7,context='B1_REP_CRS');sup=np.mean([DM[r,p]>=D[r] for p in ids]);z=classify(r,ids,x,ALGS[0],f'B1V2|REP|{r}|CRS');z.update(representation='CRS',D=D[r],P_support=sup);rows2.append(z)
        for p in ids:support.append(dict(regimen=r,profile_id=p,representation='CRS',D=D[r],observed_dmax=DM[r,p],extrapolation_required=D[r]>DM[r,p]))
        for d in [50,100,150,200,250]:
            # Same frozen operational spline, physical-dose normalized integral.
            xa=np.array([quad(lambda t:float(FS[r,p]([t])[0]),0.,d,epsabs=1e-10,epsrel=1e-10,limit=300)[0]/d for p in ids]);z=classify(r,ids,xa,ALGS[0],f'B1V2|REP|{r}|AUC{d}');z.update(representation=f'AUC[0,{d}]',D=d,P_support=np.mean([DM[r,p]>=d for p in ids]));rows2.append(z)
            scores.extend(dict(regimen=r,profile_id=p,model='SPLINE',representation=f'AUC[0,{d}]',AUC=v) for p,v in zip(ids,xa))
            for p in ids:support.append(dict(regimen=r,profile_id=p,representation=f'AUC[0,{d}]',D=d,observed_dmax=DM[r,p],extrapolation_required=d>DM[r,p]))
        for alg in ALGS:
            z=classify(r,ids,x,alg,f'B1FINAL|MATCHED_L3|{r}|{alg}');z.update(stratifier=alg,P_SF=z.get('I_SF'));rows3.append(z)
        log('B1 matched layers complete: '+r)
    out=[]
    for name,rows,key,comp,metric in [('model',rows1,'model',['S_BW','S_tau','S_match','P_fit'],'C_model'),('representation',rows2,'representation',['S_BW','S_tau','S_match','P_support'],'C_representation'),('algorithm',rows3,'stratifier',['S_BW','S_tau','S_match','P_SF'],'C_algorithm')]:
        df,ra=ranks(rows,key,comp,metric);wc(f'01_BENCHMARK1/{name}_tasks.csv',df);wc(f'01_BENCHMARK1/{name}_ranking.csv',ra)
        for z in ra.to_dict('records'):out.append(dict(layer=name,candidate=z[key],rank=z['rank'],score=z[metric],**{c:z[c] for c in comp}))
    wc('01_BENCHMARK1/profile_scores.csv',scores);wc('01_BENCHMARK1/profile_support.csv',support);wc('TRIOS_PhaseB_Benchmark1_v2.0_Summary.csv',out)
    ff=pd.DataFrame(FOLDS);wc('01_BENCHMARK1/classifier_LOO_folds.csv',ff[ff.task.str.startswith('B1')]);check('B1_CLASSIFIER_LOO_COMPLETE',bool(ff[ff.task.str.startswith('B1')].fold_valid.all()))

def benchmark2():
    tasks=[];scores=[];quadrows=[];cache={}
    for r in REG:
        ids=IDS[r]
        for k in range(3,8):
            x,g=score(r,ids,D[r],k,context='B2');cache[r,k]=x;z=classify(r,ids,x,ALGS[0],f'B2V2|{r}|K{k}');z.update(K=k,D_transport=D[r],G_T=g['G_d'],G_norm=g['G_norm']);tasks.append(z)
            for p,v in zip(ids,x):
                scores.append(dict(regimen=r,profile_id=p,K=k,CRS=v,D_transport=D[r]));f=FS[r,p];q=np.arange(k-1,0,-1);numer=sum(qj*quad(lambda t:float(f([t])[0]),a,b,epsabs=1e-10,epsrel=1e-10,limit=300)[0] for qj,a,b in zip(q,g['nodes'][:-1],g['nodes'][1:]));target=numer/g['M'];rr=float(f([D[r]])[0]-f([0.])[0]);err=abs(v-target)
                quadrows.append(dict(regimen=r,profile_id=p,K=k,continuous_target=target,CRS=v,response_range_0_D=rr,E_quad=err,E_quad_norm=err/(rr+1e-12)))
        log('B2 grids and quadrature complete: '+r)
    df=pd.DataFrame(tasks);check('B2_FULL_VALID',bool(df.valid.all()));df['C_common']=df[['S_BW','S_tau','S_match']].mean(axis=1);fidelity=[]
    for r in REG:
        ref=df[(df.regimen==r)&(df.K==7)].iloc[0];y=np.array([labels(ref.full_labels)[p] for p in IDS[r]])
        for k in range(3,8):
            z=df[(df.regimen==r)&(df.K==k)].iloc[0];pred=np.array([labels(z.full_labels)[p] for p in IDS[r]]);pairs=[a+b for a,b in zip(y,pred)]
            fidelity.append(dict(regimen=r,K=k,Spearman=spearmanr(cache[r,k],cache[r,7]).statistic,Kendall=kendalltau(cache[r,k],cache[r,7]).statistic,Pearson=pearsonr(cache[r,k],cache[r,7]).statistic,agreement=np.mean(y==pred),balanced_agreement=np.mean([np.mean(pred[y==c]==c) for c in 'LIH']),L_I=sum(p in ['LI','IL'] for p in pairs),I_H=sum(p in ['IH','HI'] for p in pairs),L_H=sum(p in ['LH','HL'] for p in pairs)))
    fi=wc('02_BENCHMARK2/fidelity_by_regimen_K.csv',fidelity);wc('02_BENCHMARK2/tasks.csv',df);wc('02_BENCHMARK2/profile_scores.csv',scores);qr=wc('02_BENCHMARK2/quadrature.csv',quadrows)
    summary=df.groupby('K',as_index=False)[['S_BW','S_tau','S_match','C_common','G_T']].mean().merge(fi.groupby('K',as_index=False)[['Spearman','Kendall','Pearson','agreement','balanced_agreement']].mean(),on='K').merge(fi.groupby('K',as_index=False)[['L_I','I_H','L_H']].sum(),on='K').merge(qr.groupby(['K','regimen'],as_index=False).E_quad_norm.mean().groupby('K',as_index=False).E_quad_norm.mean(),on='K');wc('TRIOS_PhaseB_Benchmark2_v2.0_K_Summary.csv',summary)
    ff=pd.DataFrame(FOLDS);wc('02_BENCHMARK2/classifier_LOO_folds.csv',ff[ff.task.str.startswith('B2')]);check('B2_CLASSIFIER_LOO_COMPLETE',bool(ff[ff.task.str.startswith('B2')].fold_valid.all()));return cache

def conventional_metric(r,p,model,rep,b):
    obj=OBJ[r,p,model]
    if rep=='IC50':v=NS['analytic_ic50'](obj);return -v if np.isfinite(v) else np.nan
    f0=NS['scalar'](obj,0.);fb=NS['scalar'](obj,b)
    if rep=='EMAX':return fb
    v=NS['root_interval'](obj,f0+.5*(fb-f0),b);return -v if np.isfinite(v) else np.nan
def aggregate(tasks):
    rows=[]
    for mid,g in tasks.groupby('method_id'):
        valid=g.valid.astype(bool);nv=int(valid.sum());a=nv/6;cv=g.loc[valid,'C_task'].mean() if nv else np.nan
        z=dict(method_id=mid,valid_regimen_count=nv,A_valid=a,C_conditional_valid=cv,C_operational=a*cv if nv else 0.)
        z.update({c:g.loc[valid,c].mean() for c in ['S_BW','S_tau','S_match','P_fit','I_SF']});rows.append(z)
    out=pd.DataFrame(rows);out['rank']=out.C_operational.rank(method='min',ascending=False).astype(int);return out.sort_values(['rank','method_id'])

def benchmark3(cache):
    registry=read(manifested(B3,'01_METHOD_REGISTRY/method_registry.csv'));old=read(manifested(B3,'04_FULL_DATA_STRATIFICATION/method_regimen_full_results.csv'));oldfold=read(manifested(B3,'05_LOO_PERTURBATIONS/method_native_LOO_registry.csv'));oldscore=read(manifested(B3,'02_PROFILE_LEVEL_SCORES/profile_level_method_native_scores.csv'));oldrank=read(manifested(B3,'07_OPERATIONAL_SCORING/method_level_operational_table.csv'))
    check('B3_REGISTRY_61_NO_AUC',len(registry)==61 and sum(registry.is_TRIOS)==1 and not registry.representation.eq('AUC').any());wc('03_BENCHMARK3/method_registry.csv',registry)
    tasks=[];prof=[];convfold=[];parity=[];stagefold=[]
    # All comparator metrics reconstructed from parity-verified native scores and native fold cutoffs.
    # Fold-specific finite-range endpoints are recalculated, including held-out evaluation.
    for method in registry[~registry.is_TRIOS].itertuples():
        for r in REG:
            ids=IDS[r];n=len(ids);b=min(DM[r,p] for p in ids);x=np.array([conventional_metric(r,p,method.curve_model,method.representation,b) for p in ids]);arch=oldscore[(oldscore.method_id==method.method_id)&(oldscore.regimen==r)].set_index('profile_id');stored=np.array([arch.loc[p,'oriented_score'] for p in ids]);assert np.allclose(x,stored,rtol=0,atol=1e-12,equal_nan=True)
            row=old[(old.method_id==method.method_id)&(old.regimen==r)].iloc[0].to_dict();prof.extend(dict(method_id=method.method_id,regimen=r,profile_id=p,score=v) for p,v in zip(ids,x))
            if not row['valid']:
                assert not np.isfinite(x).all() or row['invalid_reason'];tasks.append(row);continue
            y=np.array([labels(row['full_labels'])[p] for p in ids]);sb=sbw(x,y);fs=oldfold[(oldfold.method_id==method.method_id)&(oldfold.regimen==r)];assert len(fs)==n;d2=[];om=[];rt=[];R=np.ptp(x)
            for fo in fs.itertuples():
                h=list(ids).index(fo.left_out_id);keep=np.arange(n)!=h;bf=min(DM[r,p] for p in ids[keep]);xa=x if method.representation=='IC50' else np.array([conventional_metric(r,p,method.curve_model,method.representation,bf) for p in ids]);ok=bool(fo.fold_valid)
                if ok:
                    pred=np.array([NS['held'](v,fo.tau1_fold,fo.tau2_fold,False) for v in xa]);dh=.5*(((fo.tau1_fold-row['tau1'])/R)**2+((fo.tau2_fold-row['tau2'])/R)**2);ml=float(pred[h]==y[h]);mr=float(np.mean(pred[keep]==y[keep]));assert np.allclose([dh,ml,mr],[fo.d_h_squared,fo.omitted_match,fo.retained_match],rtol=0,atol=1e-12);d2.append(dh);om.append(ml);rt.append(mr)
                convfold.append(dict(method_id=method.method_id,regimen=r,left_out_id=fo.left_out_id,fold_valid=ok,fold_failure_code=fo.fold_failure_code,finite_range_upper=bf,tau1_fold=fo.tau1_fold,tau2_fold=fo.tau2_fold,d_h_squared=dh if ok else np.nan,omitted_match=ml if ok else np.nan,retained_match=mr if ok else np.nan,origin='RECONSTRUCTED_FROM_HASH_VERIFIED_NATIVE_CUTOFFS'))
            a=len(d2)/n;stc=1/(1+math.sqrt(np.mean(d2))) if d2 else np.nan;smc=harmonic(np.mean(om),np.mean(rt)) if d2 else np.nan;st=a*stc if d2 else 0.;sm=a*smc if d2 else 0.;pf=PF[r,method.curve_model];sf=float(min(sum(y==c) for c in 'LIH')>=2);ct=np.mean([sb,st,sm,pf,sf])
            parity.append(dict(method_id=method.method_id,regimen=r,max_component_error=max(abs(sb-row['S_BW']),abs(st-row['S_tau']),abs(sm-row['S_match']),abs(pf-row['P_fit']),abs(ct-row['C_task']))))
            row.update(S_BW=sb,S_tau=st,S_match=sm,P_fit=pf,I_SF=sf,C_task=ct,A_LOO=a,LOO_failure_count=n-len(d2));tasks.append(row)
        if method.stratifier=='TIED_GMM':log('B3 conventional metrics reconstructed: '+method.method_id.rsplit('|',1)[0])
    check('CONVENTIONAL_RECONSTRUCTION_PARITY',max(z['max_component_error'] for z in parity)<1e-12);wc('03_BENCHMARK3/conventional_metric_parity.csv',parity);wc('03_BENCHMARK3/conventional_native_LOO_reconstructed.csv',convfold)
    for r in REG:
        ids=IDS[r];x,g=score(r,ids,D[r],7,context='B3_FULL');check('B2_B3_EXACT_CRS_'+r,np.array_equal(x,cache[r,7]));prof.extend(dict(method_id='TRIOS',regimen=r,profile_id=p,score=v) for p,v in zip(ids,x))
        def foldscore(h,keep):
            dh,trace=stage(r,ids[keep]);stagefold.extend(dict(left_out_id=ids[h],**z) for z in trace)
            return (score(r,ids,dh,7,context='B3_LOO_'+ids[h])[0],dh) if dh is not None else (None,None)
        z=classify(r,ids,x,ALGS[0],f'B3V2|TRIOS|{r}',foldscore);z.update(method_id='TRIOS',D_transport=D[r],P_fit=PF[r,'SPLINE']);z['C_task']=np.mean([z[c] for c in ['S_BW','S_tau','S_match','P_fit','I_SF']]) if z['valid'] else np.nan;tasks.append(z);log('B3 TRIOS native LOO complete: '+r)
    ta=wc('03_BENCHMARK3/method_regimen_results.csv',tasks);pr=wc('03_BENCHMARK3/profile_scores.csv',prof);ff=pd.DataFrame(FOLDS);tf=wc('03_BENCHMARK3/TRIOS_native_LOO.csv',ff[ff.task.str.startswith('B3')]);wc('03_BENCHMARK3/TRIOS_LOO_candidate_registry.csv',stagefold);check('TRIOS_113_FOLDS',len(tf)==113 and not tf.duplicated(['regimen','left_out_id']).any())
    rank=aggregate(ta);wc('TRIOS_PhaseB_Benchmark3_v2.0_61_Method_Rank.csv',rank);return ta,pr,tf,rank

def comparisons(ta,pr,rank):
    old=read(manifested(P,'04_PHASE_B/B3/04_FULL_DATA_STRATIFICATION/method_regimen_full_results.csv','09_REPORT/TRIOS_CRS_LogSpace_Amendment_Manifest.csv'));oldpr=read(manifested(P,'04_PHASE_B/B3/02_PROFILE_LEVEL_SCORES/TRIOS_exact_logspace_scores.csv','09_REPORT/TRIOS_CRS_LogSpace_Amendment_Manifest.csv'));oldrank=aggregate(old);wc('04_OLD_VS_AMENDED/old_rank_reconstructed.csv',oldrank)
    ot=old[old.method_id=='TRIOS'].set_index('regimen');nt=ta[ta.method_id=='TRIOS'].set_index('regimen');profile=[];regrows=[]
    for r in REG:
        a=ot.loc[r];b=nt.loc[r];la=labels(a.full_labels);lb=labels(b.full_labels);av=oldpr[oldpr.regimen==r].set_index('profile_id').score;bv=pr[(pr.method_id=='TRIOS')&(pr.regimen==r)].set_index('profile_id').score
        for p in IDS[r]:profile.append(dict(regimen=r,profile_id=p,old_CRS=av[p],amended_CRS=bv[p],old_label=la[p],amended_label=lb[p],shift=abs('LIH'.index(la[p])-'LIH'.index(lb[p]))))
        rr=dict(regimen=r,CRS_Spearman=spearmanr(av.loc[IDS[r]],bv.loc[IDS[r]]).statistic)
        for c in ['D_transport','tau1','tau2','group_n_L','group_n_I','group_n_H','S_BW','S_tau','S_match','P_fit','I_SF','C_task','A_LOO','LOO_failure_count']:rr.update({c+'_old':a[c],c+'_amended':b[c],c+'_delta':b[c]-a[c]})
        regrows.append(rr)
    pc=wc('04_OLD_VS_AMENDED/profile_score_label_comparison.csv',profile);wc('04_OLD_VS_AMENDED/regimen_comparison.csv',regrows)
    a=oldrank[oldrank.method_id=='TRIOS'].iloc[0];b=rank[rank.method_id=='TRIOS'].iloc[0];rows=[dict(metric=c,old=a[c],amended=b[c],delta=b[c]-a[c]) for c in ['S_BW','S_tau','S_match','P_fit','I_SF','C_conditional_valid','A_valid','C_operational','rank']];wc('TRIOS_PhaseB_Old_vs_Amended_TRIOS_v2.0.csv',rows);return pc

def main():
    setup();benchmark1();cache=benchmark2();ta,pr,tf,rank=benchmark3(cache);pc=comparisons(ta,pr,rank)
    wc('05_CROSS_BENCHMARK_PARITY/CRS_geometry_registry.csv',GEOMS);wc('07_PROVENANCE/source_hashes_before.csv',list(SOURCES.values()))
    tr=rank[rank.method_id=='TRIOS'].iloc[0];second=rank[rank['rank']==2].iloc[0]
    js('analysis_complete.json',dict(TRIOS=tr.to_dict(),rank2=second.to_dict(),gap=tr.C_operational-second.C_operational,LOO_success=int(tf.fold_valid.sum()),LOO_failure=int((~tf.fold_valid).sum()),A_LOO=float(tf.fold_valid.mean()),label_agreement=float((pc['shift']==0).mean()),adjacent_changes=int((pc['shift']==1).sum()),extreme_changes=int((pc['shift']==2).sum()),endpoints=D))
    log('Numerical analyses complete; reporting and independent QA remain')
if __name__=='__main__':main()
