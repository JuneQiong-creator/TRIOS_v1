"""Postprocessing only: consume completed checkpoints, no fitting/scoring/classification calls."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,gzip,pickle,hashlib,csv,time,math,zipfile,importlib.util,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
O=Path(__file__).resolve().parents[1];R=O.parents[1]
P=R/'deliveries/SECTION4_READONLY_POSTPROCESSING_20260929_v1'
C=R/'04_RESULTS/TRIOS_CRS_LOGSPACE_AMENDMENT_v1.0_20260912_195937/05_PHASE_C/E4'
S1=R/'04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_4_V3/STAGE1_UPPER_DOMAIN_SUFFICIENCY_v3.0'
S2=R/'04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_4_V3/STAGE2_INTERNAL_MEASUREMENT_COMPRESSION_v3.2/EXECUTION_v3.2b'
TOL=5e-12
QA=[];INDEX=[];START=time.time()
def log(s):
    line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+s;print(line,flush=True)
    with (O/'REPORTS/POSTPROCESSING_RUNTIME.log').open('a',encoding='utf8') as f:f.write(line+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
@lru_cache(maxsize=128)
def raw_source(p):return {z['meta']['cohort_id']:z for z in load(p)['cohorts']},sha(p)
def save(rel,d):
    p=O/rel;p.parent.mkdir(parents=True,exist_ok=True)
    (d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)).to_csv(p,index=False,float_format='%.17g',compression='infer')
def js(rel,o):
    p=O/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,default=str)+'\n',encoding='utf8')
def check(name,value,detail=''):
    QA.append(dict(check=name,status='PASS' if value else 'FAIL',detail=str(detail)))
    if not value:save('EXECUTION_QA_CHECKS.csv',QA);raise AssertionError((name,detail))
def close(name,x,y,tol=TOL):check(name,np.allclose(x,y,rtol=0,atol=tol,equal_nan=True))
def writer(path,fields):
    f=gzip.open(O/path,'wt',encoding='utf8',newline='');w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();return f,w
def extract(arm,lane,registry):
    base=O/f'LANE_{lane}';paths=sorted((base/f'RAW_FITS/{arm}').glob('*.pkl.gz'))
    check(arm+'_10200_checkpoints',len(paths)==10200)
    tasks=[];ids_seen=set();profiles=folds=success=newvals=0;fitmax=0.;hashrows=[]
    ff,fw=writer(f'LANE_{lane}/RAW_FITS/{arm}_FITS.csv.gz',['arm','cohort_id','sample_id','fit_valid','lambda','GCV','EDF','SSE','solver_status','solver_iterations','intercept','coefficients','edge','raw_rows_used','gcv_evaluations'])
    sf,sw=writer(f'LANE_{lane}/TASKS/{arm}_SCORES.csv.gz',['arm','cohort_id','sample_id','CRS','label','truth_label'])
    lf,lw=writer(f'LANE_{lane}/FOLDS/{arm}_FOLDS.csv.gz',['arm','cohort_id','fold','omitted_sample_id','success','failure_code','D_transport','K','geometry_sha256','tau1','tau2','omitted_score','omitted_label','omitted_match','retained_match','d_i_squared'])
    rawsrc=None;lastsh=None;configsha=sha(O/'01_DESIGN/LOCK.json')
    try:
        for ix,p in enumerate(paths):
            q=load(p);cid=q['meta']['cohort_id'];assert q['complete'] and cid not in ids_seen;ids_seen.add(cid)
            n=len(q['ids']);profiles+=n;assert len(set(q['ids']))==n and q['raw'].shape==(n,{'J3':4,'J4':5,'J6':7,'H250':8}[arm],3)
            assert q['key'][2]==cid and q['key'][-1]==configsha
            ar=np.ascontiguousarray(q['raw']);assert hashlib.sha256(ar.tobytes()).hexdigest()==q['raw_provenance']['raw_sha256']
            sh=registry[cid]['shard']
            if sh!=lastsh:
                sourcepath=(S1/'02_PROSPECTIVE_RAW_BANKS/raw_shards' if lane=='A' else R/'work/trios_phasec_sim_e4a_v20/raw_bank')/f'raw_shard_{sh:02d}.pkl.gz'
                rawsrc,sourcehash=raw_source(sourcepath);lastsh=sh
            src=rawsrc[cid];assert src['ids']==q['ids'] and np.array_equal(src['truth'],q['truth']) and src['paired_keys']==q['paired_keys'] and q['key'][3]==sourcehash
            if arm in ('J3','J4'):assert np.array_equal(ar,src['Y_by_assay']['H150'][:,{'J3':[0,1,4,7],'J4':[0,1,3,5,7]}[arm],:])
            elif arm=='J6':
                assert np.array_equal(ar[:,[0,1,6],:],src['Y_by_assay']['H150'][:,[0,1,7],:]);newvals+=q['raw_provenance']['new_replicates'];assert q['raw_provenance']['new_replicates']==n*12
            else:assert np.array_equal(ar,src['Y_control'])
            for pid,f in zip(q['ids'],q['fits']):
                row=dict(arm=arm,cohort_id=cid,sample_id=pid,fit_valid=f is not None,raw_rows_used=ar.shape[1]*3,gcv_evaluations=21 if lane=='A' else 1)
                if f is not None:
                    assert np.isfinite([f['GCV'],f['SSE'],f['EDF']]).all() and f['lambda'] in np.logspace(-6,4,21)
                    row.update(f);row['coefficients']=json.dumps(np.asarray(f['coefficients']).tolist());row['edge']=json.dumps(list(f['edge']))
                fw.writerow(row)
            if lane=='B':
                assert q['parity']['status']=='PASS';fitmax=max(fitmax,max(max(r[k] for k in ('SSE','GCV','EDF')) for r in q['parity']['profiles']))
                rp=base/f'TASKS/H250/{cid}.pkl.gz';qq=load(rp);assert qq['key']==q['key'] and qq['fit_checkpoint_sha256']==sha(p);res=qq['result']
                hashrows.append(dict(path=str(rp.relative_to(O)),sha256=sha(rp)))
            else:res=q['result']
            t=res['task'];assert t['cohort_id']==cid and t['arm']==arm
            assert abs(t['factor_weight']-1/3/3/{12:3,18:12,24:19}[n]/20/5)<1e-20
            for gg in res['geometries'].values():
                nodes=np.asarray(gg['nodes']);beta=np.asarray(gg['beta']);assert gg['K']==7 and nodes[0]==1 and nodes[-1]==gg['D_transport'] and np.all(np.diff(nodes)>0) and np.all(beta>0) and abs(sum(beta)-1)<1e-14
                assert gg['geometry_sha256']==hashlib.sha256(np.r_[nodes,beta].tobytes()).hexdigest()
            if t['valid']:
                assert len(res['scores'])==n and len(res['loo'])==n
                scores=np.array([r['CRS'] for r in res['scores']]);labels=np.array([r['label'] for r in res['scores']]);assert [r['sample_id'] for r in res['scores']]==q['ids']
                assert np.array_equal(labels,np.where(scores<=t['tau1'],0,np.where(scores<=t['tau2'],1,2)))
                good=[]
                for h,f in enumerate(res['loo']):
                    assert f['fold']==h+1 and f['omitted_sample_id']==q['ids'][h] and f['K']==7
                    if np.isfinite(f['D_transport']):assert f['geometry_sha256']==res['geometries'][str(float(f['D_transport']))]['geometry_sha256']
                    if f['success']:
                        keep=np.arange(n)!=h;ll=np.asarray(f['retained_labels']);xx=np.asarray(f['retained_scores']);assert len(ll)==n-1
                        assert np.array_equal(ll,np.where(xx<=f['tau1'],0,np.where(xx<=f['tau2'],1,2)))
                        held=0 if f['omitted_score']<=f['tau1'] else 1 if f['omitted_score']<=f['tau2'] else 2
                        assert held==f['omitted_label'] and f['omitted_match']==float(held==labels[h]) and f['retained_match']==float(np.mean(ll==labels[keep]))
                        displacement=.5*(((f['tau1']-t['tau1'])/np.ptp(scores))**2+((f['tau2']-t['tau2'])/np.ptp(scores))**2)
                        assert abs(displacement-f['d_i_squared'])<TOL;good.append(f)
                    lw.writerow(dict(arm=arm,cohort_id=cid,**f))
                al=len(good)/n;st=al/(1+math.sqrt(np.mean([f['d_i_squared'] for f in good]))) if good else 0
                u=np.mean([f['omitted_match'] for f in good]) if good else 0;v=np.mean([f['retained_match'] for f in good]) if good else 0;sm=al*2*u*v/(u+v) if u+v else 0
                grand=np.mean(scores);between=sum(np.sum(labels==k)*(np.mean(scores[labels==k])-grand)**2 for k in range(3));within=sum(np.sum((scores[labels==k]-np.mean(scores[labels==k]))**2) for k in range(3));bw=between/(between+within)
                assert np.allclose([al,st,sm,bw,(st+sm+bw)/3],[t[k] for k in ('A_LOO','S_tau','S_match','S_BW','C_common')],rtol=0,atol=TOL)
                folds+=n;success+=len(good)
                for r in res['scores']:sw.writerow(r)
            else:assert not res['scores'] and not res['loo'] and t['failure_category']
            tasks.append(t);hashrows.append(dict(path=str(p.relative_to(O)),sha256=sha(p)))
            if (ix+1)%2000==0:log(f'{arm}: validated/exported {ix+1}/10200 checkpoints')
    finally:ff.close();sf.close();lf.close()
    check(arm+'_population',len(ids_seen)==10200 and profiles==212400 and ids_seen==set(registry))
    check(arm+'_checkpoint_raw_fits_full_native_fold_QA',True,f'profiles={profiles}, folds={folds}, successful={success}, fit_parity_max={fitmax}')
    if arm=='J6':check('J6_new_replicate_count',newvals==2548800)
    save(f'LANE_{lane}/TASKS/{arm}_TASKS.csv.gz',tasks);save(f'LANE_{lane}/QA/{arm}_CHECKPOINT_MANIFEST.csv',hashrows)
    return pd.DataFrame(tasks),dict(arm=arm,profiles=profiles,folds=folds,successful_folds=success,new_replicates=newvals,fit_parity_max=fitmax)
def summary(d):
    rows=[];metrics=['ARI','BA','MacroF1','Extreme','S_BW','S_tau','S_match','A_LOO','C_common']
    for arm,g in d.groupby('arm'):
        groups=[('ALL','ALL',g)]+[(field,str(k),z) for field in ('separation','n','family') for k,z in g.groupby(field)]
        for field,key,z in groups:
            w=z.factor_weight.to_numpy();v=z.valid.to_numpy(bool);mass=w[v].sum();r=dict(arm=arm,stratum_type=field,stratum=key,intended_tasks=len(z),valid_tasks=int(v.sum()),invalid_tasks=int((~v).sum()),intended_profiles=int(z.n.sum()),valid_profiles=int(z.loc[v,'n'].sum()),total_weight=w.sum(),valid_weight=mass,A_valid=mass/w.sum())
            for c in metrics:r[c]=np.dot(w[v],z.loc[v,c])/mass if mass else np.nan
            r['C_conditional']=r['C_common'];r['C_operational']=np.dot(w[v],z.loc[v,'C_common'])/w.sum() if mass else 0
            assert not mass or abs(r['C_operational']-r['A_valid']*r['C_conditional'])<TOL
            if arm.startswith('J'):r.update(J=int(arm[1:]),wells_per_profile=3*(int(arm[1:])+1))
            rows.append(r)
    return pd.DataFrame(rows)
def paired(d,arms,reference,directory):
    # Reuse the accepted derived-statistics functions without calling its main or writing to its parent.
    spec=importlib.util.spec_from_file_location('accepted_postprocessing',P/'code/section4_postprocess.py');pp=importlib.util.module_from_spec(spec);spec.loader.exec_module(pp)
    reg=pd.read_csv(S2/'12_BOOTSTRAP/stratum_registry.csv').sort_values('stratum_id');drawfiles=sorted((S2/'12_BOOTSTRAP').glob('draw_registry_*.csv.gz'))
    assert len(drawfiles)==50
    allrows=[];allreps=[];members=[]
    for arm in arms:
        a=d[d.arm==arm].set_index('cohort_id').sort_values(['n','pattern_id','separation','base_block_id','family']);b=d[d.arm==reference].set_index('cohort_id').loc[a.index]
        for c in ('n','pattern_id','separation','base_block_id','family','factor_weight'):assert np.allclose(a[c],b[c],rtol=0,atol=1e-18) if c=='factor_weight' else a[c].equals(b[c])
        x=pp.paired_sufficient(a,b);blocks=[]
        for sr in reg.itertuples():
            mask=((a.n==sr.n)&(a.pattern_id==sr.pattern_id)&(a.separation==sr.separation)).to_numpy();assert mask.sum()==100
            blocks.append([x[mask&(a.base_block_id.to_numpy()==k)].sum(axis=0) for k in range(1,21)])
        blocks=np.asarray(blocks);draws={s:np.empty((5000,len(pp.PAIR_FIELDS))) for s in pp.SEPS}
        for chunk,p in enumerate(drawfiles):
            z=pd.read_csv(p).sort_values(['bootstrap_id','stratum_id']);assert np.array_equal(z.bootstrap_id,np.repeat(np.arange(chunk*100+1,chunk*100+101),102))
            ix=np.array([list(map(int,s.split('|'))) for s in z.base_block_ids]).reshape(100,102,20)-1
            sums={s:np.zeros((100,x.shape[1])) for s in pp.SEPS}
            for st in range(102):ss=blocks[st][ix[:,st,:]].sum(axis=1);sums['ALL']+=ss;sums[reg.iloc[st].separation]+=ss
            for s in pp.SEPS:
                assert np.all(sums[s][:,:4]>0),'BOOTSTRAP_ZERO_DENOMINATOR';draws[s][chunk*100:(chunk+1)*100]=pp.pair_values(sums[s])[2]
        cv=a.valid.to_numpy(bool)&b.valid.to_numpy(bool)
        for sep in pp.SEPS:
            mask=np.ones(len(a),bool) if sep=='ALL' else a.separation.to_numpy()==sep;ss=x[mask].sum(axis=0);left,right,delta=pp.pair_values(ss);ci=np.quantile(draws[sep],[.025,.975],axis=0,method='linear')
            for j,c in enumerate(pp.PAIR_FIELDS):
                if c in ('C_truth','C_E4'):continue
                allrows.append(dict(arm=arm,reference=reference,separation=sep,metric=c,estimand_domain='all_intended' if c in ('A_valid','C_operational_E4') else 'common_valid',left=left[j],right=right[j],difference=delta[j],CI95_low=ci[0,j],CI95_high=ci[1,j],intended_pairs=int(mask.sum()),common_valid_pairs=int((mask&cv).sum()),total_weight=ss[0],common_valid_weight=ss[3],bootstrap_B=5000))
            reps=pd.DataFrame(draws[sep],columns=pp.PAIR_FIELDS).drop(columns=['C_truth','C_E4']);reps.insert(0,'bootstrap_id',np.arange(1,5001));reps.insert(0,'separation',sep);reps.insert(0,'reference',reference);reps.insert(0,'arm',arm);allreps.append(reps)
        mm=a[['n','pattern_id','separation','base_block_id','family','factor_weight']].copy();mm['left_valid']=a.valid;mm['right_valid']=b.valid;mm['common_valid']=cv;mm['arm']=arm;mm['reference']=reference;members.append(mm.reset_index())
        if arm=='J5' and reference=='J7':
            old=pd.read_csv(P/'tables/E4_J5_MINUS_J7_PAIRED_CONTRASTS_CI.csv',float_precision='round_trip')
            now=pd.DataFrame([r for r in allrows if r['arm']=='J5']).set_index(['separation','metric']);old=old.set_index(['separation','metric']).loc[now.index]
            for c in ('difference','CI95_low','CI95_high'):close('J5_J7_accepted_'+c,now[c],old[c])
            rr=pd.read_csv(P/'tables/E4_PAIRED_BOOTSTRAP_REPLICATES.csv.gz',float_precision='round_trip').set_index(['separation','bootstrap_id']);nn=pd.concat([r for r in allreps if r.iloc[0]['arm']=='J5']).set_index(['separation','bootstrap_id'])
            for c in pp.PAIR_FIELDS:
                if c not in ('C_truth','C_E4'):close('J5_J7_all5000_'+c,nn[c],rr.loc[nn.index,c])
        log(f'{arm} versus {reference}: original 5000 paired draws completed')
    save(directory+'/CONTRASTS.csv',allrows);save(directory+'/BOOTSTRAP_5000.csv.gz',pd.concat(allreps));save(directory+'/PAIR_MEMBERSHIP.csv.gz',pd.concat(members));return pd.DataFrame(allrows)
def main():
    log('Postprocessing started; completed science checkpoints read only; no scientific rerun')
    cfg=json.loads((O/'01_DESIGN/LOCK.json').read_text());registry={t['cohort_id']:t for t in cfg['tasks']}
    for name,h in json.loads((O/'00_AUTHORITY/EXECUTABLE_SHA.json').read_text()).items():check('execution_code_'+name,sha(O/'code'/name)==h)
    dfs=[];counts=[]
    for arm,lane in [('J3','A'),('J4','A'),('J6','A'),('H250','B')]:
        df,cc=extract(arm,lane,registry);dfs.append(df);counts.append(cc)
    old=pd.read_csv(C/'01_TASKS/E4_AMENDED_TASKS.csv.gz',float_precision='round_trip');old['source_arm']=old.arm;old['arm']=old.arm.replace({'H150':'J7'})
    alltask=pd.concat(dfs+[old],ignore_index=True);save('REPORTS/ALL_CURRENT_TASKS.csv.gz',alltask)
    summ=summary(alltask);save('REPORTS/ALL_ARM_STRATIFIED_SUMMARY.csv',summ)
    j=summ[summ.arm.isin(['J3','J4','J5','J6','J7'])];save('E4_J3_J7_BURDEN_PERFORMANCE_MASTER.csv',j[j.stratum_type=='ALL']);save('E4_SEPARATION_ROBUSTNESS.csv',j[j.stratum_type=='separation'])
    save('LANE_A/SUMMARY/ALL_N_FAMILY_SEPARATION.csv',summ[summ.arm.isin(['J3','J4','J5','J6','J7','ROUNDED_J5'])]);save('LANE_B/SUMMARY/ALL_N_FAMILY_SEPARATION.csv',summ[summ.arm.isin(['H100','J7','H200','H250'])].replace({'arm':{'J7':'H150'}}))
    save('REPORTS/COMPUTATION_COUNTS.csv',counts)
    contrasts=paired(alltask,['J3','J4','J5','J6'],'J7','LANE_A/PAIRED');save('E4_ALL_INTENDED_AND_COMMON_VALID_CONTRASTS.csv',contrasts)
    hs=alltask[alltask.arm.isin(['H100','J7','H200','H250'])].copy();hs['arm']=hs.arm.replace({'J7':'H150'});paired(hs,['H100','H150','H200'],'H250','LANE_B/PAIRED')
    fidelity(hs)
    check('all_frozen_draw_files_reused',len(list((S2/'12_BOOTSTRAP').glob('draw_registry_*.csv.gz')))==50)
    (O/'01_DESIGN/FROZEN_BOOTSTRAP_REGISTRY').mkdir(exist_ok=True)
    for p in [S2/'12_BOOTSTRAP/stratum_registry.csv']+sorted((S2/'12_BOOTSTRAP').glob('draw_registry_*.csv.gz')):shutil.copy2(p,O/'01_DESIGN/FROZEN_BOOTSTRAP_REGISTRY'/p.name)
    parents=pd.read_csv(O/'00_AUTHORITY/INPUT_SHA256_MANIFEST.csv');rows=[]
    def verify(row):
        h=sha(row.source_path);return dict(path=row.source_path,before_sha256=row.sha256,after_sha256=h,unchanged=h.lower()==row.sha256.lower())
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,parents.itertuples(index=False)))
    for x in json.loads((O/'00_AUTHORITY/PROTECTED_BASELINE.json').read_text()):
        h=sha(x['path']);rows.append(dict(path=x['path'],before_sha256=x['sha256'],after_sha256=h,unchanged=h.lower()==x['sha256'].lower()))
    save('PARENT_AND_PROTECTED_SHA_BEFORE_AFTER.csv',rows);check('all_parent_protected_sha_unchanged',all(r['unchanged'] for r in rows))
    save('EXECUTION_QA_CHECKS.csv',QA);js('REPORTS/POSTPROCESSING_STATUS.json',dict(status='NUMERICAL_POSTPROCESSING_QA_PASS_REPORT_PACKAGING_PENDING',seconds=time.time()-START));log('Numerical exports, paired analysis, fidelity and protected SHA checks PASS; report/package remain')
def fidelity(tasks):
    log('H-series common-valid score/label fidelity from stored vectors')
    score=pd.read_csv(C/'02_SCORES/E4_AMENDED_PROFILE_SCORES.csv.gz',float_precision='round_trip');score=score[score.arm.isin(['H100','H150','H200'])]
    new=pd.read_csv(O/'LANE_B/TASKS/H250_SCORES.csv.gz',float_precision='round_trip');score=pd.concat([score,new],ignore_index=True)
    vectors={(a,c):g.set_index('sample_id') for (a,c),g in score.groupby(['arm','cohort_id'],sort=False)};tt=tasks.set_index(['arm','cohort_id']);rows=[];ref=tasks[tasks.arm=='H250']
    for arm in ['H100','H150','H200','H250']:
        for r in ref.itertuples():
            a=tt.loc[(arm,r.cohort_id)];cv=bool(a.valid and r.valid);row=dict(arm=arm,cohort_id=r.cohort_id,separation=r.separation,factor_weight=r.factor_weight,common_valid=cv)
            if cv:
                x=vectors[('H250',r.cohort_id)];y=vectors[(arm,r.cohort_id)].loc[x.index];lr=x.label.to_numpy(int);la=y.label.to_numpy(int)
                row.update(CRS_spearman=float(spearmanr(y.CRS,x.CRS).statistic),balanced_label_fidelity=float(np.mean([np.mean(la[lr==k]==k) for k in range(3)])),exact_label_agreement=float(np.mean(la==lr)),transport_exact_agreement=float(a.D_transport==r.D_transport),extreme_shift_rate=float(np.mean(np.abs(la-lr)==2)))
            rows.append(row)
    d=pd.DataFrame(rows);save('LANE_B/SUMMARY/H250_COMMON_VALID_COHORT_FIDELITY.csv.gz',d);out=[]
    sm=summary(tasks)
    for arm,g in d.groupby('arm'):
        for sep in ['ALL','weak','moderate','strong']:
            z=g if sep=='ALL' else g[g.separation==sep];v=z.common_valid.to_numpy(bool);w=z.factor_weight.to_numpy();r=dict(arm=arm,separation=sep,intended_tasks=len(z),common_valid_tasks=int(v.sum()),common_valid_weight=w[v].sum(),total_weight=w.sum())
            for c in ['CRS_spearman','balanced_label_fidelity','exact_label_agreement','transport_exact_agreement','extreme_shift_rate']:r[c]=np.dot(w[v],z.loc[v,c])/w[v].sum()
            ss=sm[(sm.arm==arm)&(sm.stratum_type==('ALL' if sep=='ALL' else 'separation'))&(sm.stratum==sep)].iloc[0]
            for c in ['A_valid','C_conditional','C_operational','ARI','BA','MacroF1','Extreme']:r[c]=ss[c]
            r['operational_gate_pass']=r['A_valid']>=.95;r['strict_reference_gate_pass']=r['operational_gate_pass'] and r['CRS_spearman']>=.98 and r['balanced_label_fidelity']>=.95;r['subgroup_gate_role']='descriptive_only' if sep!='ALL' else 'original_overall'
            out.append(r)
    save('E4_H100_H250_MATCHED_STAGE1_CURRENT_EXACTLOG.csv',out)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        js('REPORTS/POSTPROCESSING_ERROR.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
