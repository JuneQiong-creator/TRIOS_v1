from d5a import *
import math,itertools,time
from scipy.stats import spearmanr
CODE={'L':0,'I':1,'H':2}
def seed(c,f,kind):return int.from_bytes(hashlib.sha256(f'D5_v1.0|{c}|{f}|{kind}'.encode()).digest()[:8],'big')
def perm(y,p,c,f,kind):
 n=len(y);k=int(p.sum());observed=abs(np.median(y[p])-np.median(y[~p])) if kind=='CONTINUOUS_CRS' else abs(y[p].mean()-y[~p].mean());total=math.comb(n,k);exact=total<=200000;B=total if exact else 100000;ge=0;sd=seed(c,f,kind);rng=np.random.Generator(np.random.PCG64(sd));it=itertools.combinations(range(n),k) if exact else None
 for start in range(0,B,2000):
  size=min(2000,B-start)
  if exact:ix=np.array(list(itertools.islice(it,size)));mask=np.zeros((size,n),bool);mask[np.arange(size)[:,None],ix]=True;positive=y[np.where(mask)[1]].reshape(size,k);negative=y[np.where(~mask)[1]].reshape(size,n-k)
  else:
   ix=rng.permuted(np.tile(np.arange(n),(size,1)),axis=1);positive=y[ix[:,:k]];negative=y[ix[:,k:]]
  stats=abs(np.median(positive,axis=1)-np.median(negative,axis=1)) if kind=='CONTINUOUS_CRS' else abs(positive.mean(axis=1)-negative.mean(axis=1));ge+=int((stats>=observed).sum())
 return (ge/B if exact else (ge+1)/(B+1)),('EXACT' if exact else 'MONTE_CARLO'),B,sd,ge
def bh(df):
 df['q']=np.nan
 for _,ix in df[df.p.notna()].groupby(['cohort_id','family']).groups.items():
  ids=list(ix);p=df.loc[ids,'p'].to_numpy();order=np.argsort(p,kind='stable');q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);df.loc[ids,'q']=q
 return df
def cliff(a,b):return float(np.sign(a[:,None]-b[None,:]).mean())
def cramer(tab):
 t=tab[:,tab.sum(axis=0)>0];n=t.sum()
 if min(t.shape)<2 or n==0 or (t.sum(axis=1)==0).any():return np.nan
 e=t.sum(axis=1)[:,None]*t.sum(axis=0)[None,:]/n;return float(np.sqrt(np.sum((t-e)**2/e)/(n*min(t.shape[0]-1,t.shape[1]-1))))
def main():
 freeze=json.loads((OUT/'D5A_FREEZE.json').read_text(encoding='utf-8'));req(all(sha(OUT/p)==h for p,h in freeze['files'].items()),'D5A freeze drift')
 req(not (OUT/'D5_PRIMARY_RESULT_FREEZE.json').exists(),'No result overwrite')
 js('00_BINDING/CURRENT_OUTCOME_OPEN.json',{'opened_utc':now(),'D5A_freeze_sha256':sha(OUT/'D5A_FREEZE.json'),'historical_numerical_biology_access':False})
 parents=json.loads((OUT/'00_BINDING/PARENTS.json').read_text(encoding='utf-8'));read={}
 def load(key,suffix):
  with zipfile.ZipFile(parents[key]['path']) as z:
    names=[n for n in z.namelist() if n.endswith(suffix)];req(len(names)==1,'Unique member'+suffix);name=names[0];b=z.read(name);mn=next(n for n in z.namelist() if n.endswith('manifest.csv'));m=pd.read_csv(io.BytesIO(z.read(mn)));r=m[m.path.map(lambda p:name.endswith(p))].iloc[0];h=hashlib.sha256(b).hexdigest().upper();req(h==r.sha256,'Member hash');read[key+':'+name]=h;return pd.read_csv(io.BytesIO(b),float_precision='round_trip')
 full=load('D2','D2_FULL_SOURCE_REFERENCE_LABEL_REGISTRY.csv').rename(columns={'ccle_cell_line_name':'cell_line'});crs=load('D2','D2_PROFILE_CRS_SCORES.csv').rename(columns={'ccle_cell_line_name':'cell_line'})
 sp=load('D3','D3_v1.1_SUBSET_PROFILE_RESULTS.csv').rename(columns={'ccle_cell_line_name':'cell_line'});ss=load('D3','D3_v1.1_SUBSET_TRANSPORT_SUMMARY.csv');oof=load('D4','D4_LOO_FOLDS.csv').rename(columns={'heldout_profile':'cell_line'})
 req(len(full)==len(crs)==424 and len(sp)==12600 and len(ss)==700 and len(oof)==11916,'Frozen cardinality')
 req(full[['cohort_id','cell_line']].duplicated().sum()==0 and sp[['cohort_id','subset_id','cell_line']].duplicated().sum()==0 and oof[['cohort_id','subset_id','cell_line']].duplicated().sum()==0,'No duplicates')
 check=full.merge(crs,on=['cohort_id','cell_line'],suffixes=('_label','_score'));req(np.array_equal(check.CRS_label,check.CRS_score),'Raw CRS identity')
 expected={'CCLE-34-Paclitaxel-SKIN':(.08,[18,10,5],100),'CCLE-11-AZD6244-LUNG':(2.53,[32,36,18],100),'CCLE-38-PD-0325901-HEM':(.25,[34,18,10],100),'CCLE-10-AZD6244-HEM':(2.53,[32,19,11],100),'CCLE-65-Topotecan-LUNG':(2.53,[22,40,25],98),'CCLE-51-PLX4720-SKIN':(8,[12,13,7],77),'CCLE-58-TAE684-HEM':(8,[15,30,17],87)}
 for c,(d,counts,v) in expected.items():
  g=full[full.cohort_id==c];req(g.D_transport_uM.eq(d).all() and [int(g.reference_label.eq(x).sum()) for x in CODE]==counts,'D2 identities');q=ss[ss.cohort_id==c];req(len(q)==100 and q.transport_status.eq('TRANSPORT_COMPATIBLE').sum()==v,'D3 status identity')
 member=pd.read_csv(BASE/'D0_DATA_CHALLENGE_SET_LOCK_v1.0/D0_PRIMARY_7_SUBSET_MEMBERSHIP_REGISTRY.csv').rename(columns={'replicate_id':'subset_id'});req(set(map(tuple,sp[['cohort_id','subset_id','cell_line']].values))==set(map(tuple,member[['cohort_id','subset_id','cell_line']].values)),'Subset membership identity')
 validsp=sp[sp.AB_status=='VALID'];req(set(map(tuple,validsp[['cohort_id','subset_id','cell_line']].values))==set(map(tuple,oof[['cohort_id','subset_id','cell_line']].values)) and oof.success.all(),'OOF identity completeness')
 js('00_BINDING/CURRENT_READOUT_MEMBER_HASHES.json',read)
 reg=pd.read_csv(OUT/'02_REGISTRY/D5_MECHANISTIC_FEATURE_REGISTRY.csv');calls=pd.read_csv(OUT/'02_CALLS/D5_MOLECULAR_CALLS.csv');full['ordinal']=full.reference_label.map(CODE);sp['ordinal']=sp.subset_label.map(CODE)
 continuous=[];strata=[];tests=[];joined=[];start=time.perf_counter()
 for j,r in enumerate(reg.itertuples(),1):
  g=calls[(calls.cohort_id==r.cohort_id)&(calls.feature==r.feature)].merge(full,on=['cohort_id','cell_line'],validate='one_to_one').sort_values('cell_line');joined.append(g)
  a=g[g.value==1];b=g[g.value==0];np0=len(a);nn=len(b);na=len(g)-np0-nn
  status='INFERENTIALLY_EVALUABLE' if np0>=5 and nn>=5 else 'SPARSE_DESCRIPTIVE_ONLY' if 2<=np0<=4 and nn>=5 else 'NOT_EVALUABLE'
  if r.family=='DESCRIPTIVE_ONLY' and status=='INFERENTIALLY_EVALUABLE':status='DESCRIPTIVE_ONLY_PRESPECIFIED'
  base=dict(cohort_id=r.cohort_id,feature=r.feature,primary=r.primary,family=r.family,expected_direction=r.expected_direction,n=len(g),n_positive=np0,n_call_negative=nn,n_NA=na,evaluability=status,modality_status='BOUND_EXACT' if not g.status.str.startswith('NOT_EVALUABLE').all() else g.status.iloc[0])
  delta=float(a.CRS.median()-b.CRS.median()) if np0 and nn else np.nan;cd=cliff(a.CRS.to_numpy(),b.CRS.to_numpy()) if np0 and nn else np.nan;od=float(a.ordinal.mean()-b.ordinal.mean()) if np0 and nn else np.nan
  row=dict(base,delta_median=delta,cliffs_delta=cd,positive_Q1=a.CRS.quantile(.25),positive_median=a.CRS.median(),positive_Q3=a.CRS.quantile(.75),negative_Q1=b.CRS.quantile(.25),negative_median=b.CRS.median(),negative_Q3=b.CRS.quantile(.75),p=np.nan)
  tab=np.array([[int(x.reference_label.eq(l).sum()) for l in CODE] for x in [b,a]]);sr=dict(base,ordinal_mean_difference=od,cramers_V=cramer(tab),contingency=json.dumps(tab.tolist()),p=np.nan)
  if status=='INFERENTIALLY_EVALUABLE':
   gg=g[g.value.notna()];p=gg.value.to_numpy()==1
   for kind,col,dest in [('CONTINUOUS_CRS','CRS',row),('REFERENCE_STRATA','ordinal',sr)]:
    pv,mode,B,sd,ge=perm(gg[col].to_numpy(),p,r.cohort_id,r.feature,kind);dest['p']=pv;tests.append(dict(cohort_id=r.cohort_id,feature=r.feature,kind=kind,n=len(gg),n_positive=np0,mode=mode,assignments=B,seed=str(sd),exceedances=ge,p=pv))
  continuous.append(row);strata.append(sr)
  if j%10==0:print(f'Full-source features {j}/83; formal tests={len(tests)}; elapsed={time.perf_counter()-start:.1f}s',flush=True)
 co=csv('03_CONTINUOUS_CRS/D5_FULL_SOURCE_CONTINUOUS_CRS_RESULTS.csv',bh(pd.DataFrame(continuous)));sr=csv('04_REFERENCE_STRATA/D5_FULL_SOURCE_REFERENCE_STRATA_RESULTS.csv',bh(pd.DataFrame(strata)));csv('03_CONTINUOUS_CRS/PERMUTATION_AUDIT.csv',tests);csv('03_CONTINUOUS_CRS/ANALYSIS_JOIN.csv',pd.concat(joined))
 frozen={p.relative_to(OUT).as_posix():sha(p) for d in ['03_CONTINUOUS_CRS','04_REFERENCE_STRATA'] for p in (OUT/d).glob('*.csv')};js('D5_PRIMARY_RESULT_FREEZE.json',{'frozen_utc':now(),'files':frozen,'historical_numerical_results_opened':False})
 # All 8300 intended subset-feature rows retained, including invalid pharmacology and unavailable modality.
 op=oof[['cohort_id','subset_id','cell_line','heldout_label']].copy();op['oof_ordinal']=op.heldout_label.map(CODE);sp=sp.merge(op,on=['cohort_id','subset_id','cell_line'],how='left',validate='one_to_one');rob=[]
 for r in reg.itertuples():
  call=calls[(calls.cohort_id==r.cohort_id)&(calls.feature==r.feature)][['cell_line','value']];g=sp[sp.cohort_id==r.cohort_id].merge(call,on='cell_line',validate='many_to_one');ref=co[(co.cohort_id==r.cohort_id)&(co.feature==r.feature)].delta_median.iloc[0]
  for sid,q in g.groupby('subset_id',sort=True):
   a=q[q.value==1];b=q[q.value==0];ph=q.AB_status.eq('VALID').all();ev=ph and len(a)>=2 and len(b)>=2;de=float(a.CRS.median()-b.CRS.median()) if ev else np.nan;od=float(a.ordinal.mean()-b.ordinal.mean()) if ev else np.nan;oe=bool(ev and q.oof_ordinal.notna().all());ov=float(a.oof_ordinal.mean()-b.oof_ordinal.mean()) if oe else np.nan
   rob.append(dict(cohort_id=r.cohort_id,feature=r.feature,subset_id=sid,n_positive=len(a),n_call_negative=len(b),pharmacology_valid=ph,evaluable=ev,status='EVALUABLE' if ev else 'NOT_EVALUABLE_PHARMACOLOGY' if not ph else 'NOT_EVALUABLE_MOLECULAR',CRS_delta_median=de,ordinal_difference=od,OOF_evaluable=oe,OOF_ordinal_difference=ov,full_source_CRS_effect=ref,CRS_sign_consistent=bool(np.sign(de)==np.sign(ref)) if ev and np.isfinite(ref) else None,ordinal_sign_consistent=bool(np.sign(od)==np.sign(ref)) if ev and np.isfinite(ref) else None,OOF_sign_consistent=bool(np.sign(ov)==np.sign(ref)) if oe and np.isfinite(ref) else None))
 rb=csv('05_B100/D5_B100_FEATURE_CONTRASTS.csv',rob);req(len(rb)==8300,'8300 intended feature-subsets');sums=[]
 for (c,f),g in rb.groupby(['cohort_id','feature'],sort=False):
  row=dict(cohort_id=c,feature=f,n_intended=100,n_pharmacology_valid=int(g.pharmacology_valid.sum()),n_evaluable=int(g.evaluable.sum()),n_OOF_evaluable=int(g.OOF_evaluable.sum()))
  for col in ['CRS_delta_median','ordinal_difference','OOF_ordinal_difference']:
   v=g[col].dropna();row.update({col+'_'+name:float(v.quantile(q)) if len(v) else np.nan for name,q in [('P05',.05),('Q1',.25),('median',.5),('Q3',.75),('P95',.95)]})
  for col in ['CRS_sign_consistent','ordinal_sign_consistent','OOF_sign_consistent']:
   v=g[col].dropna();row[col+'_count']=int(v.sum()) if len(v) else 0;row[col+'_denominator']=len(v);row[col+'_fraction']=float(v.sum()/len(v)) if len(v) else np.nan
  sums.append(row)
 su=csv('05_B100/D5_B100_DIRECTION_SUMMARY.csv',sums);csv('06_OOF/D5_OOF_DIRECTION_SUMMARY.csv',su[['cohort_id','feature','n_OOF_evaluable']+[c for c in su if c.startswith('OOF_')]])
 primary=co[co.primary].merge(su,on=['cohort_id','feature']);csv('07_SYNTHESIS/D5_SEVEN_DATASET_PRIMARY_FEATURE_TABLE.csv',primary)
 csv('07_SYNTHESIS/D5_ALL83_COHERENCE_MATRIX.csv',co.merge(su,on=['cohort_id','feature']))
 display=primary[primary.cliffs_delta.notna()].sort_values(['cliffs_delta'],key=lambda x:x.abs(),ascending=False).copy();display=display.assign(abs_effect=display.cliffs_delta.abs()).sort_values(['abs_effect','q','cohort_id'],ascending=[False,True,True]).head(2);csv('07_SYNTHESIS/D5_DISPLAY_CASES.csv',display)
 js('07_SYNTHESIS/ANALYSIS_COUNTS.json',{'registry':len(reg),'inferential':int(co.evaluability.eq('INFERENTIALLY_EVALUABLE').sum()),'sparse':int(co.evaluability.eq('SPARSE_DESCRIPTIVE_ONLY').sum()),'prespecified_descriptive':int(co.evaluability.eq('DESCRIPTIVE_ONLY_PRESPECIFIED').sum()),'not_evaluable':int(co.evaluability.eq('NOT_EVALUABLE').sum()),'formal_tests':len(tests),'robustness_rows':len(rb)})
 print('D5B SCIENTIFIC TABLES COMPLETE',flush=True)

if __name__=='__main__':main()
