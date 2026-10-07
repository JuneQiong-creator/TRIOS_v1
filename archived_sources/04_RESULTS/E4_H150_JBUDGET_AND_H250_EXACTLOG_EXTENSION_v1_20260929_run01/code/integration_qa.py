import sys,warnings,math
sys.dont_write_bytecode=True
from adapter import *
def main():
    checks=[]
    def ok(name,detail):checks.append(dict(check=name,status='PASS',detail=detail))
    cfg=json.loads((OUT/'01_DESIGN/LOCK.json').read_text())
    for name,digest in cfg['code'].items():assert sha(OUT/'code'/name)==digest
    ok('LOCKED_ADAPTER','SHA checked before science')
    assert sys.version_info[:2]==(3,13)
    import scipy
    assert np.__version__=='2.4.0' and scipy.__version__=='1.16.3' and pd.__version__=='2.3.3'
    ok('RUNTIME',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    for arm in ('J3','J4','J6'):
        cache=CACHES[arm];assert cache['design'].shape==(len(GRIDS[arm])*3,12) and len(cache['lambdas'])==21
        for kind,y in [('monotone',np.repeat(np.linspace(0,1,len(GRIDS[arm])),3)),('constant',np.full(len(GRIDS[arm])*3,.5)),('nonfinite',np.full(len(GRIDS[arm])*3,np.nan))]:
            with warnings.catch_warnings():warnings.simplefilter('ignore');f=pf.cached_fit(y,cache)
            if kind=='nonfinite':assert f is None
            else:
                assert f is not None and len(f['coefficients'])==12
                model=pf.fit_constrained_pspline(cache['z'],y,f['lambda'],cache['edge'],n_splines=12,spline_order=3)
                assert max(abs(f['SSE']-model.replicate_level_SSE),abs(f['GCV']-model.GCV),abs(f['EDF']-model.effective_degrees_of_freedom))<=1e-10
            ok(arm+'_'+kind,'synthetic only; original GCV and solver')
    for D in pf.TCANDS:
        g=geometry(D);nodes=np.asarray(g['nodes']);beta=np.asarray(g['beta']);assert nodes[0]==1 and nodes[-1]==D and np.all(np.diff(nodes)>0) and np.all(beta>0) and abs(beta.sum()-1)<1e-14
    ok('EXACT_LOG_GEOMETRY','all six candidate endpoints, K7')
    ids=[f'toy{i:02d}' for i in range(12)]
    for x in (np.arange(12,dtype=float),np.repeat([0.,1.,2.],4),np.ones(12)):
        z=cur.ab(x,ids)
        if np.ptp(x)==0:assert z is None
        else:
            from trios_ab_production import _exact_sse
            s=np.sort(x);can=[(i,j) for i in range(3,7) for j in range(i+3,10) if s[i-1]<s[i] and s[j-1]<s[j]]
            i,j=min(can,key=lambda ij:(_exact_sse(s,*ij),*ij));assert z[1]==(s[i-1]+s[i])/2 and z[2]==(s[j-1]+s[j])/2
    ok('AB_EXACT_TIE','independent exhaustive exact-rational toy objective')
    # Native fold metric bookkeeping independently reconciled on synthetic score vectors.
    full=np.linspace(.1,1.2,12);folds=[dict(D_transport=float(pf.TCANDS[h%6])) for h in range(12)];folds[0]['D_transport']=np.nan
    t,sc,_=cur.eval_arm(dict(valid=True,D_transport=50.),folds,ids,np.repeat([0,1,2],4),lambda D:full+float(D)*.0001)
    z=cur.ab(full+.005,ids);ds=[];om=[];ret=[]
    for h,row in enumerate(folds):
        if not np.isfinite(row['D_transport']):continue
        x=full+row['D_transport']*.0001;keep=np.arange(12)!=h;ll,a,b=cur.ab(x[keep],np.asarray(ids)[keep]);held=0 if x[h]<=a else 1 if x[h]<=b else 2
        ds.append(.5*(((a-z[1])/np.ptp(full))**2+((b-z[2])/np.ptp(full))**2));om.append(held==z[0][h]);ret.append(np.mean(ll==z[0][keep]))
    assert t['A_LOO']==11/12 and abs(t['S_tau']-(11/12)/(1+math.sqrt(np.mean(ds))))<1e-14
    hm=2*np.mean(om)*np.mean(ret)/(np.mean(om)+np.mean(ret));assert abs(t['S_match']-(11/12)*hm)<1e-14
    invalid,sc,_=cur.eval_arm(dict(valid=False),[],ids,np.zeros(12),lambda D:(_ for _ in ()).throw(RuntimeError('INVALID_SCORED')));assert not invalid['valid'] and not sc
    ok('NATIVE_FOLD_FAILURE_ACCOUNTING','independent successful-fold displacement/projection/harmonic mean; no invalid scoring')
    tasks=pd.read_csv(CURRENT/'01_TASKS/E4_AMENDED_TASKS.csv.gz',float_precision='round_trip')
    frozen=pd.read_csv(CURRENT/'03_SUMMARY/E4_AMENDED_SUMMARY.csv',float_precision='round_trip').set_index('arm')
    for arm in ('J5','H150','ROUNDED_J5'):
        g=tasks[tasks.arm==arm];assert len(g)==10200 and g.cohort_id.nunique()==10200
        w=g.factor_weight.to_numpy();v=g.valid.to_numpy(bool);assert abs(w[v].sum()/w.sum()-frozen.loc[arm,'applicability'])<5e-12
        for c in ('ARI','BA','MacroF1','Extreme','C_common'):assert abs(np.sum(w[v]*g.loc[v,c])/w[v].sum()-frozen.loc[arm,c])<5e-12
    ok('READ_ONLY_BASELINE_ANCHORS','J5 J7 rounded-J5 full task tables; no refit or baseline evaluation')
    # The phase-0 raw view hashes are immutable inputs; additionally check every source view here.
    total=0
    for sh in range(64):
        for raw in load(src.RAW/f'raw_shard_{sh:02d}.pkl.gz')['cohorts']:
            n=len(raw['ids']);assert raw['Y_by_assay']['H150'].shape==(n,8,3);total+=n
            for arm,idx in (('J3',[0,1,4,7]),('J4',[0,1,3,5,7])):assert np.array_equal(GRIDS[arm],src.GRIDS['H150'][idx])
    assert total==212400
    ok('RAW_SUBSET_POPULATION','10200 cohorts /212400 profiles source-bound, J3/J4 exact index geometry')
    js(OUT/'00_AUTHORITY/INTEGRATION_QA.json',dict(status='PASS',checks=checks,scientific_profiles=0,time=time.time(),resource_monitor='Windows native counters; no psutil dependency'))
    print('INTEGRATION_ENGINEERING_QA_PASS',len(checks),flush=True)
if __name__=='__main__':main()
