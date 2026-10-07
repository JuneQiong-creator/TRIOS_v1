"""Read-only source binding and synthetic QA; no scientific full-state evaluation."""
import sys,os,json,csv,hashlib,math,time,collections,platform,ast
sys.dont_write_bytecode=True
from pathlib import Path
import engine as m
from engine import A,F,O,e,rt,np
X=e.X;C=e.C;B=e.B
EXPECTED=dict(zip(e.CID,[.08,2.53,.25,2.53,2.53,8.,8.]))
def main():
    start=time.time();e.require(not (A/'00_BINDING/INITIALIZED.json').exists(),'ALREADY_INITIALIZED')
    for d in ['01_QA','02_FULL','03_B100','04_RESULTS','05_PROVENANCE']: (A/d).mkdir(exist_ok=True)
    sources={};expected={}
    for root in [F,O,m.A.parent/'CCLE_OPTION_C_B100_PILOT_20260928_v1']:
        for r in e.rows(root/'RESULT_MANIFEST_SHA256.csv'):expected[str((root/r['path']).resolve())]=r['sha256'].lower()
    for r in e.rows(F/'00_BINDING/input_code_SHA256.csv'):expected[str(Path(r['path']).resolve())]=r['sha256'].lower()
    def bind(p,role):
        p=Path(p).resolve();h=rt.sha(p)
        if str(p) in expected:e.require(h==expected[str(p)],'FROZEN_SOURCE_SHA:'+str(p))
        sources[str(p)]=dict(path=str(p),sha256=h,role=role);return h
    for r in e.rows(F/'00_BINDING/input_code_SHA256.csv'):bind(r['path'],'FROZEN_CONTRACT_OR_FIT:'+r['role'])
    bind(F/'execute_full.py','APPROVED_ORIGINAL5_CLASSIFIER_LIBRARY');bind(O/'bound_subset_evaluator.py','CONVENTIONAL_FIXED_SCORE_CACHE_PRODUCER')
    bind(O/'00_BINDING/B100_RESULT_SOURCE_INDEX.csv','700_SOURCE_CHECKPOINT_LOCATIONS')
    sourcepaths={'scores':X/'D2/02_PROFILE_CRS/D2_PROFILE_CRS_SCORES.csv','nodes':X/'D2/01_CRS_NODE_EVALUATIONS/D2_CRS_NODE_EVALUATIONS.csv','summary':X/'D2/04_REFERENCE_STRATIFICATION/D2_FULL_SOURCE_REFERENCE_STRATIFICATION_SUMMARY.csv','labels':X/'D2/04_REFERENCE_STRATIFICATION/D2_FULL_SOURCE_REFERENCE_LABEL_REGISTRY.csv','members':X/'D3/01_SUBSET_REGISTRY/D3_FROZEN_MEMBERSHIP_REGISTRY.csv','d3':X/'D3/06_SUBSET_COMPLETE_RESULTS/D3_v1.1_SUBSET_COMPLETE_RESULTS.csv','registry':B/'CCLE_PIPELINE_EXECUTION_REGISTRY_v0.3.csv','weights':C/'00_BINDING/current_profile_model_AICc_weights.csv','observed':e.R/'work/trios_phaseb_v13/prior_nonstandard_json_package/08_QSC_REFERENCE_SPACE/ccle_observed_8dose_response_registry.csv'}
    tables={}
    for name,p in sourcepaths.items():bind(p,name.upper());tables[name]=e.rows(p)
    for p in [X/'D2/BLOCK_METADATA.json',X/'D2/RUNTIME_PROVENANCE.json',X/'D3/BLOCK_METADATA.json',C/'11_PROVENANCE/run_comp_a.py',C/'00_BINDING/preflight_summary.json']:bind(p,'SOURCE_LINEAGE')
    for row in rt.read(X/'D2/BLOCK_METADATA.json')['parent_bindings']:
        e.require(bind(row['resolved_path'],'D2_PARENT')==row['sha256'].lower(),'D2_PARENT_SHA')
    e.require(set(r['method_id'] for r in tables['registry'])==set(m.METHODS) and len(tables['registry'])==81,'REGISTRY_81')
    idx={(r['dataset'],int(r['replicate_id'])):r for r in e.rows(O/'00_BINDING/B100_RESULT_SOURCE_INDEX.csv')};e.require(len(idx)==700,'700_CONTEXTS')
    mem=collections.defaultdict(list)
    for r in tables['members']:mem[r['cohort_id'],int(r['replicate_id'])].append(r)
    nodes=collections.defaultdict(list)
    for r in tables['nodes']:nodes[r['cohort_id'],r['ccle_cell_line_name']].append(r)
    contexts=[];qa=[];no_pass=[];historic_ref=[]
    for cid in e.CID:
        ss=[r for r in tables['scores'] if r['cohort_id']==cid];ids=sorted(r['ccle_cell_line_name'] for r in ss);e.require(len(ids)==len(set(ids)),'D2_ID_UNIQUE')
        summ=next(r for r in tables['summary'] if r['cohort_id']==cid);D=float(summ['D_transport_uM']);e.require(D==EXPECTED[cid] and int(summ['N'])==len(ids),'D2_DOMAIN_OR_N')
        g=e.geom['crs_geometry'](.0025,D,7)
        scores={r['ccle_cell_line_name']:{'TRIOS':float(r['CRS'])} for r in ss}
        for r in ss:
            pid=r['ccle_cell_line_name'];nn=sorted(nodes[cid,pid],key=lambda q:int(q['node_index']))
            e.require(r['scoring_geometry']=='EXACT_LOGSPACE_K7' and float(r['D_transport_uM'])==D and len(nn)==7,'D2_GEOMETRY_MARKER')
            e.require(np.array_equal([float(q['dose_uM']) for q in nn],g['nodes']) and np.array_equal([float(q['beta']) for q in nn],g['beta']),'CANONICAL_EXP_LOG_NODES_WEIGHTS')
            # QA of saved weighted terms only; do not call prediction or reconstruct scores.
            e.require(math.isclose(sum(float(q['weighted_term']) for q in nn),float(r['CRS']),rel_tol=1e-14,abs_tol=1e-14),'SAVED_CRS_TERM_CLOSURE')
        obs=[r for r in tables['observed'] if r.get('dataset_id')==cid]
        e.require(len(obs)==len(ids)*8 and {r['sample_id'] for r in obs}==set(ids),'OBSERVED_PROFILE_IDENTITY')
        for pid in ids:e.require(sorted(float(r['dose_uM']) for r in obs if r['sample_id']==pid)==e.DOSES.tolist(),'OBSERVED_EIGHT_DOSE_GRID')
        scalar=F/'03_FULL'/cid/'profile_scalar_cache.csv';bind(scalar,'FROZEN_NATIVE_SCALARS');sr=e.rows(scalar);e.require(len(sr)==len(ids)*16,'SCALAR_COVERAGE')
        for r in sr:
            e.require(r['profile_id'] in scores,'SCALAR_IDENTITY');scores[r['profile_id']][r['model']+'|'+r['representation']]=e.clean(m.num(r['oriented_score']))
        weights={pid:{} for pid in ids}
        for r in tables['weights']:
            if r['cohort_id']==cid:weights[r['sample_id']][r['model']]=float(r['AICc_weight'])
        e.require(all(set(w)=={'SPLINE',*e.MODELS} and math.isclose(sum(w.values()),1,abs_tol=1e-12) for w in weights.values()),'FROZEN_WEIGHT_IDENTITY')
        hist={}
        for scope,folder in [('FULL','01_COMP_FULLSOURCE_81'),('B100','02_COMP_B100_81')]:
            hp=C/folder/(cid.replace('-','_')+'_task_results.csv');bind(hp,'HISTORICAL_FIXED_TRANSPORT_FULL_STATES_ONLY')
            # Select labels/validity/cutoffs only; no QSC-family components or rankings imported.
            for r in e.rows(hp):
                if r['method_id']=='TRIOS':hist[scope,0 if scope=='FULL' else int(float(r['replicate_id']))]={k:r.get(k) for k in ['V','tau1','tau2','full_labels','full_failure_code']}
        e.require(len(hist)==101,'HISTORICAL_FIXED_TRANSPORT_REFERENCE_KEYS')
        for rep in [None,*range(1,101)]:
            scope='FULL' if rep is None else 'B100';unit=cid if rep is None else f'{cid}|R{rep:03d}'
            if rep is None:members=ids;cp=F/'03_FULL'/cid/'checkpoints'
            else:
                raw=sorted(mem[cid,rep],key=lambda r:int(r['member_position']));members=sorted(r['cell_line'] for r in raw);e.require(len(members)==18 and len(set(members))==18 and set(members)<=set(ids),'LOCKED_N18_MEMBERS');cp=Path(idx[cid,rep]['checkpoint_dir'])
            hashes={}
            for method in m.METHODS:
                p=cp/(m.key(method)+'.task.json');e.require(str(p.resolve()) in expected,'UNMANIFESTED_OLD_CHECKPOINT');hashes[method]=expected[str(p.resolve())]
            contexts.append(dict(dataset_id=cid,scope=scope,rep=rep,unit=unit,ids=members,checkpoint_dir=str(cp),checkpoint_sha256=hashes))
            if rep is not None:
                d3=next(r for r in tables['d3'] if r['cohort_id']==cid and int(r['subset_id'])==rep)
                if d3['terminal_status']!='VALID':no_pass.append(dict(dataset_id=cid,replicate_id=rep,historical_D3_terminal_status=d3['terminal_status'],historical_D3_domain=d3['D_transport_uM'],fixed_D2_domain=D,ids=members))
            historic_ref.append(dict(dataset_id=cid,scope=scope,rep=rep,unit=unit,**hist[scope,0 if rep is None else rep]))
        rt.atomic_json(A/'00_BINDING'/f'{cid}.json',dict(dataset=cid,D2_domain=D,ordered_full_profile_ids=ids,scores=scores,weights=weights,source_sha256={k:sources[str(p.resolve())]['sha256'] for k,p in sourcepaths.items()}),immutable=True)
        qa.append(dict(dataset_id=cid,profiles=len(ids),D2_domain=D,canonical_nodes='PASS_EXACT',score_source='D2_CRS_COLUMN_ONLY',scalar_profiles=len(sr),membership='100_LOCKED_SUBSETS'))
    e.require(len(no_pass)==38 and collections.Counter(r['dataset_id'] for r in no_pass)=={e.CID[4]:2,e.CID[5]:23,e.CID[6]:13},'D3_38_NO_PASS')
    rt.atomic_json(A/'00_BINDING/CONTEXTS.json',contexts,immutable=True);rt.atomic_json(A/'00_BINDING/HISTORICAL_FIXED_TRANSPORT_TRIOS_REFERENCE.json',historic_ref,immutable=True);rt.atomic_json(A/'00_BINDING/D3_38_NO_PASS.json',no_pass,immutable=True)
    e.csvwrite(A/'01_QA/SOURCE_PARITY.csv',qa)
    tests=[]
    def test(name,v):e.require(v,'FIXTURE:'+name);tests.append(dict(test=name,status='PASS'))
    x=np.arange(18,dtype=float);ids=np.array([str(i) for i in range(18)]);r=e.classify(x,ids,'AB','fixture')
    test('AB_n18_m3',r['valid'] and min(r['group_n_'+g] for g in 'LIH')>=3)
    tied=np.repeat(np.arange(6,dtype=float),3);r=e.classify(tied,ids,'AB','ties');test('AB_exact_ties_intact',r['valid'] and all(len(set(r['labels'][tied==v]))==1 for v in set(tied)))
    test('constant_native_invalid',not e.classify(np.ones(18),ids,'AB','constant')['valid'])
    test('cutoff_conventions',e.project([1,2],1,2,True).tolist()==['L','I'] and e.project([1,2],1,2,False).tolist()==['L','H'])
    folds=[dict(attempted=True,success=True,d_tau_squared=0.,omitted_match=1.,retained_match_fraction=1.),dict(attempted=True,success=True,d_tau_squared=1.,omitted_match=0.,retained_match_fraction=.5),dict(attempted=True,success=False)]
    z=e.task_components([0,1,2],['L','I','H'],.5,1.5,.5,folds);test('failure_A_LOO_and_unbalanced_harmonic',abs(z['S_match']-.4)<1e-14 and abs(z['S_tau']-(2/3)/(1+math.sqrt(.5)))<1e-14 and z['V']==1)
    z=e.task_components([],[],None,None,None,[],False);test('invalid_components_undefined',all(z[k] is None for k in m.COMP) and z['C_operational']==0)
    z=e.task_components([0,1,2],['L','I','H'],.5,1.5,.5,[dict(attempted=True,success=False)]*3);test('all_failed_stability_zero',z['S_tau']==z['S_match']==0 and z['V']==1)
    for alg in e.ALGS:
        r=e.classify(x,ids,alg,'SYNTHETIC_FIXED_CONTEXT');r2=e.classify(x,ids,alg,'SYNTHETIC_FIXED_CONTEXT');test('native_determinism_'+alg,r['valid']==r2['valid'] and (not r['valid'] or np.array_equal(r['labels'],r2['labels'])))
    for name in ['select_domain','quad','brentq','frozen']:
        try:getattr(e,name)();raise AssertionError(name)
        except e.IntegrityStop:test('upstream_disabled_'+name,True)
    calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(Path(m.__file__).read_text())) if isinstance(n,ast.Call)]
    test('no_upstream_calls_in_engine',not any(c in calls for c in ['e.select_domain','e.quad','e.brentq','e.frozen',"e.geom['crs_geometry']"]))
    e.csvwrite(A/'01_QA/IMPLEMENTATION_FIXTURES.csv',tests)
    env=dict(python=sys.version,numpy=np.__version__,scipy=e.scipy.__version__,pygam=e.pygam.__version__,executable=sys.executable,classifier_path=e.classifiers.__file__,threads=1,seed=e.cfg.MASTER_SEED,starts=e.cfg.CLASSIFIER_STARTS,tolerance=e.cfg.CLASSIFICATION_TOL,contract=m.CONTRACT)
    e.require(sys.version_info[:3]==(3,13,5) and np.__version__=='2.4.0' and e.scipy.__version__=='1.16.3','ENVIRONMENT_PARITY')
    rt.atomic_json(A/'00_BINDING/ENVIRONMENT.json',env,immutable=True)
    for p in [A/'engine.py',A/'durable_runtime.py',Path(__file__),A/'00_BINDING/AUTHOR_EXECUTION_INSTRUCTION.md']:bind(p,'NEW_EXECUTION_CODE_OR_AUTHORITY')
    for p in (A/'00_BINDING').glob('*.json'):bind(p,'LOCKED_EXECUTION_INPUT')
    e.csvwrite(A/'00_BINDING/SOURCE_SHA256.csv',sources.values())
    # Record parent manifests and all source checkpoint hashes without changing parent files.
    e.csvwrite(A/'00_BINDING/USED_CHECKPOINT_SHA256.csv',[dict(path=str(Path(c['checkpoint_dir'])/(m.key(method)+'.task.json')),sha256=c['checkpoint_sha256'][method]) for c in contexts for method in m.METHODS])
    protected=[]
    for p in (e.R/'05_MANUSCRIPT/BIOSTATISTICS').rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.tex','.bib','.csv','.json','.pdf']:protected.append(dict(path=str(p),sha256=rt.sha(p)))
    e.csvwrite(A/'00_BINDING/MANUSCRIPT_PROTECTION_SHA256.csv',protected)
    rt.atomic_json(A/'00_BINDING/INITIALIZED.json',dict(status='PASS',contexts=len(contexts),tasks=57267,D3_no_pass=38,fixtures=len(tests),elapsed_seconds=time.time()-start),immutable=True)
    print('Experiment A source binding and synthetic fixtures PASS',flush=True)
if __name__=='__main__':main()
