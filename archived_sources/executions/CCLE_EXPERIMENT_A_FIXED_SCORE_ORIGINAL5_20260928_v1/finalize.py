"""Summaries, independent archival verification and compact author review delivery."""
import sys
sys.dont_write_bytecode=True
import csv,json,gzip,hashlib,math,time,zipfile,io,collections
from pathlib import Path
import engine as m
import qa
from engine import A,e,rt
def csv_gz(path,records,fields):
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items() if k in fields})
def main():
    begin=time.time();contexts=rt.read(A/'00_BINDING/CONTEXTS.json');full=[];b100=[];counts=collections.Counter();tri={};source_counts=collections.Counter();nfold=0
    foldfields=['contract_id','dataset_id','scope','replicate_id','task_id','method_id','heldout_profile','fold_index','n_intended','attempted','success','status','D_full','D_fold','full_score_hash','fold_score_hash','tau1','tau2','d_tau_squared','omitted_match','retained_match_fraction','heldout_label','retained_labels','failure_code','source_kind','source_path']
    archive=A/'EXPERIMENT_A_LOO_FOLD_RECORDS.csv.gz'
    with gzip.open(archive,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=foldfields);w.writeheader()
        for ctx in contexts:
            dest=m.location(ctx);qa.check(rt.read(dest/'QA.json')['status']=='PASS','CONTEXT_QA_GATE')
            for method in m.METHODS:
                p=dest/(m.key(method)+'.json.gz');d=m.read_checkpoint(p);r=d['result'];(full if ctx['scope']=='FULL' else b100).append(r);counts.update(d['counts'])
                if method=='TRIOS':tri[ctx['unit']]=r
                for q in d['folds']:
                    row=dict(contract_id=m.CONTRACT,**q);w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,list) else v for k,v in row.items() if k in foldfields});nfold+=1;source_counts[q['source_kind']]+=1
    qa.check(len(full)==567 and len(b100)==56700,'TOTAL_TASKS')
    qa.check(nfold==sum(r['n'] for r in full+b100),'ALL_INTENDED_FOLD_SLOTS')
    fs=qa.aggregate(full,'FULL');bs=qa.aggregate(b100,'B100')
    e.csvwrite(A/'EXPERIMENT_A_FULL_81_METHOD_RESULTS.csv',fs);e.csvwrite(A/'EXPERIMENT_A_B100_ALL_81_METHOD_RESULTS.csv',bs)
    fields=list(dict.fromkeys(k for r in b100 for k in r))
    csv_gz(A/'EXPERIMENT_A_B100_ALL_TASK_RESULTS.csv.gz',b100,fields);csv_gz(A/'EXPERIMENT_A_FULL_ALL_TASK_RESULTS.csv.gz',full,list(dict.fromkeys(k for r in full for k in r)))
    for cid in e.CID:
        e.csvwrite(A/'04_RESULTS'/(cid+'_FULL_81.csv'),[r for r in fs if r['dataset_id']==cid]);e.csvwrite(A/'04_RESULTS'/(cid+'_B100_81.csv'),[r for r in bs if r['dataset_id']==cid])
    hist=rt.read(A/'00_BINDING/HISTORICAL_FIXED_TRANSPORT_TRIOS_REFERENCE.json');reference=[]
    for old in hist:
        r=tri[old['unit']];v=str(old['V']).lower() in ('1','1.0','true');labels=dict(x.split(':',1) for x in (old['full_labels'] or '').split('|') if x)
        ctx=next(c for c in contexts if c['unit']==old['unit']);match=v==bool(r['V']) and (not v or (all(labels[pid]==lab for pid,lab in zip(ctx['ids'],r['full_labels'])) and qa.close(r['tau1'],float(old['tau1'])) and qa.close(r['tau2'],float(old['tau2']))))
        reference.append(dict(dataset_id=r['dataset_id'],scope=r['scope'],replicate_id=r['replicate_id'],historical_V=int(v),new_V=r['V'],full_state_reference_parity='PASS' if match else 'DIFFERENCE_REQUIRES_REVIEW',historical_source_use='INDEPENDENT_FULL_STATE_REFERENCE_ONLY'))
    e.csvwrite(A/'01_QA/HISTORICAL_FIXED_TRANSPORT_REFERENCE_PARITY.csv',reference)
    # Do not substitute historical validity; this check is made after every new AB actually ran.
    qa.check(all(r['full_state_reference_parity']=='PASS' for r in reference),'HISTORICAL_REFERENCE_STATE_PARITY')
    reconc=[]
    for old in rt.read(A/'00_BINDING/D3_38_NO_PASS.json'):
        unit=f"{old['dataset_id']}|R{old['replicate_id']:03d}";r=tri[unit]
        reconc.append(dict(**{k:v for k,v in old.items() if k!='ids'},new_V=r['V'],new_failure_code=r['full_failure_code'],new_D=r['D'],new_tau1=r['tau1'],new_tau2=r['tau2'],new_group_n_L=r['full_labels'].count('L') if r['V'] else None,new_group_n_I=r['full_labels'].count('I') if r['V'] else None,new_group_n_H=r['full_labels'].count('H') if r['V'] else None,full_score_hash=r['full_score_hash'],C_conditional=r['C_conditional'],C_operational=r['C_operational'],n_success=r.get('n_success'),n_failed=r.get('n_failed'),interpretation='D3_NO_PASS_PRESERVED; FIXED_D2_DOWNSTREAM_AB_SEPARATE'))
    e.csvwrite(A/'EXPERIMENT_A_TRIOS_38_SUBSET_RECONCILIATION.csv',reconc);qa.check(len(reconc)==38 and all(r['new_V'] for r in reconc),'38_NEW_AB_VALIDITY')
    print('Stored numerical QA and 38-subset reconciliation complete; verifying source hashes',flush=True)
    protected_checks=0
    for name in ['SOURCE_SHA256.csv','USED_CHECKPOINT_SHA256.csv','MANUSCRIPT_PROTECTION_SHA256.csv','RAW_DATA_SHA256.csv']:
        for row in e.rows(A/'00_BINDING'/name):qa.check(rt.sha(row['path'])==row['sha256'],'PROTECTED_FILE_UNCHANGED:'+row['path']);protected_checks+=1
    # Full/partial matched payloads were never imported or dispatched by this branch.
    failures=[dict(dataset_id=r['dataset_id'],scope=r['scope'],replicate_id=r['replicate_id'],method_id=r['method_id'],level='FULL_INVALID',failure_code=r['full_failure_code']) for r in full+b100 if not r['V']]
    e.csvwrite(A/'04_RESULTS/FULL_INVALID_TASKS.csv',failures)
    runtime=rt.read(A/'EXECUTION_COMPLETE.json');env=rt.read(A/'00_BINDING/ENVIRONMENT.json')
    finalqa=dict(status='PASS',FULL_tasks=567,B100_tasks=56700,contexts=707,intended_fold_records=nfold,source_and_protection_hash_checks=protected_checks,D3_no_pass_reconciled=38,historical_reference_states=len(reference),counts=dict(counts),fold_record_sources=dict(source_counts),no_fit_calls=True,no_CRS_reconstruction=True,no_LOO_domain_selection=True,no_representation_recalculation=True,no_manuscript_edits=True,execution_seconds=runtime['elapsed_seconds'],finalization_seconds=time.time()-begin)
    rt.atomic_json(A/'01_QA/FINAL_QA.json',finalqa,immutable=True)
    protocol=f'''# Experiment A: protocol and source binding

Contract: `{m.CONTRACT}`. Author authority: `00_BINDING/AUTHOR_EXECUTION_INSTRUCTION.md` (byte-preserved). Interpretation: **conditional-on-established-domain, fixed-score comparative evaluation**. Seven datasets are reported separately; no cross-dataset ranking.

Seven fixed D2 endpoints (μM): Paclitaxel–SKIN .08; AZD6244–LUNG 2.53; PD-0325901–HEM .25; AZD6244–HEM 2.53; Topotecan–LUNG 2.53; PLX4720–SKIN 8; TAE684–HEM 8. Canonical direct exp/log K=7 nodes and endpoint-inclusive weights were checked against saved D2 nodes; the CRS column was read unchanged. No ratio-power scores were used.

Each locked subset receives its 18 D2 scores and an independent AB full partition (m_AB(18)=3), followed by classifier-only LOO of that same vector. It inherits neither full-source labels/cutoffs nor D3 validity/domain. All TRIOS full partitions and folds in this experiment were classified directly; no historical 700/700 validity was hardcoded.

The other 80 pipelines are the frozen 4 models × 4 representations × 5 stratifiers. IC50 is the archived native concentration/orientation, EC50 the frozen established endpoint definition, EMAX the response at 8 μM, and AUC the normalized physical-dose integral [.0025,8]. Scalar caches and frozen eligible CCLE AICc weights are reused; no new integral, root, fit, or upstream scoring is executed. P_fit is the profile mean of the relevant frozen model weights, using CCLE's executed parameter count. The frozen registry, fit hashes and weights are listed in `00_BINDING/SOURCE_SHA256.csv`.

Conventional full states and fold sufficient statistics are reused from SHA-bound Option C checkpoints only after exact ordered-profile/score/domain/projection checks. Their bound producer explicitly kept conventional scores fixed; all conventional successful fold vectors were compared to the new fixed full vector. Original scalar/fit/classifier/seed/tie definitions remain frozen. Failure records retain their original failure code; they are not reclassified as successes. Old QSC-family scores or rankings are not used. TRIOS native LOO folds are not reused.

ORIGINAL evaluator source: `{F_path()}` lines 127–143. For full-valid task: S_BW=B/(B+W); F_h=1 for successful fold, v=sum F_h, A_LOO=v/n; d_h²=((tau1_h−tau1)/R_x)²/2+((tau2_h−tau2)/R_x)²/2, R_x=max(x)−min(x). S_tau=A_LOO/(1+sqrt(mean_success d_h²)), zero if v=0. Let a=mean_success omitted agreement and b=mean_success unbalanced retained agreement; S_match=A_LOO·2ab/(a+b), zero if v=0 or a+b=0. I_SF=1(min full group size≥2). Task C is the mean of these five components. Full-invalid components/conditional C are undefined, operational contribution zero and all intended folds NOT_ATTEMPTED. Failed folds have undefined cutoffs/agreements and remain in n; failure does not invalidate full-data validity.

AB uses L≤tau1, I≤tau2, H>tau2; conventional uses L≤tau1, I<tau2, H≥tau2. Native classifier legal-size and exact-tie rules are unchanged. Seed context remains TRIOS_PHASEB_V12|CCLE_FULL or CCLE_N18_B100|unit|method|FULL or PERTURB|heldout; frozen seed/start/tolerance appear in `ENVIRONMENT.json`.

B100: A_valid=n_valid/100; conditional C=mean valid task C; operational C=sum valid task C/100. Each component summary is conditional on valid tasks, never invalid-filled zero. All-invalid conditional C/rank are undefined, operational C=0. Both rankings use exact equality and minimum rank separately within each dataset. FULL uses the same rule with one intended task.

Source chain: frozen observed datasets → unchanged profile fits/CCLE weights → D2 canonical scores or conventional scalar cache → task full state → fixed-score classifier fold → ORIGINAL components. `SOURCE_SHA256.csv`, `RAW_DATA_SHA256.csv`, `USED_CHECKPOINT_SHA256.csv`, per-dataset binding JSONs and checkpoint source references retain exact paths and hashes. Historical fixed-transport TRIOS full states are independent post-execution checks, not a source of validity or new metrics. Existing D3 662/700 and 38 terminal no-pass results remain unchanged. This experiment cannot establish standalone n=18 transport.

No matched ablation, manuscript edit or production promotion is included.
'''
    (A/'EXPERIMENT_A_PROTOCOL_AND_SOURCE_BINDING.md').write_text(protocol,encoding='utf-8')
    lines=['# TRIOS versus the 80 frozen comparators','', '**B100 author-review summary (seven datasets separately)**','', '|Dataset|A_valid|Conditional C|Operational C|Conditional rank|Operational rank|','|---|---:|---:|---:|---:|---:|']
    for r in bs:
        if r['method_id']=='TRIOS':lines.append(f"|{r['dataset_id']}|{r['A_valid']:.6f}|{r['C_conditional']:.15g}|{r['C_operational']:.15g}|{r['rank_conditional']}|{r['rank_operational']}|")
    for scope,rr in [('FULL',fs),('B100',bs)]:
        lines+=['',f'## {scope}','']
        for cid in e.CID:
            group=[r for r in rr if r['dataset_id']==cid];t=next(r for r in group if r['method_id']=='TRIOS');conv=[r for r in group if r['method_id']!='TRIOS']
            lines.append(f"- {cid}: TRIOS conditional {t['C_conditional']:.15g} (rank {t['rank_conditional']}/81); operational {t['C_operational']:.15g} (rank {t['rank_operational']}/81); A_valid={t['A_valid']:.6g}.")
            for field in ['C_conditional','C_operational']:
                eligible=[r for r in conv if r[field] is not None];best=max(r[field] for r in eligible);winners=[r['method_id'] for r in eligible if r[field]==best]
                lines.append(f"  Best conventional {field}: {'; '.join(winners)} = {best:.15g}. Descriptive comparison only; no new significance test.")
            lines.append(f"  Complete 81-row result: `04_RESULTS/{cid}_{scope}_81.csv`.")
    lines+=['','Conditional-on-established-domain, fixed-score comparative evaluation. Component averages condition on validity. No cross-dataset ranking; old native-LOO or QSC rankings are not substituted. D3 standalone transport retains 662/700 with 38 no-pass subsets.']
    (A/'EXPERIMENT_A_TRIOS_VS_80_COMPARATORS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (A/'EXPERIMENT_A_QA_REPORT.md').write_text('# Experiment A QA\n\nStatus: PASS; AUTHOR_REVIEW_REQUIRED.\n\n'+json.dumps(finalqa,indent=2)+'\n\nSource and synthetic gates passed before FULL. All FULL task/fold and independent component QA passed before B100. Every completed context has an immutable QA.json. All seven datasets have 81 methods for both rank definitions. Exact ties use minimum rank; undefined conditional values are unranked. Per-task QA re-derives projections, cutoff drift, omitted agreement, unbalanced retained agreement, the ORIGINAL five components and validity semantics from persisted vectors/folds. No metric is reconstructed from a historical QSC aggregate.\n\nFull raw responses and profile identities matched the seven frozen source CSVs. Each valid fold carries the same full/fold score hash; original reused conventional fold vectors are checked exactly. Engine upstream entry points are disabled. Raw/fit/source/checkpoint/manuscript hashes were checked unchanged. No historical result was overwritten. No manuscript/global source lock is declared.\n',encoding='utf-8')
    # Include all requested outputs, binding/QA evidence and executable source, but not duplicate individual checkpoints.
    selected=[p for p in A.rglob('*') if p.is_file() and not any(x in p.relative_to(A).parts for x in ['02_FULL','03_B100','07_RUNTIME']) and p.suffix not in ['.zip','.log'] and p.name not in ['STATUS.json','AUTHOR_REVIEW_ZIP_QA.json','AUTHOR_REVIEW_MANIFEST.csv','FINAL_STATUS.json']]
    manifest=[dict(path=p.relative_to(A).as_posix(),bytes=p.stat().st_size,sha256=rt.sha(p)) for p in selected];e.csvwrite(A/'AUTHOR_REVIEW_MANIFEST.csv',manifest);selected.append(A/'AUTHOR_REVIEW_MANIFEST.csv')
    out=A/'CCLE_EXPERIMENT_A_AUTHOR_REVIEW.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in selected:z.write(p,p.relative_to(A).as_posix())
    with zipfile.ZipFile(out) as z:
        qa.check(z.testzip() is None,'ZIP_CRC');qa.check(set(z.namelist())=={p.relative_to(A).as_posix() for p in selected},'ZIP_EXACT_MEMBERS')
        for row in manifest:qa.check(hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'],'ZIP_PAYLOAD_SHA')
    rt.atomic_json(A/'AUTHOR_REVIEW_ZIP_QA.json',dict(CRC='PASS',payload_SHA256='PASS',files=len(selected),zip_sha256=rt.sha(out),zip_bytes=out.stat().st_size),immutable=True)
    status=dict(status='AUTHOR_REVIEW_REQUIRED',contract=m.CONTRACT,FULL=567,B100=56700,matched_ablations=0,QA='PASS',archive=str(out),archive_SHA256=rt.sha(out),execution_seconds=runtime['elapsed_seconds'],finalization_seconds=time.time()-begin)
    rt.atomic_json(A/'FINAL_STATUS.json',status,immutable=True);rt.atomic_json(A/'STATUS.json',status);print(json.dumps(status,indent=2),flush=True)
def F_path():return str(m.F/'execute_full.py')
