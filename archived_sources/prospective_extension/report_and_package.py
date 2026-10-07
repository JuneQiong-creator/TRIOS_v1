"""Seal already validated derived results. No scientific execution."""
import os,sys,json,time,hashlib,zipfile,shutil,io,csv,difflib,gzip,pickle
from pathlib import Path
import pandas as pd
O=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(rel,text):
    p=O/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf8')
def stamp(t):return time.strftime('%Y-%m-%d %H:%M:%S %z',time.localtime(t))
def md(df):
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in df.itertuples(index=False,name=None))
def main():
    status=json.loads((O/'REPORTS/POSTPROCESSING_STATUS.json').read_text());assert status['status']=='NUMERICAL_POSTPROCESSING_QA_PASS_REPORT_PACKAGING_PENDING'
    qa=pd.read_csv(O/'EXECUTION_QA_CHECKS.csv');assert qa.status.eq('PASS').all()
    counts=pd.read_csv(O/'REPORTS/COMPUTATION_COUNTS.csv');j=pd.read_csv(O/'E4_J3_J7_BURDEN_PERFORMANCE_MASTER.csv');h=pd.read_csv(O/'E4_H100_H250_MATCHED_STAGE1_CURRENT_EXACTLOG.csv');h=h[h.separation=='ALL']
    s=pd.read_csv(O/'E4_SEPARATION_ROBUSTNESS.csv');cp=pd.read_csv(O/'E4_ALL_INTENDED_AND_COMMON_VALID_CONTRASTS.csv')
    identity=json.loads((O/'RUN_IDENTITY.json').read_text());endA=max(p.stat().st_mtime for p in (O/'LANE_A/RAW_FITS').rglob('*.pkl.gz'));endB=max(p.stat().st_mtime for p in (O/'LANE_B/TASKS/H250').glob('*.pkl.gz'))
    bpar=json.loads((O/'LANE_B/QA/FULL_POPULATION_PARITY_PASS.json').read_text());br=json.loads((O/'LANE_B/QA/SCHEDULING_RECOVERY.json').read_text())
    minJ=int(j.loc[j.A_valid>=.95,'J'].min()) if (j.A_valid>=.95).any() else None
    doses=[]
    for budget in range(3,8):
        grid=[0.]+[150**(k/(budget-1)) for k in range(budget)]
        for i,d in enumerate(grid):doses.append(dict(arm=f'J{budget}',J=budget,H=150,node_index=i,dose_uM_17g=format(d,'.17g'),binary64_hex=d.hex(),replicates=3,wells_per_profile=3*(budget+1),source='frozen J5/J7 reuse' if budget in (5,7) else 'frozen J7 subset' if budget in (3,4) else 'J7 shared' if i in (0,1,6) else 'locked latent plus original paired noise'))
    for i,d in enumerate([0.,1.,2.5099014421834109,6.2996052494743653,15.811388300841896,39.685026299204978,99.605504741461203,250.]):doses.append(dict(arm='H250',J=7,H=250,node_index=i,dose_uM_17g=format(d,'.17g'),binary64_hex=d.hex(),replicates=3,wells_per_profile=24,source='V2 REGENERATED_E1_GRID original raw'))
    pd.DataFrame(doses).to_csv(O/'01_DESIGN/EXACT_DOSE_REGISTRY.csv',index=False)
    geometries={}
    for arm in ('J3','J4','J6'):
        for p in sorted((O/f'LANE_A/RAW_FITS/{arm}').glob('*.pkl.gz')):
            with gzip.open(p,'rb') as f:q=pickle.load(f)
            for k,g in q['result']['geometries'].items():
                if k in geometries:assert geometries[k]==g
                else:geometries[k]=g
            if len(geometries)==6:break
        if len(geometries)==6:break
    write('01_DESIGN/OBSERVED_FULL_FOLD_K7_GEOMETRIES.json',json.dumps(geometries,indent=2)+'\n')
    env=json.loads((O/'00_AUTHORITY/INTEGRATION_QA.json').read_text())
    write('00_AUTHORITY/EXECUTION_ENVIRONMENT.json',json.dumps(next(x['detail'] for x in env['checks'] if x['check']=='RUNTIME'),indent=2)+'\n')
    code_manifest=[dict(path=str(p.relative_to(O)),sha256=sha(p)) for p in sorted((O/'code').glob('*.py'))]
    pd.DataFrame(code_manifest).to_csv(O/'00_AUTHORITY/FINAL_CODE_MANIFEST.csv',index=False)
    write('REPORTS/ADDITIVE_ADAPTER_DIFF.patch',''.join(difflib.unified_diff([], (O/'code/adapter.py').read_text().splitlines(True),fromfile='/dev/null',tofile='new_run/code/adapter.py')))
    tt=pd.read_csv(O/'REPORTS/ALL_CURRENT_TASKS.csv.gz');tt['failure_category']=tt.failure_category.fillna('VALID')
    tt.groupby(['arm','valid','failure_category'],dropna=False).agg(tasks=('cohort_id','size'),profiles=('n','sum'),weight_mass=('factor_weight','sum')).reset_index().to_csv(O/'REPORTS/FULL_VALIDITY_FAILURE_COUNTS.csv',index=False)
    cols=['arm','intended_tasks','valid_tasks','A_valid','C_conditional','C_operational','ARI','BA','MacroF1','Extreme']
    display=j[cols].copy()
    for c in cols[3:]:display[c]=display[c].map(lambda x:f'{x:.9f}')
    strat=s[['arm','stratum','A_valid','C_operational','ARI','BA','MacroF1','Extreme']].copy().round(9)
    contrast_display=cp[(cp.separation=='ALL') & cp.metric.isin(['A_valid','C_operational_E4','C_common','ARI','BA','MacroF1','Extreme'])][['arm','metric','estimand_domain','difference','CI95_low','CI95_high','common_valid_pairs']].round(9)
    report=f'''# E4 Lane A — completed J3/J4/J6 extension

Status: COMPLETE_REVIEW_REQUIRED. Results are an authorized post-readout extension; no historical parent or production pointer is promoted.

Run root: `{O}`. Run UUID: `{identity['run_uuid']}`.
Scientific process started: {stamp(identity['started'])}. Last Lane A checkpoint: {stamp(endA)}. Elapsed from common process start: {(endA-identity['started'])/60:.3f} min, including prefix and worker startup.

J3, J4 and J6 each contain 10,200 cohorts and 212,400 profiles: 637,200 original 21-lambda profile fits in total. Full/native-LOO outputs were read back and checked independently against stored scores, cutoffs, omitted/retained projections and component formulas. Scientific invalid tasks remain in intended denominators; no invented folds or scores. J5/J7/rounded-J5 are read-only exact-log parent rows.

## Five-budget overall results

{md(display)}

The smallest tested budget passing the locked **overall** A_valid >= 0.95 gate is **J={minJ}**. This is a descriptive gate-relative finding, not a globally optimal assay or wet-lab equivalence claim. The threshold was not applied as a new subgroup decision rule. Scoring K remains 7 for every finite full/fold domain; J counts measured positive doses plus a genuine zero control and three replicates.

J3/J4 reproduce their entire inherited raw arrays exactly; J6 shares 0/1/150-dose bytes and contains exactly 2,548,800 new interior replicate measurements under the original latent/noise keys. No biological profiles or calibration were regenerated. Exact full-precision grids and raw provenance are saved in the locked code and checkpoints.

## Separation robustness

{md(strat)}

These subgroup values are descriptive and do not introduce a subgroup eligibility gate. Own-valid truth recovery uses each arm's valid population and cannot be substituted for a paired contrast.

## Pairing and uncertainty

All four J-versus-J7 comparisons use the original 5,000-draw registry: 102 strata, 20 paired blocks per stratum, all five families together. Truth and conditional-quality contrasts are common-valid paired contrasts. Applicability and operational contrasts use all intended pairs. Every bootstrap ratio has its own reformed denominator; percentile CIs use the original linear convention. J5/J7 point differences, CIs and all 5,000 replicates reconcile to the accepted postprocessing pack within its existing 5e-12 sum-order QA tolerance. No new P-values or tuning.

Overall contrasts below are left arm minus J7. Common-valid pair counts are supplied even for all-intended metrics, whose denominator remains all intended tasks.

{md(contrast_display)}

ALL, separation, n and family summaries are in `LANE_A/SUMMARY/ALL_N_FAMILY_SEPARATION.csv`; full contrast rows, membership and replicate outputs are in `LANE_A/PAIRED/`. Four truth endpoints remain separate; C_common is valid-task conditional quality and is never substituted directly for all-intended operational quality. Rounded J5 is an independent sensitivity, not a sixth budget.

Limitations: Only the predefined exact grids were evaluated; untested/rounded J3/J4/J6 schedules, real-lab cost and clinical validity are not established. This report does not automatically select a practical assay or edit the manuscript. Full failures and separation-specific weaknesses are retained in the attached tables.
'''
    write('REPORTS/E4_LANE_A_J3_J4_J6_EXECUTION_REPORT.md',report)
    hc=['arm','A_valid','C_conditional','C_operational','CRS_spearman','balanced_label_fidelity','exact_label_agreement','operational_gate_pass','strict_reference_gate_pass']
    write('REPORTS/E4_LANE_B_H250_RECONSTRUCTION_AND_EXACTLOG_REPORT.md',f'''# E4 Lane B — prospective H250 recovery and exact-log completion

Status: COMPLETE_REVIEW_REQUIRED. Original prospective `REGENERATED_E1_GRID` only; E1 H250_CONTROL is not used.

212,400 profiles in 10,200 cohorts were reconstructed once at their recorded selected lambda. **No 21-lambda H250 search** occurred. Full-population parity passed at {stamp(bpar['time'])}, before any new exact-log downstream dispatch. Maximum fit diagnostic residual (SSE/GCV/EDF): {counts.loc[counts.arm=='H250','fit_parity_max'].iloc[0]:.17g}. Original fit status/lambda, transport/no-pass, fold endpoint identity and available historical-geometry score/label diagnostics were checked. The inherited numerical tolerance is 1e-9 from the frozen current E4 replay, not a post-readout rescue tolerance.

Current exact-log K7/AB/native LOO finished at {stamp(endB)}. Domain is reselected in each retained cohort; profile fits stay frozen. All 10,200 downstream checkpoints exist and were independently read back. Historical geometry is confined to parity diagnostics and is not a current result.

{md(h[hc].round(9))}

H100/H150/H200 are unchanged current exact-log parent records. H250 is a newly reconstructed prospective reference, not relabeled E1 control. Score Spearman, class-balanced label fidelity and exact label agreement use common-valid matched cohorts and original factor weights. Four truth outcomes, conditional/operational quality, subgroup coverage and 5,000-draw paired contrasts are provided separately. The original overall operational H150 support rule remains unchanged; strict reference-gate failures are reported as such, not hidden or substituted for operational selection.

## Recovered scheduling error

The initial coordinator hit Windows PermissionError while replacing an observational progress JSON. No fit parity test failed and no scientific checkpoint was overwritten. Its worker pool exited safely; the recovery coordinator resumed with unchanged adapter/input hashes and skipped all completed task keys. The old `LANE_B_BLOCKED_H250_FIT_PARITY` status was a misleading broad exception label; this incident was an I/O reporting failure. The original error remains archived. Recovery started {stamp(br['time'])}; `LANE_B/progress_recovery.jsonl` and full-population parity/checkpoint QA supersede the stale coordinator label.
''')
    timing=[dict(event='scientific supervisor start',time=stamp(identity['started']),source='RUN_IDENTITY.json'),dict(event='B scheduling-only resume',time=stamp(br['time']),source='LANE_B/QA/SCHEDULING_RECOVERY.json'),dict(event='B full-population recovery parity PASS',time=stamp(bpar['time']),source='LANE_B/QA/FULL_POPULATION_PARITY_PASS.json'),dict(event='B exact-log/native LOO checkpoint complete',time=stamp(endB),source='LANE_B/TASKS/H250/'),dict(event='A checkpoint complete',time=stamp(endA),source='LANE_A/RAW_FITS/'),dict(event='postprocessing QA complete',time=stamp((O/'REPORTS/POSTPROCESSING_STATUS.json').stat().st_mtime),source='REPORTS/POSTPROCESSING_RUNTIME.log')]
    pd.DataFrame(timing).to_csv(O/'REPORTS/RUN_TIMELINE.csv',index=False)
    write('REPORTS/RUN_LOG.md','# Actual execution log\n\n'+md(pd.DataFrame(timing))+'\nInitial concurrency: Lane A 16 workers, Lane B 4 workers; BLAS/OMP limits 1. Separate directories/checkpoints. Memory-pressure dispatch limit was implemented without algorithm changes. Observed console snapshots left about 14–16 GiB available. Continuous CPU utilization and a rigorous process-tree peak-memory series were **not persisted**; no retrospective peak or CPU claim is made. Prefix task durations/checkpoint sizes and subsequent completion records are retained. This telemetry limitation does not imply a scientific QA failure.\n\nRaw execution logs, completion events and the Lane B recovery log are retained unchanged. Scientific start-to-checkpoint durations are distinct from later interactive/postprocessing/delivery time.\n')
    write('REPORTS/SOURCE_TO_OUTPUT_PROVENANCE.md',f'''# Source-to-output provenance

- `00_AUTHORITY/INPUT_SHA256_MANIFEST.csv`: original 10,613 source identities; all rehashed after computation in `PARENT_AND_PROTECTED_SHA_BEFORE_AFTER.csv`, plus protected manuscript baseline.
- `00_AUTHORITY/EXECUTABLE_SHA.json` and `01_DESIGN/LOCK.json`: pre-science frozen adapter/configuration, prefix IDs, original 64-shard cohort membership. Additional scheduling-only recovery has its own hash in `LANE_B/QA/SCHEDULING_RECOVERY.json`.
- New J3/J4/J6 checkpoint `raw` is the exact inherited view or locked J6 construction; `fits` holds coefficients/selected lambda/diagnostics; `result` holds full and native fold records. Unique key binds lane/arm/cohort/source SHA/config SHA.
- H250 RAW_FITS checkpoints contain original selected-lambda reconstruction and per-profile parity; separate TASKS checkpoints bind the fitted-state SHA and current exact-log output.
- Original current exact-log E4 tasks/scores supply J5/J7/rounding and H100/H150/H200. The source H150 arm is displayed as J7 only in budget tables.
- `LANE_*/QA/*_CHECKPOINT_MANIFEST.csv`: payload hashes of every raw-fit/downstream checkpoint. Flat fits/scores/tasks/folds are exported alongside retained full checkpoints; retained arrays remain available in the latter.
- `01_DESIGN/FROZEN_BOOTSTRAP_REGISTRY/`: unchanged original stratum/draw files, no new draws.
- The new postprocessor reads stored values and independently reconciles fold arithmetic; it performs no new fit, CRS, AB or native LOO.

All numerical tables are full precision; report display rounding does not replace source values. Invalid outcomes remain explicit. No parent, manuscript, theory, NV or production pointer was changed.
''')
    final=dict(status='COMPLETE_REVIEW_REQUIRED',lanes=dict(A=dict(state='COMPLETE_REVIEW_REQUIRED',cohorts_per_arm=10200,arms=['J3','J4','J6'],profile_fits=637200),B=dict(state='COMPLETE_REVIEW_REQUIRED',cohorts=10200,profile_reconstructions=212400,full_population_parity='PASS',historical_status_label_corrected='Recovered progress-file I/O error; no fit parity failure')),qa='PASS',minimum_tested_overall_eligible_J=minJ,production_promotion=False,manuscript_modified=False,completed_at=stamp(time.time()),telemetry_limitation='Continuous CPU and rigorous process-tree peak RAM not persisted')
    # Preserve stale original coordinator state as historical evidence before replacing consolidated status.
    original=O/'FINAL_STATUS.json';archived=O/'REPORTS/ORIGINAL_COORDINATOR_FINAL_STATUS.json'
    if not archived.exists():shutil.copy2(original,archived)
    write('FINAL_STATUS.json',json.dumps(final,indent=2)+'\n')
    write('README.md',f'''# E4 authorized extension — author review

Status: COMPLETE_REVIEW_REQUIRED. Both scientific lanes and derived numerical QA complete. No production promotion or paper edits.

Start with `REPORTS/E4_LANE_A_J3_J4_J6_EXECUTION_REPORT.md`, `REPORTS/E4_LANE_B_H250_RECONSTRUCTION_AND_EXACTLOG_REPORT.md`, and `REPORTS/RUN_LOG.md`.

Primary tables: `E4_J3_J7_BURDEN_PERFORMANCE_MASTER.csv`, `E4_SEPARATION_ROBUSTNESS.csv`, `E4_ALL_INTENDED_AND_COMMON_VALID_CONTRASTS.csv`, `E4_H100_H250_MATCHED_STAGE1_CURRENT_EXACTLOG.csv`.

The compact ZIP includes reports, summaries, paired CIs and all 5000 replicate outputs, code, source manifests, logs and QA. Large individual raw/fitted-state/scientific checkpoints and flat profile/fold exports remain at `{O}`; their hashes are included. Immutable parent banks are not duplicated. Do not rerun science: all requested checkpoints exist. Full results remain recoverable from this run root.
''')
    # Complete root payload manifest (except delivery archive and this manifest itself).
    payload=[]
    for p in sorted(O.rglob('*')):
        if p.is_file() and 'DELIVERY' not in p.relative_to(O).parts and p.name!='COMPLETE_RESULT_MANIFEST.csv':payload.append(dict(path=p.relative_to(O).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    pd.DataFrame(payload).to_csv(O/'COMPLETE_RESULT_MANIFEST.csv',index=False)
    selected=[]
    for p in sorted(O.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(O);parts=rel.parts
        if 'DELIVERY' in parts or p.suffix in ('.tmp','.claim','.bin'):continue
        if 'RAW_FITS' in parts or 'TASKS' in parts or 'FOLDS' in parts:continue
        if p.name=='ALL_CURRENT_TASKS.csv.gz':continue
        selected.append(p)
    zpath=O/'DELIVERY/E4_LANEA_LANEB_AUTHOR_REVIEW.zip'
    manifest=[dict(path=p.relative_to(O).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in selected]
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in selected:z.write(p,p.relative_to(O).as_posix())
        z.writestr('AUTHOR_REVIEW_PAYLOAD_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(manifest)+1 and len(set(z.namelist()))==len(z.namelist())
        for r in manifest:
            b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    seal=dict(status='PASS',CRC='PASS',payload_SHA256='PASS',complete_manifest='PASS',archive=str(zpath),archive_sha256=sha(zpath),archive_bytes=zpath.stat().st_size,payload_files=len(manifest),complete_result_files=len(payload),verified_at=stamp(time.time()))
    write('DELIVERY/ARCHIVE_VERIFICATION.json',json.dumps(seal,indent=2)+'\n');print(json.dumps(seal,indent=2),flush=True)
if __name__=='__main__':main()
