"""Read complete binding material, verify all source hashes, then lock adapter."""
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,json,hashlib,shutil,time,uuid
ROOT=Path(__file__).resolve().parents[1]
R=ROOT.parents[1]
AUD=R/'deliveries/E4_EXTENSION_PHASE0_SOURCE_LOCK_20260929_v1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    for d in ('00_AUTHORITY','01_DESIGN','REPORTS','DELIVERY'):
        (ROOT/d).mkdir(exist_ok=True)
    for lane in ('A','B'):
        for d in ('RAW_FITS','TASKS','FOLDS','SUMMARY','PAIRED','QA'):(ROOT/f'LANE_{lane}'/d).mkdir(parents=True,exist_ok=True)
    inventory=[]
    for parent in (AUD,R/'deliveries/SECTION4_READONLY_POSTPROCESSING_20260929_v1'):
        for p in sorted(parent.rglob('*')):
            if not p.is_file():continue
            data=p.read_bytes()
            if p.suffix in ('.md','.py','.json','.csv'):data.decode('utf-8-sig')
            if p.suffix=='.json':json.loads(data)
            inventory.append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (ROOT/'00_AUTHORITY/READ_BINDING_INVENTORY.json').write_text(json.dumps(inventory,indent=2))
    rows=list(csv.DictReader((AUD/'00_SOURCE_LOCK/INPUT_SHA256_MANIFEST.csv').open(encoding='utf-8-sig')))
    for i,row in enumerate(rows):
        actual=sha(Path(row['source_path']));assert actual.lower()==row['sha256'].lower(),('SOURCE_SHA_MISMATCH',row['source_path'])
        if i%1000==0:print('verified',i,'/',len(rows),flush=True)
    shutil.copy2(AUD/'00_SOURCE_LOCK/INPUT_SHA256_MANIFEST.csv',ROOT/'00_AUTHORITY/INPUT_SHA256_MANIFEST.csv')
    for name in ('PROTECTED_BASELINE.json','FROZEN_CODE_BINDING.json'):
        shutil.copy2(AUD/'00_SOURCE_LOCK'/name,ROOT/'00_AUTHORITY'/name)
    for name in ('TRIOS_E4_LaneA_LaneB_FORMAL_EXECUTE_Authorization_20260929.md','TRIOS_E4_Prospective_Raw_Dose_Budget_Expansion_Design_v1_0_20260929.md'):
        p=Path('OMITTED_HOST_PATH/Downloads')/name;p.read_text(encoding='utf-8-sig');shutil.copy2(p,ROOT/'00_AUTHORITY'/name)
    from adapter import src,load,sha as h
    tasks=[]
    for sh in range(64):
        a=load(src.RAW/f'raw_shard_{sh:02d}.pkl.gz')['cohorts'];b=load(src.V2_WORK/'raw_bank'/f'raw_shard_{sh:02d}.pkl.gz')['cohorts']
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):
            assert x['meta']['cohort_id']==y['meta']['cohort_id'] and x['ids']==y['ids']
            m=x['meta'];tasks.append(dict(shard=sh,index=i,cohort_id=m['cohort_id'],n=int(m['n']),separation=m['separation'],family=m['family']))
    assert len(tasks)==10200 and len({r['cohort_id'] for r in tasks})==10200 and sum(r['n'] for r in tasks)==212400
    # First member in deterministic shard order for each n/separation/family cell.
    seen=set();prefix=[]
    for t in tasks:
        k=(t['n'],t['separation'],t['family'])
        if k not in seen:seen.add(k);prefix.append(t)
    assert len(prefix)==45
    config=dict(run_uuid=str(uuid.uuid4()),created=time.time(),workers_A=16,workers_B=4,threads=1,source_manifest_sha=h(ROOT/'00_AUTHORITY/INPUT_SHA256_MANIFEST.csv'),code={p.name:h(p) for p in sorted((ROOT/'code').glob('*.py'))},prefix=prefix,tasks=tasks,laneB_metric_parity_tolerance=1e-9,laneB_tolerance_source='frozen current run_phase_c_e4.py s1_worker / FINAL_STATUS',full_tasks_A=30600,full_tasks_B=10200)
    p=ROOT/'01_DESIGN/LOCK.json'
    assert not p.exists(),'LOCK_ALREADY_EXISTS';p.write_text(json.dumps(config,indent=2));print('SOURCE_LOCK_PASS',h(p),flush=True)
if __name__=='__main__':main()
