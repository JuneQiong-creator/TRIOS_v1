"""Local, independently checkpointed lane execution. Resume uses the identical lock."""
import sys,os
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time,traceback,threading,ctypes,msvcrt
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from adapter import *
CFG=json.loads((OUT/'01_DESIGN/LOCK.json').read_text());CONFIG_SHA=sha(OUT/'01_DESIGN/LOCK.json')
STATUS={};MUTEX=threading.Lock()
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('totalpage',ctypes.c_ulonglong),('availablepage',ctypes.c_ulonglong),('totalvirtual',ctypes.c_ulonglong),('availablevirtual',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
def resource():
    m=Memory();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(memory_load_percent=m.load,available_GiB=m.available/2**30,total_GiB=m.total/2**30)
def update(lane,**kwargs):
    with MUTEX:
        STATUS.setdefault(lane,{}).update(kwargs)
        js(OUT/'FINAL_STATUS.json',dict(status='RUNNING' if any(s.get('state') in ('PREFIX_RUNNING','FULL_RUNNING','H250_EXACTLOG_RUNNING') for s in STATUS.values()) else 'PARTIAL',lanes=STATUS,supervisor_pid=os.getpid(),run_uuid=CFG['run_uuid'],heartbeat=time.time(),resources=resource()))
def B3_job(arg):
    _,arm,sh,index,config=arg
    raw=load(src.V2_WORK/'raw_bank'/f'raw_shard_{sh:02d}.pkl.gz')['cohorts'][index];cid=raw['meta']['cohort_id']
    p=OUT/f'LANE_B/RAW_FITS/H250/{cid}.pkl.gz';obj=load(p);assert obj['parity']['status']=='PASS'
    dest=OUT/f'LANE_B/TASKS/H250/{cid}.pkl.gz'
    if dest.exists():
        saved=load(dest);assert saved['key']==obj['key'] and saved['complete'];return dict(lane='B',arm=arm,cohort_id=cid,profiles=len(raw['ids']),reused=True,seconds=0,path=str(dest))
    from constrained_pspline_solver import ConstrainedPSpline
    models=[ConstrainedPSpline(intercept_=f['intercept'],differences_=f['differences'],coefficients_=f['coefficients'],edge_knots_=f['edge'],selected_lambda=f['lambda'],effective_degrees_of_freedom=f['EDF'],GCV=f['GCV'],replicate_level_SSE=f['SSE']) for f in obj['fits']]
    start=time.perf_counter();result=downstream(raw,arm,obj['fits'],models=models)
    digest=atomic(dest,dict(complete=True,key=obj['key'],result=result,fit_checkpoint_sha256=sha(p),seconds=time.perf_counter()-start))
    return dict(lane='B',arm=arm,cohort_id=cid,profiles=len(raw['ids']),seconds=time.perf_counter()-start,reused=False,path=str(dest),sha256=digest)
def run_batch(lane,tasks,arms,workers,stage,fn=checkpoint_job):
    args=[(lane,a,t['shard'],t['index'],CONFIG_SHA) for t in tasks for a in arms]
    start=time.monotonic();done=profiles=0;result_rows=[]
    update(lane,state=stage,stage_intended_tasks=len(args),stage_completed=0,workers=workers)
    # Only a bounded queue is submitted. Reduced dispatch under memory pressure preserves task identity.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        it=iter(args);pending={};exhausted=False
        while pending or not exhausted:
            mem=resource();limit=workers if mem['available_GiB']>=6 else max(2,workers//2) if mem['available_GiB']>=3 else 1
            while len(pending)<limit and not exhausted:
                try:a=next(it)
                except StopIteration:exhausted=True;break
                pending[pool.submit(fn,a)]=a
            ready,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for f in ready:
                a=pending.pop(f)
                try:r=f.result()
                except BaseException:
                    for q in pending:q.cancel()
                    raise
                result_rows.append(r);done+=1;profiles+=r['profiles']
                with (OUT/f'LANE_{lane}/completion_events.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(r,default=str)+'\n')
            elapsed=time.monotonic()-start
            update(lane,state=stage,stage_completed=done,stage_profiles=profiles,stage_seconds=elapsed,profiles_per_minute=profiles/elapsed*60,workers=workers,active_submitted=len(pending),dispatch_limit=limit,stage_eta_minutes=(len(args)-done)*elapsed/done/60 if done else None)
    return result_rows
def lane_main(lane):
    try:
        arms=['J3','J4','J6'] if lane=='A' else ['H250'];workers=16 if lane=='A' else 4
        prefix=run_batch(lane,CFG['prefix'],arms,workers,'PREFIX_RUNNING')
        assert len(prefix)==45*len(arms)
        js(OUT/f'LANE_{lane}/QA/PREFIX_PASS.json',dict(status='PASS',records=prefix,resources=resource(),note='Predeclared first member of each n/separation/family cell; included exactly once through validated resume'))
        run_batch(lane,CFG['tasks'],arms,workers,'FULL_RUNNING')
        if lane=='B':
            files=list((OUT/'LANE_B/RAW_FITS/H250').glob('*.pkl.gz'));assert len(files)==10200
            count=0
            for p in files:
                q=load(p);assert q['parity']['status']=='PASS';count+=len(q['fits'])
            assert count==212400
            js(OUT/'LANE_B/QA/FULL_POPULATION_PARITY_PASS.json',dict(status='PASS',cohorts=10200,profiles=count,fit_tolerance=1e-9,time=time.time()))
            run_batch(lane,CFG['tasks'],arms,workers,'H250_EXACTLOG_RUNNING',B3_job)
        update(lane,state='COMPUTATION_DONE_SUMMARY_QA_PENDING',completed_tasks=30600 if lane=='A' else 10200)
    except BaseException as exc:
        detail=traceback.format_exc();js(OUT/f'LANE_{lane}/QA/STOP.json',dict(error=repr(exc),traceback=detail,time=time.time()))
        update(lane,state='LANE_B_BLOCKED_H250_FIT_PARITY' if lane=='B' else 'HARD_STOP',error=repr(exc));print(detail,flush=True)
def main():
    assert json.loads((OUT/'00_AUTHORITY/INTEGRATION_QA.json').read_text())['status']=='PASS'
    for name,digest in CFG['code'].items():assert sha(OUT/'code'/name)==digest
    # OS-held exclusive byte-range lock survives stale metadata and is released only on process exit.
    lock=(OUT/'RUN_LOCK.bin').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    js(OUT/'RUN_IDENTITY.json',dict(pid=os.getpid(),run_uuid=CFG['run_uuid'],started=time.time(),command=sys.argv,configuration_sha256=CONFIG_SHA))
    manifest={p.name:sha(p) for p in (OUT/'code').glob('*.py')};dest=OUT/'00_AUTHORITY/EXECUTABLE_SHA.json'
    if dest.exists():assert json.loads(dest.read_text())==manifest,'EXECUTION_CODE_CHANGED'
    else:js(dest,manifest)
    threads=[threading.Thread(target=lane_main,args=(lane,)) for lane in ('A','B')]
    for t in threads:t.start()
    for t in threads:t.join()
    print('LANE_COMPUTATION_EXIT',json.dumps(STATUS),flush=True)
if __name__=='__main__':main()
