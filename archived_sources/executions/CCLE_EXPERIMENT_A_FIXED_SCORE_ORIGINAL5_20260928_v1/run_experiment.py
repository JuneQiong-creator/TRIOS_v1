"""Independent local FULL -> QA -> B100 -> QA execution. No matched entry point."""
import sys,os,time,traceback,uuid
sys.dont_write_bytecode=True
os.environ.setdefault('OPTION_C_RUN_ID','EXPERIMENT_A_'+str(uuid.uuid4()))
from concurrent.futures import ProcessPoolExecutor,as_completed
import engine as m
from engine import A,e,rt
import qa
def work_and_qa(ctx):
    r=m.evaluate_context(ctx);qa.audit_context(ctx);return r
def main():
    e.require(rt.read(A/'00_BINDING/INITIALIZED.json')['status']=='PASS','SOURCE_FIXTURE_GATE')
    e.require(rt.read(A/'01_QA/RAW_DATA_PARITY.json')['status']=='PASS','RAW_SOURCE_GATE')
    for r in e.rows(A/'00_BINDING/SOURCE_SHA256.csv'):e.require(rt.sha(r['path'])==r['sha256'],'SOURCE_SHA_MISMATCH:'+r['path'])
    for r in rt.read(A/'00_BINDING/LAUNCH_BINDING.json')['files']:e.require(rt.sha(r['path'])==r['sha256'],'LAUNCH_SHA')
    with rt.Lease(rt.RUNTIME/'EXPERIMENT_A_CONTROLLER.lock','EXPERIMENT_A_ONLY'):
        allctx=rt.read(A/'00_BINDING/CONTEXTS.json');started=time.time()
        for scope,workers in [('FULL',7),('B100',16)]:
            if scope=='B100':e.require(rt.read(A/'01_QA/FULL_GATE.json')['status']=='PASS','FULL_QA_BEFORE_B100')
            contexts=[c for c in allctx if c['scope']==scope];done=0;stage_start=time.time()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                fs={pool.submit(work_and_qa,c):c for c in contexts}
                for future in as_completed(fs):
                    try:r=future.result()
                    except BaseException as exc:
                        rt.atomic_bytes(A/'STOP_REQUESTED',str(exc).encode());rt.atomic_json(A/'HARD_STOP.json',dict(error=str(exc),traceback=traceback.format_exc(),context=fs[future]))
                        for f in fs:f.cancel()
                        raise
                    done+=1;elapsed=time.time()-stage_start
                    rt.atomic_json(A/'STATUS.json',dict(status='RUNNING',contract=m.CONTRACT,stage=scope,workers=workers,completed_contexts=done,total_contexts=len(contexts),completed_tasks=done*81,stage_elapsed_seconds=elapsed,ETA_stage_seconds=(len(contexts)-done)*elapsed/done,identity=rt.own_identity(),last_context=r))
                    if scope=='FULL' or done%25==0:print(f'{scope} {done}/{len(contexts)} contexts QA PASS; {elapsed:.1f}s',flush=True)
            if scope=='FULL':
                rows=[m.read_checkpoint(m.location(c)/(m.key(method)+'.json.gz'))['result'] for c in contexts for method in m.METHODS]
                summary=qa.aggregate(rows,'FULL');e.csvwrite(A/'EXPERIMENT_A_FULL_81_METHOD_RESULTS.csv',summary)
                rt.atomic_json(A/'01_QA/FULL_GATE.json',dict(status='PASS',tasks=567,elapsed_seconds=time.time()-stage_start),immutable=True)
        rt.atomic_json(A/'EXECUTION_COMPLETE.json',dict(status='COMPLETE_PENDING_FINAL_QA',tasks=57267,elapsed_seconds=time.time()-started),immutable=True)
        import finalize
        finalize.main()
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        rt.atomic_json(A/'STATUS.json',dict(status='STOPPED_REVIEW_REQUIRED',error=str(exc),traceback=traceback.format_exc()));raise
