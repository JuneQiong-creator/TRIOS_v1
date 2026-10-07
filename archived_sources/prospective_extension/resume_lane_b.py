"""Scheduling-only recovery: avoid replacing a progress JSON held open by Windows readers.
Scientific adapter, input lock, task keys and completed checkpoints are unchanged.
The first B coordinator exited its pool safely; A continues in the original process.
"""
import sys,os,time,json,msvcrt
sys.dont_write_bytecode=True
import run
from adapter import OUT,sha,js
def update(lane,**kwargs):
    run.STATUS.setdefault(lane,{}).update(kwargs)
    payload=dict(lane=lane,state=run.STATUS[lane],pid=os.getpid(),time=time.time(),resources=run.resource())
    # Append-only progress is observational; it cannot contend with a reader's replace lock.
    with (OUT/'LANE_B/progress_recovery.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(payload,default=str)+'\n')
run.update=update
if __name__=='__main__':
    original=json.loads((OUT/'00_AUTHORITY/EXECUTABLE_SHA.json').read_text())
    for name,h in original.items():assert sha(OUT/'code'/name)==h
    stop=json.loads((OUT/'LANE_B/QA/STOP.json').read_text())
    assert 'PermissionError' in stop['error'] and 'FINAL_STATUS.json' in stop['traceback']
    lock=(OUT/'LANE_B/RECOVERY_LOCK.bin').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    js(OUT/'LANE_B/QA/SCHEDULING_RECOVERY.json',dict(status='RECOVERING',cause='Windows reader held replace target FINAL_STATUS.json; no scientific parity failure',prior_error_preserved=True,algorithm_changed=False,code_sha256=sha(__file__),pid=os.getpid(),time=time.time()))
    run.lane_main('B')
