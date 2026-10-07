"""Windows-only scheduling/atomic I/O. No scientific evaluation definitions."""
import sys, os, json, time, uuid, ctypes, subprocess, threading, msvcrt, hashlib
from pathlib import Path
from functools import lru_cache
from ctypes import wintypes

O = Path(__file__).resolve().parent
RUNTIME = O/'07_RUNTIME/unattended'
RUN_ID = os.environ.get('OPTION_C_RUN_ID', 'UNASSIGNED')

class IntegrityError(RuntimeError): pass
class LeaseBusy(RuntimeError): pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def birth(pid):
    k=ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    h=k.OpenProcess(0x1000,False,int(pid))
    if not h:
        if ctypes.get_last_error() in (0,87): return None
        raise OSError(ctypes.get_last_error(),'Cannot verify process identity')
    try:
        code=wintypes.DWORD()
        if not k.GetExitCodeProcess(h,ctypes.byref(code)): raise OSError('Cannot inspect process state')
        if code.value!=259:return None
        fields=[wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(h,*[ctypes.byref(x) for x in fields]):raise OSError('Cannot inspect process creation time')
        return (fields[0].dwHighDateTime<<32)|fields[0].dwLowDateTime
    finally:k.CloseHandle(h)

def same_process(identity):
    return identity.get('creation_filetime') is not None and birth(identity['pid']) == identity['creation_filetime']

@lru_cache(maxsize=1)
def own_identity():
    k=ctypes.WinDLL('kernel32');k.GetCommandLineW.restype=wintypes.LPWSTR
    return dict(pid=os.getpid(),parent_pid=os.getppid(),creation_filetime=birth(os.getpid()),command_line=k.GetCommandLineW(),run_id=RUN_ID)

def process_inventory():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],encoding='utf-8',errors='replace').strip()
    rows=json.loads(raw) if raw else []
    if isinstance(rows,dict):rows=[rows]
    result=[]
    for r in rows:
        t=birth(r['ProcessId'])
        if t is not None:result.append(dict(pid=int(r['ProcessId']),parent_pid=int(r['ParentProcessId']),creation_filetime=t,command_line=r['CommandLine']))
    return result

def atomic_bytes(path,data,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes()!=data:raise IntegrityError('CONTRADICTORY_CHECKPOINT:'+str(path))
        return
    tmp=path.parent/('tmp_'+uuid.uuid4().hex[:16])
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    for attempt in range(20):
        try:
            if immutable:
                try:os.link(tmp,path)
                except FileExistsError:
                    if path.read_bytes()!=data:raise IntegrityError('CONTRADICTORY_CHECKPOINT:'+str(path))
                # Only this newly-created temporary file is removed; never a checkpoint.
                tmp.unlink()
            else:os.replace(tmp,path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.05*(attempt+1),.5))

def atomic_json(path,value,immutable=False):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'),immutable)

def archive_file(path,reason):
    path=Path(path).resolve()
    if not path.is_relative_to(O):raise IntegrityError('ARCHIVE_OUTSIDE_EXECUTION')
    target=RUNTIME/'preserved'/(str(time.time_ns())+'_'+uuid.uuid4().hex[:8]+'.record')
    target.parent.mkdir(parents=True,exist_ok=True);before=sha(path);path.rename(target)
    if sha(target)!=before:raise IntegrityError('ARCHIVED_BYTES_CHANGED')
    atomic_json(target.with_name(target.name+'.json'),dict(original=str(path),sha256=before,reason=reason,run_id=RUN_ID,archived_at=time.time()))

class Lease:
    """OS-held byte lock plus identity metadata; TTL alone never permits takeover."""
    def __init__(self,path,key,heartbeat=15,ttl=120):
        self.path=Path(path);self.key=key;self.interval=heartbeat;self.ttl=ttl;self.stop=threading.Event();self.error=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('a+b');self.handle.seek(0)
        try:msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:self.handle.close();raise LeaseBusy('TASK_ALREADY_OS_LOCKED:'+self.key)
        if self.path.stat().st_size==0:self.handle.write(b'0');self.handle.flush()
        self.owner=self.path.with_suffix(self.path.suffix+'.owner.json')
        try:
            if self.owner.exists():
                old=read(self.owner)
                if old['key']!=self.key:raise IntegrityError('LOCK_KEY_COLLISION')
                if old.get('status')!='RELEASED' and same_process(old['identity']):
                    raise LeaseBusy('LIVE_OWNER_METADATA_WITHOUT_LOCK:'+self.key)
                archive_file(self.owner,'released_or_exited_owner')
            self.beat();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
            return self
        except BaseException:self.handle.close();raise
    def beat(self,status='ACTIVE'):
        atomic_json(self.owner,dict(key=self.key,identity=own_identity(),heartbeat_epoch=time.time(),lease_expires_epoch=time.time()+self.ttl,status=status))
    def loop(self):
        while not self.stop.wait(self.interval):
            try:self.beat()
            except BaseException as exc:self.error=exc;return
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.beat('RELEASED');self.handle.close()
        if self.error and args[0] is None:raise self.error

def checkpoint_key(q):
    if 'heldout_profile' in q:return (q.get('dataset_id'),q.get('scope'),q.get('task_id'),q.get('method_id'),q['heldout_profile'])
    if 'profile_id' in q and 'D' in q:return (q['profile_id'],q['D'])
    raise IntegrityError('UNKNOWN_APPEND_CHECKPOINT_SCHEMA')

def validated_lines(path):
    path=Path(path)
    if not path.exists():return []
    raw=path.read_text(encoding='utf-8');lines=[json.loads(x) for x in raw.splitlines()]
    keys=[checkpoint_key(q) for q in lines]
    if len(keys)!=len(set(keys)):raise IntegrityError('DUPLICATE_CHECKPOINT_KEYS:'+str(path))
    return lines

def append_json(path,value):
    path=Path(path);rows=validated_lines(path);key=checkpoint_key(value)
    for old in rows:
        if checkpoint_key(old)==key:
            if old!=value:raise IntegrityError('CONTRADICTORY_FOLD_CHECKPOINT:'+str(path))
            return
    original=path.read_bytes() if path.exists() else b''
    if original and not original.endswith(b'\n'):raise IntegrityError('CHECKPOINT_LINE_NOT_TERMINATED:'+str(path))
    newline=b'\r\n' if b'\r\n' in original else b'\n'
    addition=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')+newline
    atomic_bytes(path,original+addition)

def install_io(e):
    e.js=lambda p,v:atomic_json(p,e.clean(v),immutable=str(p).endswith('.task.json'))
    e.append=lambda p,v:append_json(p,e.clean(v))

def resource_memory():
    class Mem(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(k,ctypes.c_ulonglong) for k in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
    m=Mem();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('Memory status failed')
    return dict(load_percent=m.load,available_GB=m.available/2**30)
