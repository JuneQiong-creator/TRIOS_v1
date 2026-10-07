from __future__ import annotations
import argparse, ast, csv, gzip, hashlib, importlib.util, json, math, os, pickle, shutil, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import t as tdist
from sklearn.metrics import adjusted_rand_score,balanced_accuracy_score,f1_score

ROOT=Path(r"SOURCE_PROJECT"); DL=Path(r"OMITTED_HOST_PATH/Downloads")
OUT=ROOT/"04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_5/FORMAL_STRUCTURAL_STRESS_v1.0"; WORK=ROOT/"work/trios_phasec_sim_e5_v10"; CK=WORK/"checkpoints"
E1SRC=ROOT/"04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_1/TRIOS_PHASEC_SIM_E1_v1.0/13_SOURCE_CONFIG/run_formal.py"; E1OUT=ROOT/"04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_1/TRIOS_PHASEC_SIM_E1_v1.0"; E1BANK=ROOT/"work/trios_phasec_sim_e1_v10/checkpoints_authoritative"
E4SRC=ROOT/"04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_4_V3/PRACTICAL_DOSE_ROUNDING_AUDIT_v3.3/src/frozen_1_run_e4a.py"
LOCK=ROOT/"04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_5/CALIBRATION_v1.1/04_STRESS_LOCK/E5_STRESS_CALIBRATION_LOCK_v1.1.json"; GENLOCK=E1OUT/"01_CALIBRATION/generator_lock_v1.0.json"
SPEC=DL/"TRIOS_PhaseC_Simulation_Experiment5_Structural_Assumption_Stress_Tests_Frozen_Spec_v1.0.md"; DIRECT=DL/"CODEX_TRIOS_PhaseC_Simulation_Experiment5_Structural_Assumption_Stress_Tests_v1.0_DIRECT_EXECUTE.md"; PHASEA=ROOT.parent/"meeting/meeting_3/TRIOS_PhaseA_Complete_Methodological_Architecture_v3.0.pdf"
EXPECTED_LOCK="001DF989090299F8172A82E879F83E0EEC1A371008DCDFBBCA164121AF8B646E"; EXPECTED_GEN="2C76CCACA01082FC7C72D7F7B0820C567298541715D9D2085F46E87037951974"
DOSES=np.array([0,1,2.5099014421834109,6.2996052494743653,15.811388300841896,39.685026299204978,99.605504741461203,250.]); TCANDS=np.array([30.,50.,100.,150.,200.,250.]); P=np.array([.02,.04,.08,.16,.30,.60,1.]);BETA=np.array([6,16,26,37,51,50,20])/206
PN={12:3,18:12,24:19}; WORKERS=16; SCENARIOS=[("E5A_MILD","A","mild"),("E5A_MODERATE","A","moderate"),("E5A_SEVERE","A","severe"),("E5B_HETEROSCEDASTIC","B","heteroscedastic"),("E5B_T3","B","t3"),("E5B_CONTAMINATION","B","contamination"),("E5C_MILD","C","mild"),("E5C_MODERATE","C","moderate"),("E5C_SEVERE","C","severe"),("E5D_LAMBDA_1P25","D",1.25),("E5D_LAMBDA_1P50","D",1.5),("E5D_LAMBDA_2P00","D",2.0)]
PROHIBITED={k:0 for k in "STRESS_SCENARIO_ADDED_AFTER_RESULTS STRESS_SCENARIO_REMOVED_AFTER_RESULTS STRESS_SEVERITY_CHANGED_AFTER_RESULTS BIOLOGICAL_TRUTH_RESAMPLED TRUTH_LABEL_CHANGED ASSAY_GRID_CHANGED E5A_CONSTANT_CHANGED E5A_LATENT_CLIPPING E5A_LATENT_REPAIR E5B_NOISE_SCALE_TUNED_TO_RESULT E5B_CONTAMINATION_RATE_TUNED_TO_RESULT E5C_GAMMA_CHANGED E5C_TRUTH_RELABELING E5D_LAMBDA_CHANGED SPLINE_DEFINITION_CHANGED GCV_GRID_CHANGED TRANSPORT_RULE_CHANGED TRANSPORT_THRESHOLD_CHANGED TRANSPORT_FAILURE_REPAIRED CRS_K_CHANGED CRS_P_CHANGED CRS_BETA_CHANGED AB_RULE_CHANGED TRUTH_METRIC_CHANGED TRUTH_COMPOSITE_CREATED PRIMARY_ENDPOINT_REDEFINED STRESS_PASS_FAIL_THRESHOLD_ADDED EQUIVALENCE_MARGIN_ADDED NONINFERIORITY_MARGIN_ADDED COMPARATOR_RUN METHOD_RANKING_OPENED PRODUCTION_METHOD_CHANGED FORMAL_PDF_CREATED EXPERIMENT6_STARTED".split()}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest().upper()
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp")
 with gzip.open(q,"wb",compresslevel=3) as f:pickle.dump(x,f,protocol=5)
 q.replace(p)
def load(p):
 with gzip.open(p,"rb") as f:return pickle.load(f)
def log(msg):
 OUT.mkdir(parents=True,exist_ok=True);s=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}";print(s,flush=True)
 with open(OUT/"progress.log","a",encoding="utf-8") as f:f.write(s+"\n")
def weight(m):return 1/3/3/PN[int(m["n"])]/20/5
def bern(key):return int(hashlib.sha256(("TRIOS_E5_B3_CONTAMINATION_V1|"+key).encode()).hexdigest()[:16],16)/2**64<.02

e1=e4=pf=s1=None; CACHE=None; LOCKOBJ=None
def init_worker(cache):
 global e1,e4,pf,s1,CACHE,LOCKOBJ
 e1=mod(E1SRC,"e1formal_worker");e4=mod(E4SRC,"e4helper_worker");pf=e4.pf;s1=e4.s1;CACHE=cache;LOCKOBJ=json.loads(LOCK.read_text())
def stress_data(cp,scenario):
 m=dict(cp["meta"]);b=e1.common_block(m);spec=e1.LOCK["families"][m["family"]];K,lp=e1.latent_builder(m["family"],spec["center"],spec["deltas"][m["separation"]],b);n=int(m["n"]);kind=next(x[1] for x in SCENARIOS if x[0]==scenario);arg=next(x[2] for x in SCENARIOS if x[0]==scenario)
 def pred(i,d):
  d=np.asarray(d,float);base=lp(i,d)
  if kind=="A":
   c=next(x["c_F_severity"] for x in LOCKOBJ["E5A"]["constants"] if x["family"]==m["family"] and x["severity"]==arg);bump=np.exp(-(np.log(np.maximum(d,1e-300)/(2*K[i]))**2)/(2*.35**2));bump=np.where(d==0,0,bump);return base*(1-c*bump)
  if kind=="C":
   g=next(x["gamma"] for x in LOCKOBJ["E5C"]["constants"] if x["severity"]==arg);sg={"L":1,"I":0,"H":-1}[b["g"][i]];new=1/(1+np.exp(-(math.log(b["amp"][i]/(1-b["amp"][i]))+g*sg)));return base*new/b["amp"][i]
  if kind=="D":return lp(i,d/float(arg))
  return base
 mu=np.stack([pred(i,DOSES) for i in range(n)]);z=b["noise"]/.06
 if kind in "ACD":eps=b["noise"]
 elif arg=="heteroscedastic":
  q=.5+mu/b["amp"][:,None];scale=(np.mean(q*q,axis=1))**-.5;eps=.06*scale[:,None,None]*q[:,:,None]*z
 elif arg=="t3":eps=.06/math.sqrt(3)*tdist.ppf(ndtr(z),3)
 else:
  sigma=.06/math.sqrt(.98+.02*25);masks=np.empty_like(z,dtype=bool)
  for i in range(n):
   for j in range(8):
    for r in range(3):masks[i,j,r]=bern(f"{m['n']}|{m['pattern_id']}|{m['separation']}|{m['base_block_id']}|S{i+1}|D{j}|R{r+1}")
  eps=sigma*z*np.where(masks,5,1)
 return m,b,K,pred,mu[:,:,None]+eps,eps
def process_cp(path,scenario):
 with open(path,"rb") as f:cp=pickle.load(f)
 m,b,K,lp,Y,eps=stress_data(cp,scenario);n=int(m["n"]);ids=[x["sample_id"] for x in cp["latent"]];truth=np.array([{"L":0,"I":1,"H":2}[x] for x in b["g"]],np.int8);base={k:m[k] for k in ("cohort_id","n","pattern_id","pattern_index","separation","base_block_id","family","n_L","n_I","n_H")}|{"scenario_id":scenario,"factor_weight":weight(m)}
 fits=[];fitrows=[]
 for i,pid in enumerate(ids):
  f=pf.cached_fit(Y[i].reshape(-1),CACHE);fits.append(f);fitrows.append(base|{"sample_id":pid,"fit_valid":f is not None,"selected_lambda":f["lambda"] if f else np.nan,"GCV":f["GCV"] if f else np.nan,"EDF":f["EDF"] if f else np.nan,"SSE":f["SSE"] if f else np.nan})
 latent=[lambda d,i=i:lp(i,d) for i in range(n)];kind=next(x[1] for x in SCENARIOS if x[0]==scenario);oracle=cp["oracle"] if kind=="B" else e1.transport(latent,n)[0]
 out={"tasks":[],"fits":fitrows,"transport":[],"loo":[],"scores":[],"diagnostics":[]}
 def add(level,D,scores):
  bb=base|{"level":level,"D_oracle":oracle if oracle is not None else np.nan}
  if D is None or scores is None or not np.isfinite(scores).all():out["tasks"].append(bb|{"valid":False,"failure_category":"ENDPOINT_OR_FIT_FAILURE"});return
  z=s1.ab(scores,ids)
  if z is None:out["tasks"].append(bb|{"valid":False,"failure_category":"AB_NO_ADMISSIBLE_PARTITION"});return
  lab,t1,t2,nl,ni,nh=z;ari,ba,mf,ex=s1.truth_metrics(truth,lab);row=bb|{"valid":True,"failure_category":"","D_used":float(D),"tau1":t1,"tau2":t2,"n_L_pred":nl,"n_I_pred":ni,"n_H_pred":nh,"ARI":ari,"BA":ba,"MacroF1":mf,"Extreme":ex}
  out["tasks"].append(row);out["scores"].extend(base|{"level":level,"sample_id":ids[i],"truth_label":int(truth[i]),"CRS":float(scores[i]),"label":int(lab[i])} for i in range(n))
 add("A",oracle,e1.crs(latent,oracle) if oracle is not None else None)
 if any(f is None for f in fits):add("B",None,None);add("C",None,None);return out
 A=np.empty((n,6));T=np.empty((n,6));C=np.empty((n,6));bad=False
 for i,f in enumerate(fits):A[i],T[i],C[i],_,q=s1.derived_arrays(f,CACHE);bad|=q
 if bad:add("B",None,None);add("C",None,None);return out
 add("B",oracle,C[:,np.where(TCANDS==oracle)[0][0]] if oracle is not None else None)
 j,treg=s1.transport_index(A,T);out["transport"]=[base|{"candidate_index":k,"candidate_uM":u,"E_stage":E,"A_e":ae,"T_e":te,"candidate_pass":ok,"stage_evaluable_n":nev,"N_intended":ni} for k,(u,E,ae,te,ok,nev,ni) in enumerate(treg)]
 if j is None:add("C",None,None);return out
 sc=C[:,j];z=s1.ab(sc,ids)
 if z is None:add("C",None,None);return out
 lab,t1,t2,nl,ni,nh=z;sbw=s1.quality(sc,lab);Aloo,stc,st,smc,sm,ml,mr,lrows=s1.loo_metrics(A,T,C,ids,lab,t1,t2,j);ari,ba,mf,ex=s1.truth_metrics(truth,lab);cc=float(np.mean([sbw,st,sm]));out["tasks"].append(base|{"level":"C","valid":True,"failure_category":"","D_oracle":oracle if oracle is not None else np.nan,"D_used":float(TCANDS[j]),"tau1":t1,"tau2":t2,"n_L_pred":nl,"n_I_pred":ni,"n_H_pred":nh,"ARI":ari,"BA":ba,"MacroF1":mf,"Extreme":ex,"S_BW":sbw,"A_LOO":Aloo,"S_tau":st,"S_match":sm,"C_common":cc});out["loo"]=[base|x for x in lrows];out["scores"].extend(base|{"level":"C","sample_id":ids[i],"truth_label":int(truth[i]),"CRS":float(sc[i]),"label":int(lab[i])} for i in range(n))
 out["diagnostics"].append(base|{"oracle_endpoint":oracle if oracle is not None else np.nan,"estimated_endpoint":float(TCANDS[j]),"estimated_no_pass":False,"exact":oracle is not None and float(TCANDS[j])==oracle,"under":oracle is not None and float(TCANDS[j])<oracle,"over":oracle is not None and float(TCANDS[j])>oracle})
 return out
def process_shard(args):
 scenario,sh,paths=args;dest=CK/scenario/f"shard_{sh:02d}.pkl.gz"
 if dest.exists():
  try:
   x=load(dest)
   if x.get("complete") and x.get("scenario")==scenario:return {"scenario":scenario,"shard":sh,"status":"REUSED","cohorts":len(x["cohorts"]),"fits":sum(len(z["fits"]) for z in x["cohorts"])}
  except Exception:pass
 rows=[]
 for p in paths:rows.append(process_cp(str(p),scenario))
 dump(dest,{"complete":True,"scenario":scenario,"shard":sh,"cohorts":rows});return {"scenario":scenario,"shard":sh,"status":"COMPUTED","cohorts":len(rows),"fits":sum(len(z["fits"]) for z in rows)}
def prepare():
 for d in ["00_BINDING","01_STRESS_REGISTRY","02_RAW_GENERATION_PROVENANCE","03_SPLINE_FITS","04_ORACLE_STAGE","05_ESTIMATED_TRANSPORT","06_CRS_AB","07_TRUTH_RECOVERY","08_ERROR_DECOMPOSITION","09_OPERATIONAL","10_STRESS_BASELINE_CONTRASTS","11_ROBUSTNESS","12_BOOTSTRAP","13_FIGURES","14_QA","15_PROVENANCE","src"]:(OUT/d).mkdir(parents=True,exist_ok=True)
 binds=[]
 for role,p,exp in [("STRESS_LOCK",LOCK,EXPECTED_LOCK),("GENERATOR_LOCK",GENLOCK,EXPECTED_GEN),("E1_FINAL_STATUS",E1OUT/"FINAL_STATUS.json",None),("FROZEN_SPEC",SPEC,None),("DIRECT_EXECUTE",DIRECT,None),("PHASE_A_V3",PHASEA,None)]:binds.append({"role":role,"path":str(p),"sha256":sha(p) if p.exists() else "MISSING","expected_sha256":exp or "","status":"PASS" if p.exists() and (not exp or sha(p)==exp) else "FAIL"})
 pd.DataFrame(binds).to_csv(OUT/"00_BINDING/parent_binding_registry.csv",index=False)
 files=sorted(E1BANK.glob("*.pkl"));profiles=sum(len(pickle.load(open(p,"rb"))["latent"]) for p in files)
 if any(x["status"]!="PASS" for x in binds) or len(files)!=10200 or profiles!=212400:raise SystemExit("STOP_E5_PARENT_BINDING_FAIL")
 rows=[]
 for i,(sid,b,arg) in enumerate(SCENARIOS,1):rows.append({"execution_order":i,"scenario_id":sid,"block":b,"severity_or_parameter":arg,"profiles":212400,"expected_spline_fits":212400})
 pd.DataFrame(rows).to_csv(OUT/"01_STRESS_REGISTRY/E5_STRESS_SCENARIO_REGISTRY_v1.0.csv",index=False)
 json.dump({"workers":16,"threads_per_worker":1,"cohorts":10200,"profiles":212400,"scenarios":12,"expected_fits":2548800,"assay_grid":DOSES.tolist()},open(OUT/"00_BINDING/run_config.json","w"),indent=2)
 for p in [SPEC,DIRECT]:shutil.copy2(p,OUT/"src"/p.name)
 log("PREPARE PASS: 10200 cohorts, 212400 profiles, 12 frozen scenarios")
def run():
 prepare(); global e4,pf,s1
 e4=mod(E4SRC,"e4cache_main");pf=e4.pf;s1=e4.s1;cache=e4.assay_cache(DOSES,"E5_ORIGINAL_8DOSE")
 paths=sorted(E1BANK.glob("*.pkl"));shards=[paths[i::64] for i in range(64)]
 for si,(scenario,_,_) in enumerate(SCENARIOS,1):
  done=[];start=time.time();log(f"SCENARIO_START {si}/12 {scenario}")
  with ProcessPoolExecutor(max_workers=WORKERS,initializer=init_worker,initargs=(cache,)) as ex:
   fut=[ex.submit(process_shard,(scenario,i,shards[i])) for i in range(64)]
   for k,f in enumerate(as_completed(fut),1):
    r=f.result();done.append(r)
    if k%4==0:
     fits=sum(x["fits"] for x in done);rate=fits/max(time.time()-start,1);log(f"{scenario} shards={k}/64 fits={fits}/212400 rate={rate:.1f}/s ETA={(212400-fits)/max(rate,1)/60:.1f}m")
  pd.DataFrame(done).sort_values("shard").to_csv(OUT/"15_PROVENANCE"/f"{scenario}_checkpoint_registry.csv",index=False);log(f"SCENARIO_COMPLETE {scenario}")
 log("ALL_SCENARIOS_COMPLETE; run --finalize")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("mode",choices=["prepare","run"]);x=a.parse_args();prepare() if x.mode=="prepare" else run()
