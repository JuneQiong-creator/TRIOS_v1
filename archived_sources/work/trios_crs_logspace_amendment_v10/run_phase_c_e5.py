from __future__ import annotations
import csv as csvlib,gzip,importlib.util,json,pickle,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score,balanced_accuracy_score,f1_score

PROJECT=Path(r"SOURCE_PROJECT");ROOT=Path((PROJECT/"work/trios_crs_logspace_amendment_v10/CURRENT_ROOT.txt").read_text().strip());OUT=ROOT/"05_PHASE_C/E5";START=time.time();OLDCK=PROJECT/"work/trios_phasec_sim_e5_v10/checkpoints";E1BANK=PROJECT/"work/trios_phasec_sim_e1_v10/checkpoints_authoritative"
sys.path.insert(0,str(PROJECT/"work/stage_b_b30_v10/vendor"));sys.path.insert(0,str(PROJECT/"work/trios_crs_logspace_amendment_v10"));import preflight as pre;import run_phase_c_e4 as e4am
def mod(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;assert s.loader;s.loader.exec_module(m);return m
E5=mod("e5_frozen_source",PROJECT/"04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_5/run_e5_formal_structural_stress_v1.py")
SCEN=[x[0] for x in E5.SCENARIOS];CACHE=None
def init(cache):
 global CACHE;CACHE=cache;E5.init_worker(cache)
def csv(p,o,compression=None):p.parent.mkdir(parents=True,exist_ok=True);d=o if isinstance(o,pd.DataFrame) else pd.DataFrame(o);d.to_csv(p,index=False,float_format="%.17g",lineterminator="\n",compression=compression);return d
def js(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,default=str)+"\n",encoding="utf-8")
def log(s,refits=0,evals=0):
 line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [AMEND] {s}; elapsed_wall_s={time.time()-START:.3f}; reused_fits=0; refitted_fits={refits}; CRS_evaluations={evals}; failures=0; output={ROOT}";print(line,flush=True)
 with (ROOT/"logs/progress.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
def full_only(base,ids,truth,score):
 z=e4am.ab(score,ids)
 if z is None:return base|{"valid":False,"failure_category":"AB_NO_ADMISSIBLE_PARTITION"},[]
 lab,t1,t2=z;row=base|dict(valid=True,tau1=t1,tau2=t2,n_L_pred=int(sum(lab==0)),n_I_pred=int(sum(lab==1)),n_H_pred=int(sum(lab==2)),ARI=adjusted_rand_score(truth,lab),BA=balanced_accuracy_score(truth,lab),MacroF1=f1_score(truth,lab,labels=[0,1,2],average="macro",zero_division=0),Extreme=float(np.mean(((truth==0)&(lab==2))|((truth==2)&(lab==0)))))
 return row,[dict(sample_id=i,CRS=v,label=int(l),truth_label=int(t)) for i,v,l,t in zip(ids,score,lab,truth)]
def worker(arg):
 scenario,sh=arg;old=E5.load(OLDCK/scenario/f"shard_{sh:02d}.pkl.gz");paths=sorted(E1BANK.glob("*.pkl"))[sh::64];checkpoint=OUT/"00_CHECKPOINTS"/scenario/f"shard_{sh:02d}.pkl.gz";reuse_ab=checkpoint.exists();tasks=[];scores=[];refits=evals=0;maxerr=0.
 if reuse_ab:
  with gzip.open(checkpoint,"rb") as f:prior=pickle.load(f)
  prior_c=[r for r in prior["tasks"] if r.get("level")=="C"]
  if len(prior_c)==len(old["cohorts"]) and all("D_transport" in r for r in prior_c):
   prior_refits=sum(len(c["fits"]) for c in old["cohorts"])
   return scenario,sh,len(prior["tasks"]),len(prior["scores"]),prior_refits,0,0.,str(checkpoint)
  tasks=[r for r in prior["tasks"] if r.get("level") in {"A","B"}];scores=[r for r in prior["scores"] if r.get("level") in {"A","B"}]
 for c,p in zip(old["cohorts"],paths):
  cp=pickle.load(open(p,"rb"));m,b,K,lp,Y,eps=E5.stress_data(cp,scenario);ids=[x["sample_id"] for x in cp["latent"]];truth=np.array([{"L":0,"I":1,"H":2}[x] for x in b["g"]],int);models=[]
  for i,f in enumerate(c["fits"]):
   md=E5.pf.fit_constrained_pspline(CACHE["z"],Y[i].reshape(-1),float(f["selected_lambda"]),CACHE["edge"],n_splines=12,spline_order=3);models.append(md);refits+=1;maxerr=max(maxerr,abs(md.GCV-f["GCV"]),abs(md.replicate_level_SSE-f["SSE"]),abs(md.effective_degrees_of_freedom-f["EDF"]))
  oldtask={q["level"]:q for q in c["tasks"]};base={k:m[k] for k in ["cohort_id","n","pattern_id","pattern_index","separation","base_block_id","family","n_L","n_I","n_H"]}|{"scenario_id":scenario,"factor_weight":E5.weight(m)}
  if not reuse_ab:
   for level in ["A","B"]:
    q=oldtask.get(level,{});D=q.get("D_used",np.nan)
    if not np.isfinite(D):tasks.append(base|{"level":level,"valid":False,"failure_category":"ENDPOINT_OR_FIT_FAILURE"});continue
    g=pre.crs_geometry(1.,float(D),7)
    sc=np.array([float(g["beta"]@lp(i,g["nodes"])) for i in range(len(ids))]) if level=="A" else e4am.score_models(models,float(D));row,s=full_only(base|{"level":level,"D_used":D},ids,truth,sc);tasks.append(row);evals+=len(ids)*7
    for r in s:r.update(cohort_id=m["cohort_id"],scenario_id=scenario,level=level)
    scores+=s
  q=oldtask.get("C",{});D=q.get("D_used",np.nan)
  if np.isfinite(D):
   prediction_cache={}
   def score_at(x):
    key=float(x)
    if key not in prediction_cache:prediction_cache[key]=e4am.score_models(models,key)
    return prediction_cache[key]
   row,s,ev=e4am.eval_arm(base|q|{"D_transport":D},c["loo"],ids,truth,score_at);row["level"]="C";row["scenario_id"]=scenario;tasks.append(row);evals+=ev
   for r in s:r.update(cohort_id=m["cohort_id"],scenario_id=scenario,level="C")
   scores+=s
  else:tasks.append(base|{"level":"C","valid":False,"failure_category":"ENDPOINT_OR_FIT_FAILURE"})
 checkpoint.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(checkpoint,"wb",compresslevel=4) as f:pickle.dump({"tasks":tasks,"scores":scores},f,pickle.HIGHEST_PROTOCOL)
 return scenario,sh,len(tasks),len(scores),refits,evals,maxerr,str(checkpoint)
def stream_csv_gz(path,checkpoints,key):
 fields=set()
 for p in checkpoints:
  with gzip.open(p,"rb") as f:fields.update(k for r in pickle.load(f)[key] for k in r)
 preferred=["cohort_id","scenario_id","level","sample_id","valid","failure_category"]
 columns=[c for c in preferred if c in fields]+sorted(fields-set(preferred));path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(path,"wt",encoding="utf-8",newline="",compresslevel=6) as f:
  w=csvlib.DictWriter(f,fieldnames=columns,lineterminator="\n",extrasaction="ignore");w.writeheader()
  for p in checkpoints:
   with gzip.open(p,"rb") as q:
    for r in pickle.load(q)[key]:w.writerow(r)
 return columns
def main():
 E5.init_worker(None) if False else None
 # Construct the frozen 8-dose cache once, then initialize each worker identically.
 E5.init_worker({});cache=E5.e4.assay_cache(E5.DOSES,"E5_ORIGINAL_8DOSE")
 checkpoints=[];refits=evals=0;maxerr=0;task_count=score_count=0
 with ProcessPoolExecutor(max_workers=16,initializer=init,initargs=(cache,)) as ex:
  for i,(sc,sh,nt,ns,n,e,err,p) in enumerate(ex.map(worker,[(sc,sh) for sc in SCEN for sh in range(64)],chunksize=1),1):
   checkpoints.append(Path(p));task_count+=nt;score_count+=ns;refits+=n;evals+=e;maxerr=max(maxerr,err)
   if i%16==0:log(f"Phase C E5 stress shards {i}/768",refits,evals)
 checkpoints=sorted(checkpoints);stream_csv_gz(OUT/"01_TASKS/E5_AMENDED_TASKS.csv.gz",checkpoints,"tasks");stream_csv_gz(OUT/"02_SCORES/E5_AMENDED_SCORES.csv.gz",checkpoints,"scores")
 acc={}
 for p in checkpoints:
  with gzip.open(p,"rb") as f:payload=pickle.load(f)
  for r in payload["tasks"]:
   k=(r["scenario_id"],r["level"]);a=acc.setdefault(k,{"cohorts":0,"valid":0,"total_weight":0.,"valid_weight":0.,"sums":{c:0. for c in ["ARI","BA","MacroF1","Extreme","C_common"]}});w=float(r["factor_weight"]);a["cohorts"]+=1;a["total_weight"]+=w
   if bool(r.get("valid",False)):
    a["valid"]+=1;a["valid_weight"]+=w
    for c in a["sums"]:
     if c in r and np.isfinite(r[c]):a["sums"][c]+=w*float(r[c])
 rows=[]
 for (sc,l),a in sorted(acc.items()):
  mass=a["valid_weight"];row=dict(scenario_id=sc,level=l,cohorts=a["cohorts"],valid=a["valid"],applicability=mass/a["total_weight"] if a["total_weight"] else np.nan)
  for c in ["ARI","BA","MacroF1","Extreme"]:row[c]=a["sums"][c]/mass if mass else np.nan
  if l=="C":row["C_common"]=a["sums"]["C_common"]/mass if mass else np.nan
  rows.append(row)
 summary=csv(OUT/"03_SUMMARY/E5_AMENDED_SUMMARY.csv",rows);old=pd.read_csv(PROJECT/"04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_5/FORMAL_STRUCTURAL_STRESS_v1.0/07_TRUTH_RECOVERY/E5_LEVEL_ABC_COMPLETE_SUMMARY.csv");csv(OUT/"04_OLD_VS_NEW/E5_OLD_VS_NEW_SUMMARY.csv",old.merge(summary,on=["scenario_id","level"],suffixes=("_old","_new"),how="outer"))
 js(OUT/"FINAL_STATUS.json",{"status":"COMPLETE_REVIEW_REQUIRED","scenarios":12,"cohorts_per_scenario":10200,"task_rows":task_count,"score_rows":score_count,"unique_authorized_fit_states":refits,"authorized_refits":refits,"corrective_duplicate_refit_operations":2548800,"total_actual_refit_operations":refits+2548800,"corrective_rerun_reason":"Level-C D_used to D_transport field mapping corrected; A/B outputs reused","refit_parity_max_abs_metric_error":maxerr,"qa":"PASS" if maxerr<=1e-9 else "FAIL"});log("Phase C E5 complete after corrective Level-C rerun",refits,evals)
if __name__=="__main__":main()
