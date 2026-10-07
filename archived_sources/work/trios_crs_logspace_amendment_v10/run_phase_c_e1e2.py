from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score


PROJECT=Path(r"SOURCE_PROJECT")
ROOT=Path((PROJECT/"work/trios_crs_logspace_amendment_v10/CURRENT_ROOT.txt").read_text().strip())
PC=PROJECT/"04_RESULTS/PHASE_C_SIMULATION"
E1OLD=PC/"EXPERIMENT_1/TRIOS_PHASEC_SIM_E1_v1.0"
STATEDIR=PC/"EXPERIMENT_1/TRIOS_PHASEC_SIM_E1_v1.0_ARCHIVE_v1.1_FITTED_CURVE_RECOVERED_FROZEN/05_SPLINE_FITS/recovered_curve_representation"
CHECK=PROJECT/"work/trios_phasec_sim_e1_v10/checkpoints_authoritative"
OUT1=ROOT/"05_PHASE_C/E1";OUT2=ROOT/"05_PHASE_C/E2_LOGSPACE_K"
START=time.time();FIXED=[50.,100.,150.,200.,250.];CANDS=[30.,50.,100.,150.,200.,250.]


def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(m);return m


pre=mod("pre_c12",PROJECT/"work/trios_crs_logspace_amendment_v10/preflight.py")
sim=mod("e1_frozen_impl",PROJECT/"work/trios_phasec_sim_e1_v10/run_formal.py")
solver=sys.modules["constrained_pspline_solver"]


def csv(path,obj,compression=None):
    path.parent.mkdir(parents=True,exist_ok=True);df=obj if isinstance(obj,pd.DataFrame) else pd.DataFrame(obj)
    df.to_csv(path,index=False,lineterminator="\n",float_format="%.17g",compression=compression);return df
def js(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
def log(stage,reused=212400,evals=0,failures=0):
    line=(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [AMEND] {stage}; elapsed_wall_s={time.time()-START:.3f}; reused_fits={reused}; "
          f"refitted_fits=0; CRS_evaluations={evals}; failures={failures}; output={ROOT}")
    print(line,flush=True)
    with (ROOT/"logs/progress.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
def recover(truth,pred):
    return dict(ARI=adjusted_rand_score(truth,pred),balanced_accuracy=balanced_accuracy_score(truth,pred),
                macro_f1=f1_score(truth,pred,labels=list("LIH"),average="macro",zero_division=0),
                extreme_error=float(np.mean(((truth=="L")&(pred=="H"))|((truth=="H")&(pred=="L")))))
def fit_scores(intercepts,coefs,nodes,beta):
    B=solver._basis(np.log1p(nodes),[0.,np.log1p(250.)],12,3)
    return intercepts+(coefs@B.T)@beta


def fitted_wide(endpoints):
    outdir=OUT1/"01_PROFILE_SCORES/FITTED_PARTS";outdir.mkdir(parents=True,exist_ok=True)
    allparts=[];evals=0
    for ix,p in enumerate(sorted(STATEDIR.glob("curve_state_part_*.csv.gz"))):
        d=pd.read_csv(p,usecols=["sample_id","cohort_id","intercept","coefficients"])
        co=np.asarray([json.loads(x) for x in d.coefficients],float);it=d.intercept.to_numpy(float);z=d[["cohort_id","sample_id"]].copy()
        for D in FIXED:
            g=pre.crs_geometry(1.,D,7);z[f"K7_D{int(D)}"]=fit_scores(it,co,g["nodes"],g["beta"]);evals+=len(d)*7
        for K in range(3,7):
            g=pre.crs_geometry(1.,50.,K);z[f"K{K}_D50"]=fit_scores(it,co,g["nodes"],g["beta"]);evals+=len(d)*K
        # Endpoint-specific score from the unchanged oracle/estimated transport states.
        for kind in ["oracle","estimated"]:
            vals=np.full(len(d),np.nan)
            ep=d.cohort_id.map(endpoints[kind])
            for D in CANDS:
                mask=ep.eq(D).to_numpy()
                if mask.any():
                    g=pre.crs_geometry(1.,D,7);vals[mask]=fit_scores(it[mask],co[mask],g["nodes"],g["beta"]);evals+=int(mask.sum())*7
            z[f"K7_{kind}"]=vals
        op=outdir/f"fitted_scores_part_{ix:03d}.csv.gz";csv(op,z,compression="gzip");allparts.append(z)
        if (ix+1)%10==0 or ix+1==len(list(STATEDIR.glob('curve_state_part_*.csv.gz'))):log(f"Phase C E1 fitted-state evaluation {ix+1}/51",evals=evals)
    return pd.concat(allparts,ignore_index=True),evals


def latent_wide(endpoints):
    rows=[];evals=0;paths=sorted(CHECK.glob("*.pkl"))
    for ix,p in enumerate(paths,1):
        with p.open("rb") as f:o=pickle.load(f)
        meta=o["meta"];b=sim.common_block(meta);spec=sim.LOCK["families"][meta["family"]];Kpar,lp=sim.latent_builder(meta["family"],spec["center"],spec["deltas"][meta["separation"]],b);n=int(meta["n"]);ids=[x["sample_id"] for x in o["latent"]]
        z={"cohort_id":[meta["cohort_id"]]*n,"sample_id":ids,"truth":b["g"].astype(str)}
        for D in FIXED:
            g=pre.crs_geometry(1.,D,7);z[f"K7_D{int(D)}"]=[float(g["beta"]@lp(i,g["nodes"])) for i in range(n)];evals+=n*7
        for K in range(3,7):
            g=pre.crs_geometry(1.,50.,K);z[f"K{K}_D50"]=[float(g["beta"]@lp(i,g["nodes"])) for i in range(n)];evals+=n*K
        for kind in ["oracle","estimated"]:
            D=endpoints[kind].get(meta["cohort_id"],np.nan)
            if np.isfinite(D):
                g=pre.crs_geometry(1.,float(D),7);z[f"K7_{kind}"]=[float(g["beta"]@lp(i,g["nodes"])) for i in range(n)];evals+=n*7
            else:z[f"K7_{kind}"]=[np.nan]*n
        rows.append(pd.DataFrame(z))
        if ix%1000==0 or ix==len(paths):log(f"Phase C E1 latent evaluation {ix}/{len(paths)} cohorts",evals=evals)
    out=pd.concat(rows,ignore_index=True);csv(OUT1/"01_PROFILE_SCORES/latent_scores.csv.gz",out,compression="gzip");return out,evals


def scenario_metrics(fitted,latent):
    merged=fitted.merge(latent,on=["cohort_id","sample_id"],suffixes=("_fitted","_latent"),validate="one_to_one")
    e1=[];e2=[];labels=[]
    for count,(cid,g) in enumerate(merged.groupby("cohort_id",sort=True),1):
        ids=g.sample_id.astype(str).tolist();truth=g.truth.to_numpy(str)
        for D in FIXED:
            for layer,suf in [("FITTED","_fitted"),("LATENT","_latent")]:
                col=f"K7_D{int(D)}{suf}";score=g[col].to_numpy(float);a=sim.ab(score,ids)
                if a is None:e1.append(dict(cohort_id=cid,scenario=f"FIXED_CRS_D{int(D)}_{layer}",layer=layer,D=D,valid=False));continue
                pred,t1,t2=a;e1.append(dict(cohort_id=cid,scenario=f"FIXED_CRS_D{int(D)}_{layer}",layer=layer,D=D,valid=True,tau1=t1,tau2=t2,**recover(truth,pred)))
        for scenario,layer,col in [("E1C_A_LATENT_ORACLE","LATENT","K7_oracle_latent"),("E1C_B_FITTED_ORACLE","FITTED","K7_oracle_fitted"),("E1C_C_COMPLETE_ESTIMATED","FITTED","K7_estimated_fitted")]:
            score=g[col].to_numpy(float)
            if not np.isfinite(score).all() or (a:=sim.ab(score,ids)) is None:e1.append(dict(cohort_id=cid,scenario=scenario,layer=layer,D=np.nan,valid=False));continue
            pred,t1,t2=a;e1.append(dict(cohort_id=cid,scenario=scenario,layer=layer,D=np.nan,valid=True,tau1=t1,tau2=t2,**recover(truth,pred)))
        for layer,suf in [("FITTED","_fitted"),("LATENT","_latent")]:
            refscore=g[f"K7_D50{suf}"].to_numpy(float);ref=sim.ab(refscore,ids)
            if ref is None:continue
            reflab=ref[0]
            for K in range(3,8):
                col=f"K{K}_D50{suf}" if K<7 else f"K7_D50{suf}";score=g[col].to_numpy(float);a=sim.ab(score,ids)
                if a is None:e2.append(dict(cohort_id=cid,layer=layer,K=K,valid=False));continue
                pred,t1,t2=a
                e2.append(dict(cohort_id=cid,layer=layer,K=K,valid=True,score_spearman=pd.Series(score).corr(pd.Series(refscore),method="spearman"),
                               class_balanced_label_fidelity=float(np.mean([np.mean(pred[reflab==v]==v) for v in "LIH"])),
                               K7_label_agreement=float(np.mean(pred==reflab)),tau1=t1,tau2=t2,**recover(truth,pred)))
        if count%1000==0 or count==10200:log(f"Phase C E1/E2 AB and truth recovery {count}/10200 cohorts")
    return pd.DataFrame(e1),pd.DataFrame(e2)


def bootstrap(df,groups,metrics,path,seed):
    rng=np.random.default_rng(seed);rows=[]
    for key,g in df.groupby(groups,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for metric in metrics:
            v=g.loc[g.valid.astype(bool),metric].dropna().to_numpy(float)
            if not len(v):continue
            reps=[]
            for _ in range(20):
                ix=rng.integers(0,len(v),size=(100,len(v)));reps.extend(v[ix].mean(axis=1))
            rows.append({**dict(zip(groups,key)),"metric":metric,"estimate":float(v.mean()),"bootstrap_B":2000,"CI95_low":float(np.quantile(reps,.025)),"CI95_high":float(np.quantile(reps,.975)),"seed":seed})
    csv(path,rows)


def main():
    tr=pd.read_csv(E1OLD/"08_TRUTH_RECOVERY/transport_recovery.csv",float_precision="round_trip")
    endpoints={"oracle":tr.set_index("cohort_id").D_oracle.replace("ORACLE_NO_PASS",np.nan).astype(float).to_dict(),
               "estimated":tr.set_index("cohort_id").D_estimated.replace("NO_PASS",np.nan).astype(float).to_dict()}
    if len(endpoints["oracle"])!=10200:raise SystemExit("STOP_TRUTH_BANK_IDENTITY_DRIFT")
    fitted,fe=fitted_wide(endpoints);latent,le=latent_wide(endpoints)
    if len(fitted)!=212400 or len(latent)!=212400:raise SystemExit("STOP_FIT_STATE_PROVENANCE_FAIL")
    e1,e2=scenario_metrics(fitted,latent)
    csv(OUT1/"02_TRUTH_RECOVERY/E1_CRS_SCENARIO_METRICS.csv.gz",e1,compression="gzip")
    e1sum=e1.groupby(["scenario","layer"],as_index=False).agg(cohorts=("cohort_id","size"),valid=("valid","sum"),ARI=("ARI","mean"),balanced_accuracy=("balanced_accuracy","mean"),macro_f1=("macro_f1","mean"),extreme_error=("extreme_error","mean"));e1sum["applicability"]=e1sum.valid/e1sum.cohorts;csv(OUT1/"03_SUMMARY/E1_AMENDED_SUMMARY.csv",e1sum)
    bootstrap(e1,["scenario","layer"],["ARI","balanced_accuracy","macro_f1","extreme_error"],OUT1/"04_BOOTSTRAP/E1_BOOTSTRAP.csv",2026091201)
    old=pd.read_csv(E1OLD/"08_TRUTH_RECOVERY/scenario_recovery_metrics.csv",float_precision="round_trip");old=old[old.scenario.isin(e1.scenario.unique())]
    cmp=old.merge(e1,on=["cohort_id","scenario"],suffixes=("_old","_new"));
    for m in ["ARI","balanced_accuracy","macro_f1","extreme_error"]:cmp["delta_"+m]=cmp[m+"_new"]-cmp[m+"_old"]
    csv(OUT1/"05_OLD_VS_NEW/E1_OLD_VS_NEW.csv.gz",cmp,compression="gzip")
    csv(OUT2/"02_TRUTH_RECOVERY/E2_LOGSPACE_K_METRICS.csv.gz",e2,compression="gzip")
    e2sum=e2.groupby(["layer","K"],as_index=False).agg(cohorts=("cohort_id","size"),valid=("valid","sum"),score_fidelity=("score_spearman","mean"),class_balanced_label_fidelity=("class_balanced_label_fidelity","mean"),ARI=("ARI","mean"),balanced_accuracy=("balanced_accuracy","mean"),macro_f1=("macro_f1","mean"),extreme_error=("extreme_error","mean"));e2sum["applicability"]=e2sum.valid/e2sum.cohorts
    geoms=[dict(K=K,nodes_json=json.dumps(pre.crs_geometry(1.,50.,K)["nodes"].tolist()),beta_json=json.dumps(pre.crs_geometry(1.,50.,K)["beta"].tolist()),G_d=pre.crs_geometry(1.,50.,K)["G_d"],G_norm=pre.crs_geometry(1.,50.,K)["G_norm"]) for K in range(3,8)]
    csv(OUT2/"01_GEOMETRY/E2_LOGSPACE_K_GEOMETRY.csv",geoms);csv(OUT2/"03_SUMMARY/E2_LOGSPACE_K_SUMMARY.csv",e2sum);bootstrap(e2,["layer","K"],["score_spearman","class_balanced_label_fidelity","ARI","balanced_accuracy","macro_f1","extreme_error"],OUT2/"04_BOOTSTRAP/E2_BOOTSTRAP.csv",2026091202)
    legacy=PC/"EXPERIMENT_2/TRIOS_PHASEC_SIM_E2_v1.1_PARENT_AMENDED_complete_delivery.zip";js(ROOT/"05_PHASE_C/LEGACY_E2_57_GEOMETRY_REFERENCE/reference.json",{"path":str(legacy),"sha256":hashlib.sha256(legacy.read_bytes()).hexdigest().upper(),"status":"SUPERSEDED_PRIMARY_REFERENCE_ONLY"})
    js(OUT1/"FINAL_STATUS.json",{"status":"COMPLETE_REVIEW_REQUIRED","cohorts":10200,"profiles":212400,"truth_bank_reused":True,"fit_states_reused":212400,"refits":0,"qa":"PASS"})
    js(OUT2/"FINAL_STATUS.json",{"status":"COMPLETE_REVIEW_REQUIRED","cohorts":10200,"K_values":[3,4,5,6,7],"production_K":7,"refits":0,"qa":"PASS"})
    log("Phase C E1 complete",evals=fe+le);log("Phase C E2 complete",evals=fe+le)


if __name__=="__main__":main()
