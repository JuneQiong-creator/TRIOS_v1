"""One existing HB regimen: frozen fit states -> stage -> exact-log CRS -> historical AB.
No refitting, LOO, new observations, resampling or scientific result overwrite.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trios_release import hb_stage
from trios_release.hb_frozen import spline_from_parameters,extension
from trios_release.geometry import crs_geometry
from trios_release.ab_hb import fit_ms

def run():
    raw=pd.read_csv(ROOT/'data/hb/hb_pdo_drug_response.csv')
    membership=pd.read_csv(ROOT/'data/hb/source_to_eligible_membership.csv')
    params=pd.read_csv(ROOT/'data/hb/frozen_fit_parameters.csv')
    status=pd.read_csv(ROOT/'data/hb/frozen_fit_status.csv')
    reference=pd.read_csv(ROOT/'figures/Figure2_3/data/figure3_scores_labels.csv',float_precision='round_trip')
    reg='5FU'
    selected=membership[membership.regimen.eq(reg)&membership.included_in_QA_clean_source.astype(str).str.lower().isin(['true','1','1.0'])]
    ids=np.array(sorted(selected.pdo_id.astype(str)),str)
    hb_stage.LOCAL=pd.read_csv(ROOT/'results/hb/profile_candidate_quantities.csv',float_precision='round_trip')
    hb_stage.CSETS={reg:sorted(hb_stage.LOCAL.loc[hb_stage.LOCAL.regimen.eq(reg),'candidate_uM'].unique())}
    D,trace=hb_stage.stage(reg,ids)
    assert D==50.,'Frozen stage endpoint did not replay'
    geom=crs_geometry(1.,D,7)
    scores=[]
    for pid in ids:
        obj=spline_from_parameters(reg,pid,params,status,raw)
        dm=float(raw.loc[raw.regimen.eq(reg)&raw.PDO_ID.eq(pid),'dose_uM'].max())
        f,_,_,material=extension(obj,dm)
        assert not material,'Unexpected frozen continuation diagnostic'
        scores.append(float(geom['beta']@f(geom['nodes'])))
    scores=np.array(scores)
    output=fit_ms(scores,ids)
    frozen=reference[reference.regimen.eq(reg)].set_index('profile_id').loc[ids]
    assert output['valid'] and np.array_equal(output['labels'],frozen.label.to_numpy())
    difference=float(np.max(np.abs(scores-frozen.score.to_numpy())))
    assert difference<=1e-12,('Cross-platform diagnostic score discrepancy',difference)
    assert abs(output['tau1']-float(frozen.tau1.iloc[0]))<=1e-12
    assert abs(output['tau2']-float(frozen.tau2.iloc[0]))<=1e-12
    return dict(status='PASS',regimen=reg,profiles=len(ids),selected_domain_uM=D,K=7,
                score_max_absolute_difference=difference,partition_labels_match=True,
                tau1=output['tau1'],tau2=output['tau2'],
                stage_input='Frozen profile-level candidate quantities',
                fit_input='Original frozen spline coefficients/status; no refit',
                ab_version='Historical HB binary64 SSE tuple, source hash 7eb37341623b10b4...',
                scope='One existing full-data regimen path; no benchmark, LOO, fitting or complete raw-data recomputation')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
