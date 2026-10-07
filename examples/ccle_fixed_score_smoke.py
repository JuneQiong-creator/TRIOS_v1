"""Synthetic n=12 fixture only. Does not execute a stored CCLE cohort or benchmark."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trios_release.fixed_score import evaluate_fixed_scores
from trios_release.evaluation import task_components
from trios_release.ab_hb import fit_ms as historical_ab
from trios_release.ab_ccle import fit_ms as ccle_ab
from trios_release.geometry import crs_geometry

def run():
    x=np.arange(12,dtype=float);ids=np.array([f'SYNTH-{i:02d}' for i in range(12)])
    labels=np.array(['L']*4+['I']*4+['H']*4)
    result,folds=evaluate_fixed_scores(x,ids,labels,3.5,7.5,.5,'AB','PUBLIC_SMOKE_ONLY')
    assert result['V']==1 and len(folds)==12 and all(f['success'] for f in folds)
    assert result['P_fit']==.5 and result['I_SF']==1 and 0<=result['S_match']<=1
    expected_folds=[dict(attempted=True,success=True,d_tau_squared=0.,omitted_match=1.,retained_match_fraction=1.),
                    dict(attempted=True,success=True,d_tau_squared=1.,omitted_match=0.,retained_match_fraction=.5),
                    dict(attempted=True,success=False)]
    z=task_components([0,1,2],['L','I','H'],.5,1.5,.5,expected_folds)
    assert abs(z['S_match']-.4)<1e-14
    assert abs(z['S_tau']-(2/3)/(1+math.sqrt(.5)))<1e-14
    invalid=task_components([],[],None,None,None,[],False)
    assert invalid['C_conditional'] is None and invalid['C_operational']==0
    fixture=np.array([0,0,0,0,1,2,2,2,2,4,4,4],float)
    a=historical_ab(fixture,ids);b=ccle_ab(fixture,ids)
    historical_cut=(a['group_n_L'],a['group_n_L']+a['group_n_I'])
    current_cut=(b['group_n_L'],b['group_n_L']+b['group_n_I'])
    assert historical_cut==(5,9) and current_cut==(4,9)
    try:crs_geometry(.0025,.0025,7);raise AssertionError('Degenerate nodes accepted')
    except ValueError:pass
    return dict(status='PASS',synthetic_profiles=12,synthetic_fold_count=12,
                original_five_components=result,failure_accounting_control='PASS',
                historical_HB_tie_fixture_cut=historical_cut,current_CCLE_tie_fixture_cut=current_cut,
                scope='Synthetic fixed-score classifier-only and boundary fixtures; not a CCLE benchmark or global AB exactness certification')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
