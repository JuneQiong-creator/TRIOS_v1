"""Lightweight integrity and regression anchors; no new rankings or inference."""
from pathlib import Path
import argparse,csv,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_manifest import verify
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(actual,expected,tol=5e-12):
    assert abs(float(actual)-expected)<=tol,(actual,expected)
def check():
    result=verify()
    for scope,filename in [('FULL','EXPERIMENT_A_FULL_81_METHOD_RESULTS.csv'),('B100','EXPERIMENT_A_B100_ALL_81_METHOD_RESULTS.csv')]:
        rows=read('results/ccle/'+filename)
        assert len(rows)==567 and len({r['dataset_id'] for r in rows})==7
        assert len({r['method_id'] for r in rows})==81
        assert {r['scope'] for r in rows}=={scope}
        assert {r['contract_id'] for r in rows}=={'CCLE_EXPERIMENT_A_FIXED_SCORE_ORIGINAL5'}
        trios=[r for r in rows if r['method_id']=='TRIOS']
        assert len(trios)==7
        assert all(float(r['A_valid'])==1 and int(r['rank_conditional'])==int(r['rank_operational'])==1 for r in trios)
        assert sum(int(r['intended_tasks']) for r in rows)==(567 if scope=='FULL' else 56700)
        for r in trios:close(r['C_conditional'],float(r['C_operational']))
    e3=next(r for r in read('results/simulation/E3_81METHOD_TRUTH_METRIC_SUMMARY.csv') if r['method_id']=='TRIOS')
    # Published rounded anchors are diagnostics, not substitutes for source rows.
    close(e3['ARI'],.527579,5e-7);close(e3['BA'],.807502,5e-7)
    close(e3['MacroF1'],.785824,5e-7);close(e3['Extreme_LH'],.003977691,5e-10)
    hb=read('results/hb/ranking.csv')
    trios=next(r for r in hb if r['method_id']=='TRIOS')
    quality_key=next(k for k in trios if k in ['C_operational','operational_C'])
    close(trios[quality_key],.8943306487982025)
    result.update(current_CCLE_full_tasks=567,current_CCLE_B100_tasks=56700,
                  current_pipeline_count=81,E3_source='Current Q50 81-method summary only',
                  verification='Stored fields, declared contract and regression anchors; no independent scientific reproduction')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=check();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
