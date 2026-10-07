"""Current CCLE entry: stored-result checks or documented synthetic fixed-score smoke.
Full real-data recomputation is not a verified portable path in this scoped release.
"""
from pathlib import Path
import argparse,runpy,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['verify-stored','synthetic-smoke','full-recompute'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if a.mode=='full-recompute':
        p.error('Complete real-data input/score/fit/fold bank is not bundled or portability-verified. See data/ccle/README.md and docs/KNOWN_LIMITATIONS.md; no alternative result/version is substituted.')
    target=ROOT/('scripts/verify_stored_results.py' if a.mode=='verify-stored' else 'examples/ccle_fixed_score_smoke.py')
    sys.argv=[str(target),'--out',a.out];runpy.run_path(str(target),run_name='__main__')
