"""Compare archived raw response registry with the seven frozen CCLE CSVs."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import engine as m
from engine import A,e,rt
base=Path('OMITTED_HOST_PATH/cohorts')
registry=e.rows(e.R/'work/trios_phaseb_v13/prior_nonstandard_json_package/08_QSC_REFERENCE_SPACE/ccle_observed_8dose_response_registry.csv')
out=[]
for cid in e.CID:
    path=base/(cid+'.csv');raw=e.rows(path)
    actual={(r['ccle_cell_line_name'],float(r['dose_uM'])):-float(r['activity_median_raw'])/100 for r in raw}
    expected={(r['sample_id'],float(r['dose_uM'])):float(r['oriented_response']) for r in registry if r['dataset_id']==cid}
    e.require(len(actual)==len(raw) and actual.keys()==expected.keys() and max(abs(actual[k]-expected[k]) for k in actual)<1e-14,'RAW_RESPONSE_PARITY:'+cid)
    out.append(dict(dataset_id=cid,path=str(path),sha256=rt.sha(path),profiles=len(actual)//8,rows=len(actual),status='PASS'))
e.csvwrite(A/'00_BINDING/RAW_DATA_SHA256.csv',out)
rt.atomic_json(A/'01_QA/RAW_DATA_PARITY.json',dict(status='PASS',datasets=7,profiles=sum(r['profiles'] for r in out),rows=sum(r['rows'] for r in out),sources=out),immutable=True)
print('Seven frozen raw datasets: identity and response parity PASS')
