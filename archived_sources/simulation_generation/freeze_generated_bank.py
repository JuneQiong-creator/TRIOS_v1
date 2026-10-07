import hashlib,json
from pathlib import Path

r=Path(r"SOURCE_PROJECT\04_RESULTS\PHASE_C_SIMULATION\EXPERIMENT_1\TRIOS_PHASEC_SIM_E1_v1.0")
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest().upper()
def rows(p):
 with p.open('rb') as f:return sum(chunk.count(b'\n') for chunk in iter(lambda:f.read(8<<20),b''))-1
raw=r/'03_GENERATED_BANK/formal_raw_observations.csv';lat=r/'03_GENERATED_BANK/formal_latent_parameters.csv';design=r/'02_DESIGN_REGISTRY/formal_design_registry.csv';seed=r/'02_DESIGN_REGISTRY/seed_registry.csv'
obj={'status':'PASS','chronology_state':'FORMAL_GENERATED_BANK_FROZEN_BEFORE_AUTHORITATIVE_ANALYSIS','formal_cohorts':10200,'formal_profiles':rows(lat),'raw_observed_rows':rows(raw),'files':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in (raw,lat,design,seed)}}
assert obj['formal_profiles']==212400 and obj['raw_observed_rows']==5097600
(r/'03_GENERATED_BANK/generated_bank_lock.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
