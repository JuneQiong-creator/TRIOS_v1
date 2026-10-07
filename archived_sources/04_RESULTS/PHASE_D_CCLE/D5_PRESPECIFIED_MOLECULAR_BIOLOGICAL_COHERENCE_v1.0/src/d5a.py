import os,sys,json,hashlib,shutil,gzip,re,ast,zipfile,io
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
ROOT=Path(r'SOURCE_PROJECT');BASE=ROOT/'04_RESULTS/PHASE_D_CCLE';OUT=BASE/'D5_PRESPECIFIED_MOLECULAR_BIOLOGICAL_COHERENCE_v1.0';EXT=ROOT.parent/'external-dataset/CCLE'
HIST=ROOT/'04_RESULTS/TRIOS_PHASEC_BIOLOGICAL_COHERENCE_AND_MATCHED_ABLATION_v1.1/02_BIO_REGISTRY'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest().upper()
def req(x,s):
 if not bool(x):raise RuntimeError(s)
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def js(p,x):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv(p,x):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);d=pd.DataFrame(x);d.to_csv(p,index=False,lineterminator='\n');return d
def now():return datetime.now(timezone.utc).isoformat()
def main():
 req(not (OUT/'D5A_FREEZE.json').exists(),'Frozen output collision');OUT.mkdir(parents=True,exist_ok=True)
 parents={}
 for key,pattern,h in [('A','TRIOS_PHASEA_v3.1_PHASED_D1_D2_IDENTITY_REBIND_COMPACT_REVIEW.zip','372AED536F72130E581BD35956590AFE49BDEC774A27953305F02146ECEF9698'),('D2','TRIOS_PHASED_CCLE_D2*COMPACT*.zip','530B7DFF0479F8F95945D0B80B2AB39972E67D07588251BFDC0BC45CE3F97C5C'),('D3','TRIOS_PHASED_CCLE_D3*v1.1*COMPACT*.zip','36DF5CEAE1CE952D32F1FE1EC4AB0EC83BDB5FD5B8A95CA8A8F1246F45762D57'),('D4','TRIOS_PHASED_CCLE_D4*COMPACT*.zip','7FEC51EBE89355512D71C480DAE83462A2DFF99121E6B0D28A14BFD08ADC3F8F')]:
  found=list((ROOT/'04_RESULTS' if key=='A' else BASE).glob(pattern));req(len(found)==1,'Parent unique '+key);p=found[0];req(sha(p)==h,'Parent SHA '+key);parents[key]={'path':str(p),'sha256':h}
 source={}
 for name,h in [('CCLE_hybrid_capture1650_hg19_NoCommonSNPs_NoNeutralVariants_CDS_2012.05.07.maf.gz','5021F4D1499E11A631A06D1BB36E5680F6E9DDC70CE6F865BCF46CA8742A5087'),('CCLE_hybrid_capture1650_HGNC_info_2012.02.20.txt','62F3388EE6848479F119FC769EA2EA91448D6626285E873659E0BB5B7813706A'),('CCLE_sample_info_file_2012-10-18.txt','5B5BB14B100EADAC6FAB52988DF5659EFF9EDC0E08845B13330C77673B2BA89A')]:
  p=EXT/name;req(sha(p)==h,'Molecular source SHA');source[name]={'path':str(p),'sha256':h}
 # Inventory filenames only. No historical biological result payloads opened.
 inventory=[];candidates=[]
 for root in [ROOT,EXT]:
  for p in root.rglob('*'):
   if not p.is_file():continue
   if p.suffix.lower() in ['.gct','.txt','.csv','.tsv','.gz','.xls','.xlsx','.rds']:
    n=p.name.lower()
    if any(t in n for t in ['expression','fusion','rearrang','gene_exp','rnaseq','rna_seq','u133']) or re.search(r'(^|[_\-.])(rma|gnf)([_\-.]|$)',n):
     category='NON_SOURCE_CODE_OR_AUDIT' if any(t in str(p).lower() for t in ['site-packages','pydeps','vendor','confusion','summary','audit','registry','manifest','provenance','report','status']) else 'CANDIDATE_NAME'
     inventory.append({'path':str(p),'category':category})
     if category=='CANDIDATE_NAME':candidates.append(str(p))
 csv('01_SOURCE_AUDIT/LOCAL_MODALITY_FILENAME_SEARCH.csv',inventory)
 # GNF is drug screening, not an expression source. All candidates retained for audit.
 req(all('CCLE_GNF_data_090613.xls' in p for p in candidates),'Unexpected molecular candidate requires source inspection')
 js('01_SOURCE_AUDIT/MODALITY_STATUS.json',{'MUTATION':'BOUND_EXACT','EXPRESSION':'NOT_EVALUABLE_MODALITY','FUSION':'NOT_EVALUABLE_MODALITY','ALK_ACTIVATING_MUTATION':'NOT_EVALUABLE_DEFINITION','reason':'No uniquely provenance-bound legacy expression/fusion source in established local tree; prior ALK definition names feature but gives no exact activating variant rule','candidate_paths':candidates})
 # Only historical registry/contract definitions, not numerical readouts.
 old=pd.read_csv(HIST/'frozen_feature_registry.csv');contract=json.loads((HIST/'mutation_call_contract.json').read_text(encoding='utf-8'));white=set(contract['qualifying_whitelist'])
 src=ROOT/'work/ccle_two_cohort_mutation_v10/run_mutation.py';tree=ast.parse(src.read_text(encoding='utf-8'));constants={}
 for n in tree.body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name) and t.id in ['PACG','PDG','PACMOD','WHITE','LOF']:constants[t.id]=ast.literal_eval(n.value)
 req(white==constants['WHITE'],'Inherited whitelist exact')
 reg=[]
 for r in old.itertuples():
  c,f=r.dataset_id,r.feature_name
  if 'Paclitaxel' in c:fam='PAC_CORE' if f in ['FBXW7','TUBB'] else 'PAC_MODULE' if f in ['FBXW7_MITOTIC_APOPTOSIS','APOPTOSIS_COMPETENCE'] else 'STRUCTURAL_NON_TEST' if f=='SPINDLE_CHECKPOINT' else 'PAC_TIER2'
  elif 'PLX4720' in c:fam='PLX_FUNCTION_AWARE' if f in ['BRAF_V600','RAS_HOTSPOT','NF1_LOF'] else 'DESCRIPTIVE_ONLY'
  elif 'TAE684' in c:fam='TAE_ALK'
  elif 'Topotecan' in c:fam='TOPOTECAN_EXPRESSION'
  else:fam='MODULE' if f=='CANONICAL_MAPK_ACTIVATION' else 'CORE_MECH' if f in ['RAS_HOTSPOT','BRAF_V600','NF1_LOF'] else 'BROAD_GENE'
  expected='LOWER_CRS' if ('PLX4720' in c and f in ['RAS_HOTSPOT','NF1_LOF']) or f=='FBXW7_MITOTIC_APOPTOSIS' else 'HIGHER_CRS' if bool(r.primary_feature) else 'CONTEXT_TWO_SIDED'
  reg.append(dict(cohort_id=c,feature=f,primary=bool(r.primary_feature),feature_type=r.feature_type,definition=r.definition,family=fam,expected_direction=expected))
 reg=pd.DataFrame(reg);req(len(reg)==83 and reg.groupby('cohort_id').size().sort_values().tolist()==[2,2,8,16,16,16,23],'Registry83')
 d0=BASE/'D0_DATA_CHALLENGE_SET_LOCK_v1.0';membership=pd.read_csv(d0/'D0_PRIMARY_7_SUBSET_MEMBERSHIP_REGISTRY.csv');models=membership[['cohort_id','cell_line']].drop_duplicates();req(len(models)==424,'Outcome blind membership424')
 si=pd.read_csv(EXT/'CCLE_sample_info_file_2012-10-18.txt',sep='\t').set_index('CCLE name');panel=pd.read_csv(EXT/'CCLE_hybrid_capture1650_HGNC_info_2012.02.20.txt',sep='\t');genes=set(panel['HGNC Symbol'].dropna());req(si.index.is_unique and models.cell_line.isin(si.index).all(),'Exact sample mapping')
 maps=models.copy();maps['hybrid_capture_assayed']=[str(si.loc[x,'Hybrid Capture Sequencing']).lower().strip()=='yes' for x in maps.cell_line];csv('01_SOURCE_AUDIT/SAMPLE_MAPPING.csv',maps)
 maf=pd.read_csv(EXT/'CCLE_hybrid_capture1650_hg19_NoCommonSNPs_NoNeutralVariants_CDS_2012.05.07.maf.gz',sep='\t',comment='#',usecols=['Hugo_Symbol','Tumor_Sample_Barcode','Variant_Classification','Protein_Change','UniProt_AApos','Genome_Change'])
 maf=maf[maf.Hugo_Symbol.isin(set(constants['PACG']+constants['PDG']))&maf.Tumor_Sample_Barcode.isin(models.cell_line)].copy();maf['qualifying']=maf.Variant_Classification.isin(white);q=maf[maf.qualifying].copy();csv('02_CALLS/FROZEN_CANDIDATE_VARIANT_EVIDENCE.csv',maf)
 def codon(r):
  m=re.search(r'(\d+)',str(r.Protein_Change))
  if m:return int(m[1])
  try:return int(float(r.UniProt_AApos))
  except:return None
 q['codon']=[codon(r) for r in q.itertuples()];calls=[]
 for r in reg.itertuples():
  for x in maps[maps.cohort_id==r.cohort_id].itertuples():
   variants=q[q.Tumor_Sample_Barcode==x.cell_line];flags={'RAS_HOTSPOT':((variants.Hugo_Symbol.isin(['KRAS','NRAS','HRAS']))&variants.codon.isin([12,13,61])).any(),'BRAF_V600':((variants.Hugo_Symbol=='BRAF')&(variants.codon==600)).any(),'NF1_LOF':((variants.Hugo_Symbol=='NF1')&variants.Variant_Classification.isin(constants['LOF'])).any()};flags['CANONICAL_MAPK_ACTIVATION']=any(flags.values())
   f=r.feature;members=constants['PACMOD'].get(f,[f]);structural=f not in flags and not all(g in genes for g in members)
   status='NOT_EVALUABLE_MODALITY' if 'Topotecan' in r.cohort_id or f=='ALK_REARRANGEMENT' else 'NOT_EVALUABLE_DEFINITION' if f=='ALK_ACTIVATING_MUTATION' else 'NA_STRUCTURALLY_NOT_EVALUABLE_PANEL' if structural else 'NA_NOT_HYBRID_CAPTURE_ASSAYED' if not x.hybrid_capture_assayed else 'MUTATION_CALL_POSITIVE' if flags.get(f,variants.Hugo_Symbol.isin(members).any()) else 'NO_QUALIFYING_MUTATION_CALL'
   value=1 if status=='MUTATION_CALL_POSITIVE' else 0 if status=='NO_QUALIFYING_MUTATION_CALL' else np.nan
   calls.append(dict(cohort_id=r.cohort_id,feature=f,cell_line=x.cell_line,status=status,value=value))
 csv('02_CALLS/D5_MOLECULAR_CALLS.csv',calls);csv('02_REGISTRY/D5_MECHANISTIC_FEATURE_REGISTRY.csv',reg)
 csv('02_REGISTRY/MULTIPLICITY_REGISTRY.csv',reg[['cohort_id','feature','family']].assign(continuous_family=lambda d:d.family+'_CONTINUOUS_CRS',strata_family=lambda d:d.family+'_REFERENCE_STRATA'))
 js('02_REGISTRY/MUTATION_DEFINITION_LOCK.json',{'whitelist':sorted(white),'NF1_LOF_classes':sorted(constants['LOF']),'gene_lists':{k:constants[k] for k in ['PACG','PDG','PACMOD']},'seed_conversion':'first 8 SHA256 digest bytes, unsigned big-endian, PCG64','continuous_namespace':'D5_v1.0|cohort_id|feature_id|CONTINUOUS_CRS','secondary_namespace':'D5_v1.0|cohort_id|feature_id|REFERENCE_STRATA','zero_effect_sign_rule':'sign(effect)==sign(full-source CRS effect), finite effects only'})
 js('00_BINDING/PARENTS.json',parents);js('00_BINDING/MOLECULAR_SOURCES.json',source)
 js('00_BINDING/ALLOWED_HISTORICAL_DEFINITION_SOURCES.json',{str(p):sha(p) for p in [HIST/'frozen_feature_registry.csv',HIST/'mutation_call_contract.json',src]})
 for name in ['TRIOS_PhaseD_CCLE_D5_Prespecified_Molecular_Biological_Coherence_Validation_Frozen_Execution_Spec_v1.0.md','CODEX_TRIOS_PhaseD_CCLE_D5_Prespecified_Molecular_Biological_Coherence_Validation_v1.0_DIRECT_EXECUTE.md']:
  p=Path('OMITTED_HOST_PATH/Downloads')/name;shutil.copy2(p,OUT/'00_BINDING'/name)
 (OUT/'src').mkdir(exist_ok=True);shutil.copy2(__file__,OUT/'src'/Path(__file__).name)
 frozen={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
 js('D5A_FREEZE.json',{'frozen_utc':now(),'files':frozen,'feature_rows':83,'historical_numerical_biology_opened':False,'current_pharmacology_outcomes_opened':False,'D5A_STATUS':'FROZEN'})
 print('D5A FROZEN',sha(OUT/'D5A_FREEZE.json'),flush=True)

if __name__=='__main__':main()
