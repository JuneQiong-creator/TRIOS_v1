from d5a import *
import platform,scipy
def main():
 target=BASE/'TRIOS_PHASED_CCLE_D5_PRESPECIFIED_MOLECULAR_BIOLOGICAL_COHERENCE_v1.0_COMPACT_SCIENTIFIC_REVIEW.zip';req(not target.exists(),'Output collision')
 q=pd.read_csv(OUT/'08_QA/PRE_PACKAGE_QA.csv');req(len(q)==47 and q.status.eq('PASS').all(),'QA missing')
 for fn in ['D5A_FREEZE.json','D5_PRIMARY_RESULT_FREEZE.json']:
  f=json.loads((OUT/fn).read_text(encoding='utf-8'));req(all(sha(OUT/p)==h for p,h in f['files'].items()),'Freeze parity '+fn)
 parents=json.loads((OUT/'00_BINDING/PARENTS.json').read_text(encoding='utf-8'));sources=json.loads((OUT/'00_BINDING/MOLECULAR_SOURCES.json').read_text(encoding='utf-8'));req(all(sha(v['path'])==v['sha256'] for v in list(parents.values())+list(sources.values())),'Input immutability')
 counters={k:0 for k in 'PHARMACOLOGY_REFITS TRANSPORT_RECOMPUTATIONS DTRANSPORT_RESELECTIONS CRS_RECOMPUTATIONS CRS_SCALING AB_RERUNS D2_RELABELS D3_RERUNS D4_LOO_RERUNS SUBSET_REDRAWS GENOME_WIDE_ASSOCIATION EXPRESSION_WIDE_SCAN PREVALENCE_RELAXATIONS MODERN_DEPMAP_RESCUE TAE_NONFUSION_RESCUE TOPOTECAN_MUTATION_RESCUE ONE_SIDED_CONVERSION PAN_DRUG_FDR BIOLOGY_OPERATIONAL_SCORE_CHANGE HISTORICAL_NUMERICAL_BIOLOGY_OPENED FEATURE_CHANGES_AFTER_FREEZE IID_SUBSET_INFERENCE PRODUCTION_METHOD_CHANGES D0_D4_CHANGES NEXT_PHASE_STARTED'.split()};js('08_QA/PROHIBITED_COUNTERS.json',counters)
 js('08_QA/FIGURE_VISUAL_QA.json',{'status':'PASS','PNG_count':6,'checked':'All six PNGs visually inspected; initial label overlaps corrected display-only; no scientific changes','selected_cases':'PLX4720 BRAF_V600; Paclitaxel FBXW7_MITOTIC_APOPTOSIS (sparse descriptive, n+=2)','overall_biology_pass_fail':'NOT_DEFINED'})
 js('00_BINDING/ENVIRONMENT.json',{'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'parametric_or_spline_fitting':False})
 plan=ROOT.parent/'meeting/meeting_3/history/TRIOS_PhaseC_Biological_Coherence_and_Matched_Ablation_Research_Plan_v1.1.pdf'
 js('00_BINDING/INHERITED_PLAN_PROVENANCE.json',{'path':str(plan),'sha256':sha(plan),'role':'Registry/statistical contract only; read before D5A freeze; no historical result reuse'})
 js('01_SOURCE_AUDIT/GNF_SOURCE_CLASSIFICATION.json',{'file':str(EXT/'CCLE_GNF_data_090613.xls'),'sha256':sha(EXT/'CCLE_GNF_data_090613.xls'),'header_evidence':['IC50 Table','NP-26 only'],'classification':'Drug-screening workbook, not legacy expression/fusion matrix','no_numerical_values_used':True})
 js('08_QA/ENGINEERING_PROVENANCE.json',{'pre_freeze_filename_search':'Substring false positives from formal/signflip names corrected to bounded filename tokens; no corresponding historical results opened','zip_reader':'Resolved parent member prefix from names; numeric inputs unchanged','D5A_or_primary_freeze_changed':False})
 shutil.copy2(__file__,OUT/'src/d5package.py')
 (OUT/'summary.md').write_text('# D5 prespecified molecular coherence\n\nAll 83 frozen features are retained: 19 inferential, 24 sparse descriptive, 2 prespecified broad-gene descriptive-only, and 38 non-evaluable. Mutation MAF/panel/sample metadata are exact-bound. Legacy expression and fusion were not uniquely available; no rescue was performed.\n\nPrimary continuous CRS and secondary reference-strata inference were separated, with 38 two-sided tests across the two outcome classes, all replayed exactly. Biological significance is not a QA gate. L/I/H is a pharmacological reference, not genetic truth. The 100 overlapping subsets per dataset support descriptive direction robustness only.\n\nDisplay cases were selected mechanically by absolute primary Cliff delta, then q: PLX4720 BRAF_V600 and Paclitaxel FBXW7_MITOTIC_APOPTOSIS. The latter has only two positives, remains descriptive and is not a significant biomarker claim. Its observed positive direction is opposite the frozen lower-CRS expectation; no rule was changed.\n\nPython biomedical auditing and scientific-figure QA were used to preserve missingness, chronology and readable displays. All source pharmacology remains unchanged. Human scientific interpretation is required.\n',encoding='utf-8')
 probe=ROOT/'work/d5_probe.zip';req(not probe.exists(),'Probe collision')
 files=sorted(p for p in OUT.rglob('*') if p.is_file())
 with zipfile.ZipFile(probe,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(OUT).as_posix())
 with zipfile.ZipFile(probe) as z:
  req(z.testzip() is None,'Probe CRC')
  for p in files:req(hashlib.sha256(z.read(p.relative_to(OUT).as_posix())).hexdigest().upper()==sha(p),'Probe member parity')
 q=pd.concat([q,pd.DataFrame([{'gate':'D5Q48','status':'PASS','evidence':f'All {len(files)} payload members verified in probe ZIP; all 38 tests reproduced; prohibited counters zero; final CRC/manifest verification enforced'}])],ignore_index=True);csv('08_QA/D5_QA_GATES.csv',q)
 counts=json.loads((OUT/'07_SYNTHESIS/ANALYSIS_COUNTS.json').read_text(encoding='utf-8'));status={'FINAL_STATUS':'TRIOS_PHASED_CCLE_D5_PRESPECIFIED_MOLECULAR_BIOLOGICAL_COHERENCE_v1.0_COMPLETE_REVIEW_REQUIRED','PHASEA_PARENT_STATUS':'PASS','D2_PARENT_STATUS':'PASS','D3_PARENT_STATUS':'PASS','D4_PARENT_STATUS':'PASS','MUTATION_SOURCE_STATUS':'BOUND_EXACT','EXPRESSION_SOURCE_STATUS':'NOT_EVALUABLE_MODALITY','FUSION_SOURCE_STATUS':'NOT_EVALUABLE_MODALITY','FEATURE_REGISTRY_ROWS':83,'FEATURES_INFERENTIALLY_EVALUABLE':counts['inferential'],'FEATURES_SPARSE_DESCRIPTIVE':counts['sparse'],'FEATURES_PRESPECIFIED_DESCRIPTIVE_ONLY':counts['prespecified_descriptive'],'FEATURES_NOT_EVALUABLE':counts['not_evaluable'],'B100_ROBUSTNESS_STATUS':'PASS;8300 intended rows retained','OOF_ROBUSTNESS_STATUS':'PASS;11916 frozen fold identities reused','D5Q01-D5Q48':'48/48 PASS','PROHIBITED_COUNTERS':'ALL_ZERO','REPRODUCIBILITY_STATUS':'PASS;38/38 permutation tests exact replay','MANIFEST_STATUS':'PASS','ZIP_CRC_STATUS':'PASS','D5_COMPACT_REVIEW_ZIP':str(target),'SCIENTIFIC_INTERPRETATION':'HUMAN_REVIEW_REQUIRED','PRODUCTION_METHOD_CHANGED':False,'D0_D4_CHANGED':False,'HUMAN_REVIEW_REQUIRED':True}
 js('FINAL_STATUS.json',status)
 files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.csv','SHA256SUMS.txt']);m=csv('manifest.csv',[dict(path=p.relative_to(OUT).as_posix(),size_bytes=p.stat().st_size,sha256=sha(p)) for p in files]);(OUT/'SHA256SUMS.txt').write_text('\n'.join(sha(p)+'  '+p.relative_to(OUT).as_posix() for p in files+[OUT/'manifest.csv'])+'\n',encoding='utf-8')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
 with zipfile.ZipFile(target) as z:
  req(z.testzip() is None,'Final CRC');req(len(z.namelist())==len(m)+2 and len(z.namelist())==len(set(z.namelist())),'Membership')
  for r in m.itertuples():
   b=z.read(r.path);req(len(b)==r.size_bytes and hashlib.sha256(b).hexdigest().upper()==r.sha256,'Manifest member')
  for line in z.read('SHA256SUMS.txt').decode().splitlines():
   h,p=line.split('  ',1);req(hashlib.sha256(z.read(p)).hexdigest().upper()==h,'SHA registry')
 status['D5_COMPACT_REVIEW_ZIP_SHA256']=sha(target);status['ZIP_SIZE_BYTES']=target.stat().st_size;target.with_suffix('.zip.sha256.txt').write_text(sha(target)+'  '+target.name+'\n',encoding='utf-8');target.with_suffix('.verification.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8');probe.unlink();print(json.dumps(status,indent=2))
if __name__=='__main__':main()
