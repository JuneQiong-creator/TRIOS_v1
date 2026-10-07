"""Replay existing scientific figure tables. No producer or inference is imported."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
GROUPS={'2_3':'plot_hb_figures.py','4':'plot_figure4.py','5':'plot_figure5.py','6':'plot_figure6.py','7_8':'plot_figures7_8.py'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--include-figure1',action='store_true');p.add_argument('--group',choices=['all',*GROUPS],default='all');a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
    for group,script in GROUPS.items():
        if a.group!='all' and group!=a.group:continue
        folder='Figure2_3' if group=='2_3' else 'Figure7_8' if group=='7_8' else 'Figure'+group
        src=ROOT/'figures'/folder;dest=out/folder;dest.mkdir(exist_ok=True)
        if group=='2_3':
            for f in (src/'data').iterdir():shutil.copy2(f,dest/f.name)
            env['TRIOS_FIGURE_OUTPUT']=str(dest)
            command=[sys.executable,str(src/script)]
        else:command=[sys.executable,str(src/script),'--out',str(dest)]
        subprocess.run(command,check=True,env=env)
    if a.include_figure1:
        assert shutil.which('pdflatex'),'Figure 1 requires pdflatex + article/geometry + TikZ; no TeX distribution is installed by this script.'
        dest=out/'Figure1';dest.mkdir(exist_ok=True)
        for f in (ROOT/'figures/Figure1').glob('*.tex'):shutil.copy2(f,dest/f.name)
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','figure1.tex'],cwd=dest,check=True,env=env,stdout=subprocess.DEVNULL)
    files=sorted(str(f.relative_to(out)) for f in out.rglob('*') if f.suffix in ['.pdf','.svg','.png'])
    (out/'REPLAY_SCOPE.json').write_text(json.dumps(dict(scope='Frozen-table redraw only; not full scientific reproduction',files=files),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',figure_files=len(files),scope='Figure replay only')))
