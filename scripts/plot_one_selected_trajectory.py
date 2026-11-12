#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
METHODS=['no_cbf','cbf_rl','bcbf_rl']; LABELS={'no_cbf':'No CBF','cbf_rl':'CBF-CF','bcbf_rl':'bCBF-CF'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--map',required=True,choices=['shifted','corridor','clustered','slalom','bottleneck']); p.add_argument('--case-id',required=True,type=int); p.add_argument('--output',required=True); a=p.parse_args()
 md=Path(a.run_dir).expanduser()/a.map; fig,ax=plt.subplots(figsize=(7,7)); mp=pd.read_csv(md/'map.csv')
 if {'x','y','radius'}.issubset(mp.columns):
  for _,r in mp.iterrows(): ax.add_patch(plt.Circle((r.x,r.y),r.radius,fill=False))
 elif {'center_x','center_y','radius'}.issubset(mp.columns):
  for _,r in mp.iterrows(): ax.add_patch(plt.Circle((r.center_x,r.center_y),r.radius,fill=False))
 plotted=False
 for m in METHODS:
  f=md/m/f'{m}_trajectory.csv'
  if not f.is_file(): continue
  d=pd.read_csv(f); sub=d[d.case_id==a.case_id]
  if sub.empty: continue
  xcol='x' if 'x' in sub.columns else 'position_x'; ycol='y' if 'y' in sub.columns else 'position_y'; ax.plot(sub[xcol],sub[ycol],label=LABELS[m]); plotted=True
 if not plotted: raise RuntimeError('No trajectory rows found; rerun with --log-trajectories.')
 sc=pd.read_csv(md/'scenarios.csv'); r=sc[sc.case_id==a.case_id].iloc[0]; ax.scatter([r.start_x],[r.start_y],s=80,label='Start'); ax.scatter([r.goal_x],[r.goal_y],marker='*',s=140,label='Goal')
 ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_title(f'{a.map.capitalize()} map, case {a.case_id}'); ax.grid(alpha=.3); ax.legend(); out=Path(a.output).expanduser(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=300,bbox_inches='tight'); print(f'Saved: {out}')
if __name__=='__main__': main()
