#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
METHOD_LABELS={'no_cbf':'No CBF','cbf_rl':'CBF-CF','bcbf_rl':'bCBF-CF'}
METHOD_ORDER=['no_cbf','cbf_rl','bcbf_rl']
MAP_ORDER=['shifted','corridor','clustered','slalom','bottleneck']
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--original-dir',required=True); p.add_argument('--unseen-dir',required=True); p.add_argument('--output-dir',default='logs/evaluation/figures_current'); p.add_argument('--formats',nargs='+',default=['png','pdf'],choices=['png','pdf','svg']); p.add_argument('--dpi',type=int,default=300); return p.parse_args()
def save(fig,stem,formats,dpi):
 stem.parent.mkdir(parents=True,exist_ok=True)
 for fmt in formats: fig.savefig(stem.with_suffix('.'+fmt),dpi=dpi,bbox_inches='tight')
 plt.close(fig)
def grouped(df,cat,metrics,labels,title,ylabel,out,formats,dpi,percent=False):
 x=np.arange(len(df)); width=0.8/len(metrics); fig,ax=plt.subplots(figsize=(8.5,4.8))
 for i,(m,l) in enumerate(zip(metrics,labels)):
  vals=df[m].to_numpy(float); vals=100*vals if percent else vals
  ax.bar(x+(i-(len(metrics)-1)/2)*width,vals,width,label=l)
 ax.set_xticks(x); ax.set_xticklabels(df[cat].tolist()); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y',alpha=.3); ax.legend(); save(fig,out,formats,dpi)
def main():
 a=parse_args(); od=Path(a.original_dir).expanduser(); ud=Path(a.unseen_dir).expanduser(); out=Path(a.output_dir).expanduser(); out.mkdir(parents=True,exist_ok=True)
 orig=pd.read_csv(od/'combined_summary.csv'); unseen=pd.read_csv(ud/'combined_summary_by_map.csv')
 orig['method_label']=orig.method.map(METHOD_LABELS); orig['rank']=orig.method.map({m:i for i,m in enumerate(METHOD_ORDER)}); orig=orig.sort_values('rank')
 grouped(orig,'method_label',['success_rate','collision_rate','timeout_rate'],['Success','Collision','Timeout'],'Original environment outcomes','Rate (%)',out/'original_outcomes',a.formats,a.dpi,True)
 grouped(orig,'method_label',['mean_success_goal_time_s'],['Goal time'],'Original environment successful goal time','Time (s)',out/'original_goal_time',a.formats,a.dpi)
 grouped(orig,'method_label',['mean_success_path_length_m'],['Path length'],'Original environment successful path length','Path length (m)',out/'original_path_length',a.formats,a.dpi)
 grouped(orig,'method_label',['mean_minimum_clearance_m','minimum_clearance_overall_m'],['Mean minimum clearance','Worst-case clearance'],'Original environment clearance','Clearance (m)',out/'original_clearance',a.formats,a.dpi)
 filt=orig[orig.method!='no_cbf']; grouped(filt,'method_label',['mean_filter_active_fraction'],['Filter active fraction'],'Original environment safety-filter activity','Fraction',out/'original_filter_activity',a.formats,a.dpi)
 unseen['method_label']=unseen.method.map(METHOD_LABELS)
 for metric,title,ylabel,pct in [('success_rate','Success rate on unseen maps','Success rate (%)',True),('collision_rate','Collision rate on unseen maps','Collision rate (%)',True),('mean_success_goal_time_s','Successful goal time on unseen maps','Time (s)',False),('mean_success_path_length_m','Successful path length on unseen maps','Path length (m)',False),('mean_minimum_clearance_m','Mean minimum clearance on unseen maps','Clearance (m)',False),('mean_filter_active_fraction','Safety-filter activity on unseen maps','Fraction',False)]:
  piv=unseen.pivot(index='map_name',columns='method',values=metric).reindex(MAP_ORDER); x=np.arange(len(piv)); fig,ax=plt.subplots(figsize=(9.5,4.8)); width=.24
  for i,m in enumerate(METHOD_ORDER):
   vals=piv[m].to_numpy(float); vals=100*vals if pct else vals; ax.bar(x+(i-1)*width,vals,width,label=METHOD_LABELS[m])
  ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in piv.index]); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y',alpha=.3); ax.legend(); save(fig,out/f'unseen_{metric}',a.formats,a.dpi)
 agg=unseen.groupby('method',as_index=False).agg(success_rate=('success_rate','mean'),collision_rate=('collision_rate','mean'),timeout_rate=('timeout_rate','mean'),mean_success_goal_time_s=('mean_success_goal_time_s','mean'),mean_success_path_length_m=('mean_success_path_length_m','mean'),mean_minimum_clearance_m=('mean_minimum_clearance_m','mean'),mean_filter_active_fraction=('mean_filter_active_fraction','mean'),mean_command_correction_norm=('mean_command_correction_norm','mean'))
 agg['method_label']=agg.method.map(METHOD_LABELS); agg['rank']=agg.method.map({m:i for i,m in enumerate(METHOD_ORDER)}); agg=agg.sort_values('rank'); agg.to_csv(out/'unseen_aggregate_summary.csv',index=False)
 grouped(agg,'method_label',['success_rate','collision_rate','timeout_rate'],['Success','Collision','Timeout'],'Aggregate unseen-map outcomes','Rate (%)',out/'unseen_aggregate_outcomes',a.formats,a.dpi,True)
 orig.to_csv(out/'original_summary_copy.csv',index=False); unseen.to_csv(out/'unseen_summary_copy.csv',index=False)
 print(f'Saved figures to: {out}')
if __name__=='__main__': main()
