"""Độ nhạy của PGExplainer với số luồng CPU: so hai lượt chạy lưới PG (cùng seed, khác số luồng).

Lượt 22/11 (luồng mặc định, ``PN2_H2_pgexplainer_scores_20261122.zip`` trên Drive) so với lượt cuối kỳ
(``OMP_NUM_THREADS=2``). Với mỗi (g, e): κ@5 giữa hai lượt (trung bình theo node, m > 5) và AUROC motif
trung bình của từng lượt. Chỉ BA-Shapes. Ghi ``results/analysis/a2/pg_thread_sensitivity.csv``.

Ví dụ:
  python scripts/analyze_pg_thread_sensitivity.py --old results/test-assets/pg_run_20261122       --new results/scores --out results/analysis/a2/pg_thread_sensitivity.csv
"""
import argparse
from pathlib import Path

import pandas as pd, numpy as np, torch
from sklearn.metrics import roc_auc_score
from sna.evaluation.core import agreement, rank_edges
d=torch.load('results/data/bashapes/data.pt',weights_only=True,map_location='cpu')
ei=d['edge_index']; gt=d['gt_edge_mask'].bool()
motif={(min(int(a),int(b)),max(int(a),int(b))) for a,b in ei[:,gt].t()}
ap=argparse.ArgumentParser(); ap.add_argument('--old',type=Path,required=True); ap.add_argument('--new',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args(); OLD,NEW=args.old.as_posix(),args.new.as_posix()
rows=[]
for g in range(1000,1005):
  for e in range(5):
    o=pd.read_csv(f'{OLD}/bashapes/gcn_s{g}/pgexplainer_e{e}.csv'); n=pd.read_csv(f'{NEW}/bashapes/gcn_s{g}/pgexplainer_e{e}.csv')
    ka=[];au=[[],[]]
    for node,(go,gn) in enumerate(zip(o.groupby('node_id'),n.groupby('node_id'))):
        (n1,a),(n2,b)=go,gn; assert n1==n2 and (a.u.values==b.u.values).all()
        sa,sb=a.importance_score.values,b.importance_score.values
        m=len(sa)
        if m>5:
            ka.append(agreement(sa,sb,5,seed=0)['kappa'])
        y=np.array([(min(u,v),max(u,v)) in motif for u,v in zip(a.u,a.v)])
        if 0<y.sum()<len(y):
            au[0].append(roc_auc_score(y,sa)); au[1].append(roc_auc_score(y,sb))
    rows.append(dict(g=g,e=e,kappa5_old_vs_new=np.nanmean(ka),auc_old=np.mean(au[0]),auc_new=np.mean(au[1])))
r=pd.DataFrame(rows); pd.set_option('display.float_format',lambda x:f'{x:.3f}')
print(r[r.g==1000].to_string(index=False))
print('mean kappa@5 old vs new (25 cfgs): %.3f  | mean |AUC old-new|: %.3f  max %.3f'%(r.kappa5_old_vs_new.mean(),(r.auc_old-r.auc_new).abs().mean(),(r.auc_old-r.auc_new).abs().max()))
print('AUC range across 25 cfg old: %.3f-%.3f ; new: %.3f-%.3f'%(r.auc_old.min(),r.auc_old.max(),r.auc_new.min(),r.auc_new.max()))
args.out.parent.mkdir(parents=True,exist_ok=True); r.to_csv(args.out,index=False)
