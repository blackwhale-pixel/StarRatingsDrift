import pandas as pd, numpy as np
TR='/home/claude/e/transformer_out/preds'; TF='/home/claude/tp/preds'
yrs=list(range(2017,2024)); seeds=[42,43,44]; B=1000; rng=np.random.default_rng(2026)
def mf1(y,p,cls=None):
    fs=[]
    for c in range(3):
        tp=((y==c)&(p==c)).sum(-1);fp=((y!=c)&(p==c)).sum(-1);fn=((y==c)&(p!=c)).sum(-1);fs.append(2*tp/np.maximum(2*tp+fp+fn,1))
    return fs[cls] if cls is not None else np.mean(fs,0)
for v in ['matched','natural']:
  for cls,name in [(None,'macro'),(1,'neu')]:
    F={'tr':np.zeros((len(yrs),B)),'tf':np.zeros((len(yrs),B))}; P={'tr':np.zeros(len(yrs)),'tf':np.zeros(len(yrs))}
    for i,y in enumerate(yrs):
        base=None; data={}
        for m,d in [('tr',TR),('tf',TF)]:
            for s in seeds:
                x=pd.read_csv(f'{d}/seed{s}_{y}_{v}.csv').sort_values('rid').reset_index(drop=True)
                if base is None: base=x[['rid','y']]
                assert (x.rid.values==base.rid.values).all() and (x.y.values==base.y.values).all()
                data[(m,s)]=x.pred.values
        Y=base.y.values; idx=rng.integers(0,len(Y),(B,len(Y)))
        for m in ['tr','tf']:
            F[m][i]=np.mean([mf1(Y[idx],data[(m,s)][idx],cls) for s in seeds],0)
            P[m][i]=np.mean([mf1(Y,data[(m,s)],cls) for s in seeds])
    sl={m:np.polyfit(yrs,F[m],1)[0] for m in F}; d=sl['tr']-sl['tf']
    pt={m:np.polyfit(yrs,P[m],1)[0] for m in P}
    rel={m:pt[m]/P[m][0] for m in P}
    lvl=(F['tr']-F['tf']).mean(0)
    print(f"{v:8s} {name:5s} slope TR {pt['tr']:.4f} [{np.percentile(sl['tr'],2.5):.4f},{np.percentile(sl['tr'],97.5):.4f}]  TF {pt['tf']:.4f} [{np.percentile(sl['tf'],2.5):.4f},{np.percentile(sl['tf'],97.5):.4f}]  diff {pt['tr']-pt['tf']:.4f} [{np.percentile(d,2.5):.4f},{np.percentile(d,97.5):.4f}] P(diff<0)={np.mean(d<0):.3f}  rel/yr TR {rel['tr']*100:.2f}% TF {rel['tf']*100:.2f}%  level gain {(P['tr']-P['tf']).mean():.4f} [{np.percentile(lvl,2.5):.4f},{np.percentile(lvl,97.5):.4f}]")
