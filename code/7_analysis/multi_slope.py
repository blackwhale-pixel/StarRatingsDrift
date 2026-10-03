import pandas as pd, numpy as np
M={'RoBERTa':'/home/claude/rb/roberta_out/preds','Distil':'/home/claude/e/transformer_out/preds','TF200':'/home/claude/t2k/preds'}
yrs=list(range(2017,2024)); B=1000; rng=np.random.default_rng(11)
def mf1(y,p,cls=None):
    fs=[]
    for c in range(3):
        tp=((y==c)&(p==c)).sum(-1);fp=((y!=c)&(p==c)).sum(-1);fn=((y==c)&(p!=c)).sum(-1);fs.append(2*tp/np.maximum(2*tp+fp+fn,1))
    return fs[cls] if cls is not None else np.mean(fs,0)
for v in ['matched','natural']:
  for cls,cn in [(None,'macro'),(1,'neu')]:
    F={k:np.zeros((7,B)) for k in M}; P={k:np.zeros(7) for k in M}
    for i,y in enumerate(yrs):
        base=None; pr={}
        for k,d in M.items():
            for s in [42,43,44]:
                x=pd.read_csv(f'{d}/seed{s}_{y}_{v}.csv').sort_values('rid').reset_index(drop=True)
                if base is None: base=x[['rid','y']]
                assert (x.rid.values==base.rid.values).all()
                pr[(k,s)]=x.pred.values
        Y=base.y.values; idx=rng.integers(0,len(Y),(B,len(Y)))
        for k in M:
            F[k][i]=np.mean([mf1(Y[idx],pr[(k,s)][idx],cls) for s in [42,43,44]],0); P[k][i]=np.mean([mf1(Y,pr[(k,s)],cls) for s in [42,43,44]])
    sl={k:np.polyfit(yrs,F[k],1)[0] for k in M}; pt={k:np.polyfit(yrs,P[k],1)[0] for k in M}
    print(f'--- {v} {cn}')
    for k in M: print(f"  {k:8s} 2017 {P[k][0]:.4f} 2022 {P[k][5]:.4f} slope {pt[k]:.4f} [{np.percentile(sl[k],2.5):.4f},{np.percentile(sl[k],97.5):.4f}] rel {pt[k]/P[k][0]*100:.2f}%/yr")
    for a,b in [('RoBERTa','Distil'),('RoBERTa','TF200')]:
        d=sl[a]-sl[b]; lv=(F[a]-F[b]).mean(0); print(f"  {a}-{b}: slope diff {pt[a]-pt[b]:.4f} [{np.percentile(d,2.5):.4f},{np.percentile(d,97.5):.4f}] | level diff {(P[a]-P[b]).mean():.4f} [{np.percentile(lv,2.5):.4f},{np.percentile(lv,97.5):.4f}]")
