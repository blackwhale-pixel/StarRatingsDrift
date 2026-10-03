import pandas as pd, numpy as np, glob
f=[x for x in glob.glob('/home/claude/zr/**/zs__*.jsonl',recursive=True)]; print(f)
z=pd.read_json(f[0],lines=True); z=z[z.label.notna()].drop_duplicates('rid',keep='last')
z['zp']=z.label.map({'NEGATIVE':0,'NEUTRAL':1,'POSITIVE':2})
yrs=list(range(2017,2024)); B=1000; rng=np.random.default_rng(7)
def mf1(y,p):
    fs=[]
    for c in range(3):
        tp=((y==c)&(p==c)).sum(-1);fp=((y!=c)&(p==c)).sum(-1);fn=((y==c)&(p!=c)).sum(-1);fs.append(2*tp/np.maximum(2*tp+fp+fn,1))
    return np.mean(fs,0)
models={'TR':'/home/claude/e/transformer_out/preds','TF200':'/home/claude/t2k/preds','RB':'/home/claude/rb/roberta_out/preds'}
F={k:np.zeros((7,B)) for k in ['ZS','TR','TF200','RB']}; P={k:np.zeros(7) for k in F}; match=[]
for i,y in enumerate(yrs):
    zz=z[z['sample']==f'{y}_matched'][['rid','zp']]
    # check reproduction of subset
    x=pd.read_csv(f'{models["TR"]}/seed42_{y}_matched.csv'); match.append(set(x.sample(2000,random_state=2026).rid)==set(zz.rid))
    base=x[['rid','y']].merge(zz,on='rid').sort_values('rid').reset_index(drop=True)
    preds={}
    for k,d in models.items():
        preds[k]=[pd.read_csv(f'{d}/seed{s}_{y}_matched.csv').set_index('rid').loc[base.rid,'pred'].values for s in [42,43,44]]
    Y=base.y.values; idx=rng.integers(0,len(Y),(B,len(Y)))
    F['ZS'][i]=mf1(Y[idx],base.zp.values[idx]); P['ZS'][i]=mf1(Y,base.zp.values)
    for k in models:
        F[k][i]=np.mean([mf1(Y[idx],p[idx]) for p in preds[k]],0); P[k][i]=np.mean([mf1(Y,p) for p in preds[k]])
print('subset reproduced exactly:',match)
sl={k:np.polyfit(yrs,F[k],1)[0] for k in F}
for k in F: print(f"{k:6s} slope {np.polyfit(yrs,P[k],1)[0]:.4f} [{np.percentile(sl[k],2.5):.4f},{np.percentile(sl[k],97.5):.4f}] rel/yr {np.polyfit(yrs,P[k],1)[0]/P[k][0]*100:.2f}%  F1 {[round(v,3) for v in P[k]]}")
for k in ['TR','TF200','RB']:
    d=sl[k]-sl['ZS']; print(f"{k} - ZS slope diff {np.polyfit(yrs,P[k],1)[0]-np.polyfit(yrs,P['ZS'],1)[0]:.4f} [{np.percentile(d,2.5):.4f},{np.percentile(d,97.5):.4f}]")
# agreement of zero-shot with trained models over years (do they diverge from labels jointly?)
