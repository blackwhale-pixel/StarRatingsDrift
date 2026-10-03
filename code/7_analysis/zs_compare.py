import pandas as pd, numpy as np
yrs=list(range(2017,2024))
def mf1(y,p):
    fs=[]
    for c in range(3):
        tp=((y==c)&(p==c)).sum(); fp=((y!=c)&(p==c)).sum(); fn=((y==c)&(p!=c)).sum(); fs.append(2*tp/max(2*tp+fp+fn,1))
    return np.mean(fs), fs[1]
zs=[0.6628,0.6565,0.6403,0.6576,0.6352,0.6110,0.6375]
for name,d in [('Transformer','/home/claude/e/transformer_out/preds'),('TF-IDF 200k','/home/claude/t2k/preds'),('TF-IDF 500k','/home/claude/tp/preds')]:
    full=[];sub=[];neu=[]
    for y in yrs:
        a=[];b=[];c=[]
        for s in [42,43,44]:
            x=pd.read_csv(f'{d}/seed{s}_{y}_matched.csv'); xs=x.sample(2000,random_state=2026)
            a.append(mf1(x.y.values,x.pred.values)[0]); m,n_=mf1(xs.y.values,xs.pred.values); b.append(m); c.append(n_)
        full.append(np.mean(a)); sub.append(np.mean(b)); neu.append(np.mean(c))
    print(f"{name:12s} full slope {np.polyfit(yrs,full,1)[0]:.4f} | same-2000-subset slope {np.polyfit(yrs,sub,1)[0]:.4f} | subset F1 {[round(v,3) for v in sub]}")
print(f"{'zero-shot':12s} subset slope {np.polyfit(yrs,zs,1)[0]:.4f} | F1 {zs}")
