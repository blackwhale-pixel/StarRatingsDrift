import pandas as pd, numpy as np
mo=pd.read_csv('/home/claude/tm/test_months.csv').set_index('rid').month
M={'RoBERTa-base':'/home/claude/rb/roberta_out/preds','DistilRoBERTa':'/home/claude/e/transformer_out/preds','TF-IDF 200k':'/home/claude/t2k/preds','TF-IDF 500k':'/home/claude/tp/preds'}
yrs=list(range(2017,2024)); S=[42,43,44]; B=1000; rng=np.random.default_rng(9)
def f1(y,p):
    fs=[]
    for c in range(3):
        tp=((y==c)&(p==c)).sum(-1); fp=((y!=c)&(p==c)).sum(-1); fn=((y==c)&(p!=c)).sum(-1); fs.append(2*tp/np.maximum(2*tp+fp+fn,1))
    return np.mean(fs,0)
for v in ['matched','natural']:
    print(f'== {v}')
    comp=[]
    F={k:np.zeros((7,B)) for k in M}; P={k:np.zeros(7) for k in M}; Pall={k:np.zeros(7) for k in M}
    for i,y in enumerate(yrs):
        base=pd.read_csv(f'{M["TF-IDF 200k"]}/seed42_{y}_{v}.csv').sort_values('rid').reset_index(drop=True)
        m=base.rid.map(mo).values; keep=m<=9
        comp.append((y,len(base),keep.mean().round(3), np.bincount(base.y[keep],minlength=3)/keep.sum()))
        Y=base.y.values[keep]; idx=rng.integers(0,len(Y),(B,len(Y)))
        for k,d in M.items():
            pr=[pd.read_csv(f'{d}/seed{s}_{y}_{v}.csv').sort_values('rid').pred.values for s in S]
            F[k][i]=np.mean([f1(Y[idx],p[keep][idx]) for p in pr],0); P[k][i]=np.mean([f1(Y,p[keep]) for p in pr]); Pall[k][i]=np.mean([f1(base.y.values,p) for p in pr])
    for y,n,share,dist in comp: print(f'   {y}: Jan-Sep share {share}  label dist (neg,neu,pos) {np.round(dist,3)}')
    sl={k:np.polyfit(yrs,F[k],1)[0] for k in M}
    for k in M:
        print(f'   {k:14s} all months slope {np.polyfit(yrs,Pall[k],1)[0]:+.4f} | Jan-Sep slope {np.polyfit(yrs,P[k],1)[0]:+.4f} [{np.percentile(sl[k],2.5):+.4f},{np.percentile(sl[k],97.5):+.4f}] | 2017 {P[k][0]:.3f} 2022 {P[k][5]:.3f} 2023 {P[k][6]:.3f}')
    d=sl['RoBERTa-base']-sl['TF-IDF 200k']; print(f'   RoBERTa - TF200 slope diff [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]')
