import pandas as pd, numpy as np
M={'RoBERTa-base':'/home/claude/rb/roberta_out/preds','DistilRoBERTa':'/home/claude/e/transformer_out/preds',
   'TF-IDF 200k':'/home/claude/t2k/preds','TF-IDF 500k':'/home/claude/tp/preds'}
S=[42,43,44]; B=1000; rng=np.random.default_rng(31)
def f1m(y,p,classes):
    fs=[]
    for c in classes:
        tp=((y==c)&(p==c)).sum(-1); fp=((y!=c)&(p==c)).sum(-1); fn=((y==c)&(p!=c)).sum(-1); fs.append(2*tp/np.maximum(2*tp+fp+fn,1))
    return np.mean(fs,0)
def load(y,v):
    base=None; P={}
    for k,d in M.items():
        for s in S:
            x=pd.read_csv(f'{d}/seed{s}_{y}_{v}.csv').sort_values('rid').reset_index(drop=True)
            if base is None: base=x[['rid','y']]
            P[(k,s)]=x
    return base,P
def run(years, v, mode):
    F={k:np.zeros((len(years),B)) for k in M}; Pt={k:np.zeros(len(years)) for k in M}
    for i,y in enumerate(years):
        base,P=load(y,v); Y=base.y.values
        if mode=='binary':
            keep=Y!=1; Yb=Y[keep]; idx=rng.integers(0,len(Yb),(B,len(Yb)))
            for k in M:
                preds=[np.where(P[(k,s)].p_neg.values[keep]>P[(k,s)].p_pos.values[keep],0,2) for s in S]
                F[k][i]=np.mean([f1m(Yb[idx],p[idx],[0,2]) for p in preds],0); Pt[k][i]=np.mean([f1m(Yb,p,[0,2]) for p in preds])
        else:
            idx=rng.integers(0,len(Y),(B,len(Y)))
            for k in M:
                preds=[P[(k,s)].pred.values for s in S]
                F[k][i]=np.mean([f1m(Y[idx],p[idx],[0,1,2]) for p in preds],0); Pt[k][i]=np.mean([f1m(Y,p,[0,1,2]) for p in preds])
    sl={k:np.polyfit(years,F[k],1)[0] for k in M}; pt={k:np.polyfit(years,Pt[k],1)[0] for k in M}
    out={}
    for k in M: out[k]=(Pt[k][0],Pt[k][-1],pt[k],np.percentile(sl[k],2.5),np.percentile(sl[k],97.5),pt[k]/Pt[k][0]*100)
    d=sl['RoBERTa-base']-sl['TF-IDF 200k']; out['diff RoBERTa-TF200']=(pt['RoBERTa-base']-pt['TF-IDF 200k'],np.percentile(d,2.5),np.percentile(d,97.5))
    return out
for title,years,v,mode in [('A. 2017-2022, macro-F1 (3 classes)',list(range(2017,2023)),'matched','macro'),
                           ('A. 2017-2022, natural',list(range(2017,2023)),'natural','macro'),
                           ('B. binary (neg vs pos, neutral items removed), 2017-2023',list(range(2017,2024)),'matched','binary'),
                           ('B. binary, natural, 2017-2023',list(range(2017,2024)),'natural','binary')]:
    print('===',title)
    for k,v_ in run(years,v,mode).items():
        if k.startswith('diff'): print(f'  {k}: {v_[0]:+.4f} [{v_[1]:+.4f},{v_[2]:+.4f}]')
        else: print(f'  {k:14s} first {v_[0]:.4f} last {v_[1]:.4f} slope {v_[2]:+.4f} [{v_[3]:+.4f},{v_[4]:+.4f}] rel {v_[5]:+.2f}%/yr')
