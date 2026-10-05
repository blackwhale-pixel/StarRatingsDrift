import pandas as pd, numpy as np, statsmodels.formula.api as smf
from scipy import stats
# ---------- (1)(2) annotation: merged neutral+mixed, no-majority handling ----------
a=pd.read_csv('mo/annotation_key_meta.csv'); L=pd.read_csv('ar/annotation_results/annotation_items_blind_labels.csv',index_col=0)
d=a.set_index('item_id').join(L[['majority']],how='inner'); t=d[d.rating_star==3].copy()
keep=t.cat_l1.value_counts(); keep=keep[keep>=30].index; t['cat']=np.where(t.cat_l1.isin(keep),t.cat_l1,'(other)'); t['yc']=t.yr-2013
t['neg']=(t.majority=='NEGATIVE').astype(int); t['nm']=t.majority.isin(['NEUTRAL','MIXED']).astype(int); t['neu']=(t.majority=='NEUTRAL').astype(int)
print('yearly n:',t.yr.value_counts().sort_index().to_dict()); print('no-majority/unclear:',t.majority.isin(['NO_MAJORITY','UNCLEAR']).sum())
def lg(y,df,f='{y} ~ yc + C(cat)'):
    m=smf.logit(f.format(y=y),data=df).fit(disp=0); ci=np.exp(m.conf_int().loc['yc']); return np.exp(m.params['yc']),ci[0],ci[1],m.pvalues['yc']
for y in ['nm','neu','neg']:
    print(f'{y}: share early {t[t.yr<=2016][y].mean():.3f} late {t[t.yr.between(2019,2022)][y].mean():.3f} | OR (cat FE) %.3f [%.3f, %.3f] p=%.1e' % lg(y,t))
tt=t[~t.majority.isin(['NO_MAJORITY','UNCLEAR'])]
print('excluding no-majority/unclear n=',len(tt),'neg OR %.3f [%.3f, %.3f] p=%.1e' % lg('neg',tt))
# ---------- (3)(4) per-seed slopes and cluster bootstrap ----------
M={'RoBERTa-base':'/home/claude/rb/roberta_out/preds','DistilRoBERTa':'/home/claude/e/transformer_out/preds','TF-IDF 200k':'/home/claude/t2k/preds','TF-IDF 500k':'/home/claude/tp/preds'}
rm=pd.read_csv('/home/claude/pid/rid_map.csv',usecols=['rid','user_id','parent_asin']).set_index('rid')
yrs=list(range(2017,2024)); S=[42,43,44]
def wf1(y,p,w):
    fs=[]
    for c in range(3):
        tp=((y==c)&(p==c))@w.T; fp=((y!=c)&(p==c))@w.T; fn=((y==c)&(p!=c))@w.T; fs.append(2*tp/np.maximum(2*tp+fp+fn,1e-9))
    return np.mean(fs,0)
data={}
for y in yrs:
    base=None
    for k,dd in M.items():
        for s in S:
            x=pd.read_csv(f'{dd}/seed{s}_{y}_matched.csv').sort_values('rid').reset_index(drop=True)
            if base is None: base=x[['rid','y']].join(rm,on='rid')
            data[(y,k,s)]=x.pred.values
    data[y]=base
print('\nper-seed slopes (prior-matched, 2017-2023):')
for k in M:
    sl=[np.polyfit(yrs,[wf1(data[y].y.values,data[(y,k,s)],np.ones((1,len(data[y]))))[0] for y in yrs],1)[0] for s in S]
    print(f'  {k:14s}', ' '.join(f'{v:+.4f}' for v in sl), f'| mean {np.mean(sl):+.4f} sd {np.std(sl,ddof=1):.4f}')
B=500; rng=np.random.default_rng(5)
for clus in ['parent_asin','user_id']:
    F={k:np.zeros((len(yrs),B)) for k in M}
    for i,y in enumerate(yrs):
        base=data[y]; codes,uniq=pd.factorize(base[clus]); nC=len(uniq)
        W=np.zeros((B,len(base)))
        for b in range(B):
            cw=np.bincount(rng.integers(0,nC,nC),minlength=nC); W[b]=cw[codes]
        Y=base.y.values
        for k in M: F[k][i]=np.mean([wf1(Y,data[(y,k,s)],W) for s in S],0)
        print(f'  {clus} {y}: items {len(base)}, clusters {nC}', flush=True) if i==0 else None
    sl={k:np.polyfit(yrs,F[k],1)[0] for k in M}
    print(f'\ncluster bootstrap by {clus}:')
    for k in M: print(f'  {k:14s} 95% CI [{np.percentile(sl[k],2.5):+.4f}, {np.percentile(sl[k],97.5):+.4f}]')
    for a_,b_ in [('RoBERTa-base','TF-IDF 200k'),('RoBERTa-base','DistilRoBERTa')]:
        dd=sl[a_]-sl[b_]; print(f'  diff {a_} - {b_}: [{np.percentile(dd,2.5):+.4f}, {np.percentile(dd,97.5):+.4f}]')
