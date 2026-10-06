"""Shift-share decomposition of neutral-class recall on three-star reviews (Section 4.9, Table 10, Table B6).
Models trained on 2013-2016 reviews (TF-IDF 200k and DistilRoBERTa, seeds 42-44) were applied to the
LLM-annotated reviews; predictions are in results/4_annotation/annotation_preds/.
Inputs (all in this repository, no review text). Run from the repository root:
    python code/7_analysis/shift_share_decomposition.py
Output: results/4_annotation/shift_share_results.csv
"""
import pandas as pd, numpy as np
A='data/annotation/'; P='results/4_annotation/annotation_preds/'
k=pd.read_csv(A+'annotation_key.csv').set_index('item_id')
at=pd.read_csv(A+'annotation_item_attributes.csv').set_index('item_id')
L=pd.read_csv(A+'annotation_labels.csv').set_index('item_id')
H=pd.read_csv(A+'human_validation_labels.csv').set_index('annotation_item_id')
d=k.join(at).join(L[['majority']]).join(H[['adjudicated']])
d['validated']=d.index.isin(H.index)
K=['NEGATIVE','MIXED','NEUTRAL','POSITIVE','OTHER']          # no majority / unclear -> OTHER (not renormalized)
lab=lambda s: s if s in K[:4] else 'OTHER'
models={'TF-IDF+LR (200k)':'tfidf_200k','DistilRoBERTa':'distilroberta'}
for m,f in models.items():
    ps=[pd.read_csv(f'{P}{f}/seed{s}.csv').set_index('item_id') for s in (42,43,44)]
    for c,j in [('neg',0),('neu',1),('pos',2)]:
        d[f'{m}_{c}']=sum((p.pred==j).astype(float) for p in ps)/3   # share of seeds predicting class j
base=d[(d.rating_star==3)&d.english&~d.in_train_2013_2016_500k].copy()
base['per']=np.select([base.yr<=2016, base.yr.between(2019,2022)],['early','late'],None)
base=base[base.per.notna()]
def decomp(df,m):
    out=[]
    for per in ('early','late'):
        x=df[df.per==per]
        w=np.array([(x.k==c).mean() for c in K])
        r=np.array([x.loc[x.k==c,m+'_neu'].mean() if (x.k==c).any() else 0 for c in K])
        out.append((w,r,x[m+'_neu'].mean()))
    (we,re_,Re),(wl,rl,Rl)=out
    return dict(R_e=Re,R_l=Rl,total=Rl-Re,comp=((wl-we)*re_).sum(),within=(we*(rl-re_)).sum(),
                inter=((wl-we)*(rl-re_)).sum()),we,wl,re_,rl
rng=np.random.default_rng(11); rows=[]
for src,col in [('LLM majority','majority'),('Human adjudicated','adjudicated')]:
    df=base[base.validated].copy() if src.startswith('Human') else base.copy()
    df['k']=df[col].map(lab)
    e,l=df[df.per=='early'],df[df.per=='late']
    for m in models:
        est,we,wl,re_,rl=decomp(df,m)
        bs=pd.DataFrame([decomp(pd.concat([e.sample(len(e),replace=True,random_state=rng.integers(1e9)),
                                            l.sample(len(l),replace=True,random_state=rng.integers(1e9))]),m)[0]
                         for _ in range(1000)])
        r={'labels':src,'model':m,'n_early':len(e),'n_late':len(l),
           'other_early':int((e.k=='OTHER').sum()),'other_late':int((l.k=='OTHER').sum())}
        for t in ['R_e','R_l','total','comp','within','inter']:
            r[t]=est[t]; r[t+'_lo']=bs[t].quantile(.025); r[t+'_hi']=bs[t].quantile(.975)
        r['sum_minus_total']=est['comp']+est['within']+est['inter']-est['total']
        for i,c in enumerate(K): r[f'w_e_{c}'],r[f'w_l_{c}'],r[f'r_e_{c}'],r[f'r_l_{c}']=we[i],wl[i],re_[i],rl[i]
        for per,x in (('early',e),('late',l)):          # where negative-text three-star predictions go
            s=x[x.k=='NEGATIVE']; r[f'negtext_n_{per}']=len(s)
            for c in ('neg','neu','pos'): r[f'negtext_pred_{c}_{per}']=s[f'{m}_{c}'].mean()
        rows.append(r)
out=pd.DataFrame(rows); out.to_csv('results/4_annotation/shift_share_results.csv',index=False)
cols=['labels','model','n_early','n_late','R_e','R_l','total','comp','within','inter','sum_minus_total']
print(out[cols].round(3).to_string(index=False))
print(out[['labels','model']+[c for c in out.columns if c.startswith('negtext')]].round(3).T)
