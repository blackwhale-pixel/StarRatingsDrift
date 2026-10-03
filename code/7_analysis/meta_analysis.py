import pandas as pd, numpy as np, statsmodels.formula.api as smf
a=pd.read_csv('mo/annotation_key_meta.csv'); L=pd.read_csv('ar/annotation_results/annotation_items_blind_labels.csv',index_col=0)
d=a.set_index('item_id').join(L[['majority']],how='inner')
t=d[d.rating_star==3].copy()
top=t.cat_l1.value_counts(); keep=top[top>=30].index
t['cat']=np.where(t.cat_l1.isin(keep),t.cat_l1,'(other)')
t['neg']=(t.majority=='NEGATIVE').astype(int); t['mix']=(t.majority=='MIXED').astype(int)
t['yc']=t.yr-2013; t['ver']=t.verified_purchase.astype(str).str.lower().eq('true').astype(int)
print('n 3-star',len(t),'cats',t.cat.nunique())
for y in ['neg','mix']:
  for f,lab in [(f'{y} ~ yc','no controls'),(f'{y} ~ yc + C(cat)','+category FE'),(f'{y} ~ yc + C(cat) + ver + np.log1p(helpful_vote)','+cat+verified+helpful')]:
    m=smf.logit(f,data=t).fit(disp=0); lp=smf.ols(f,data=t).fit(cov_type='HC1')
    ci=m.conf_int().loc['yc']
    print(f"{y} {lab:24s} logit b={m.params['yc']:.4f} OR/yr={np.exp(m.params['yc']):.3f} [{np.exp(ci[0]):.3f},{np.exp(ci[1]):.3f}] p={m.pvalues['yc']:.1e} | LPM {lp.params['yc']*100:.2f}pp/yr p={lp.pvalues['yc']:.1e}")
# drift reweighting: per-category F1 with fixed 2017 composition
tm=pd.read_csv('mo/test_items_meta.csv'); tm=tm[tm.test_set.str.endswith('matched')]
TR='/home/claude/e/transformer_out/preds'; TF='/home/claude/tp/preds'
def f1(y,p):
    return np.mean([2*((y==c)&(p==c)).sum()/max(2*((y==c)&(p==c)).sum()+((y!=c)&(p==c)).sum()+((y==c)&(p!=c)).sum(),1) for c in range(3)])
yrs=range(2017,2024); res={}
ref=tm[tm.test_set=='2017_matched'].cat_l1.fillna('(none)').value_counts(normalize=True)
for m,dd in [('TR',TR),('TF',TF)]:
  raw=[];rw=[]
  for y in yrs:
    meta=tm[tm.test_set==f'{y}_matched'][['rid','cat_l1']]
    vals=[];vals_rw=[]
    for s in [42,43,44]:
      x=pd.read_csv(f'{dd}/seed{s}_{y}_matched.csv').merge(meta,on='rid'); x['cat_l1']=x.cat_l1.fillna('(none)')
      vals.append(f1(x.y.values,x.pred.values))
      # reweight items to 2017 category composition
      cur=x.cat_l1.value_counts(normalize=True); w=x.cat_l1.map(lambda c: ref.get(c,0)/cur[c]).values
      # weighted macro F1
      fs=[]
      for c in range(3):
        tp=(w*((x.y==c)&(x.pred==c))).sum(); fp=(w*((x.y!=c)&(x.pred==c))).sum(); fn=(w*((x.y==c)&(x.pred!=c))).sum(); fs.append(2*tp/(2*tp+fp+fn))
      vals_rw.append(np.mean(fs))
    raw.append(np.mean(vals)); rw.append(np.mean(vals_rw))
  res[m]=(raw,rw)
  print(m,'raw slope',round(np.polyfit(list(yrs),raw,1)[0],4),'reweighted slope',round(np.polyfit(list(yrs),rw,1)[0],4), [round(v,3) for v in rw])
# composition dissimilarity
c=pd.read_csv('mo/category_by_year.csv').pivot(index='cat',columns='yr',values='share').fillna(0)
for y0,y1 in [(2013,2022),(2016,2022),(2017,2023)]: print('dissimilarity',y0,y1,round(0.5*abs(c[y0]-c[y1]).sum(),3))
