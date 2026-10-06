"""Length control for the three-star (and four-star) negative-share trend (Section 3.7, Section 4.4, Table 6).
Inputs (all in this repository, no review text):
  data/annotation/annotation_key_with_category.csv, annotation_labels.csv, annotation_item_attributes.csv,
  human_validation_labels.csv
Run from the repository root:  python code/7_analysis/length_control.py
"""
import pandas as pd, numpy as np, statsmodels.formula.api as smf
A='data/annotation/'
k=pd.read_csv(A+'annotation_key_with_category.csv').set_index('item_id')
L=pd.read_csv(A+'annotation_labels.csv').set_index('item_id')
at=pd.read_csv(A+'annotation_item_attributes.csv').set_index('item_id')
d=k.join(L[['majority']],how='inner').join(at[['text_len']])
d['yc']=d.yr-2013; d['loglen']=np.log(d.text_len)
d['ver']=d.verified_purchase.astype(str).str.lower().eq('true').astype(int)
d['neg']=(d.majority=='NEGATIVE').astype(int)
def prep(star):
    t=d[d.rating_star==star].copy()
    top=t.cat_l1.value_counts(); keep=top[top>=30].index
    t['cat']=np.where(t.cat_l1.isin(keep),t.cat_l1,'(other)'); return t
def orr(m,v='yc'):
    ci=m.conf_int().loc[v]; return f"OR={np.exp(m.params[v]):.3f} [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}] p={m.pvalues[v]:.2g}"
t=prep(3); print('three-star n =',len(t))
for f,lab in [('neg ~ yc + C(cat)','category FE'),
              ('neg ~ yc + C(cat) + loglen','+ log length'),
              ('neg ~ yc + C(cat) + loglen + ver + np.log1p(helpful_vote)','+ log length, verified, helpful')]:
    m=smf.logit(f,data=t).fit(disp=0); print(f'  {lab:32s} per year {orr(m)}' + (f" | log length p={m.pvalues['loglen']:.2g}" if 'loglen' in f else ''))
t['tert']=pd.qcut(t.text_len,3,labels=['short','medium','long'])
for g,x in t.groupby('tert',observed=True):
    m=smf.logit('neg ~ yc + C(cat)',data=x).fit(disp=0); print(f'  tertile {g:6s} n={len(x)} median length={x.text_len.median():.0f} per year {orr(m)}')
t4=prep(4); m=smf.logit('neg ~ yc + C(cat) + loglen',data=t4).fit(disp=0); print('four-star + log length per year',orr(m))
# human validation sample: period odds ratio with and without log length
h=pd.read_csv(A+'human_validation_labels.csv'); h=h[h.rating_star==3].copy()
h=h.join(at[['text_len']],on='annotation_item_id'); h['late']=(h.period=='late').astype(int)
h['neg']=(h.adjudicated=='NEGATIVE').astype(int); h['loglen']=np.log(h.text_len)
print('human three-star median length early/late', h.groupby('period').text_len.median().to_dict())
for f in ['neg ~ late','neg ~ late + loglen']:
    m=smf.logit(f,data=h).fit(disp=0); print(f'  human {f:22s} {orr(m,"late")}')
