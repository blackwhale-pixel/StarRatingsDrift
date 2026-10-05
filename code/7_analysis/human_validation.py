"""Human validation analysis (Section 4.4): human vs LLM agreement, three-star negative share by period,
sensitivity checks. Runs on the public files in data/annotation/ (no review text needed)."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint
H = pd.read_csv('data/annotation/human_validation_labels.csv')
L = pd.read_csv('data/annotation/annotation_labels.csv').rename(columns={'item_id': 'annotation_item_id'})
D = H.merge(L[['annotation_item_id', 'majority']], on='annotation_item_id')
ok = D[D.majority != 'NO_MAJORITY']
print('human vs LLM majority: n', len(ok), 'agreement', round((ok.majority == ok.adjudicated).mean(), 3),
      'kappa', round(cohen_kappa_score(ok.adjudicated, ok.majority), 3))
for p in ('early', 'late'):
    x = ok[ok.period == p]; print(f'  {p}: kappa {cohen_kappa_score(x.adjudicated, x.majority):.3f}')
print('annotator 1 vs 2 kappa', round(cohen_kappa_score(D.annotator_1, D.annotator_2), 3))
t = D[D.rating_star == 3]
def trend(df, col, name):
    y = (df[col] == 'NEGATIVE').astype(int); late = (df.period == 'late').astype(int)
    m = smf.logit('y ~ late', data=pd.DataFrame({'y': y, 'late': late})).fit(disp=0); ci = np.exp(m.conf_int().loc['late'])
    sh = [y[df.period == p].mean() for p in ('early', 'late')]
    print(f'  {name:32s} n={len(df):3d} early {sh[0]:.3f} late {sh[1]:.3f} OR {np.exp(m.params.late):.2f} [{ci[0]:.2f}, {ci[1]:.2f}]')
print('three-star negative share by period:')
for col, name in (('adjudicated', 'human, adjudicated'), ('majority', 'LLM majority'), ('annotator_1', 'annotator 1'), ('annotator_2', 'annotator 2')):
    trend(t, col, name)
trend(t[t.annotator_1 == t.annotator_2], 'adjudicated', 'both annotators agree')
trend(t[t.annotator_2_uncertain != 'Y'], 'adjudicated', 'excluding annotator-2 uncertain')
