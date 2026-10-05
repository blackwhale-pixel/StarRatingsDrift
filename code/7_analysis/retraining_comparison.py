import pandas as pd, numpy as np
OLD={'TF-IDF 200k':'/home/claude/t2k/preds','DistilRoBERTa':'/home/claude/e/transformer_out/preds','RoBERTa-base':'/home/claude/rb/roberta_out/preds'}
NEW={'TF-IDF 200k':'/home/claude/rt/tfidf200k_recent_out/tfidf200k_recent_out/preds','DistilRoBERTa':'/home/claude/rt/distil_recent_out/distil_recent_out/preds','RoBERTa-base':'/home/claude/rt/roberta_recent_out/roberta_recent_out/preds'}
S=[42,43,44]; B=1000; rng=np.random.default_rng(77)
def f1(y,p,cls=None):
    fs=[]
    for c in range(3):
        tp=((y==c)&(p==c)).sum(-1); fp=((y!=c)&(p==c)).sum(-1); fn=((y==c)&(p!=c)).sum(-1); fs.append(2*tp/np.maximum(2*tp+fp+fn,1))
    return fs[cls] if cls is not None else np.mean(fs,0)
def load(d,s,ts):
    return pd.read_csv(f'{d}/seed{s}_{ts}.csv').sort_values('rid').reset_index(drop=True)
for v in ['matched','natural']:
    print(f'\n===== {v} test sets, 2022 and 2023 =====')
    res={}
    for yr in (2022,2023):
        ts=f'{yr}_{v}'; base=load(OLD['TF-IDF 200k'],42,ts)[['rid','y']]; Y=base.y.values; idx=rng.integers(0,len(Y),(B,len(Y)))
        for k in OLD:
            for tag,src in (('old',OLD),('new',NEW)):
                P=[load(src[k],s,ts) for s in S]; assert all((p.rid.values==base.rid.values).all() for p in P)
                for cls,cn in ((None,'macro'),(1,'neu'),(0,'neg'),(2,'pos')):
                    res.setdefault((k,tag,cn,'pt'),[]).append(np.mean([f1(Y,p.pred.values,cls) for p in P]))
                    res.setdefault((k,tag,cn,'bs'),[]).append(np.mean([f1(Y[idx],p.pred.values[idx],cls) for p in P],0))
    for cn in ('macro','neu','neg','pos'):
        print(f'-- {cn} F1 (mean of 2022, 2023)')
        for k in OLD:
            o=np.mean(res[(k,'old',cn,'pt')]); n=np.mean(res[(k,'new',cn,'pt')])
            d=np.mean(res[(k,'new',cn,'bs')],0)-np.mean(res[(k,'old',cn,'bs')],0)
            print(f'   {k:14s} old {o:.4f} new {n:.4f} gain {n-o:+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]')
    # compare gains: retraining vs model size vs pretrained (old models)
    g=lambda k,tag: np.mean(res[(k,tag,'macro','bs')],0)
    rt=g('DistilRoBERTa','new')-g('DistilRoBERTa','old'); sz=g('RoBERTa-base','old')-g('DistilRoBERTa','old')
    print(f'   retrain gain (Distil) {np.mean(np.mean(res[("DistilRoBERTa","new","macro","pt")]))-np.mean(res[("DistilRoBERTa","old","macro","pt")]):+.4f} vs size gain (RoBERTa-Distil, old) {np.mean(res[("RoBERTa-base","old","macro","pt")])-np.mean(res[("DistilRoBERTa","old","macro","pt")]):+.4f}; diff [{np.percentile(rt-sz,2.5):+.4f},{np.percentile(rt-sz,97.5):+.4f}]')
    nt=g('TF-IDF 200k','new')-g('RoBERTa-base','old')
    print(f'   TF-IDF retrained - RoBERTa old: {np.mean(res[("TF-IDF 200k","new","macro","pt")])-np.mean(res[("RoBERTa-base","old","macro","pt")]):+.4f} [{np.percentile(nt,2.5):+.4f},{np.percentile(nt,97.5):+.4f}]')
    for k in OLD:
        print(f'   {k}: old 2017 ref -> see main; new by year 2022 {res[(k,"new","macro","pt")][0]:.4f} 2023 {res[(k,"new","macro","pt")][1]:.4f}')
# how much of the 2017->2022/23 decline is recovered (matched)
print('\n== recovery of decline (matched): old 2017 vs old 2022-23 vs new 2022-23')
for k in OLD:
    o17=np.mean([f1(load(OLD[k],s,'2017_matched').y.values,load(OLD[k],s,'2017_matched').pred.values) for s in S])
    o2=np.mean([np.mean([f1(load(OLD[k],s,f'{y}_matched').y.values,load(OLD[k],s,f'{y}_matched').pred.values) for s in S]) for y in (2022,2023)])
    n2=np.mean([np.mean([f1(load(NEW[k],s,f'{y}_matched').y.values,load(NEW[k],s,f'{y}_matched').pred.values) for s in S]) for y in (2022,2023)])
    print(f'   {k:14s} old@2017 {o17:.4f} old@22-23 {o2:.4f} new@22-23 {n2:.4f} | loss {o2-o17:+.4f}, recovered {(n2-o2)/(o17-o2)*100:.0f}%')
