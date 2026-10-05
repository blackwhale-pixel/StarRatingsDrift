"""Figures for the manuscript. Reads per-item prediction files and annotation outputs; writes PNG (300 dpi) and PDF."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
OUT = "/mnt/user-data/outputs/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False})
YRS = list(range(2017, 2024)); SEEDS = [42, 43, 44]
M = {"RoBERTa-base": ("/home/claude/rb/roberta_out/preds", "#1b4f72", "o"),
     "DistilRoBERTa": ("/home/claude/e/transformer_out/preds", "#5dade2", "s"),
     "TF-IDF + LR (200k)": ("/home/claude/t2k/preds", "#c0392b", "^")}
def f1s(y, p):
    out = []
    for c in range(3):
        tp = ((y == c) & (p == c)).sum(-1); fp = ((y != c) & (p == c)).sum(-1); fn = ((y == c) & (p != c)).sum(-1)
        out.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
    return out
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Figure 1: workflow ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec="#34495e", lw=0.8))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.6, weight="bold")
    ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=6.2, linespacing=1.3)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=0.8))
# row 1: data preparation
box(2, 48, 28, 14, "1. Raw data", "Amazon Reviews 2023 (Electronics)\nreview and metadata files\n43.9M reviews", "#eaf2f8")
box(36, 48, 28, 14, "2. Audit and cleaning", "duplicate and leakage audit\n478,014 redundant rows removed\nyears 2013-2023 kept", "#eaf2f8")
box(70, 48, 28, 14, "3. Analysis pool", "41.2M reviews\nlabels: 1-2 neg, 3 neu, 4-5 pos", "#eaf2f8")
arrow(30.3, 55, 35.7, 55); arrow(64.3, 55, 69.7, 55)
# row 2: three analyses
box(2, 25, 28, 16, "4a. Split experiment", "random, product, user, temporal\nstage A: 500k-review samples\nstage B: full training portion", "#fef5e7")
box(36, 25, 28, 16, "4b. Temporal evaluation", "train 2013-2016, test 2017-2023\nTF-IDF+LR, DistilRoBERTa,\nRoBERTa-base; 3 seeds", "#fdedec")
box(70, 25, 28, 16, "4c. Rating-text analysis", "3,014 reviews, 3 LLM annotators\nSemEval-2014 external check\n400 reviews checked by humans", "#e8f8f5")
for x in (16, 50, 84): arrow(84, 47.7, x, 41.3)
# row 3: follow-up tests
box(36, 2, 28, 16, "5. Zero-shot test", "Mistral 7B, no task-specific\nfine-tuning; 2,000 items/year\npaired with 4b, same items", "#f5eef8")
box(70, 2, 28, 16, "6. Robustness checks", "reweighting, equal training size\nexcluding 2023, binary task\nretraining on 2019-2020", "#f4f6f6")
arrow(50, 24.7, 50, 18.3); arrow(84, 24.7, 84, 18.3); arrow(64.3, 33, 69.7, 10)
save(fig, "Figure1_workflow")

# ---------- Figure 2: yearly decline (prior-matched) ----------
B = 1000; rng = np.random.default_rng(11)
res = {k: {"macro": [], "neu": [], "lo": [], "hi": [], "nlo": [], "nhi": []} for k in M}
for y in YRS:
    base = None; P = {}
    for k, (d, _, _) in M.items():
        for s in SEEDS:
            x = pd.read_csv(f"{d}/seed{s}_{y}_matched.csv").sort_values("rid").reset_index(drop=True)
            base = x[["rid", "y"]] if base is None else base; P[(k, s)] = x.pred.values
    Y = base.y.values; idx = rng.integers(0, len(Y), (B, len(Y)))
    for k in M:
        point = [f1s(Y, P[(k, s)]) for s in SEEDS]; boot = [f1s(Y[idx], P[(k, s)][idx]) for s in SEEDS]
        mac = np.mean([np.mean(f) for f in point]); neu = np.mean([f[1] for f in point])
        bm = np.mean([np.mean(b, 0) for b in boot], 0); bn = np.mean([b[1] for b in boot], 0)
        r = res[k]; r["macro"].append(mac); r["neu"].append(neu)
        r["lo"].append(np.percentile(bm, 2.5)); r["hi"].append(np.percentile(bm, 97.5))
        r["nlo"].append(np.percentile(bn, 2.5)); r["nhi"].append(np.percentile(bn, 97.5))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, key, lo, hi, title, ylab in [(axes[0], "macro", "lo", "hi", "(a) Macro-F1", "Macro-F1"),
                                     (axes[1], "neu", "nlo", "nhi", "(b) Neutral-class F1", "F1 (3-star class)")]:
    for k, (_, col, mk) in M.items():
        r = res[k]; ax.fill_between(YRS, r[lo], r[hi], color=col, alpha=0.15, lw=0)
        sl = np.polyfit(YRS, r[key], 1)[0]
        ax.plot(YRS, r[key], marker=mk, ms=3.5, color=col, lw=1.4, label=f"{k}  ({sl:+.4f}/yr)")
    ax.set_title(title, loc="left"); ax.set_ylabel(ylab); ax.set_xticks(YRS); ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.legend(fontsize=6.8, loc="upper right")
fig.text(0.5, -0.02, "Test year (prior-matched sets; models trained on 2013-2016; seed means with 95% bootstrap bands)", ha="center", fontsize=7.5)
fig.tight_layout(); save(fig, "Figure2_temporal_decline")
pd.DataFrame({f"{k}|{m}": res[k][m] for k in M for m in ["macro", "lo", "hi", "neu"]}, index=YRS).round(4).to_csv(f"{OUT}/Figure2_data.csv")

# ---------- Figure 3: three-star text composition ----------
L = pd.read_csv("/home/claude/ar/annotation_results/annotation_items_blind_labels.csv", index_col=0)
K = pd.read_csv("/home/claude/f/annotation_key.csv").set_index("item_id"); D = L.join(K, how="inner")
t = D[D.rating_star == 3]
comp = pd.crosstab(t.yr, t.majority, normalize="index")
order = [("NEGATIVE", "#c0392b", "Negative"), ("MIXED", "#e67e22", "Mixed"), ("NEUTRAL", "#95a5a6", "Neutral"), ("POSITIVE", "#27ae60", "Positive")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]; bottom = np.zeros(len(comp))
for lab, col, name in order:
    v = comp.get(lab, pd.Series(0, index=comp.index)).values; ax.bar(comp.index, v, bottom=bottom, color=col, width=0.75, label=name); bottom += v
other = 1 - bottom; ax.bar(comp.index, other, bottom=bottom, color="#ecf0f1", width=0.75, label="No majority / unclear")
ax.set_ylim(0, 1); ax.set_ylabel("Share of 3-star reviews"); ax.set_title("(a) Majority label of 3-star review text", loc="left")
ax.set_xticks(comp.index); ax.set_xticklabels(comp.index, rotation=45, fontsize=7); ax.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax = axes[1]
for star, col, mk in [(2, "#7b241c", "s"), (3, "#c0392b", "o"), (4, "#f1948a", "^")]:
    s = D[D.rating_star == star]; g = s.groupby("yr").majority.apply(lambda v: (v == "NEGATIVE").mean())
    r = stats.linregress(s.yr, (s.majority == "NEGATIVE").astype(int))
    ax.plot(g.index, g.values, marker=mk, ms=3.5, lw=0.9, color=col, label=f"{star}-star ({r.slope*100:+.1f} pp/yr, p={r.pvalue:.1g})")
ax.set_ylim(0, 1); ax.set_ylabel("Share labeled negative"); ax.set_title("(b) Negative text by star rating", loc="left")
ax.set_xticks(list(range(2013, 2024, 2))); ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, 0.44)); ax.grid(axis="y", lw=0.3, alpha=0.6)
fig.tight_layout(); save(fig, "Figure3_three_star_composition")
comp.round(4).to_csv(f"{OUT}/Figure3_data.csv")

# ---------- Figure 4: zero-shot vs trained on the same items ----------
z = pd.read_json("/home/claude/zr/zs_runs/zs__mistral.jsonl", lines=True); z = z[z.label.notna()].drop_duplicates("rid", keep="last")
z["zp"] = z.label.map({"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2})
ser = {"Zero-shot Mistral 7B (no fine-tuning)": ("#7d3c98", "D")}; ser.update({k: (v[1], v[2]) for k, v in M.items()})
vals = {k: [] for k in ser}; predneg = []
for y in YRS:
    zz = z[z["sample"] == f"{y}_matched"][["rid", "zp"]]
    base = pd.read_csv(f"{M['DistilRoBERTa'][0]}/seed42_{y}_matched.csv")[["rid", "y"]].merge(zz, on="rid")
    Y = base.y.values; vals["Zero-shot Mistral 7B (no fine-tuning)"].append(np.mean(f1s(Y, base.zp.values))); predneg.append((base.zp == 0).mean())
    for k, (d, _, _) in M.items():
        vals[k].append(np.mean([np.mean(f1s(Y, pd.read_csv(f"{d}/seed{s}_{y}_matched.csv").set_index("rid").loc[base.rid, "pred"].values)) for s in SEEDS]))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
for k, (col, mk) in ser.items():
    sl = np.polyfit(YRS, vals[k], 1)[0]; ax.plot(YRS, vals[k], marker=mk, ms=3.5, lw=1.3 if "Zero" in k else 1.0, color=col,
                                                ls="-" if "Zero" in k else "--", label=f"{k}  ({sl:+.4f}/yr)")
ax.set_title("(a) Same 2,000 items per year", loc="left"); ax.set_ylabel("Macro-F1 vs. star labels"); ax.set_xticks(YRS)
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); ax.grid(axis="y", lw=0.3, alpha=0.6)
ax = axes[1]; ax.bar(YRS, np.array(predneg) * 100, color="#7d3c98", width=0.65)
ax.axhline(14.6, color="#34495e", lw=0.8, ls=":"); ax.text(2016.6, 15.4, "negative share of star labels (14.6%)", fontsize=6.3)
ax.set_ylim(0, 25); ax.set_ylabel("Predicted negative (%)"); ax.set_title("(b) Zero-shot negative predictions", loc="left"); ax.set_xticks(YRS)
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout(); save(fig, "Figure4_zero_shot")
print("done")
