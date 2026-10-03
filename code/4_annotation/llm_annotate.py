#!/usr/bin/env python3
"""
三星評論 LLM 盲標註流程（多模型、外部基準驗證、一致性分析）

子指令：
  annotate      用指定模型標註一份 CSV（欄位 item_id, text），結果寫入 JSONL，可中斷續跑
  prep-semeval  把 SemEval-2014 Task 4 的 XML 轉成句子層級的標註題目與標準答案
  evaluate      彙整多個模型的標註：模型間一致性、多數決、依星數與年份的分布、SemEval 準確度

範例（PowerShell）：
  $env:ANTHROPIC_API_KEY="..."
  python llm_annotate.py annotate --items annotation_items_blind.csv --backend anthropic --model claude-sonnet-5-5
  python llm_annotate.py annotate --items annotation_items_blind.csv --backend openai --model <OpenAI 模型名稱>
  python llm_annotate.py prep-semeval --xml Laptop_Train_v2.xml Laptops_Test_Gold.xml --out semeval_laptop
  python llm_annotate.py evaluate --runs runs --key annotation_key.csv --semeval_gold semeval_laptop_gold.csv
需求：pip install anthropic openai pandas scikit-learn
"""
import argparse, os, re, sys, json, time, hashlib, threading, datetime
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, numpy as np

LABELS = ["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNCLEAR"]

CODEBOOK = """You are an expert annotator of customer product reviews. Classify the OVERALL sentiment the reviewer expresses in the review into exactly one label. You will NOT see the star rating; judge only from the text.

LABELS
POSITIVE - The evaluations are predominantly favorable. Negatives are absent, or are minor and explicitly downplayed by the writer (e.g., "only complaint is the short cable, but still love it").
NEGATIVE - The evaluations are predominantly unfavorable. Positives are absent, or minor and clearly outweighed (e.g., "looks nice but stopped working after a week, avoid").
MIXED - The review contains at least one substantive positive evaluation AND at least one substantive negative evaluation, and neither clearly dominates (e.g., "sound quality is excellent, but the Bluetooth keeps disconnecting").
NEUTRAL - The review expresses no clear evaluative stance (factual description, usage notes, a question, setup instructions), OR only a lukewarm, middling overall judgment with no substantive praise or complaint (e.g., "it's okay", "average, does the job").
UNCLEAR - The text cannot be judged: unintelligible, off-topic, not about the purchase, sarcasm whose direction cannot be determined, or not in English.

RULES
1. Evaluations of the product, its features, price/value, durability, seller, shipping, and packaging all count as evaluations.
2. Decide by substance, not by counting sentences. A single serious defect (e.g., failure, safety issue) is substantive.
3. MIXED requires both sides to be substantive. A trivial caveat does not make a review MIXED.
4. Do not infer a star rating; do not reward politeness or length.

OUTPUT
Return only a JSON object, no other text:
{"label": "<one of POSITIVE, NEGATIVE, MIXED, NEUTRAL, UNCLEAR>", "reason": "<at most 20 words>"}"""
PROMPT_SHA = hashlib.sha256(CODEBOOK.encode()).hexdigest()[:12]

# ---------------- 呼叫模型 ----------------
PRESETS = {"openai": (None, "OPENAI_API_KEY"),
           "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY")}

def make_caller(backend, model):
    if backend == "mock":   # 測試用：不呼叫任何 API
        def call(text):
            t = text.lower(); p = any(w in t for w in ["good", "great", "love", "excellent"])
            n = any(w in t for w in ["bad", "broke", "terrible", "not ", "stopped"])
            lab = "MIXED" if p and n else "POSITIVE" if p else "NEGATIVE" if n else "NEUTRAL"
            return json.dumps({"label": lab, "reason": "mock"})
        return call
    if backend == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        def call(text):
            r = client.messages.create(model=model, max_tokens=200, temperature=0, system=CODEBOOK,
                                       messages=[{"role": "user", "content": f'Review:\n"""{text}"""'}])
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return call
    if backend == "ollama":   # 改用 Ollama 原生介面：關閉思考模式＋以 JSON schema 限制輸出格式
        import urllib.request
        schema = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS},
                  "reason": {"type": "string"}}, "required": ["label", "reason"]}
        state = {"think": False}
        def post(payload):
            req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())
        def call(text):
            payload = {"model": model, "stream": False, "format": schema,
                       "options": {"temperature": 0, "num_predict": 300, "seed": 42},
                       "messages": [{"role": "system", "content": CODEBOOK},
                                    {"role": "user", "content": f'Review:\n"""{text}"""'}]}
            if state["think"] is not None: payload["think"] = False
            try: out = post(payload)
            except urllib.error.HTTPError as e:   # 不支援 think 參數的模型：拿掉後重送
                body = e.read().decode(errors="ignore")
                if "think" in body and state["think"] is not None:
                    state["think"] = None; payload.pop("think"); out = post(payload)
                else: raise RuntimeError(f"HTTP {e.code}: {body[:200]}")
            return out.get("message", {}).get("content", "")
        return call
    from openai import OpenAI
    base, env = PRESETS[backend]
    client = OpenAI(base_url=base, api_key=os.environ.get(env))
    state = {"legacy": True}
    def call(text):
        msgs = [{"role": "system", "content": CODEBOOK}, {"role": "user", "content": f'Review:\n"""{text}"""'}]
        if state["legacy"]:
            try:
                r = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=200)
                return r.choices[0].message.content or ""
            except Exception as e:   # 部分新模型不接受 temperature / max_tokens
                if "temperature" in str(e) or "max_tokens" in str(e): state["legacy"] = False
                else: raise
        r = client.chat.completions.create(model=model, messages=msgs, max_completion_tokens=2000)
        return r.choices[0].message.content or ""
    return call

def parse(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)   # 去除推理模型的思考段落
    for cand in reversed(re.findall(r"\{[^{}]*\}", raw)):          # 取最後一個合法 JSON 物件
        try: d = json.loads(cand)
        except json.JSONDecodeError: continue
        lab = str(d.get("label", "")).strip().upper()
        if lab in LABELS: return lab, str(d.get("reason", ""))[:300]
    return None, None

def cmd_annotate(a):
    items = pd.read_csv(a.items)
    if a.limit: items = items.head(a.limit)
    os.makedirs(a.runs, exist_ok=True)
    tag = f"{os.path.splitext(os.path.basename(a.items))[0]}__{a.backend}__{re.sub(r'[^A-Za-z0-9._-]', '_', a.model)}"
    out = os.path.join(a.runs, tag + ".jsonl")
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["item_id"] for l in f if l.strip() and json.loads(l).get("label")}
    todo = items[~items.item_id.isin(done)]
    print(f"輸出：{out}\n已完成 {len(done)} 筆，待標註 {len(todo)} 筆｜prompt 版本 {PROMPT_SHA}", flush=True)
    call, lock, n_ok, n_fail = make_caller(a.backend, a.model), threading.Lock(), [0], [0]
    def work(row):
        lab = reason = raw = err = None
        for attempt in range(4):
            try:
                raw = call(str(row.text)); lab, reason = parse(raw)
                if lab: break
                err = "unparseable"
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"; time.sleep(2 ** attempt * 3)
        rec = {"item_id": row.item_id, "label": lab, "reason": reason, "raw": None if lab else raw, "error": None if lab else err,
               "backend": a.backend, "model": a.model, "prompt_sha": PROMPT_SHA,
               "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        with lock:
            with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (n_ok if lab else n_fail)[0] += 1
            k = n_ok[0] + n_fail[0]
            if k % 100 == 0 or k == len(todo): print(f"  {k}/{len(todo)}（失敗 {n_fail[0]}）", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, r) for r in todo.itertuples()]): f.result()
    print(f"完成：成功 {n_ok[0]}，失敗 {n_fail[0]}（失敗的題目重跑同一指令即可補標）")

# ---------------- SemEval ----------------
def cmd_prep_semeval(a):
    rows = []
    for path in a.xml:
        for s in ET.parse(path).getroot().iter("sentence"):
            pols = {t.get("polarity") for t in s.iter("aspectTerm")}
            if not pols: continue          # 沒有面向標註的句子不納入
            if "conflict" in pols or {"positive", "negative"} <= pols: gold = "MIXED"
            elif pols == {"neutral"}: gold = "NEUTRAL"
            elif "positive" in pols: gold = "POSITIVE"
            else: gold = "NEGATIVE"
            rows.append({"item_id": f"S{os.path.basename(path)[:3]}_{s.get('id')}", "text": s.findtext("text"), "gold": gold})
    d = pd.DataFrame(rows).drop_duplicates("item_id")
    d[["item_id", "text"]].to_csv(a.out + "_items.csv", index=False, encoding="utf-8-sig")
    d[["item_id", "gold"]].to_csv(a.out + "_gold.csv", index=False)
    print(f"{len(d)} 句；標準答案分布：{d.gold.value_counts().to_dict()}")

# ---------------- 語言篩選（規則式，不依賴模型） ----------------
EN = set("the a an and or but is are was were it this that to of in on for with my i you not have has had be as at so they them very would will can just no do does did its if all one these works work great good".split())
OTHER = {
 "es": set("de la que el en y los se del las un por con una su para es al lo como más pero sus le ya este porque esta muy sin también me hay todo nada mucho bien funciona producto calidad".split()),
 "pt": set("de que não um uma para com os no na do da em é muito mas produto bom qualidade".split()),
 "fr": set("le la les et est un une des pour pas avec dans ce qui très mais produit".split()),
 "de": set("der die das und ist nicht ein eine mit für sehr auf ich es zu aber".split()),
}
def is_english(text):
    w = re.findall(r"[^\W\d_]+", str(text).lower())
    if len(w) < 3: return True
    en = sum(x in EN for x in w)
    oth = max(sum(x in s for x in w) for s in OTHER.values())
    return not (oth >= 2 and oth > en)

# ---------------- 一致性分析 ----------------
def fleiss(mat):   # mat: n_items x n_categories 的計票矩陣
    n = mat.sum(1)[0]; p = mat.sum(0) / mat.sum()
    P = ((mat ** 2).sum(1) - n) / (n * (n - 1)); Pb, Pe = P.mean(), (p ** 2).sum()
    return (Pb - Pe) / (1 - Pe) if Pe < 1 else np.nan

def cmd_evaluate(a):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    runs = {}
    for f in sorted(os.listdir(a.runs)):
        if not f.endswith(".jsonl"): continue
        d = pd.read_json(os.path.join(a.runs, f), lines=True)
        d = d[d.label.notna()].drop_duplicates("item_id", keep="last")
        src, name = f.split("__", 1); runs.setdefault(src, {})[name[:-6]] = d.set_index("item_id").label
    os.makedirs(a.out, exist_ok=True); report = {"prompt_sha": PROMPT_SHA}
    non_en = set()
    if a.items and os.path.exists(a.items):
        it = pd.read_csv(a.items); it = it[~it.text.map(is_english)]
        non_en = set(it.item_id); it.to_csv(os.path.join(a.out, "excluded_non_english.csv"), index=False, encoding="utf-8-sig")
        report["excluded_non_english"] = len(non_en); print(f"規則式語言篩選：排除 {len(non_en)} 則非英文評論")
    for src, models in runs.items():
        df = pd.DataFrame(models).dropna(); names = list(models)
        df = df[~df.index.isin(non_en)]
        print(f"\n=== {src}：{len(names)} 個模型，{len(df)} 筆共同完成 ===")
        r = {"n_items": len(df), "models": names, "pairwise_cohen_kappa": {}}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                k = cohen_kappa_score(df[names[i]], df[names[j]]); r["pairwise_cohen_kappa"][f"{names[i]} vs {names[j]}"] = round(k, 4)
                print(f"  Cohen's kappa {names[i]} vs {names[j]}: {k:.3f}")
        if len(names) >= 2:
            mat = np.stack([(df.values == l).sum(1) for l in LABELS], 1)
            if len(names) >= 3: r["fleiss_kappa"] = round(float(fleiss(mat)), 4); print(f"  Fleiss' kappa: {r['fleiss_kappa']:.3f}")
            top = mat.max(1); df["majority"] = np.where(top > len(names) / 2, np.array(LABELS)[mat.argmax(1)], "NO_MAJORITY")
            df["unanimous"] = top == len(names)
        else:
            df["majority"] = df[names[0]]; df["unanimous"] = True
        df.to_csv(os.path.join(a.out, f"{src}_labels.csv"))
        is_sem = bool(a.semeval_gold) and src.startswith(os.path.basename(a.semeval_gold).replace("_gold.csv", ""))
        if a.key and os.path.exists(a.key) and not is_sem:
            k = pd.read_csv(a.key).set_index("item_id"); m = df.join(k, how="inner")
            dist = pd.crosstab(m.rating_star, m.majority, normalize="index").round(4)
            dist.to_csv(os.path.join(a.out, f"{src}_dist_by_star.csv")); print("\n  依星數的多數決分布：\n", dist.to_string())
            ag = m.groupby("rating_star").unanimous.mean().round(4)
            ag.to_csv(os.path.join(a.out, f"{src}_unanimity_by_star.csv")); print("\n  全體一致比例（依星數）：\n", ag.to_string())
            m3 = m[m.rating_star == 3]
            by_yr = pd.crosstab(m3.yr, m3.majority, normalize="index").round(4)
            by_yr.to_csv(os.path.join(a.out, f"{src}_3star_dist_by_year.csv"))
            r["three_star_distribution"] = pd.Series(m3.majority).value_counts(normalize=True).round(4).to_dict()
        if is_sem:
            g = pd.read_csv(a.semeval_gold).set_index("item_id").gold; m = df.join(g, how="inner")
            r["semeval"] = {}
            for col in names + ["majority"]:
                acc = float((m[col] == m.gold).mean()); kap = cohen_kappa_score(m[col], m.gold)
                rec = {l: round(float(((m[col] == l) & (m.gold == l)).sum() / max((m.gold == l).sum(), 1)), 4) for l in ["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL"]}
                r["semeval"][col] = {"accuracy": round(acc, 4), "kappa": round(kap, 4), "recall_by_gold": rec}
                print(f"  SemEval｜{col}: accuracy={acc:.3f} kappa={kap:.3f} 各類召回={rec}")
                pd.DataFrame(confusion_matrix(m.gold, m[col], labels=LABELS), index=[f"gold_{l}" for l in LABELS],
                             columns=[f"pred_{l}" for l in LABELS]).to_csv(os.path.join(a.out, f"semeval_cm_{col}.csv"))
        report[src] = r
    json.dump(report, open(os.path.join(a.out, "agreement_report.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n報告已寫入 {a.out}")

def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("annotate"); s.add_argument("--items", required=True); s.add_argument("--backend", required=True,
        choices=["anthropic", "openai", "gemini", "ollama", "mock"]); s.add_argument("--model", required=True)
    s.add_argument("--runs", default="runs"); s.add_argument("--workers", type=int, default=4); s.add_argument("--limit", type=int, default=0)
    s = sp.add_parser("prep-semeval"); s.add_argument("--xml", nargs="+", required=True); s.add_argument("--out", default="semeval_laptop")
    s = sp.add_parser("evaluate"); s.add_argument("--runs", default="runs"); s.add_argument("--key", default="annotation_key.csv")
    s.add_argument("--semeval_gold", default=""); s.add_argument("--out", default="annotation_results")
    s.add_argument("--items", default="annotation_items_blind.csv", help="用於規則式排除非英文評論")
    s = sp.add_parser("show-prompt")
    a = p.parse_args()
    if a.cmd == "show-prompt": print(CODEBOOK); print("\nprompt_sha:", PROMPT_SHA); return
    {"annotate": cmd_annotate, "prep-semeval": cmd_prep_semeval, "evaluate": cmd_evaluate}[a.cmd](a)

if __name__ == "__main__":
    main()
