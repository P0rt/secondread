"""Score the calibration panel (3 AI annotator lenses) against WMT24 en-es human MQM labels.

Inputs (all under data/realdocs/gold/calib/, git-ignored):
  annotations/<lens>_batch_XX.jsonl   one record per uid per lens {lens, uid, severity, confidence, category, span_mt, explanation}
  key_human.json                      uid -> human label ("Major" = any major/critical span in the
                                      meaning family, scripts/convert.py), spans, system, chunk, stratum, weight
  batch_01.jsonl, batch_02.jsonl      the blind units (src, mt) for the disagreement dump
Outputs:
  metrics.json          every number quoted in RESULTS.md / docs/ANNOTATION_PROTOCOL.md
  disagreements.jsonl   every unit where any lens or the majority disagrees with the human label
  (stdout)              markdown tables

Definitions
  annotator Major  = severity in {major, critical}
  majority (MV)    = >= 2 of 3 lenses say Major (ties impossible with 3)
  P/R/F1           = of the class "Major" vs the human label
  kappa            = Cohen's kappa (binary Major/OK) vs the human label
  Fleiss           = Fleiss' kappa among the 3 lenses (binary; also 3-class none/minor/major+)
  weighted         = post-stratification weights from SAMPLING.json (natural Major rate, 5.7%)
  CI               = 95% percentile cluster bootstrap over the 22 source chunks (2000 resamples, seed 0);
                     units of one source chunk (all systems) are resampled together
  adjudication     = pre-registered rule: 2-1 split whose minority says major/critical with confidence >= 0.7,
                     or any lens says critical; amended_A1 additionally takes every 1-of-3 Major split

usage: PYTHONPATH=. uv run --frozen python scripts/gold_calib_eval.py
"""
import collections
import glob
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.path.join(ROOT, "data/realdocs/gold/calib")
LENSES = ("mqm", "posteditor", "reader")
MAJ = {"major", "critical"}
B = 2000


def load():
    key = json.load(open(os.path.join(CAL, "key_human.json"), encoding="utf-8"))
    ann = {l: {} for l in LENSES}
    for fn in sorted(glob.glob(os.path.join(CAL, "annotations", "*_batch_*.jsonl"))):
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            assert r["uid"] not in ann[r["lens"]], (fn, r["uid"])
            ann[r["lens"]][r["uid"]] = r
    units = {}
    for fn in sorted(glob.glob(os.path.join(CAL, "batch_*.jsonl"))):
        for line in open(fn, encoding="utf-8"):
            u = json.loads(line)
            units[u["uid"]] = u
    uids = sorted(key)
    for l in LENSES:
        missing = set(uids) - set(ann[l])
        extra = set(ann[l]) - set(uids)
        assert not missing and not extra, (l, sorted(missing)[:5], sorted(extra)[:5])
        bad = [u for u, r in ann[l].items() if r["severity"] not in ("none", "minor", "major", "critical")]
        assert not bad, (l, bad[:5])
    return key, ann, units, uids


def conf(pred, gold, w=None):
    tp = fp = fn = tn = 0.0
    for i, (p, g) in enumerate(zip(pred, gold)):
        x = 1.0 if w is None else w[i]
        if p and g:
            tp += x
        elif p:
            fp += x
        elif g:
            fn += x
        else:
            tn += x
    return tp, fp, fn, tn


def stats(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return dict(P=p, R=r, F1=f, kappa=k, acc=po)


def cohen(a, b):
    return stats(*conf(a, b))["kappa"]


def fleiss(rows, cats):
    """rows: list of per-unit lists of category labels (same rater count)."""
    n = len(rows[0])
    N = len(rows)
    pj = collections.Counter()
    Pi = []
    for r in rows:
        c = collections.Counter(r)
        for k in cats:
            pj[k] += c[k]
        Pi.append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
    Pbar = sum(Pi) / N
    Pe = sum((pj[k] / (N * n)) ** 2 for k in cats)
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else 0.0


def cluster_boot(uids, key, fn, seed=0):
    by = collections.defaultdict(list)
    for u in uids:
        by[key[u]["chunk"].split("#")[0] + "#" + key[u]["chunk"].split("#")[1]].append(u)
    groups = list(by.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        samp = [u for _ in groups for u in rng.choice(groups)]
        vals.append(fn(samp))
    out = {}
    for k in vals[0]:
        xs = sorted(v[k] for v in vals)
        out[k] = (xs[int(0.025 * B)], xs[int(0.975 * B) - 1])
    return out


def main():
    key, ann, units, uids = load()
    gold = {u: key[u]["label"] == "Major" for u in uids}
    gold_any = {u: key[u]["label_any_major"] for u in uids}
    w = {u: key[u]["weight"] for u in uids}
    maj = {l: {u: ann[l][u]["severity"] in MAJ for u in uids} for l in LENSES}
    votes = {u: sum(maj[l][u] for l in LENSES) for u in uids}
    raters = dict(maj)
    raters["majority"] = {u: votes[u] >= 2 for u in uids}
    raters["any (>=1)"] = {u: votes[u] >= 1 for u in uids}
    raters["unanimous (3/3)"] = {u: votes[u] == 3 for u in uids}

    M = {"n_units": len(uids), "n_human_major": sum(gold.values()),
         "human_major_rate": sum(gold.values()) / len(uids),
         "human_major_rate_weighted": sum(w[u] for u in uids if gold[u]) / sum(w.values()),
         "raters": {}}
    for name, pr in raters.items():
        def f(sample, pr=pr):
            s = stats(*conf([pr[u] for u in sample], [gold[u] for u in sample]))
            return {"F1": s["F1"], "kappa": s["kappa"], "P": s["P"], "R": s["R"]}
        c = conf([pr[u] for u in uids], [gold[u] for u in uids])
        cw = conf([pr[u] for u in uids], [gold[u] for u in uids], [w[u] for u in uids])
        ca = conf([pr[u] for u in uids], [gold_any[u] for u in uids])
        M["raters"][name] = {
            "tp_fp_fn_tn": c, "pred_major": c[0] + c[1], **stats(*c),
            "ci95": cluster_boot(uids, key, f),
            "weighted": stats(*cw),
            "vs_label_any_major": stats(*ca),
        }

    # inter-annotator agreement
    M["fleiss_binary"] = fleiss([[maj[l][u] for l in LENSES] for u in uids], [True, False])
    def three(s):
        return "maj" if s in MAJ else s
    M["fleiss_3class"] = fleiss([[three(ann[l][u]["severity"]) for l in LENSES] for u in uids],
                                ["none", "minor", "maj"])
    M["fleiss_binary_ci95"] = cluster_boot(
        uids, key, lambda s: {"fleiss": fleiss([[maj[l][u] for l in LENSES] for u in s], [True, False])})["fleiss"]
    M["pairwise_cohen"] = {f"{a}~{b}": cohen([maj[a][u] for u in uids], [maj[b][u] for u in uids])
                           for i, a in enumerate(LENSES) for b in LENSES[i + 1:]}
    M["vote_distribution"] = dict(collections.Counter(votes.values()))
    M["vote_x_human"] = {f"{v}/3": {"human_major": sum(1 for u in uids if votes[u] == v and gold[u]),
                                    "units": sum(1 for u in uids if votes[u] == v)} for v in range(4)}

    # adjudication rule simulation
    #   R  = pre-registered rule: any critical, or a 2-1 split whose minority says major/critical with conf >= 0.7
    #   RA = R + calibration amendment A1: every unit with exactly 1 of 3 Major votes (any confidence)
    mv = raters["majority"]
    def rule(u, amended):
        if any(ann[l][u]["severity"] == "critical" for l in LENSES):
            return "critical"
        if votes[u] == 1:
            lone = [l for l in LENSES if maj[l][u]][0]
            if float(ann[lone][u].get("confidence", 0)) >= 0.7:
                return "minority_major_conf>=0.7"
            return "minority_major_low_conf" if amended else None
        return None
    M["adjudication"] = {}
    for name, amended in (("pre_registered", False), ("amended_A1", True)):
        adj = {u: rule(u, amended) for u in uids if rule(u, amended)}
        M["adjudication"][name] = {
            "units": len(adj), "rate": len(adj) / len(uids),
            "by_reason": dict(collections.Counter(adj.values())),
            "human_major_among_adjudicated": sum(gold[u] for u in adj),
            "mv_wrong_among_adjudicated": sum(mv[u] != gold[u] for u in adj),
            "mv_wrong_total": sum(mv[u] != gold[u] for u in uids),
            "oracle_after_adjudication": stats(*conf([gold[u] if u in adj else mv[u] for u in uids],
                                                     [gold[u] for u in uids])),
            "uids": adj,
        }
    M["minority_major_confidences"] = sorted(
        float(ann[l][u]["confidence"]) for u in uids if votes[u] == 1 for l in LENSES if maj[l][u])
    M["mv_major_max_conf_vs_human"] = {}
    for lo, hi in ((0, 0.7), (0.7, 0.8), (0.8, 1.01)):
        us = [u for u in uids if mv[u] and lo <= max(float(ann[l][u]["confidence"]) for l in LENSES if maj[l][u]) < hi]
        M["mv_major_max_conf_vs_human"][f"[{lo},{min(hi, 1.0)})"] = (sum(gold[u] for u in us), len(us))
    M["by_batch"] = {}
    for b in ("cal-a", "cal-b"):
        us = [u for u in uids if u.startswith(b)]
        M["by_batch"][b] = {n: stats(*conf([raters[n][u] for u in us], [gold[u] for u in us]))
                            for n in (*LENSES, "majority")}
        M["by_batch"][b]["fleiss_binary"] = fleiss([[maj[l][u] for l in LENSES] for u in us], [True, False])
    M["all_three_same_3class"] = sum(len({three(ann[l][u]["severity"]) for l in LENSES}) == 1 for u in uids) / len(uids)
    M["severity_counts"] = {l: dict(collections.Counter(ann[l][u]["severity"] for u in uids)) for l in LENSES}

    # confidence vs correctness of Major calls
    M["major_call_precision_by_conf"] = {}
    for l in LENSES:
        row = {}
        for lo, hi in ((0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
            calls = [u for u in uids if maj[l][u] and lo <= float(ann[l][u]["confidence"]) < hi]
            row[f"[{lo},{min(hi, 1.0)})"] = (sum(gold[u] for u in calls), len(calls))
        M["major_call_precision_by_conf"][l] = row

    # human-major recall by human category; FP by human max severity
    cats = collections.defaultdict(list)
    for u in uids:
        if gold[u]:
            c = sorted({s["category"] for s in key[u]["spans"]
                        if s["family"] == "meaning" and s["severity"] in MAJ})
            cats["+".join(c)].append(u)
    M["recall_by_human_category"] = {c: {"n": len(us), **{n: sum(raters[n][u] for u in us)
                                                          for n in (*LENSES, "majority")}}
                                     for c, us in sorted(cats.items(), key=lambda t: -len(t[1]))}
    M["fp_by_human_max_severity"] = {}
    for n in (*LENSES, "majority"):
        c = collections.Counter(key[u]["meaning_max_severity"] if key[u]["max_severity"] != "major" or not key[u]["label_any_major"]
                                else "nonmeaning-major" for u in uids if raters[n][u] and not gold[u])
        M["fp_by_human_max_severity"][n] = dict(c)
    M["by_domain"] = {}
    for d in sorted({key[u]["domain"] for u in uids}):
        us = [u for u in uids if key[u]["domain"] == d]
        M["by_domain"][d] = {"units": len(us), "human_major": sum(gold[u] for u in us),
                             **{n: stats(*conf([raters[n][u] for u in us], [gold[u] for u in us]))
                                for n in (*LENSES, "majority")}}
    # source-side: units whose MT is a non-translation (LLM meta-text / refusal) per annotators
    json.dump(M, open(os.path.join(CAL, "metrics.json"), "w"), indent=1, default=list)

    with open(os.path.join(CAL, "disagreements.jsonl"), "w", encoding="utf-8") as fo:
        for u in uids:
            if all(raters[n][u] == gold[u] for n in (*LENSES, "majority")):
                continue
            k = key[u]
            fo.write(json.dumps({
                "uid": u, "human": k["label"], "human_max_severity": k["max_severity"],
                "human_spans": [(s["text"], s["category"], s["severity"]) for s in k["spans"]],
                "votes": votes[u], "mv": "Major" if mv[u] else "OK",
                "src": units[u]["src"], "mt": units[u]["mt"],
                "lenses": {l: (ann[l][u]["severity"], ann[l][u].get("confidence"), ann[l][u].get("category"),
                               ann[l][u].get("span_mt"), ann[l][u].get("explanation")) for l in LENSES},
            }, ensure_ascii=False) + "\n")

    # markdown
    print(f"units {len(uids)}, human Major {sum(gold.values())} "
          f"({M['human_major_rate']:.1%}; natural/weighted {M['human_major_rate_weighted']:.1%})\n")
    print("| rater | pred Major | TP | FP | FN | P | R | F1 (95% CI) | kappa (95% CI) | weighted P / R / F1 / kappa |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, r in M["raters"].items():
        tp, fp, fn_, tn = r["tp_fp_fn_tn"]
        ci = r["ci95"]
        ww = r["weighted"]
        print(f"| {n} | {int(tp + fp)} | {int(tp)} | {int(fp)} | {int(fn_)} | {r['P']:.2f} | {r['R']:.2f} | "
              f"{r['F1']:.2f} ({ci['F1'][0]:.2f}-{ci['F1'][1]:.2f}) | {r['kappa']:.2f} ({ci['kappa'][0]:.2f}-{ci['kappa'][1]:.2f}) | "
              f"{ww['P']:.2f} / {ww['R']:.2f} / {ww['F1']:.2f} / {ww['kappa']:.2f} |")
    print(f"\nFleiss binary {M['fleiss_binary']:.3f} (CI {M['fleiss_binary_ci95'][0]:.2f}-{M['fleiss_binary_ci95'][1]:.2f}); "
          f"3-class {M['fleiss_3class']:.3f}; pairwise {json.dumps({k: round(v, 3) for k, v in M['pairwise_cohen'].items()})}")
    print("votes x human:", M["vote_x_human"])
    for name, a in M["adjudication"].items():
        print(f"adjudication {name}: {a['units']} units ({a['rate']:.1%}) {a['by_reason']}; human Major among them {a['human_major_among_adjudicated']}; "
              f"MV wrong among them {a['mv_wrong_among_adjudicated']}/{a['mv_wrong_total']}; "
              f"oracle after adjudication {json.dumps({k: round(v, 3) for k, v in a['oracle_after_adjudication'].items()})}")
    print("minority Major confidences:", M["minority_major_confidences"])
    print("MV-Major by max confidence (human Major, n):", M["mv_major_max_conf_vs_human"])
    for b, r in M["by_batch"].items():
        print("batch", b, {n: (round(r[n]["F1"], 2), round(r[n]["kappa"], 2)) for n in (*LENSES, "majority")}, "fleiss", round(r["fleiss_binary"], 3))
    print("all three lenses same 3-class severity:", round(M["all_three_same_3class"], 3))
    print("severity counts:", M["severity_counts"])
    print("major-call precision by confidence:", json.dumps(M["major_call_precision_by_conf"]))
    print("recall by human category:", json.dumps(M["recall_by_human_category"]))
    print("FP by human severity:", json.dumps(M["fp_by_human_max_severity"]))
    for d, r in M["by_domain"].items():
        print(d, r["units"], r["human_major"], {n: round(r[n]["F1"], 2) for n in (*LENSES, "majority")})
    print("vs label_any_major:", {n: round(r["vs_label_any_major"]["kappa"], 3) for n, r in M["raters"].items()})


if __name__ == "__main__":
    main()
