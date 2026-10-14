"""Compare scored jsonl files (probe_local.py / eval_tinker.py output) as a markdown table.

usage: PYTHONPATH=. uv run python scripts/report.py NAME=SCORED.jsonl[,DEV_SCORED.jsonl] ... \
           [--ref data/splits/en-ru.test.jsonl ...] [--by synth_type] [--out report.md]

  NAME=SCORED          a system's scores on the evaluation split (test, usually);
  ,DEV_SCORED          the same system's scores on dev: its F1-optimal threshold is chosen
                       there and applied unchanged to SCORED. Without it, F1 is left out.
  --ref SPLIT.jsonl    split files used to fill in `doc`/`lp` for older scored files that
                       lack them (e.g. results/base_enru_dev300.jsonl).

All systems are compared on the ids they have in common (--all-rows to disable), and all
CIs come from one shared document-level bootstrap (--n-boot replicates, resampling whole
documents, since segments of one document are correlated). Sharing the replicates makes
the "vs first system" differences paired. The first NAME is the baseline.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from secondread import metrics as M


def fmt(x: float | None, ci: tuple[float, float] | None = None, signed: bool = False) -> str:
    if x is None or np.isnan(x):
        return "n/a"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f.format(x) if ci is None else f"{f.format(x)} [{f.format(ci[0])}, {f.format(ci[1])}]"


def parse_run(spec: str) -> tuple[str, str, str | None]:
    if "=" not in spec:
        raise SystemExit(f"bad run spec {spec!r}: expected NAME=SCORED.jsonl[,DEV_SCORED.jsonl]")
    name, paths = spec.split("=", 1)
    test, _, dev = paths.partition(",")
    return name, test, dev or None


def fill_docs(rows: list[dict], ref: dict[str, dict], path: str) -> None:
    missing = 0
    for r in rows:
        if r.get("doc") is None and r["id"] in ref:
            r["doc"], r["lp"] = ref[r["id"]]["doc"], ref[r["id"]]["lp"]
        missing += r.get("doc") is None
    if missing:
        print(f"WARNING: {path}: {missing} rows without doc (bootstrapped as single segments); pass --ref",
              file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown comparison of scored jsonl files.")
    ap.add_argument("runs", nargs="+", help="NAME=SCORED.jsonl[,DEV_SCORED.jsonl]")
    ap.add_argument("--ref", nargs="*", default=[], help="split jsonl files to recover doc/lp by id")
    ap.add_argument("--frac", type=float, default=0.10, help="review budget for recall@k%%")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-rows", action="store_true", help="do not restrict systems to common ids")
    ap.add_argument("--by", help="extra table: recall at the dev threshold per value of this field (e.g. synth_type)")
    ap.add_argument("--out", help="also write the markdown here")
    a = ap.parse_args()

    ref = {}
    for path in a.ref:
        for r in M.load_scored(path):
            ref[r["id"]] = {"doc": r["doc"], "lp": r["lp"], "by": r.get(a.by) if a.by else None}

    runs = []
    for spec in a.runs:
        name, test, dev = parse_run(spec)
        rows = M.load_scored(test)
        fill_docs(rows, ref, test)
        dev_rows = M.load_scored(dev) if dev else None
        runs.append({"name": name, "path": test, "rows": rows, "dev": dev, "dev_rows": dev_rows})

    notes = []
    if not a.all_rows and len(runs) > 1:
        common = set.intersection(*({r["id"] for r in run["rows"]} for run in runs))
        for run in runs:
            if len(run["rows"]) != len(common):
                notes.append(f"{run['name']}: {len(run['rows']) - len(common)} rows not scored by every system were dropped")
        order = sorted(common)
        for run in runs:
            by_id = {r["id"]: r for r in run["rows"]}
            run["rows"] = [by_id[i] for i in order]
        labels = [[r["label"] for r in run["rows"]] for run in runs]
        if any(lab != labels[0] for lab in labels[1:]):
            raise SystemExit("systems disagree on gold labels for the same ids; are these the same split?")
    paired = not a.all_rows or len(runs) == 1

    shared_w = M.bootstrap_weights(M.group_keys(runs[0]["rows"]), a.n_boot, a.seed) if paired and a.n_boot else None
    k = f"recall@{a.frac:.0%}"
    for run in runs:
        y, s = M.to_arrays(run["rows"])
        run["y"], run["s"] = y, s
        if run["dev_rows"]:
            dy, ds = M.to_arrays(run["dev_rows"])
            run["thr"] = M.best_f1_threshold(dy, ds)
        else:
            run["thr"] = None
        w = shared_w if shared_w is not None else (
            M.bootstrap_weights(M.group_keys(run["rows"]), a.n_boot, a.seed) if a.n_boot else None)
        run["w"] = w
        run["sum"] = M.summarize(run["rows"], threshold=run["thr"], frac=a.frac, weights=w)

    out = []
    first = runs[0]["sum"]
    out.append(f"Evaluated on n={first['n']} segments from {first['n_docs']} documents"
               f" (Major rate {first['major_rate']:.3f}); 95% CIs from {a.n_boot} document-level bootstrap replicates."
               if paired else "Systems evaluated on their own rows (--all-rows); CIs are not paired.")
    out.append("")
    out.append(f"| system | n | AUROC | PR-AUC | {k} | threshold (dev) | precision | recall | F1 |")
    out.append("|---|---:|---|---|---|---:|---:|---:|---|")
    for run in runs:
        sm, ci = run["sum"], run["sum"].get("ci", {})
        thr = run["thr"]
        out.append("| " + " | ".join([
            run["name"], str(sm["n"]), fmt(sm["auroc"], ci.get("auroc")), fmt(sm["pr_auc"], ci.get("pr_auc")),
            fmt(sm[k], ci.get(k)), "n/a" if thr is None else f"{thr:.3f}",
            fmt(sm.get("precision")), fmt(sm.get("recall")), fmt(sm.get("f1"), ci.get("f1")),
        ]) + " |")
    out.append("")
    out.append(f"Random ranking: AUROC 0.500, PR-AUC = Major rate, {k} = {a.frac:.3f}. "
               "F1 uses the threshold that maximises F1 on the system's own dev scores.")

    if paired and len(runs) > 1 and shared_w is not None:
        base = runs[0]
        out += ["", f"Paired differences vs **{base['name']}** (same bootstrap replicates):", "",
                f"| system | ΔAUROC | ΔPR-AUC | Δ{k} |", "|---|---|---|---|"]
        fns = {"auroc": M.auroc, "pr_auc": M.pr_auc, k: lambda y, s, w=None: M.recall_at(y, s, a.frac, w)}
        for run in runs[1:]:
            cells = []
            for key, f in fns.items():
                d = run["sum"][key] - base["sum"][key]
                reps = [f(run["y"], run["s"], w=w) - f(base["y"], base["s"], w=w) for w in shared_w]
                cells.append(fmt(d, M.percentile_ci(reps), signed=True))
            out.append(f"| {run['name']} | " + " | ".join(cells) + " |")

    if a.by:
        # One value per id for every system: from --ref, else the first scored file that has it.
        by_id = {i: v["by"] for i, v in ref.items() if v["by"] is not None}
        for run in runs:
            for r in run["rows"]:
                if r.get(a.by) is not None:
                    by_id.setdefault(r["id"], r[a.by])
        group = lambda r: str(by_id.get(r["id"], "(none)"))
        values = sorted({group(r) for run in runs for r in run["rows"]})
        out += ["", f"Recall at the dev threshold by `{a.by}` (Major rows only):", "",
                "| system | " + " | ".join(values) + " |", "|---|" + "---:|" * len(values)]
        for run in runs:
            cells = []
            for v in values:
                idx = [i for i, r in enumerate(run["rows"]) if group(r) == v and r["label"] == M.POSITIVE]
                if not idx or run["thr"] is None:
                    cells.append("n/a" if idx else "-")
                else:
                    hit = sum(run["s"][i] >= run["thr"] for i in idx)
                    cells.append(f"{hit / len(idx):.2f} ({len(idx)})")
            out.append(f"| {run['name']} | " + " | ".join(cells) + " |")

    if notes:
        out += [""] + [f"- {n}" for n in notes]
    out += ["", "Files: " + "; ".join(f"{r['name']}: `{r['path']}`" + (f" (dev `{r['dev']}`)" if r["dev"] else "")
                                       for r in runs)]
    text = "\n".join(out)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
