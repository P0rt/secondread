"""Compare scored jsonl files (probe_local.py / eval_tinker.py output) as a markdown table.

usage: PYTHONPATH=. uv run python scripts/report.py NAME=SCORED.jsonl[,DEV_SCORED.jsonl[,PARENTS.jsonl]] ... \
           [--ref data/splits/en-ru.test.jsonl ...] [--lp en-ru] [--by synth_type] [--diff A:B ...] [--out report.md]

  NAME=SCORED          a system's scores on the evaluation split (test, usually);
  ,DEV_SCORED          the same system's scores on dev: its F1-optimal threshold is chosen
                       there and applied unchanged to SCORED. Without it, F1 is left out.
  ,PARENTS             synthetic test sets only: the same system's scores on the split the
                       synthetic rows were made from (e.g. results/local_v1_enru_test.jsonl).
                       With --by, each group is also scored as AUROC of the group's rows
                       against their own clean parents (`synth_parent`, taken from --ref).
  --ref SPLIT.jsonl    split files used to fill in `doc`/`lp` (and `--by` values and
                       `synth_parent`) for scored files that lack them.
  --lp LP              keep only rows of this language pair (e.g. a synthetic test set that
                       mixes en-ru and en-es, scored with the en-ru dev threshold).
  --diff A:B           extra paired differences A - B (AUROC, PR-AUC, recall@k, F1 at each
                       system's own dev threshold), on the same bootstrap replicates.

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


def parse_run(spec: str) -> tuple[str, str, str | None, str | None]:
    if "=" not in spec:
        raise SystemExit(f"bad run spec {spec!r}: expected NAME=SCORED.jsonl[,DEV_SCORED.jsonl[,PARENTS.jsonl]]")
    name, paths = spec.split("=", 1)
    parts = paths.split(",")
    if len(parts) > 3:
        raise SystemExit(f"bad run spec {spec!r}: at most SCORED,DEV,PARENTS")
    test, dev, parents = (parts + [None, None])[:3]
    return name, test, dev or None, parents or None


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
    ap.add_argument("--lp", help="keep only rows of this language pair")
    ap.add_argument("--diff", action="append", default=[], metavar="A:B",
                    help="paired differences A - B between two named systems (repeatable)")
    ap.add_argument("--out", help="also write the markdown here")
    a = ap.parse_args()

    ref = {}
    for path in a.ref:
        for r in M.load_scored(path):
            ref[r["id"]] = {"doc": r["doc"], "lp": r["lp"], "by": r.get(a.by) if a.by else None,
                            "parent": r.get("synth_parent")}

    runs, notes = [], []
    for spec in a.runs:
        name, test, dev, parents = parse_run(spec)
        rows = M.load_scored(test)
        fill_docs(rows, ref, test)
        if a.lp:
            rows = [r for r in rows if r.get("lp") == a.lp]
            if not rows:
                raise SystemExit(f"{test}: no rows with lp={a.lp}")
        dev_rows = M.load_scored(dev) if dev else None
        if dev_rows and a.lp and any(r.get("lp") not in (None, a.lp) for r in dev_rows):
            notes.append(f"{name}: threshold from dev rows of another language pair than {a.lp} (`{dev}`)")
        par = {r["id"]: r for r in M.load_scored(parents)} if parents else None
        runs.append({"name": name, "path": test, "rows": rows, "dev": dev, "dev_rows": dev_rows,
                     "parents": parents, "parent_rows": par})

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

    if a.diff and shared_w is not None:
        by_name = {run["name"]: run for run in runs}
        out += ["", "Paired differences A − B (same bootstrap replicates; F1 at each system's own dev threshold):", "",
                f"| A − B | ΔAUROC | ΔPR-AUC | Δ{k} | ΔF1 |", "|---|---|---|---|---|"]
        for spec in a.diff:
            an, _, bn = spec.partition(":")
            if an not in by_name or bn not in by_name:
                raise SystemExit(f"--diff {spec}: unknown system (have {sorted(by_name)})")
            ra, rb = by_name[an], by_name[bn]
            fns = {"auroc": M.auroc, "pr_auc": M.pr_auc, k: lambda y, s, w=None: M.recall_at(y, s, a.frac, w)}
            cells = []
            for key, f in fns.items():
                d = ra["sum"][key] - rb["sum"][key]
                reps = [f(ra["y"], ra["s"], w=w) - f(rb["y"], rb["s"], w=w) for w in shared_w]
                cells.append(fmt(d, M.percentile_ci(reps), signed=True))
            if ra["thr"] is not None and rb["thr"] is not None:
                f1 = lambda run, w=None: M.prf_at(run["y"], run["s"], run["thr"], w)["f1"]
                reps = [f1(ra, w) - f1(rb, w) for w in shared_w]
                cells.append(fmt(f1(ra) - f1(rb), M.percentile_ci(reps), signed=True))
            else:
                cells.append("n/a")
            out.append(f"| {an} − {bn} | " + " | ".join(cells) + " |")

    if a.by:
        # One value per id for every system: from --ref, else the first scored file that has it.
        by_id = {i: v["by"] for i, v in ref.items() if v["by"] is not None}
        for run in runs:
            for r in run["rows"]:
                if r.get(a.by) is not None:
                    by_id.setdefault(r["id"], r[a.by])
        group = lambda r: str(by_id.get(r["id"], "(none)"))
        values = sorted({group(r) for run in runs for r in run["rows"]})
        parent_of = {i: v["parent"] for i, v in ref.items() if v.get("parent")}
        with_parents = any(run["parent_rows"] for run in runs)
        out += ["", f"Recall at the dev threshold by `{a.by}` for groups of Major rows; for groups of OK rows "
                    "(no Major row) the false-flag rate, marked `fp`. Row counts in brackets"
                    + ("; the last column is the false-flag rate on the distinct clean parents of all rows" if with_parents else "")
                    + ":", "",
                "| system | " + " | ".join(values + (["clean parents"] if with_parents else [])) + " |",
                "|---|" + "---:|" * (len(values) + with_parents)]
        for run in runs:
            cells = []
            for v in values:
                rows_v = [i for i, r in enumerate(run["rows"]) if group(r) == v]
                idx = [i for i in rows_v if run["rows"][i]["label"] == M.POSITIVE]
                if run["thr"] is None:
                    cells.append("n/a")
                elif idx:
                    hit = sum(run["s"][i] >= run["thr"] for i in idx)
                    cells.append(f"{hit / len(idx):.2f} ({len(idx)})")
                elif rows_v:
                    hit = sum(run["s"][i] >= run["thr"] for i in rows_v)
                    cells.append(f"fp {hit / len(rows_v):.2f} ({len(rows_v)})")
                else:
                    cells.append("-")
            if with_parents:
                par = run["parent_rows"]
                pids = sorted({parent_of.get(r["id"]) for r in run["rows"]} & set(par or {}))
                if run["thr"] is None or not pids:
                    cells.append("n/a")
                else:
                    hit = sum(float(par[p]["risk"]) >= run["thr"] for p in pids)
                    cells.append(f"fp {hit / len(pids):.2f} ({len(pids)})")
            out.append(f"| {run['name']} | " + " | ".join(cells) + " |")

        if with_parents:
            # Each group's rows (positives) against the distinct clean parents they were made from
            # (negatives), all scored by the same system. For a Major group this is how well an
            # injected error is ranked above the unedited sentence; for an OK group (meaning-
            # preserving edit) 0.5 is ideal, and above 0.5 means the edit alone raises the risk.
            out += ["", f"AUROC of each `{a.by}` group against its own clean parents (`synth_parent`, "
                        "scored by the same system on the parent split); for OK groups 0.5 is ideal. "
                        "Pair win rate (edited row riskier than its parent, ties 1/2) in brackets:", "",
                    "| system | " + " | ".join(values) + " |", "|---|" + "---:|" * len(values)]
            for run in runs:
                par = run["parent_rows"]
                if not par:
                    out.append(f"| {run['name']} | " + " | ".join("n/a" for _ in values) + " |")
                    continue
                cells = []
                for v in values:
                    rows_v = [(i, parent_of.get(r["id"])) for i, r in enumerate(run["rows"]) if group(r) == v]
                    rows_v = [(i, p) for i, p in rows_v if p in par]
                    if not rows_v:
                        cells.append("-")
                        continue
                    pids = sorted({p for _, p in rows_v})
                    bad = [p for p in pids if par[p]["label"] == M.POSITIVE]
                    if bad:
                        raise SystemExit(f"{run['parents']}: parent {bad[0]} is labelled Major; parents must be clean")
                    pos = np.array([run["s"][i] for i, _ in rows_v])
                    neg = np.array([float(par[p]["risk"]) for p in pids])
                    yy = np.r_[np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
                    au = M.auroc(yy, np.r_[pos, neg])
                    win = np.mean([1.0 if run["s"][i] > float(par[p]["risk"]) else
                                   0.5 if run["s"][i] == float(par[p]["risk"]) else 0.0 for i, p in rows_v])
                    cells.append(f"{au:.2f} ({win:.2f})")
                out.append(f"| {run['name']} | " + " | ".join(cells) + " |")

    if notes:
        out += [""] + [f"- {n}" for n in notes]
    out += ["", "Files: " + "; ".join(f"{r['name']}: `{r['path']}`" + (f" (dev `{r['dev']}`)" if r["dev"] else "")
                                       + (f" (parents `{r['parents']}`)" if r["parents"] else "")
                                       for r in runs)]
    if a.lp:
        out.append(f"Rows restricted to lp={a.lp}.")
    text = "\n".join(out)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
