"""Build the SFT conversation file for Tinker from the train splits (+ optional synthetic data).

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/build_sft.py \
           [--synthetic FILE.jsonl ...] [--out data/sft_v1.jsonl] [--seed 0]

Mix (v1):
  * en-ru: every train unit;
  * en-es: every Major train unit plus --es-ok-ratio (3) x as many OK units, sampled
    with --seed (77%+ of en-es is OK; the full set would drown the signal);
  * synthetic: every row of each --synthetic file (secondread/synth.py schema; needs
    src, mt, lp, label and should carry its parent's `doc`). A row whose doc is a
    dev/test doc, or whose source sentence occurs in dev/test, aborts the build
    (private prep notes, 02-corrections.md #4).

Each conversation is rendered with the training renderer (secondread.prompt). Rows
longer than --max-tokens are dropped, because Tinker's max_length truncates from the
right and would silently cut the label off (corrections #5). Every kept row is also
checked so that its training prefix is exactly the inference prompt
(generation_prompt_ids) followed by the label token.

Output: one JSON object per line, {"messages": [...], "id", "lp", "label", "source",
"doc", "n_tokens"}; `messages` is what tinker_cookbook's FromConversationFileBuilder
and scripts/train_tinker.py read. The file is shuffled with --seed.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

from secondread.prompt import LABEL_IDS, LABELS, messages, renderer

TRAIN_PRICE_PER_M = 0.737  # USD per 1M training tokens, Qwen/Qwen3.5-4B on Tinker (private prep notes, 03 section 3)
# Tolerate the converter's old spelling but never let a 2-token label through (corrections #14).
LABEL_ALIASES = {"Major": "Major", "MAJOR": "Major", "major": "Major", "OK": "OK", "ok": "OK", "Ok": "OK"}


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of_doc(splits_dir: str, lp: str) -> dict[str, str]:
    path = os.path.join(splits_dir, f"{lp}.docs.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def heldout_sources(splits_dir: str, lp: str) -> set[str]:
    """Normalised source sentences of the dev and test splits (text-level leak guard)."""
    out: set[str] = set()
    for split in ("dev", "test"):
        path = os.path.join(splits_dir, f"{lp}.{split}.jsonl")
        if os.path.exists(path):
            out |= {" ".join(r["src"].lower().split()) for r in read_jsonl(path)}
    return out


def select_wmt(splits_dir: str, es_ok_ratio: float, rng: random.Random) -> list[dict]:
    ru = read_jsonl(os.path.join(splits_dir, "en-ru.train.jsonl"))
    es = read_jsonl(os.path.join(splits_dir, "en-es.train.jsonl"))
    for r in ru + es:
        r["source"] = "wmt"
    es_major = [r for r in es if r["label"] == "Major"]
    es_ok = sorted((r for r in es if r["label"] == "OK"), key=lambda r: r["id"])  # order-independent sample
    n_ok = min(len(es_ok), round(es_ok_ratio * len(es_major)))
    return ru + es_major + rng.sample(es_ok, n_ok)


def load_synthetic(paths: list[str], splits_dir: str) -> list[dict]:
    """Synthetic rows (secondread/synth.py schema: split-file fields + synth_type). Aborts on
    any row whose document is not a train document or whose source sentence occurs in dev/test."""
    rows, no_doc, unknown_doc = [], 0, 0
    doc_maps: dict[str, dict[str, str]] = {}
    held: dict[str, set[str]] = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        for i, r in enumerate(read_jsonl(path)):
            where = f"{path}:{i + 1}"
            missing = [k for k in ("src", "mt", "lp", "label") if k not in r]
            if missing:
                sys.exit(f"{where}: missing fields {missing}")
            if r["label"] not in LABEL_ALIASES:
                sys.exit(f"{where}: unknown label {r['label']!r}")
            if r.get("doc") is None:
                no_doc += 1
            else:
                split = doc_maps.setdefault(r["lp"], split_of_doc(splits_dir, r["lp"])).get(r["doc"])
                if split is None:
                    unknown_doc += 1
                elif split != "train":
                    sys.exit(f"{where}: doc {r['doc']!r} is in the {split} split -> leak, refusing")
            if " ".join(r["src"].lower().split()) in held.setdefault(r["lp"], heldout_sources(splits_dir, r["lp"])):
                sys.exit(f"{where}: source sentence also occurs in {r['lp']} dev/test -> leak, refusing")
            kind = r.get("synth_type") or r.get("type") or name
            rows.append({**r, "id": r.get("id", f"syn|{name}|{i}"), "source": f"synthetic:{kind}"})
    if no_doc or unknown_doc:
        print(f"WARNING: synthetic rows without a doc: {no_doc}, with a doc not in docs.json: {unknown_doc} "
              "(only the source-text leak guard applied to them)")
    return rows


def render(rows: list[dict], max_tokens: int):
    """Render every row; return (kept rows with n_tokens/weighted counts, dropped rows)."""
    from tinker_cookbook.renderers import TrainOnWhat

    ren = renderer()
    kept, dropped = [], []
    for r in rows:
        label = LABEL_ALIASES[r["label"]]
        msgs = messages(r["src"], r["mt"], r["lp"], label)
        model_input, weights = ren.build_supervised_example(msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        n = model_input.length
        if n > max_tokens:
            dropped.append({**r, "n_tokens": n})
            continue
        # The training sequence must be <inference prompt> + <label token> + ..., or the
        # adapter would learn a prompt the scorer never sends.
        tokens = model_input.to_ints()
        prompt = ren.build_generation_prompt(msgs[:1]).to_ints()
        assert tokens[: len(prompt)] == prompt, f"{r['id']}: training prefix != generation prompt"
        assert tokens[len(prompt)] == LABEL_IDS[label], f"{r['id']}: label is not the single token {LABEL_IDS[label]}"
        n_weighted = int((weights > 0).sum())
        kept.append({"messages": msgs, "id": r["id"], "lp": r["lp"], "label": label, "source": r["source"],
                     "doc": r.get("doc"), "n_tokens": n, "_weighted": n_weighted})
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--synthetic", nargs="*", default=[], help="extra jsonl files (src, mt, lp, label, doc[, type])")
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--es-ok-ratio", type=float, default=3.0, help="en-es OK units per en-es Major unit")
    ap.add_argument("--max-tokens", type=int, default=1000, help="drop rows whose rendered length exceeds this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64, help="only for the steps-per-epoch estimate")
    ap.add_argument("--epochs", type=int, nargs="*", default=[1, 2], help="epoch counts to price")
    ap.add_argument("--price", type=float, default=TRAIN_PRICE_PER_M, help="USD per 1M training tokens")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = select_wmt(a.splits_dir, a.es_ok_ratio, rng) + load_synthetic(a.synthetic, a.splits_dir)
    assert all(LABEL_ALIASES[r["label"]] in LABELS for r in rows)
    ids = collections.Counter(r["id"] for r in rows)
    dup = [i for i, c in ids.items() if c > 1]
    if dup:
        sys.exit(f"duplicate ids, e.g. {dup[:3]}")

    kept, dropped = render(rows, a.max_tokens)
    rng.shuffle(kept)

    # ---- report
    def key(r):
        return r["source"] if r["source"] != "wmt" else f"wmt {r['lp']}"

    by = collections.defaultdict(collections.Counter)
    for r in kept:
        by[key(r)][r["label"]] += 1
    drop_by = collections.defaultdict(list)
    for r in dropped:
        drop_by[key(r)].append(r)
    print(f"{'group':24s} {'kept':>6s} {'Major':>6s} {'OK':>6s} {'%Major':>7s} {'dropped(>' + str(a.max_tokens) + ')':>14s}")
    for g in sorted(set(by) | set(drop_by)):
        c = by[g]; n = c["Major"] + c["OK"]
        d = drop_by.get(g, [])
        dl = f" (Major={sum(LABEL_ALIASES[x['label']] == 'Major' for x in d)}, max={max(x['n_tokens'] for x in d)} tok)" if d else ""
        print(f"{g:24s} {n:6d} {c['Major']:6d} {c['OK']:6d} {100 * c['Major'] / max(n, 1):6.1f}% {len(d):6d}{dl}")
    tot = collections.Counter(r["label"] for r in kept)
    print(f"{'TOTAL':24s} {len(kept):6d} {tot['Major']:6d} {tot['OK']:6d} {100 * tot['Major'] / max(len(kept), 1):6.1f}% {len(dropped):6d}")

    pairs = collections.defaultdict(set)
    for r in kept:
        s, t = r["messages"][0]["content"], r["label"]
        pairs[s].add(t)
    conflicts = sum(len(v) > 1 for v in pairs.values())
    print(f"identical prompts: {len(kept) - len(pairs)} repeats, {conflicts} prompts with both labels (label noise)")

    # Tinker sees len-1 input tokens per example (the last token is only a target);
    # cookbook train.py logs exactly this as num_tokens. Weighted = loss weight > 0.
    seq = sum(r["n_tokens"] for r in kept)
    sent = seq - len(kept)
    weighted = sum(r["_weighted"] for r in kept)
    lens = sorted(r["n_tokens"] for r in kept)
    print(f"tokens/example: mean={seq / len(kept):.1f} p50={lens[len(lens) // 2]} p99={lens[int(0.99 * len(lens))]} max={lens[-1]}")
    print(f"tokens per epoch: all={sent:,} (rendered {seq:,})  weighted={weighted:,} "
          f"({weighted / len(kept):.1f}/example: label + <|im_end|>)")
    steps = len(kept) // a.batch_size
    for e in a.epochs:
        print(f"  {e} epoch(s): {steps * e} steps @ batch {a.batch_size}; "
              f"cost if all tokens billed ${sent * e * a.price / 1e6:.2f}, "
              f"if only weighted ${weighted * e * a.price / 1e6:.3f}  (@ ${a.price}/M)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            r.pop("_weighted")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {len(kept)} conversations")


if __name__ == "__main__":
    main()
