"""Machine-translate the real EN documents into Spanish with open MT models, as an agency's MT step would.

usage: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run --frozen --with sentencepiece==0.2.1 --with sacremoses==0.1.1 \
           python scripts/realdocs_mt.py [--systems marian,nllb] [--docs id1,id2] [--device mps|cpu]

For every doc in DOCS and every system, reads data/realdocs/<doc>/aligned.jsonl ({id, src, ref_es,
section}) and writes data/realdocs/<doc>/mt_<system>.jsonl ({id, src, mt, ref_es, section, lp, system}).
Each aligned segment is split into sentences first (CAT-tool style; almost every segment is one
sentence already), the sentences are translated one per input with beam search, and the
translations are joined with a single space. The run's settings, model revisions and licences go
to data/realdocs/mt_meta.json.

Systems (pinned revisions, both in the HF cache):
  marian  Helsinki-NLP/opus-mt-en-es (MarianMT, OPUS Tatoeba-Challenge eng-spa 2020-08-18), Apache-2.0
  nllb    facebook/nllb-200-distilled-600M (eng_Latn -> spa_Latn), CC-BY-NC-4.0: research use only,
          its outputs are not shown in the public demo
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "realdocs"
DOCS = ["eu-air-passenger-rights", "niddk-hypoglycemia", "ready-hurricanes",
        "irs-failure-to-file-penalty", "fda-medicine-disposal"]

SYSTEMS = {
    "marian": {
        "model_id": "Helsinki-NLP/opus-mt-en-es",
        "revision": "5bc4493d463cf000c1f0b50f8d56886a392ed4ab",
        "architecture": "MarianMT transformer-base (OPUS Tatoeba-Challenge eng-spa, opus-2020-08-18)",
        "license": "apache-2.0",
        "license_evidence": "model card front matter 'license: apache-2.0' (README.md at the pinned revision)",
        "public_demo": True,
        "note": "Apache-2.0: outputs may be shown in the public demo with attribution to Helsinki-NLP / OPUS-MT.",
    },
    "nllb": {
        "model_id": "facebook/nllb-200-distilled-600M",
        "revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "architecture": "NLLB-200 distilled 600M (M2M-100), forced target spa_Latn",
        "license": "cc-by-nc-4.0",
        "license_evidence": "model card front matter 'license: cc-by-nc-4.0' and 'License: CC-BY-NC'; "
                            "intended use: research; card says it is not intended for medical or legal "
                            "domain texts nor for production deployment",
        "public_demo": False,
        "note": "CC-BY-NC-4.0 research model: used here for a non-commercial research comparison only; "
                "outputs are labelled 'NLLB-200 (CC-BY-NC-4.0, research only)' and not shown in the public demo.",
    },
}
GEN = {"num_beams": 4, "do_sample": False, "early_stopping": True, "max_new_tokens_rule": "min(512, 2*len(src_ids)+16)"}
BATCH = 16

# Sentence split: after . ! ? (plus closing quotes/brackets), before an upper-case letter, digit or
# opening quote. Not after list numbering ("1.", "(a)"), single initials or common abbreviations.
_BOUNDARY = re.compile(r"(?<=[.!?])([\"”’)]*)\s+(?=[\"“‘(]?[A-Z0-9])")
_ABBR = re.compile(r"\b(?:No|Nos|Art|Arts|e\.g|i\.e|U\.S|Dr|Mr|Mrs|Ms|St|vs|approx|Inc|Co|Ltd|Jan|Feb|Mar|Apr|"
                   r"Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|p|pp|para|Reg|cf|[A-Z])\.$")
_NUMBERING = re.compile(r"\s*(?:\(?\d+[a-z]?\)?\.|\(?[a-z]{1,3}\)|\d+(?:\.\d+)*)\s*")


def _version(pkg: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def split_sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        left = text[start:m.start()].rstrip("\"”’)")
        if _NUMBERING.fullmatch(left) or _ABBR.search(left):
            continue
        out.append(text[start:m.end(1)])
        start = m.end()
    out.append(text[start:])
    return [s for s in out if s.strip()]


def load(system: str, device: str):
    cfg = SYSTEMS[system]
    kw = {"revision": cfg["revision"]}
    if system == "nllb":
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], src_lang="eng_Latn", **kw)
    else:
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], **kw)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_id"], dtype=torch.float32, **kw).to(device).eval()
    extra = {"forced_bos_token_id": tok.convert_tokens_to_ids("spa_Latn")} if system == "nllb" else {}
    return tok, model, extra


@torch.inference_mode()
def translate(sentences: list[str], tok, model, extra: dict, device: str) -> list[str]:
    order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))
    out: list[str | None] = [None] * len(sentences)
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        enc = tok([sentences[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        max_new = min(512, 2 * int(enc["input_ids"].shape[1]) + 16)
        gen = model.generate(**enc, num_beams=GEN["num_beams"], do_sample=False, early_stopping=True,
                             max_new_tokens=max_new, **extra)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t.strip()
    return out  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="marian,nllb")
    ap.add_argument("--docs", default=",".join(DOCS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    docs = a.docs.split(",")
    meta_path = REAL / "mt_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"systems": {}}
    for system in a.systems.split(","):
        t0 = time.perf_counter()
        tok, model, extra = load(system, a.device)
        n_seg = n_sent = 0
        for doc in docs:
            rows = [json.loads(l) for l in open(REAL / doc / "aligned.jsonl", encoding="utf-8")]
            parts = [split_sentences(r["src"]) for r in rows]
            flat = [s for p in parts for s in p]
            mt = translate(flat, tok, model, extra, a.device)
            k = 0
            with open(REAL / doc / f"mt_{system}.jsonl", "w", encoding="utf-8") as f:
                for r, p in zip(rows, parts):
                    seg_mt = " ".join(mt[k:k + len(p)])
                    k += len(p)
                    f.write(json.dumps({"id": r["id"], "src": r["src"], "mt": seg_mt, "ref_es": r["ref_es"],
                                        "section": r.get("section"), "lp": "en-es", "system": system},
                                       ensure_ascii=False) + "\n")
            n_seg += len(rows)
            n_sent += len(flat)
            print(f"{system} {doc}: {len(rows)} segments, {len(flat)} sentences", flush=True)
        dt = time.perf_counter() - t0
        meta["systems"][system] = {
            **SYSTEMS[system], "decoding": GEN, "batch_size": BATCH, "device": a.device, "dtype": "float32",
            "segments": n_seg, "sentences": n_sent, "wall_s": round(dt, 1),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "machine": platform.machine(),
            "sentencepiece": _version("sentencepiece"), "sacremoses": _version("sacremoses"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "docs": docs,
            "files": [f"data/realdocs/{d}/mt_{system}.jsonl" for d in docs],
        }
        del model
        if a.device == "mps":
            torch.mps.empty_cache()
    meta["segmentation"] = ("each aligned segment split into sentences (rule in scripts/realdocs_mt.py: "
                            "split_sentences), one sentence per MT input, outputs joined with one space")
    meta["public_demo_systems"] = [s for s, c in SYSTEMS.items() if c["public_demo"]]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
