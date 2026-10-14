"""Checks and conversion steps behind scripts/export_adapter.sh.

Subcommands (run from the repo root, with HF_HUB_OFFLINE=1):

  to-peft ADAPTER_DIR OUT_DIR [--base base35]
      Validate a Tinker LoRA checkpoint, convert it to PEFT with
      tinker_cookbook.weights.build_lora_adapter against the header-only base stub,
      then check every PEFT tensor against the base model's real names and shapes.

  check-gguf LORA_GGUF --peft PEFT_DIR [--base-gguf models/Qwen3.5-4B-Q4_K_M.gguf]
      Replay llama.cpp's adapter loader checks offline (llama-adapter.cpp at b11429):
      general.type/arch/adapter.type, a/b pairing, every target tensor exists in the
      base GGUF, and ne[0]/ne[1] agree. Also checks alpha and tensor count vs PEFT.

  smoke --port PORT
      Ask a llama-server started with --lora for P(Major) on a few built-in
      examples at LoRA scale 0 and 1, and fail if the adapter is not listed or
      has no effect at all.

Every failure exits non-zero with a one-line "error: ..." that says what to do.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GGUF_PY = REPO / "tools" / "llama.cpp" / "gguf-py"

# Modules SecondRead trains (create_lora_training_client(train_mlp=True, train_attn=False,
# train_unembed=False)). These are the only ones this path is verified for.
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}
# Modules known to break the GGUF path, with the reason and the fix.
BROKEN_MODULES = {
    "in_proj_q": "Tinker trains split linear-attention q/k/v, but HF and llama.cpp have a fused "
                 "in_proj_qkv: convert_lora_to_gguf.py fails with 'Can not map tensor'. Retrain with "
                 "train_attn=False, or fuse first with scripts/fuse_qkv.py",
    "unembed_tokens": "Qwen3.5-4B ties embeddings, so an unembed LoRA is remapped onto embed_tokens "
                      "with a layout llama.cpp rejects ('incorrect shape'). Retrain with train_unembed=False",
}
BROKEN_MODULES["in_proj_k"] = BROKEN_MODULES["in_proj_v"] = BROKEN_MODULES["in_proj_q"]
BROKEN_MODULES["embed_tokens"] = BROKEN_MODULES["lm_head"] = BROKEN_MODULES["unembed_tokens"]

LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")
GGUF_LORA_NAME = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)\.weight\.lora_[ab]$")


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def pair_lora(tensors: dict, where: str) -> dict[str, tuple]:
    """Group `<module>.lora_A.weight` / `.lora_B.weight` into {module: (A, B)}."""
    pairs: dict[str, dict[str, object]] = {}
    for key, t in tensors.items():
        m = LORA_KEY.match(key)
        if m is None:
            die(f"{where}: unexpected tensor {key!r} (only LoRA A/B matrices are supported; "
                "a checkpoint with full weights or modules_to_save cannot go through this path)")
        pairs.setdefault(m["module"], {})[m["ab"]] = t
    for module, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            die(f"{where}: {module} has lora_{''.join(ab)} but not its pair")
    return {module: (ab["A"], ab["B"]) for module, ab in pairs.items()}


# ---------------------------------------------------------------------------
# to-peft
# ---------------------------------------------------------------------------

def to_peft(adapter_dir: Path, out_dir: Path, base: Path) -> None:
    import torch
    from safetensors.torch import load_file
    from tinker_cookbook.weights import build_lora_adapter
    from tinker_cookbook.weights._artifacts import get_model_state_shapes

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    for k in ("r", "lora_alpha"):
        if k not in cfg:
            die(f"{adapter_dir}/adapter_config.json has no {k!r}: is this a Tinker LoRA checkpoint?")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])

    # 1. The Tinker checkpoint itself: module set, rank, and whether it was trained at all.
    pairs = pair_lora(load_file(str(adapter_dir / "adapter_model.safetensors")), "Tinker checkpoint")
    leaves = {m.rsplit(".", 1)[-1] for m in pairs}
    for leaf in sorted(leaves & BROKEN_MODULES.keys()):
        die(f"checkpoint has a LoRA on {leaf!r}. {BROKEN_MODULES[leaf]}")
    if leaves - MLP_MODULES:
        warn(f"non-MLP LoRA modules {sorted(leaves - MLP_MODULES)}: not verified on this path; "
             "the GGUF and llama-server checks below will still catch name/shape problems")
    if any(".experts" in m for m in pairs):
        die("checkpoint has MoE expert LoRA; Qwen3.5-4B is dense, so this is the wrong checkpoint")
    for module, (a, b) in pairs.items():
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != r or b.shape[1] != r:
            die(f"{module}: lora_A {tuple(a.shape)} / lora_B {tuple(b.shape)} do not match r={r}")
    dtypes = sorted({str(a.dtype).removeprefix("torch.") for a, _ in pairs.values()})
    zero_b = sum(int(torch.count_nonzero(b) == 0) for _, b in pairs.values())
    if zero_b == len(pairs):
        warn("every lora_B is all zeros: the adapter is a no-op (an untrained/initial checkpoint?)")
    print(f"tinker checkpoint: {len(pairs)} LoRA modules ({', '.join(sorted(leaves))}), "
          f"r={r}, alpha={alpha:g} (scale {alpha / r:g}), dtype {'/'.join(dtypes)}, "
          f"{zero_b} all-zero B")

    # 2. Tinker -> PEFT (remaps model.layers.* -> model.language_model.layers.*).
    #    base must be the local stub: a hub id makes the cookbook snapshot_download 9.3 GB.
    build_lora_adapter(base_model=str(base), adapter_path=str(adapter_dir), output_path=str(out_dir))

    # 3. build_lora_adapter does not check non-fused targets against the base model, so a
    #    naming mismatch would only surface later as a cryptic converter/loader error.
    base_shapes = get_model_state_shapes(base)
    peft = pair_lora(load_file(str(out_dir / "adapter_model.safetensors")), "PEFT output")
    for module, (a, b) in peft.items():
        target = module.removeprefix("base_model.model.") + ".weight"
        if target not in base_shapes:
            die(f"PEFT tensor {module} targets {target!r}, which is not in the base model. "
                "Tinker's key naming probably changed; inspect the checkpoint keys")
        out_f, in_f = base_shapes[target]
        if a.shape[1] != in_f or b.shape[0] != out_f:
            die(f"{target}: base weight is [{out_f}, {in_f}] but LoRA gives "
                f"[{b.shape[0]}, {a.shape[1]}] (adapter trained on a different base model?)")
    if len(peft) != len(pairs):
        die(f"PEFT has {len(peft)} modules but the checkpoint has {len(pairs)}")
    print(f"peft adapter: {len(peft)} modules, all present in the base with matching shapes -> {out_dir}")


# ---------------------------------------------------------------------------
# check-gguf
# ---------------------------------------------------------------------------

def _field(reader, key: str):
    f = reader.get_field(key)
    return None if f is None else f.contents()


def check_gguf(lora_path: Path, base_path: Path, peft_dir: Path) -> None:
    sys.path.insert(0, str(GGUF_PY))  # the gguf-py that matches the converter clone
    import gguf

    peft_cfg = json.loads((peft_dir / "adapter_config.json").read_text())
    lora, base = gguf.GGUFReader(lora_path), gguf.GGUFReader(base_path)

    for key, want in (("general.type", "adapter"), ("adapter.type", "lora"),
                      ("general.architecture", _field(base, "general.architecture"))):
        if (got := _field(lora, key)) != want:
            die(f"{lora_path.name}: {key} = {got!r}, expected {want!r}")
    if (got := _field(lora, "adapter.lora.alpha")) != float(peft_cfg["lora_alpha"]):
        die(f"adapter.lora.alpha = {got!r}, but the PEFT config says {peft_cfg['lora_alpha']}")

    base_ne = {t.name: [int(x) for x in t.shape] for t in base.tensors}
    pairs: dict[str, dict[str, list[int]]] = {}
    for t in lora.tensors:
        stem, _, ab = t.name.rpartition(".lora_")
        if ab not in ("a", "b"):
            die(f"LoRA tensor {t.name!r} has an unexpected suffix")
        if stem == "token_embd.weight":
            # Verified: llama-server b11429 exits with "tensor 'token_embd.weight' has incorrect shape".
            die("token_embd LoRA (from train_unembed=True on a tied-embedding model): "
                f"{BROKEN_MODULES['embed_tokens']}")
        if not GGUF_LORA_NAME.match(t.name):
            warn(f"non-MLP LoRA tensor {t.name} (not verified on this path)")
        pairs.setdefault(stem, {})[ab] = [int(x) for x in t.shape]
    for stem, ab in pairs.items():
        if set(ab) != {"a", "b"}:
            die(f"LoRA tensor pair for {stem!r} is missing one component")
        if stem not in base_ne:
            die(f"LoRA tensor {stem!r} does not exist in {base_path.name} (wrong base model?)")
        a, b, w = ab["a"], ab["b"], base_ne[stem]
        # Same conditions as llama_adapter_lora_init_impl (src/llama-adapter.cpp, b11429).
        if w[0] != a[0] or w[1] != b[1]:
            die(f"{stem}: base ne={w}, lora_a ne={a}, lora_b ne={b}: incorrect shape")
        if a[1] != b[0]:
            die(f"{stem}: lora_a ne={a} and lora_b ne={b} disagree on the rank")

    n_peft = _count_lora_pairs(peft_dir / "adapter_model.safetensors")
    if len(pairs) != n_peft:
        die(f"GGUF has {len(pairs)} LoRA pairs but the PEFT adapter has {n_peft}")
    size_mb = lora_path.stat().st_size / 1e6
    print(f"gguf: arch={_field(lora, 'general.architecture')} type=adapter alpha={peft_cfg['lora_alpha']} "
          f"{len(pairs)} pairs, all present in {base_path.name} with loader-compatible shapes, {size_mb:.1f} MB")


def _count_lora_pairs(path: Path) -> int:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:  # header only, no tensor data
        return sum(k.endswith(".lora_A.weight") for k in f.keys())  # noqa: SIM118


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

# Hand-written en-ru smoke examples: dropped negation, wrong number, faithful.
SMOKE = [
    ("Do not remove the cover while the device is switched on.",
     "Снимите крышку, пока устройство включено.", "Major"),
    ("The warranty is valid for 24 months from the date of purchase.",
     "Гарантия действительна в течение 12 месяцев с даты покупки.", "Major"),
    ("Tap the Settings icon at the top of the screen.",
     "Нажмите на значок «Настройки» в верхней части экрана.", "OK"),
]


def _post(port: int, path: str, body: dict | None = None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def _risk(port: int, ids: list[int], scale: float, label_ids: dict[str, int]) -> tuple[float, float]:
    res = _post(port, "/completion", {"prompt": ids, "n_predict": 1, "n_probs": 200, "temperature": 0,
                                      "cache_prompt": False, "lora": [{"id": 0, "scale": scale}]})
    top = {c["id"]: c["logprob"] for c in res["completion_probabilities"][0]["top_logprobs"]}
    pm, po = (math.exp(top.get(label_ids[k], -30.0)) for k in ("Major", "OK"))
    return pm, pm / (pm + po)


def smoke(port: int) -> None:
    sys.path.insert(0, str(REPO))
    from secondread.prompt import LABEL_IDS, generation_prompt_ids

    adapters = _post(port, "/lora-adapters")
    if not adapters:
        die(f"llama-server on :{port} has no LoRA adapter loaded (start it with --lora)")
    print(f"server :{port} adapters: {[a['path'] for a in adapters]}")
    max_delta = 0.0
    for src, mt, gold in SMOKE:
        ids = generation_prompt_ids(src, mt, "en-ru")
        (pm0, r0), (pm1, r1) = _risk(port, ids, 0.0, LABEL_IDS), _risk(port, ids, 1.0, LABEL_IDS)
        max_delta = max(max_delta, abs(pm1 - pm0))
        print(f"  gold={gold:5}  P(Major) scale0={pm0:.4f} scale1={pm1:.4f}   risk scale0={r0:.3f} scale1={r1:.3f}")
    if max_delta == 0.0:
        die("P(Major) is identical at LoRA scale 0 and 1: the adapter has no effect")
    print(f"max |P(Major) scale1 - scale0| = {max_delta:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("to-peft", help="validate a Tinker checkpoint and convert it to PEFT")
    p.add_argument("adapter", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--base", type=Path, default=REPO / "base35")
    g = sub.add_parser("check-gguf", help="replay llama.cpp's LoRA loader checks offline")
    g.add_argument("lora", type=Path)
    g.add_argument("--peft", type=Path, required=True)
    g.add_argument("--base-gguf", type=Path, default=REPO / "models" / "Qwen3.5-4B-Q4_K_M.gguf")
    s = sub.add_parser("smoke", help="P(Major) at LoRA scale 0 vs 1 on a running llama-server")
    s.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "to-peft":
        to_peft(a.adapter, a.out, a.base)
    elif a.cmd == "check-gguf":
        check_gguf(a.lora, a.base_gguf, a.peft)
    else:
        smoke(a.port)


if __name__ == "__main__":
    main()
