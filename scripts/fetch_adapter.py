"""Download a trained LoRA from Tinker as files, without the checkpoint-archive endpoint.

usage:
  set -a && . ./.env && set +a
  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/fetch_adapter.py \
      tinker://<run-id>/weights/final [--out ckpt] [--name final]
  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/fetch_adapter.py \
      tinker://<run-id>/external_weights/<name> [--out ckpt]

Why: `tinker checkpoint download` (tinker 0.33.1) goes through
RestClient.get_checkpoint_archive_url, which the SDK marks deprecated ("the server may
already reject it for your organization"). On 2026-10-14 it hung for over 10 minutes
without an error for a 69 MB sampler checkpoint. The supported path is
TrainingClient.save_weights_external() + RestClient.get_external_weights_urls(), and that
needs a training client holding the weights:

  1. a `.../weights/<name>` (training state) path: create an MLP-only LoRA training
     client with the same flags as train_tinker.py (never from checkpoint metadata, whose
     missing flags the SDK defaults to True), load the weights (no optimizer state, no
     training tokens), and save them with save_weights_external(<name>);
  2. an `.../external_weights/<name>` path: already exported, just download.

Files land in OUT/<run-id>_external_weights_<name>/, the same layout as
`tinker checkpoint download` (OUT/<run-id>_<kind>_<name>/), so
`scripts/export_adapter.sh OUT` finds the adapter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import tinker

from secondread.prompt import BASE_MODEL

LORA = dict(train_mlp=True, train_attn=False, train_unembed=False)  # = scripts/train_tinker.py


async def export(state_path: str, name: str, rank: int) -> str:
    service = tinker.ServiceClient()
    info = await service.create_rest_client().get_weights_info_by_tinker_path(state_path).result_async()
    if info.lora_rank != rank or info.train_attn or info.train_unembed:
        raise SystemExit(f"{state_path}: weights info {info} does not match rank={rank} {LORA}")
    tc = await service.create_lora_training_client_async(base_model=info.base_model or BASE_MODEL, rank=rank, **LORA)
    await (await tc.load_state_async(state_path)).result_async()
    t0 = time.perf_counter()
    res = await (await tc.save_weights_external_async(name)).result_async()
    print(f"save_weights_external({name!r}) -> {res.path} in {time.perf_counter() - t0:.0f}s")
    return res.path


def download(ext_path: str, out: Path, force: bool) -> Path:
    rest = tinker.ServiceClient().create_rest_client()
    urls = rest.get_external_weights_urls(ext_path).result().urls
    target = out / ext_path.removeprefix("tinker://").replace("/", "_")
    if target.exists():
        if not force:
            raise SystemExit(f"{target} exists (use --force)")
        shutil.rmtree(target)
    for rel, url in sorted(urls.items()):
        dest = (target / rel).resolve()
        if target.resolve() not in dest.parents:
            raise SystemExit(f"refusing file path outside the target dir: {rel!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(dest)
        print(f"  {rel}: {dest.stat().st_size / 1e6:.1f} MB")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="tinker://<run>/weights/<name> or tinker://<run>/external_weights/<name>")
    ap.add_argument("--out", type=Path, default=Path("ckpt"))
    ap.add_argument("--name", help="external weights name (default: the checkpoint's name)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="replace an existing download dir")
    a = ap.parse_args()

    parsed = tinker.types.ParsedCheckpointTinkerPath.from_tinker_path(a.path)
    kind, _, ckpt_name = parsed.checkpoint_id.partition("/")
    if kind == "weights":
        ext = asyncio.run(export(a.path, a.name or ckpt_name, a.rank))
    elif kind == "external_weights":
        ext = a.path
    else:
        sys.exit(f"{a.path}: need a weights/ (training state) or external_weights/ path, not {kind}/ "
                 "(a sampler_weights/ path cannot be loaded into a training client; use the run's weights/ path)")
    target = download(ext, a.out, a.force)
    cfg = target / "adapter_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        print(f"adapter_config: r={c.get('r')} lora_alpha={c.get('lora_alpha')} "
              f"target_modules={c.get('target_modules')}")
    print(f"downloaded to {target}\nnext: scripts/export_adapter.sh {target}")


if __name__ == "__main__":
    main()
