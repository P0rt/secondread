"""Write a FAKE Tinker-format, MLP-only LoRA adapter for testing the export path.

usage: uv run python scripts/mk_fake_adapter.py OUT_DIR [--base base35] [--rank 16]
       [--alpha 32] [--std 0.02] [--dtype bf16|f32] [--seed 0]

The output mimics what `tinker checkpoint download` extracts for a run trained with
create_lora_training_client(train_mlp=True, train_attn=False, train_unembed=False):

    OUT_DIR/adapter_model.safetensors   base_model.model.model.layers.N.mlp.{gate,up,down}_proj.lora_{A,B}.weight
    OUT_DIR/adapter_config.json         {"r": ..., "lora_alpha": ...}

Key names follow Tinker's internal (text-only) naming, which build_lora_adapter remaps
to the HF `model.language_model.*` prefix. Weights are random and NON-zero (unlike a
freshly initialised LoRA, whose B is zero), so the adapter visibly shifts the output
distribution once llama-server applies it. It is not a meaningful classifier.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", type=Path)
ap.add_argument("--base", type=Path, default=Path("base35"), help="dir with the base config.json")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--std", type=float, default=0.02, help="std of the random A and B entries")
ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = json.loads((a.base / "config.json").read_text())
text = cfg.get("text_config", cfg)  # Qwen3.5 is a VL model: LM dims live under text_config
hidden, inter, n_layers = text["hidden_size"], text["intermediate_size"], text["num_hidden_layers"]
dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[a.dtype]

g = torch.Generator().manual_seed(a.seed)
# (in_features, out_features); PEFT/Tinker layout is A [r, in], B [out, r]
shapes = {"gate_proj": (hidden, inter), "up_proj": (hidden, inter), "down_proj": (inter, hidden)}
tensors = {}
for layer in range(n_layers):
    for proj, (d_in, d_out) in shapes.items():
        key = f"base_model.model.model.layers.{layer}.mlp.{proj}"
        tensors[f"{key}.lora_A.weight"] = (torch.randn(a.rank, d_in, generator=g) * a.std).to(dtype)
        tensors[f"{key}.lora_B.weight"] = (torch.randn(d_out, a.rank, generator=g) * a.std).to(dtype)

a.out.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(a.out / "adapter_model.safetensors"))
(a.out / "adapter_config.json").write_text(json.dumps({"r": a.rank, "lora_alpha": a.alpha}, indent=2) + "\n")
size_mb = (a.out / "adapter_model.safetensors").stat().st_size / 1e6
print(f"wrote {len(tensors)} tensors ({n_layers} layers x 3 MLP projections x A/B), r={a.rank}, "
      f"alpha={a.alpha}, {a.dtype}, {size_mb:.1f} MB -> {a.out}")
