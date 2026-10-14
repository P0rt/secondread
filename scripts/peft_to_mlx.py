# Convert a PEFT LoRA (Tinker export, MLP/attn linear layers only) to mlx-lm adapter format.
# PEFT: lora_A [r,in], lora_B [out,r], scaling alpha/r  ->  MLX: lora_a [in,r], lora_b [r,out], scale
import json, os, re, sys
import mlx.core as mx
src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "adapter_config.json")))
w = mx.load(os.path.join(src, "adapter_model.safetensors"))
out, layers, keys = {}, set(), set()
for k, v in w.items():
    m = re.match(r"base_model\.model\.model\.language_model\.(layers\.(\d+)\.(.+))\.lora_([AB])\.weight$", k)
    if not m: raise SystemExit(f"unsupported key {k}")
    path, L, mod, ab = m.groups()
    layers.add(int(L)); keys.add(mod)
    out[f"language_model.model.{path}.lora_{ab.lower()}"] = v.T
os.makedirs(dst, exist_ok=True)
mx.save_safetensors(os.path.join(dst, "adapters.safetensors"), out)
json.dump({"fine_tune_type": "lora", "num_layers": -1,
           "lora_parameters": {"rank": cfg["r"], "scale": cfg["lora_alpha"] / cfg["r"], "dropout": 0.0, "keys": sorted(keys)}},
          open(os.path.join(dst, "adapter_config.json"), "w"), indent=1)
print("wrote", len(out), "tensors; keys", sorted(keys))
