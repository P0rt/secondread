# Fuse Tinker's split linear_attn.in_proj_{q,k,v} LoRA into HF's fused in_proj_qkv (rank 3r).
# llama.cpp scales every tensor by alpha / rank_of_that_tensor, so B is multiplied by 3 to keep alpha/r.
import json, os, re, shutil, sys, torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "adapter_config.json")))
w = load_file(os.path.join(src, "adapter_model.safetensors"))
out, groups = {}, {}
for k, v in w.items():
    m = re.match(r"(.*\.linear_attn)\.in_proj_([qkv])\.lora_([AB])\.weight$", k)
    if m: groups.setdefault(m.group(1), {})[(m.group(2), m.group(3))] = v
    else: out[k] = v
for pfx, g in groups.items():
    A = torch.cat([g[("q","A")], g[("k","A")], g[("v","A")]], 0)          # [3r, H]
    B = 3.0 * torch.block_diag(g[("q","B")], g[("k","B")], g[("v","B")])  # [q+k+v, 3r]
    out[f"{pfx}.in_proj_qkv.lora_A.weight"], out[f"{pfx}.in_proj_qkv.lora_B.weight"] = A.contiguous(), B.contiguous()
os.makedirs(dst, exist_ok=True)
save_file(out, os.path.join(dst, "adapter_model.safetensors"))
cfg["target_modules"] = sorted({t for t in cfg["target_modules"] if t not in ("in_proj_q","in_proj_k","in_proj_v")} | ({"in_proj_qkv"} if groups else set()))
json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"), indent=1)
# self-check: fused delta (alpha/(3r)) == concatenated split deltas (alpha/r)
r, a = cfg["r"], cfg["lora_alpha"]
for pfx, g in groups.items():
    ref = torch.cat([g[(p,"B")] @ g[(p,"A")] for p in "qkv"], 0) * (a / r)
    A, B = out[f"{pfx}.in_proj_qkv.lora_A.weight"], out[f"{pfx}.in_proj_qkv.lora_B.weight"]
    got = (B @ A) * (a / A.shape[0])
    print(pfx, "max abs diff", float((ref - got).abs().max()))
