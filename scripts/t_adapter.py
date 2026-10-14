import json, sys, os, torch
from safetensors.torch import save_file, load_file
stub, fake, out = sys.argv[1:4]
r, H, I, L = 16, 2560, 9216, 32
w = {}
for i in range(L):
    p = f"base_model.model.model.layers.{i}.mlp"
    for proj, (din, dout) in {"gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H)}.items():
        w[f"{p}.{proj}.lora_A.weight"] = torch.randn(r, din, dtype=torch.bfloat16) * 0.01
        w[f"{p}.{proj}.lora_B.weight"] = torch.randn(dout, r, dtype=torch.bfloat16) * 0.01
save_file(w, f"{fake}/adapter_model.safetensors")
json.dump({"r": r, "lora_alpha": 32}, open(f"{fake}/adapter_config.json", "w"))
from tinker_cookbook.weights import build_lora_adapter
build_lora_adapter(base_model=stub, adapter_path=fake, output_path=out)
o = load_file(f"{out}/adapter_model.safetensors")
ks = sorted(o)
print("n tensors", len(o), "example", ks[0], tuple(o[ks[0]].shape), o[ks[0]].dtype)
print(open(f"{out}/adapter_config.json").read())
print("size MB", os.path.getsize(f"{out}/adapter_model.safetensors") / 1e6)
