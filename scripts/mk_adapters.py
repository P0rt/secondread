import json, sys, os, torch
from safetensors.torch import save_file
out = sys.argv[1]
H, I, r = 2560, 9216, 8
torch.manual_seed(0)
def write(name, tensors, targets):
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3.5-4B", "r": r, "lora_alpha": 32,
               "target_modules": targets, "bias": "none", "fan_in_fan_out": False, "task_type": "CAUSAL_LM",
               "rank_pattern": {}, "alpha_pattern": {}}, open(os.path.join(d, "adapter_config.json"), "w"))
P = "base_model.model.model.language_model.layers"
# A: MLP-only (layers 0 and 3) + full-attention q/o on layer 3 + linear-attn z/out on layer 0
t = {}
for L in (0, 3):
    for m, (i, o) in {"gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H)}.items():
        t[f"{P}.{L}.mlp.{m}.lora_A.weight"] = torch.randn(r, i) * 0.01
        t[f"{P}.{L}.mlp.{m}.lora_B.weight"] = torch.randn(o, r) * 0.01
write("mlp_only", {k: v for k, v in t.items()}, ["gate_proj", "up_proj", "down_proj"])
# B: Tinker-style split linear-attn q/k/v (layer 0)
lk, lv, dk, dv = 16, 32, 128, 128
t2 = {}
for m, o in {"in_proj_q": lk*dk, "in_proj_k": lk*dk, "in_proj_v": lv*dv}.items():
    t2[f"{P}.0.linear_attn.{m}.lora_A.weight"] = torch.randn(r, H) * 0.01
    t2[f"{P}.0.linear_attn.{m}.lora_B.weight"] = torch.randn(o, r) * 0.01
write("split_qkv", t2, ["in_proj_q", "in_proj_k", "in_proj_v"])
# C: fused in_proj_qkv + in_proj_z + out_proj on layer 0 (what a fixed-up adapter would look like)
t3 = {f"{P}.0.linear_attn.in_proj_qkv.lora_A.weight": torch.randn(r, H) * 0.01,
      f"{P}.0.linear_attn.in_proj_qkv.lora_B.weight": torch.randn(2*lk*dk + lv*dv, r) * 0.01,
      f"{P}.0.linear_attn.in_proj_z.lora_A.weight": torch.randn(r, H) * 0.01,
      f"{P}.0.linear_attn.in_proj_z.lora_B.weight": torch.randn(lv*dv, r) * 0.01,
      f"{P}.0.linear_attn.out_proj.lora_A.weight": torch.randn(r, lv*dv) * 0.01,
      f"{P}.0.linear_attn.out_proj.lora_B.weight": torch.randn(H, r) * 0.01,
      f"{P}.3.self_attn.q_proj.lora_A.weight": torch.randn(r, H) * 0.01,
      f"{P}.3.self_attn.q_proj.lora_B.weight": torch.randn(2*16*256, r) * 0.01}
write("fused_linattn", t3, ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj"])
# D: Tinker unembed remapped onto tied embed_tokens (linear orientation A[r,H], B[V,r])
V = 248320
t4 = {f"{P[:-len('.layers')]}.embed_tokens.lora_A.weight": torch.randn(r, H) * 0.01,
      f"{P[:-len('.layers')]}.embed_tokens.lora_B.weight": torch.randn(V, r) * 0.01}
write("unembed_tied", t4, ["embed_tokens"])
print("ok", sorted(os.listdir(out)))
