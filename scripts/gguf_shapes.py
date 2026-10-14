import struct, sys, urllib.request
url = sys.argv[1]
r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gguf-head"}))
nread = 0
def rd(n):
    global nread
    b = b""
    while len(b) < n:
        c = r.read(n - len(b))
        if not c: raise EOFError
        b += c
    nread += n
    return b
def u32(): return struct.unpack("<I", rd(4))[0]
def u64(): return struct.unpack("<Q", rd(8))[0]
def s(): return rd(u64()).decode("utf-8", "replace")
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def val(t):
    if t == 8: return s()
    if t == 9:
        at = u32(); n = u64()
        if at == 8:
            for _ in range(n): s()
        else: rd(SZ[at]*n)
        return f"<array type={at} n={n}>"
    b = rd(SZ[t])
    fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}[t]
    return struct.unpack(fmt, b)[0]
assert rd(4) == b"GGUF"
ver = u32(); nt = u64(); nkv = u64()
kv = {}
for _ in range(nkv):
    k = s(); t = u32(); kv[k] = val(t)
names = []
types = {}
for _ in range(nt):
    n = s(); nd = u32(); dims = [u64() for _ in range(nd)]; tt = u32(); off = u64()
    names.append(n); types[n] = (tt, dims)
arch = kv.get("general.architecture")
print("hadamard/prism keys:", [k for k in kv if "hadamard" in k or "prism" in k or "hdmd" in k]); print("arch", arch, "n_tensors", nt, "header_bytes_read", nread)
for k in ["general.file_type", f"{arch}.block_count", f"{arch}.context_length", "tokenizer.ggml.model", "tokenizer.ggml.pre"]:
    print(" ", k, kv.get(k))
print("  has output.weight:", "output.weight" in names, types.get("output.weight"))
print("  token_embd:", types.get("token_embd.weight"))
[print("  ", n, types[n]) for n in names if n.startswith(("blk.0.attn_qkv","blk.0.attn_gate","blk.0.ssm_out","blk.0.ffn_up","blk.3.attn_q.","blk.3.attn_output"))]
