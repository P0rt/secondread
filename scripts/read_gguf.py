import sys
sys.path.insert(0, sys.argv[1])
import gguf
for p in sys.argv[2:]:
    r = gguf.GGUFReader(p)
    kv = {f.name: f for f in r.fields.values()}
    def get(k):
        f = kv.get(k)
        if f is None: return None
        v = f.parts[f.data[0]]
        return bytes(v).decode() if f.types[0] == gguf.GGUFValueType.STRING else v.tolist()
    print(p.split("/")[-1], "arch", get("general.architecture"), "type", get("general.type"), "adapter.type", get("adapter.type"), "alpha", get("adapter.lora.alpha"), "n_tensors", len(r.tensors))
