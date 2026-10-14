import sys
sys.path.insert(0, sys.argv[1])  # llama.cpp gguf-py
from gguf import GGUFReader
r = GGUFReader(sys.argv[2])
names = [t.name for t in r.tensors]
print(names[:3], {k: r.fields[k].parts[-1].tobytes()[:40] for k in ["general.architecture", "general.type"] if k in r.fields})
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
tok = get_tokenizer("Qwen/Qwen3.5-4B")
ren = get_renderer("qwen3_5_disable_thinking", tok)
for lab in ["Major", "OK", "ERR", "MAJOR"]:
    msgs = [{"role": "user", "content": "SRC: Do not pay.\nMT: Paga.\nAnswer Major or OK."}, {"role": "assistant", "content": lab}]
    mi, wts = ren.build_supervised_example(msgs)
    toks = mi.to_ints()
    on = [(toks[i], tok.decode([toks[i]])) for i in range(len(toks)) if wts[i] > 0]
    print(lab, "weighted tokens:", on)
    gp = ren.build_generation_prompt(msgs[:1]).to_ints()
    print("  gen prompt tail:", repr(tok.decode(gp[-8:])), "len", len(gp))
