import json, sys, mlx.core as mx
from mlx_lm.utils import load_model, load_adapters
from mlx_lm.models import qwen3_5
cfg = json.load(open(sys.argv[1]))
cfg["text_config"]["num_hidden_layers"] = 4; cfg["text_config"]["vocab_size"] = 1024
cfg["text_config"]["layer_types"] = cfg["text_config"]["layer_types"][:4]
args = qwen3_5.ModelArgs.from_dict(cfg)
model = qwen3_5.Model(args)
model = load_adapters(model, sys.argv[2])
x = mx.array([[1, 2, 3, 4]])
logits = model(x)
lp = logits[0, -1] - mx.logsumexp(logits[0, -1])
print("adapter loaded OK; logits", logits.shape, "sum p =", float(mx.exp(lp).sum()))
