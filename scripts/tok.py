import sys
from tokenizers import Tokenizer
for p in sys.argv[1:]:
    t = Tokenizer.from_file(p)
    print("==", p)
    for s in ["MAJOR", " MAJOR", "OK", " OK", "MINOR", " MINOR", "YES", "NO", "ERR", "BAD", "GOOD", "major", "ok"]:
        e = t.encode(s, add_special_tokens=False)
        print(f"  {s!r:10} -> {e.ids} {e.tokens}")
