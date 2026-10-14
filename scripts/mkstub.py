import struct, sys, urllib.request, os
repo, outdir = sys.argv[1], sys.argv[2]
sizes = dict(a.split('=') for a in sys.argv[3:])
for fn, size in sizes.items():
    url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    def rng(a, b):
        r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        return urllib.request.urlopen(r).read()
    n = struct.unpack("<Q", rng(0, 7))[0]
    hdr = rng(8, 8 + n - 1)
    p = os.path.join(outdir, fn)
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", n) + hdr)
        f.truncate(int(size))   # sparse: logical size = real shard, no data blocks written
    print(fn, "header bytes", n)
