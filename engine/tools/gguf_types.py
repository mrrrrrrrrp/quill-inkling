"""Minimal GGUF header reader: prints tensor name, ggml type and size, grouped for expert tensors."""
import struct, sys, collections

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}
# (block elements, bytes per block)
BLK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22),
       "Q5_1": (32, 24), "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
       "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82),
       "IQ3_XXS": (256, 98), "IQ3_S": (256, 110), "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ4_NL": (32, 18),
       "IQ4_XS": (256, 136), "MXFP4": (32, 17), "NVFP4": (64, 36)}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rstr(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def rval(f, t):
    if t == 8:
        return rstr(f)
    if t == 9:
        (at,) = rd(f, "I")
        (n,) = rd(f, "Q")
        return [rval(f, at) for _ in range(n)]
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
    return rd(f, fmt)[0]


def main(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    (ver,) = rd(f, "I")
    nt, nkv = rd(f, "QQ")
    kv = {}
    for _ in range(nkv):
        k = rstr(f)
        (t,) = rd(f, "I")
        v = rval(f, t)
        kv[k] = v if not isinstance(v, list) or len(v) < 16 else f"<array {len(v)}>"
    for k, v in kv.items():
        if "token" not in k:
            print("kv", k, "=", v)
    groups = collections.defaultdict(list)
    total = 0
    for _ in range(nt):
        name = rstr(f)
        (nd,) = rd(f, "I")
        dims = rd(f, "Q" * nd)
        (tt,) = rd(f, "I")
        (off,) = rd(f, "Q")
        tn = GGML_TYPES.get(tt, str(tt))
        ne = 1
        for d in dims:
            ne *= d
        be, bb = BLK.get(tn, (1, 0))
        nbytes = ne // be * bb
        total += nbytes
        key = name.split(".")
        if "exps" in name or "blk" in name:
            groups[(name.split(".", 2)[2] if name.startswith("blk.") else name)].append((int(key[1]) if key[0] == "blk" else -1, tn, nbytes, dims))
        else:
            print("tensor", name, tn, dims, nbytes)
    for k in sorted(groups):
        items = sorted(groups[k])
        s = " ".join(f"{l}:{t}" for l, t, _, _ in items)
        print(f"{k:40s} {sum(b for _, _, b, _ in items)/1e9:7.3f} GB  dims={items[0][3]}  {s}")
    print("total tensor bytes", total / 1e9)


if __name__ == "__main__":
    main(sys.argv[1])
