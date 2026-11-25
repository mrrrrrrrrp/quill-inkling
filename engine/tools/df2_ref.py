"""NumPy reference of one DFlash2 draft against the engine's dump (QUILL_DF2_DUMP).

usage: python -I df2_ref.py DUMP_DIR DRAFT_DIR TARGET_DIR
Reports, per stage, the relative error between the engine and this float32 reference built from the
checkpoints (llama.cpp src/models/dflash*.cpp semantics).
"""
import json
import struct
import sys

import numpy as np

H, NH, NKV, HD, QD, KVD, INTER, WIN, SLOTS, BLOCK = 2304, 32, 8, 128, 4096, 1024, 6912, 2048, 2056, 8
EPS, THETA, MASK = 1e-6, 10000.0, 200056


class ST:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def get(self, name, rows=None):
        t = self.h[name]
        a, b = t["data_offsets"]
        shape = t["shape"]
        if rows is not None:  # rows of a 2D tensor
            rowb = shape[1] * 2
            out = []
            for r in rows:
                self.f.seek(self.base + a + r * rowb)
                out.append(np.frombuffer(self.f.read(rowb), dtype=np.uint16))
            u = np.stack(out)
        else:
            self.f.seek(self.base + a)
            u = np.frombuffer(self.f.read(b - a), dtype=np.uint16).reshape(shape)
        assert t["dtype"] == "BF16", (name, t["dtype"])
        return (u.astype(np.uint32) << 16).view(np.float32)


def bf16file(path, shape):
    u = np.fromfile(path, dtype=np.uint16).reshape(shape)
    return (u.astype(np.uint32) << 16).view(np.float32)


def bfr(x):  # round to bf16
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope(x, pos):  # x [..., HD], NeoX halves
    d = np.arange(HD // 2, dtype=np.float64)
    inv = THETA ** (-2 * d / HD)
    ang = np.float64(pos) * inv
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[..., : HD // 2], x[..., HD // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main(dump, draft_dir, target_dir):
    anchor, P, cstart, cn = map(int, open(dump + "/meta.txt").read().split())
    print(f"anchor {anchor} P {P} last commit {cstart}+{cn}")
    d = ST(draft_dir + "/model.safetensors")
    t = ST(target_dir + "/model-00001-of-00001.safetensors")
    ring_pos = np.fromfile(dump + "/ring_pos.i32", dtype=np.int32)
    rk = [bf16file(f"{dump}/ringk_l{l}.bf16", (SLOTS, NKV, HD)) for l in range(5)]
    rv = [bf16file(f"{dump}/ringv_l{l}.bf16", (SLOTS, NKV, HD)) for l in range(5)]
    xs = np.fromfile(dump + "/x_layers.f32", dtype=np.float32).reshape(5, BLOCK, H)

    # ---- injection check: last committed rows -> ring rows of layer 0..4
    feats = bf16file(dump + "/feats.bf16", (BLOCK, 4 * H))[:cn]
    fc = d.get("fc.weight")
    g = bfr(rms(bfr(feats @ fc.T), d.get("hidden_norm.weight")))
    for l in range(5):
        p = f"layers.{l}.self_attn."
        k = bfr(g @ d.get(p + "k_proj.weight").T).reshape(cn, NKV, HD)
        v = bfr(g @ d.get(p + "v_proj.weight").T).reshape(cn, NKV, HD)
        k = bfr(rms(k, d.get(p + "k_norm.weight")))
        k = np.stack([rope(k[i], cstart + i) for i in range(cn)])
        slots = [(cstart + i) % WIN for i in range(cn)]
        print(f"inject l{l}: k rel {rel(rk[l][slots], k):.4f}  v rel {rel(rv[l][slots], v):.4f}")

    # ---- block forward from the engine's ring context
    emb = t.get("model.llm.embed.weight", rows=[anchor, MASK])
    emb = bfr(rms(emb, t.get("model.llm.embed_norm.weight")))  # Inkling get_input_embeddings: embed_norm(embed)
    x = np.stack([emb[0]] + [emb[1]] * 7).astype(np.float32)
    valid = np.zeros((BLOCK, SLOTS), dtype=bool)
    for i in range(BLOCK):
        q = P + i
        valid[i, :WIN] = (ring_pos >= 0) & (ring_pos < P) & (q - ring_pos < WIN)
        valid[i, WIN:] = True
    print("context keys valid:", int(valid[0, :WIN].sum()))
    for l in range(5):
        p = f"layers.{l}."
        base_a = d.get(p + "attention_conv.base_kernel").reshape(2, 2, H)
        base_m = d.get(p + "mlp_conv.base_kernel").reshape(2, 2, H)

        def conv(inp, dyn, base, side):
            gi = np.arange(H) // 16
            w0 = base[side, 0][None, :] + dyn[:, side * 288 + gi]
            w1 = base[side, 1][None, :] + dyn[:, side * 288 + 144 + gi]
            prev = np.concatenate([np.zeros((1, H), np.float32), inp[:-1]], 0)
            return w0 * inp + w1 * prev

        n = bfr(rms(x, d.get(p + "input_layernorm.weight")))
        dyn = bfr(n @ d.get(p + "attention_conv.kernel_projection.weight").T)
        nc = bfr(conv(n, dyn, base_a, 0))
        q = bfr(nc @ d.get(p + "self_attn.q_proj.weight").T).reshape(BLOCK, NH, HD)
        k = bfr(nc @ d.get(p + "self_attn.k_proj.weight").T).reshape(BLOCK, NKV, HD)
        v = bfr(nc @ d.get(p + "self_attn.v_proj.weight").T).reshape(BLOCK, NKV, HD)
        q = bfr(rms(q, d.get(p + "self_attn.q_norm.weight")))
        k = bfr(rms(k, d.get(p + "self_attn.k_norm.weight")))
        q = np.stack([rope(q[i], P + i) for i in range(BLOCK)])
        k = np.stack([rope(k[i], P + i) for i in range(BLOCK)])
        K = np.concatenate([rk[l][:WIN], k], 0)  # [SLOTS, 8, 128]
        V = np.concatenate([rv[l][:WIN], v], 0)
        print(f"  l{l} block k vs engine ring rows: {rel(rk[l][WIN:], k):.4f}")
        att = np.zeros((BLOCK, NH, HD), np.float32)
        for h in range(NH):
            gk = h // 4
            s = (q[:, h] @ K[:, gk].T) / np.sqrt(HD)
            s = np.where(valid, s, -np.inf)
            s = s - s.max(-1, keepdims=True)
            pr = np.exp(s)
            pr /= pr.sum(-1, keepdims=True)
            att[:, h] = pr @ V[:, gk]
        o = bfr(bfr(att.reshape(BLOCK, QD)) @ d.get(p + "self_attn.o_proj.weight").T)
        x = x + conv(o, dyn, base_a, 1)
        f = bfr(rms(x, d.get(p + "post_attention_layernorm.weight")))
        dyn = bfr(f @ d.get(p + "mlp_conv.kernel_projection.weight").T)
        fc_ = bfr(conv(f, dyn, base_m, 0))
        gg = fc_ @ d.get(p + "mlp.gate_proj.weight").T
        uu = fc_ @ d.get(p + "mlp.up_proj.weight").T
        act = bfr(bfr(gg) / (1 + np.exp(-bfr(gg))) * bfr(uu))
        dn = bfr(act @ d.get(p + "mlp.down_proj.weight").T)
        x = x + conv(dn, dyn, base_m, 1)
        print(f"layer {l}: x rel {rel(xs[l], x):.4f}  (|x| {np.linalg.norm(x):.1f})")
    h = bfr(rms(x, d.get("norm.weight")))
    he = bf16file(dump + "/h.bf16", (BLOCK, H))
    print(f"h rel {rel(he, h):.4f}")
    cand = np.fromfile(dump + "/cand_id.i32", dtype=np.int32).reshape(7, 16)
    U = t.get("model.llm.unembed.weight", rows=None)[:200058]
    lg = h[1:] @ U.T
    ref_top = np.argsort(-lg, -1)[:, :16]
    for i in range(7):
        print(f"row {i + 1}: ref top5 {ref_top[i, :5].tolist()}  engine {sorted(cand[i].tolist())[:5]}...  overlap "
              f"{len(set(ref_top[i].tolist()) & set(cand[i].tolist()))}/16")


if __name__ == "__main__":
    main(*sys.argv[1:4])
