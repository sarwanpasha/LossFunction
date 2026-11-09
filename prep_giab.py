"""Clean (label-free) pileup features for GIAB chr20 candidate SNV sites.

Scenarios produced (data/giab_<scenario>.npz, split: chr20 < 40 Mb train, >= 40 Mb test):
  HG002_35x, HG003_35x, HG004_35x            germline SNV calling at 35x
  HG003_10x                                   germline at in-silico 10x (binomial thinning)
  mosaic_f10, mosaic_f05                      HG002 reads mixed into HG003 at donor
                                              fraction f; positives = HG002-private SNVs
                                              (VAF ~2-5%), negatives = all other candidates.
Features never touch the truth set: per-offset (+-5bp) base-count fractions in 4 read
groups (all / BQ>=20 / MQ>=20 / reverse strand), relative depth, plus reference context.
"""
import argparse, gzip, os, sys
import numpy as np
import pysam
from multiprocessing import Pool

D = "giab_raw"
CHR = "chr20"
CHR_LEN = 64444167
SPLIT_POS = 40_000_000
W = 5
SAMPLES = ["HG002", "HG003", "HG004"]
BASES = "ACGT"
BIDX = {b: i for i, b in enumerate(BASES)}


def bam_path(s):
    return os.path.join(D, f"{s}.chr20.bam")


def load_ref():
    with gzip.open(os.path.join(D, "chr20.fa.gz"), "rt") as fh:
        seq = "".join(l.strip() for l in fh if not l.startswith(">"))
    return seq.upper()


def load_bed(s):
    iv = []
    with open(os.path.join(D, f"{s}.bed")) as fh:
        for l in fh:
            c, a, b = l.split()[:3]
            if c == CHR:
                iv.append((int(a), int(b)))
    iv.sort()
    return np.array(iv, dtype=np.int64)


def in_bed(iv, pos):
    i = np.searchsorted(iv[:, 0], pos, side="right") - 1
    ok = i >= 0
    out = np.zeros(len(pos), bool)
    out[ok] = pos[ok] < iv[i[ok], 1]
    return out


def load_truth(s):
    snv, anyv = {}, set()
    vf = pysam.VariantFile(os.path.join(D, f"{s}.vcf.gz"))
    for r in vf.fetch(CHR):
        for k in range(r.start, r.start + len(r.ref)):
            anyv.add(k)
        if len(r.ref) == 1 and all(a is not None and len(a) == 1 for a in r.alts):
            snv[r.start] = r.alts[0]
    return snv, anyv


def counts(s, a, b):
    bam = pysam.AlignmentFile(bam_path(s))
    base_ok = lambda r: not (r.is_unmapped or r.is_secondary or r.is_supplementary
                             or r.is_duplicate or r.is_qcfail)
    groups = [
        bam.count_coverage(CHR, a, b, quality_threshold=0, read_callback=base_ok),
        bam.count_coverage(CHR, a, b, quality_threshold=20, read_callback=base_ok),
        bam.count_coverage(CHR, a, b, quality_threshold=0,
                           read_callback=lambda r: base_ok(r) and r.mapping_quality >= 20),
        bam.count_coverage(CHR, a, b, quality_threshold=0,
                           read_callback=lambda r: base_ok(r) and r.is_reverse),
    ]
    return np.stack([np.stack([np.asarray(g[i], np.int32) for i in range(4)]) for g in groups])  # (4grp,4base,L)


def thin(C, f, rng):
    return rng.binomial(C, f).astype(np.int32)


def featurize(C, ref, a, cand_rel, alt_idx, med_depth):
    """C: (4,4,L) counts for window [a, a+L). cand_rel: candidate offsets in window."""
    G = C.shape[0]
    L = C.shape[2]
    feats = []
    ref_idx = np.array([BIDX.get(ref[a + i], 0) for i in range(L)])
    depth = C.sum(1).astype(np.float64) + 1e-9          # (G,L)
    for o in range(-W, W + 1):
        j = cand_rel + o
        rf = C[:, ref_idx[j], j] / depth[:, j]          # (G,n)
        af = C[:, alt_idx, j] / depth[:, j]
        mx = C[:, :, j].astype(np.float64)
        mx[:, ref_idx[j], np.arange(len(j))] = -1
        other = np.clip(mx.max(1), 0, None) / depth[:, j]
        rd = depth[:, j] / med_depth
        feats += [rf, af, other, rd]
    F = np.concatenate([f.T for f in feats], 1)        # (n, 4*G*(2W+1))
    c = cand_rel
    altc = C[:, alt_idx, c].T.astype(np.float64)
    hp = np.array([homopolymer(ref, a + k) for k in c])
    gc = np.array([sum(ch in "GC" for ch in ref[max(0, a + k - 10): a + k + 11]) / 21 for k in c])
    ti = np.array([{("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}.__contains__((ref[a + k], BASES[x]))
                   for k, x in zip(c, alt_idx)], float)
    return np.c_[F, np.log1p(altc), np.log1p(depth[0, c]), hp, gc, ti]


def homopolymer(ref, p):
    b = ref[p]; l = p; r = p
    while l > 0 and ref[l - 1] == b and p - l < 20: l -= 1
    while r < len(ref) - 1 and ref[r + 1] == b and r - p < 20: r += 1
    return r - l + 1


def chunk_job(args):
    a, b, seed = args
    rng = np.random.default_rng(seed + a)
    ref = REF
    pa, pb = max(0, a - W), min(CHR_LEN, b + W)
    C = {s: counts(s, pa, pb) for s in SAMPLES}
    out = {}
    scen = {
        "HG002_35x": (C["HG002"], ["HG002"], "germ", 0.05),
        "HG003_35x": (C["HG003"], ["HG003"], "germ", 0.05),
        "HG004_35x": (C["HG004"], ["HG004"], "germ", 0.05),
        "HG003_10x": (thin(C["HG003"], 0.3, rng), ["HG003"], "germ", 0.05),
        "mosaic_f10": (C["HG003"] + thin(C["HG002"], 0.10, rng), ["HG002", "HG003"], "mos", 0.01),
        "mosaic_f05": (C["HG003"] + thin(C["HG002"], 0.05, rng), ["HG002", "HG003"], "mos", 0.01),
    }
    for name, (Cs, beds, kind, afmin) in scen.items():
        L = Cs.shape[2]
        allc = Cs[0]                                    # (4,L)
        depth = allc.sum(0)
        ref_idx = np.array([BIDX.get(ref[pa + i], -1) for i in range(L)])
        nr = allc.copy(); valid = ref_idx >= 0
        nr[ref_idx[valid], np.where(valid)[0]] = -1
        alt_idx = nr.argmax(0); altc = nr.max(0)
        rel = np.arange(L)
        abs_pos = pa + rel
        ok = (abs_pos >= a) & (abs_pos < b) & valid & (depth >= 4) & (altc >= 2) & \
             (altc >= afmin * depth) & (rel >= W) & (rel < L - W)
        for s in beds:
            ok &= in_bed(BEDS[s], abs_pos)
        cand = rel[ok]
        if len(cand) == 0:
            continue
        pos = pa + cand
        if kind == "germ":
            snv, anyv = TRUTH[beds[0]]
            y = np.array([p in snv for p in pos], np.int8)
            keep = np.array([(p in snv) or (p not in anyv) for p in pos])
        else:
            snv2, any2 = TRUTH["HG002"]; snv3, any3 = TRUTH["HG003"]
            y = np.array([(p in snv2) and (p not in any3) for p in pos], np.int8)
            keep = np.array([(p in snv2 and p not in any3) or (p not in any2) or (p in any3) for p in pos])
        cand, pos, y = cand[keep], pos[keep], y[keep]
        if len(cand) == 0:
            continue
        X = featurize(Cs, ref, pa, cand, alt_idx[cand], 35.0 if "10x" not in name else 10.5)
        out[name] = (X.astype(np.float32), y, pos)
    return out


def init(seed):
    global REF, BEDS, TRUTH
    REF = load_ref()
    BEDS = {s: load_bed(s) for s in SAMPLES}
    TRUTH = {s: load_truth(s) for s in SAMPLES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--start", type=int, default=60_000)
    ap.add_argument("--end", type=int, default=CHR_LEN)
    a = ap.parse_args()
    jobs = [(s, min(s + a.chunk, a.end), 7) for s in range(a.start, a.end, a.chunk)]
    acc = {}
    with Pool(a.workers, initializer=init, initargs=(7,)) as pool:
        for i, res in enumerate(pool.imap_unordered(chunk_job, jobs)):
            for k, v in res.items():
                acc.setdefault(k, []).append(v)
            print(f"chunk {i+1}/{len(jobs)}", flush=True)
    os.makedirs("data", exist_ok=True)
    for k, parts in acc.items():
        X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        pos = np.concatenate([p[2] for p in parts])
        split = (pos >= SPLIT_POS).astype(np.int8)
        np.savez_compressed(f"data/giab_{k}.npz", X=X, y=y, split=split, pos=pos)
        print(k, X.shape, "pi=", y.mean(), "test frac", split.mean(), flush=True)


if __name__ == "__main__":
    main()
