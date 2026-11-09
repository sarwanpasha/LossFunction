"""GIAB preparation without pysam (works on any OS): downloads the chr20 BAMs and truth
sets, builds per-base count tensors with a numba BAM parser (streamed and memory-mapped,
single process), and reuses prep_giab.chunk_job for featurization. Produces the same
outputs as prep_giab.py (data/giab_*.npz)."""
import gzip, os, struct, sys, time, types, zlib, urllib.request
import numpy as np

sys.modules.setdefault("pysam", types.ModuleType("pysam"))   # prep_giab imports pysam at top
import prep_giab as P

try:
    from numba import njit
except Exception:                                            # pure-python fallback (slow; tests)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

D = P.D
DV = "https://storage.googleapis.com/deepvariant/case-study-testdata/"
NIST = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab"
TRIO = {"HG002": "HG002_NA24385_son", "HG003": "HG003_NA24149_father", "HG004": "HG004_NA24143_mother"}
SLEEP = float(os.environ.get("LF_SLEEP", "0"))


def get(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    t = time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(out + ".part", "wb") as fh:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            fh.write(b)
    os.replace(out + ".part", out)
    print("OK", url, f"{os.path.getsize(out)/1e6:.0f}MB {time.time()-t:.0f}s", flush=True)


def download():
    os.makedirs(D, exist_ok=True)
    get("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz", f"{D}/chr20.fa.gz")
    for s, d in TRIO.items():
        base = f"{NIST}/release/AshkenazimTrio/{d}/NISTv4.2.1/GRCh38/{s}_GRCh38_1_22_v4.2.1_benchmark"
        get(base + ".vcf.gz", f"{D}/{s}.vcf.gz")
        try:
            get(base + "_noinconsistent.bed", f"{D}/{s}.bed")
        except Exception:
            get(base + ".bed", f"{D}/{s}.bed")
        get(DV + f"{s}.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam", f"{D}/{s}.chr20.bam")


# ---------------------------------------------------------------- BAM -> counts
NIB = np.full(16, -1, np.int64); NIB[1] = 0; NIB[2] = 1; NIB[4] = 2; NIB[8] = 3   # =ACMGRSVTWYHKDBN


@njit(cache=True)
def u8(b, p):
    return np.int64(b[p])


@njit(cache=True)
def _i32(b, p):
    v = u8(b, p) | (u8(b, p + 1) << 8) | (u8(b, p + 2) << 16) | (u8(b, p + 3) << 24)
    return v - (1 << 32) if v >= (1 << 31) else v


@njit(cache=True)
def _inc(cnt, g, bi, r):
    if cnt[g, bi, r] < 255:
        cnt[g, bi, r] += 1


@njit(cache=True)
def parse_records(buf, n, cnt, chr_id, L, nib):
    off = 0
    nread = 0
    while off + 4 <= n:
        bs = _i32(buf, off)
        if off + 4 + bs > n:
            break
        p = off + 4
        ref = _i32(buf, p); pos = _i32(buf, p + 4)
        lrn = u8(buf, p + 8); mapq = u8(buf, p + 9)
        ncig = u8(buf, p + 12) | (u8(buf, p + 13) << 8)
        flag = u8(buf, p + 14) | (u8(buf, p + 15) << 8)
        lseq = _i32(buf, p + 16)
        if ref == chr_id and (flag & 0xF04) == 0:
            nread += 1
            cig = p + 32 + lrn
            sq = cig + 4 * ncig
            ql = sq + (lseq + 1) // 2
            rev = (flag & 16) != 0
            rpos = pos; qpos = 0
            for k in range(ncig):
                c = u8(buf, cig + 4 * k) | (u8(buf, cig + 4 * k + 1) << 8) | (u8(buf, cig + 4 * k + 2) << 16) | (u8(buf, cig + 4 * k + 3) << 24)
                op = c & 15; ln = c >> 4
                if op == 0 or op == 7 or op == 8:
                    for i in range(ln):
                        r = rpos + i; qi = qpos + i
                        if r < 0 or r >= L or qi >= lseq:
                            continue
                        byte = u8(buf, sq + qi // 2)
                        code = (byte >> 4) if qi % 2 == 0 else (byte & 15)
                        bi = nib[code]
                        if bi < 0:
                            continue
                        _inc(cnt, 0, bi, r)
                        if u8(buf, ql + qi) >= 20:
                            _inc(cnt, 1, bi, r)
                        if mapq >= 20:
                            _inc(cnt, 2, bi, r)
                        if rev:
                            _inc(cnt, 3, bi, r)
                    rpos += ln; qpos += ln
                elif op == 1 or op == 4:
                    qpos += ln
                elif op == 2 or op == 3:
                    rpos += ln
        off += 4 + bs
    return off, nread


def bgzf_stream(path, target=1 << 26):
    """Yield decompressed byte chunks (~target bytes) of a BGZF file."""
    out = []; size = 0
    with open(path, "rb") as fh:
        while True:
            h = fh.read(18)
            if len(h) < 18:
                break
            assert h[:4] == b"\x1f\x8b\x08\x04", "not BGZF"
            bsize = struct.unpack("<H", h[16:18])[0] + 1
            body = fh.read(bsize - 18)
            data = zlib.decompress(body[:-8], -15)
            if data:
                out.append(data); size += len(data)
            if size >= target:
                yield b"".join(out); out = []; size = 0
    if out:
        yield b"".join(out)


def bam_counts(path, out_path, L=P.CHR_LEN, chrom=P.CHR):
    if os.path.exists(out_path):
        return np.load(out_path, mmap_mode="r")
    cnt = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(4, 4, L))
    it = bgzf_stream(path)
    carry = b""; hdr_done = False; chr_id = -1; nread = 0; t = time.time(); k = 0
    for chunk in it:
        buf = carry + chunk
        if not hdr_done:
            assert buf[:4] == b"BAM\x01"
            lt = struct.unpack("<i", buf[4:8])[0]; p = 8 + lt
            nref = struct.unpack("<i", buf[p:p + 4])[0]; p += 4
            for i in range(nref):
                ln = struct.unpack("<i", buf[p:p + 4])[0]
                name = buf[p + 4:p + 4 + ln - 1].decode()
                if name == chrom:
                    chr_id = i
                p += 8 + ln
            assert chr_id >= 0, "chrom not in header"
            buf = buf[p:]; hdr_done = True
        arr = np.frombuffer(buf, np.uint8)
        used, nr = parse_records(arr, len(arr), cnt, chr_id, L, NIB)
        nread += nr; carry = buf[used:]; k += 1
        if k % 10 == 0:
            print(f"  {os.path.basename(path)}: {nread/1e6:.1f}M reads {time.time()-t:.0f}s", flush=True)
        if SLEEP:
            time.sleep(SLEEP)
    cnt.flush(); del cnt
    os.replace(out_path + ".tmp.npy", out_path)
    print(f"  {path}: {nread} reads -> {out_path} ({time.time()-t:.0f}s)", flush=True)
    return np.load(out_path, mmap_mode="r")


# ---------------------------------------------------------------- truth without pysam
def load_truth(s):
    snv, anyv = {}, set()
    with gzip.open(os.path.join(D, f"{s}.vcf.gz"), "rt") as fh:
        for l in fh:
            if l.startswith("#"):
                continue
            f = l.split("\t", 6)
            if f[0] != P.CHR:
                continue
            start = int(f[1]) - 1; ref = f[3]; alts = f[4].split(",")
            anyv.update(range(start, start + len(ref)))
            if len(ref) == 1 and all(len(a) == 1 for a in alts):
                snv[start] = alts[0]
    return snv, anyv


CNT = {}


def counts(s, a, b):
    return np.asarray(CNT[s][:, :, a:b], np.int32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    if not a.no_download:
        download()
    for s in P.SAMPLES:
        CNT[s] = bam_counts(P.bam_path(s), os.path.join(D, f"{s}.cnt.npy"))
    P.counts = counts; P.load_truth = load_truth
    P.REF = P.load_ref(); P.BEDS = {s: P.load_bed(s) for s in P.SAMPLES}
    P.TRUTH = {s: load_truth(s) for s in P.SAMPLES}
    print("truth loaded", {s: len(v[0]) for s, v in P.TRUTH.items()}, flush=True)
    jobs = [(st, min(st + a.chunk, P.CHR_LEN), 7) for st in range(60_000, P.CHR_LEN, a.chunk)]
    acc = {}
    for i, j in enumerate(jobs):
        for k, v in P.chunk_job(j).items():
            acc.setdefault(k, []).append(v)
        print(f"chunk {i+1}/{len(jobs)}", flush=True)
        if SLEEP:
            time.sleep(SLEEP)
    os.makedirs("data", exist_ok=True)
    for k, parts in acc.items():
        X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        pos = np.concatenate([p[2] for p in parts])
        split = (pos >= P.SPLIT_POS).astype(np.int8)
        np.savez_compressed(f"data/giab_{k}.npz", X=X, y=y, split=split, pos=pos)
        print(k, X.shape, "pi=", y.mean(), "test frac", split.mean(), flush=True)
    print("GIABDONE", flush=True)


if __name__ == "__main__":
    main()
