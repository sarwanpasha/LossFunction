"""Fetch chr20 BAMs, GIAB v4.2.1 truth VCF/BED and the chr20 reference into giab_raw/.
Tries several public mirrors per file; BAMs are region-sliced to chr20 remotely."""
import os, subprocess, sys
import pysam

D = "giab_raw"; os.makedirs(D, exist_ok=True)
DV = "https://storage.googleapis.com/deepvariant/case-study-testdata/"
NIST = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab"
TRIO = {"HG002": "HG002_NA24385_son", "HG003": "HG003_NA24149_father", "HG004": "HG004_NA24143_mother"}


def get(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return True
    r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", out + ".part", url])
    if r.returncode == 0:
        os.rename(out + ".part", out); print("OK", url, flush=True); return True
    print("FAIL", url, flush=True)
    return False


get("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz", f"{D}/chr20.fa.gz")
for s, d in TRIO.items():
    base = f"{NIST}/release/AshkenazimTrio/{d}/NISTv4.2.1/GRCh38/{s}_GRCh38_1_22_v4.2.1_benchmark"
    get(base + ".vcf.gz", f"{D}/{s}.vcf.gz"); get(base + ".vcf.gz.tbi", f"{D}/{s}.vcf.gz.tbi")
    get(base + "_noinconsistent.bed", f"{D}/{s}.bed") or get(base + ".bed", f"{D}/{s}.bed")
    out = f"{D}/{s}.chr20.bam"
    if os.path.exists(out + ".bai"):
        continue
    ok = False
    for url in [DV + f"{s}.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"]:
        if get(url, out) and get(url + ".bai", out + ".bai"):
            ok = True; break
    if not ok:  # remote slice of the GIAB 2x250 novoalign BAM
        for url in [f"{NIST}/data/AshkenazimTrio/{d}/NIST_Illumina_2x250bps/novoalign_bams/{s}.GRCh38.2x250.bam"]:
            try:
                pysam.view("-b", "-o", out, url, "chr20", catch_stdout=False)
                pysam.index(out); ok = True; print("OK slice", url, flush=True); break
            except Exception as e:
                print("FAIL slice", url, e, flush=True)
    if not ok:
        print("NO BAM for", s, file=sys.stderr)
print("done")
