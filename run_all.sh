#!/bin/bash
# End-to-end pipeline. Set CUDA_VISIBLE_DEVICES to choose a GPU.
set -e
python prep_tabular.py                     # imbalanced-learn benchmarks + credit card fraud
python prep_cifar.py                       # CIFAR-10 one-vs-rest long-tailed tasks (ResNet-18 features)
python prep_giab_nopysam.py                # GIAB chr20 germline and mosaic scenarios
for d in data/*.npz; do
  python run_dataset.py --data "$d" --out results_es --seeds 5 --workers 1 --steps 3000 --es --minpos 50
done
python aggregate.py results_es 50          # ranking transfer, theory validation, methods
python theory_fs.py results_es             # finite-sample corrected F1 predictions
python summarize.py results_es             # summary statistics and Wilcoxon tests

# Post-hoc threshold policies and probe ablation
for d in data/*.npz; do
  python run_thresh.py --data "$d" --out results_thr --profiles results_es --seeds 5
done
python probe_ablation.py results_es
python rev_summary.py results_es results_thr

# Multi-class long-tailed CIFAR (imbalance ratio 100, ResNet-32)
for ds in cifar10 cifar100; do
  python mc_profile.py --dataset $ds --out results_mc
  for s in 0 1 2; do
    for l in ce focal wce_drw cb_drw la ldam_drw pa_ce_drw paf_drw; do
      python mc_lt.py --dataset $ds --loss $l --seed $s --out results_mc
    done
  done
done
python mc_aggregate.py results_mc
