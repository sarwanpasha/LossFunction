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
