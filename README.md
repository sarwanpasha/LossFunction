# LossFunction

Code for studying how imbalance-aware binary losses behave through their *difficulty profile*, the distribution of the class posterior η(x) = P(y = 1 | x).

Each loss trains a classifier whose decision threshold on η is fixed by the loss itself. Its F1 score and its risk of collapsing to no positive predictions can therefore be predicted from the profile before any training. The code also:

- builds profile-adapted losses;
- generates difficulty-matched synthetic data for choosing a loss without the real data;
- applies a finite-sample correction to the predicted F1.

## Contents

| File | Purpose |
|---|---|
| `lf_core.py` | Loss pool, implied thresholds, F1 prediction from a profile, cross-fitted probe, data generators (Gaussian, copula, GMM, DMS-T, DMS-L, noisy-histogram DMS-L) |
| `lf_train.py` | MLP training under a given loss, with optional validation early stopping and the self-calibrating focal loss (SCFL) |
| `run_dataset.py` | Runs every (world, loss, seed) cell for one dataset; the losses include PAF and PA-BCE |
| `aggregate.py` | Ranking transfer, theory validation and method comparison tables |
| `theory_fs.py` | Noisy-threshold (finite-sample) correction of the predicted F1 and its gap bound |
| `summarize.py` | Per-dataset summaries and paired Wilcoxon tests with Holm correction |
| `prep_tabular.py` | Imbalanced-learn benchmark datasets and the credit card fraud dataset |
| `prep_cifar.py` | CIFAR-10 one-vs-rest long-tailed tasks on frozen ResNet-18 features |
| `prep_giab.py`, `prep_giab_nopysam.py`, `download_giab.py` | GIAB chr20 germline and mosaic variant scenarios (with or without pysam) |
| `run_all.sh` | End-to-end pipeline |

## Usage

```bash
pip install -r requirements.txt
bash run_all.sh
```

A single dataset can be run with:

```bash
python run_dataset.py --data data/imb_ecoli.npz --out results_es --seeds 5 --es --minpos 50
```

## License

MIT
