# miDGD: Deep Generative Decoder for Joint mRNA–miRNA Modelling

miDGD is a deep generative model that jointly models mRNA and miRNA expression across cancer and normal tissue types. Given only mRNA expression data, the trained model can predict the corresponding miRNA expression profile.

## Repository Structure

```
miDGD_paper/
├── base/               # Core model and data modules
│   ├── data/           # Dataset classes (GeneExpressionDatasetCombined)
│   ├── dgd/            # DGD model, GMM, representation layer
│   ├── engine/         # Training and prediction loops
│   ├── model/          # Decoder architecture (NB output module)
│   └── utils/          # Helpers (set_seed, get_activation)
├── data/               # Datasets
├── models/             # Trained model checkpoints (.pth)
├── notebook/           # Analysis notebooks and helper functions
├── plots/              # Generated figures (SVG)
├── predictions/        # Model prediction outputs (TSV)
├── scripts/            # Preprocessing scripts
└── supplementary/      # Supplementary files
```

## Setup

**Requirements**: Python 3.10+, conda/mamba recommended.

```bash
# create environment
conda create -n midgd python=3.10
conda activate midgd

# install dependencies
pip install -r requirement.txt
```

## Reproducing Results

Notebooks are numbered in execution order. Run them from the `notebook/` directory:

```bash
cd notebook
jupyter nbconvert --to notebook --execute --inplace 1-midgd-tcga.ipynb
jupyter nbconvert --to notebook --execute --inplace 2-midgd-gtex.ipynb
jupyter nbconvert --to notebook --execute --inplace 3a-midgd-tcga-gtex.ipynb
jupyter nbconvert --to notebook --execute --inplace 3b-midgd-tcga-gtex-collapsed.ipynb
jupyter nbconvert --to notebook --execute --inplace 4a-midgd-tcga-gtex-r2.ipynb
jupyter nbconvert --to notebook --execute --inplace 4b-midgd-tcga-gtex-r2-collapsed.ipynb
jupyter nbconvert --to notebook --execute --inplace 5-tcga-gtex-r2-smartseq.ipynb
jupyter nbconvert --to notebook --execute --inplace pscsr-smartseq-latent-space.ipynb
```

| Notebook | Description |
|----------|-------------|
| `1-midgd-tcga.ipynb` | Train miDGD on TCGA |
| `2-midgd-gtex.ipynb` | Train miDGD on GTEx normal tissue |
| `3a-midgd-tcga-gtex.ipynb` | Joint TCGA + GTEx model; full miRNA names |
| `3b-midgd-tcga-gtex-collapsed.ipynb` | Same model, collapsed miRNA names |
| `4a-midgd-tcga-gtex-r2.ipynb` | Model with additional dataset: R2 RNA Atlas |
| `4b-midgd-tcga-gtex-r2-collapsed.ipynb` | Same as 4a, with collapsed miRNA |
| `5-tcga-gtex-r2-smartseq.ipynb` | Model with additional dataset: SmartSeq-total (single-cell) |
| `pscsr-smartseq-latent-space.ipynb` | Analysis of single-cell datasets |

## Minimal Usage

See [`setup_test.ipynb`](setup_test.ipynb) for an end-to-end example: load a trained model, prepare new mRNA data, and predict miRNA expression.

## Downsampled Data

To reproduce the downsampled TCGA mRNA data used in robustness experiments:

```bash
python scripts/make_downsampled_data.py
```

This generates `data/downsampled/TCGA_mrna_downsampled_{lib_size}.tsv` at ten library sizes and runs a built-in statistical verification step.

## bioRxiv

Refer to this preprint: 

miDGD: a multi-modal deep generative model predicts microRNA expression from bulk or single-cell mRNA expression (https://doi.org/10.64898/2026.05.29.727918)
