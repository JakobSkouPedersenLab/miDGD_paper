"""
Multinomial downsampling of TCGA mRNA counts to simulate lower sequencing depths.

Python rewrite of scripts/make_downsampled_data.R.
Input:  data/TCGA/miDGD/tcga_mrna.tsv  (samples × genes, raw counts)
Output: data/downsampled/TCGA_mrna_downsampled_{lib_size}.tsv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

LIBRARY_SIZES = [
    1_000, 5_000, 10_000, 50_000, 100_000,
    200_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
]

SEED = 42


def _downsample_sample(counts: np.ndarray, lib_size: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomially downsample one sample to lib_size total reads."""
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)
    probs = counts / total
    return rng.multinomial(lib_size, probs).astype(np.float32)


def downsample(mrna: pd.DataFrame, lib_size: int, n_jobs: int = -1) -> pd.DataFrame:
    """Downsample all samples in mrna to lib_size reads in parallel."""
    counts = mrna.values.astype(np.float64)

    def _worker(i):
        rng = np.random.default_rng(SEED + i)
        return _downsample_sample(counts[i], lib_size, rng)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker)(i) for i in range(len(mrna))
    )

    return pd.DataFrame(
        np.stack(results),
        index=mrna.index,
        columns=mrna.columns,
    )


# ---------------------------------------------------------------------------
# Statistical verification
# ---------------------------------------------------------------------------

def verify_downsampled(original: pd.DataFrame, downsampled: pd.DataFrame, lib_size: int,
                       n_check: int = 20) -> bool:
    """
    Two checks:
    1. Every sample's total read count equals lib_size exactly.
    2. Spearman correlation between original proportions and downsampled proportions
       is > 0.7 for a random subset of samples (weak filter to catch obvious bugs).

    Returns True when all checks pass; prints a summary either way.
    """
    passed = True

    # --- check 1: library sizes ---
    sums = downsampled.sum(axis=1)
    wrong = (sums != lib_size).sum()
    if wrong > 0:
        print(f"  [FAIL] library size check: {wrong} samples do not sum to {lib_size}")
        passed = False
    else:
        print(f"  [OK]   library size: all {len(sums)} samples sum to {lib_size}")

    # --- check 2: rank correlation of proportions ---
    rng = np.random.default_rng(0)
    idx = rng.choice(len(original), size=min(n_check, len(original)), replace=False)
    corrs = []
    for i in idx:
        orig_prop = original.iloc[i].values.astype(float)
        orig_total = orig_prop.sum()
        if orig_total == 0:
            continue
        orig_prop = orig_prop / orig_total

        down_prop = downsampled.iloc[i].values.astype(float)
        down_total = down_prop.sum()
        if down_total == 0:
            continue
        down_prop = down_prop / down_total

        # Only non-zero genes in original (zeros always stay zero)
        mask = orig_prop > 0
        if mask.sum() < 2:
            continue
        rho, _ = stats.spearmanr(orig_prop[mask], down_prop[mask])
        corrs.append(rho)

    if corrs:
        mean_rho = np.mean(corrs)
        min_rho = np.min(corrs)
        threshold = 0.7
        status = "[OK]  " if mean_rho >= threshold else "[FAIL]"
        passed = passed and (mean_rho >= threshold)
        print(
            f"  {status} proportion correlation (Spearman): "
            f"mean={mean_rho:.3f}, min={min_rho:.3f} over {len(corrs)} samples"
        )
    else:
        print("  [WARN] could not compute proportion correlation (all-zero samples?)")

    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multinomial downsampling of TCGA mRNA counts")
    parser.add_argument(
        "--input", default="data/TCGA/miDGD/tcga_mrna.tsv",
        help="Path to input TSV (samples × genes, raw counts)",
    )
    parser.add_argument(
        "--outdir", default="data/downsampled",
        help="Output directory",
    )
    parser.add_argument(
        "--lib-sizes", nargs="+", type=int, default=LIBRARY_SIZES,
        help="Library sizes to simulate",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs (-1 = all cores)",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Run statistical verification after each downsampling",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = repo_root / args.input
    outdir = repo_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} …")
    mrna = pd.read_table(input_path, sep="\t", index_col=0)
    print(f"  {mrna.shape[0]} samples × {mrna.shape[1]} genes")

    all_passed = True
    for lib_size in args.lib_sizes:
        print(f"\nDownsampling to {lib_size:,} reads …")
        ds = downsample(mrna, lib_size, n_jobs=args.n_jobs)

        if args.verify:
            ok = verify_downsampled(mrna, ds, lib_size)
            all_passed = all_passed and ok

        out_path = outdir / f"TCGA_mrna_downsampled_{lib_size}.tsv"
        ds.to_csv(out_path, sep="\t")
        print(f"  Saved → {out_path}")

    print("\nDone!" + (" All checks passed." if all_passed else " Some checks FAILED."))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
