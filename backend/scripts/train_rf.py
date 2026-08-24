"""Train the RF intervention model: spatial-block CV (the honest score) vs.
random-split CV (which leaks due to spatial autocorrelation and looks better
than it should), a mean-baseline comparison, and permutation importances.

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.ml.train_rf import train_and_evaluate  # noqa: E402


def _print_scores(scores: dict) -> None:
    print(f"  R2   = {scores['r2']:.3f}")
    print(f"  MAE  = {scores['mae']:.3f} C")
    print(f"  RMSE = {scores['rmse']:.3f} C")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train the RF intervention model")
    parser.add_argument("--training-set", default="../data/raw/training_set.parquet")
    parser.add_argument("--output", default="../models/rf_intervention.pkl")
    args = parser.parse_args()

    training_df = pd.read_parquet(args.training_set)
    result = train_and_evaluate(training_df, output_path=Path(args.output))

    print(f"trained on {result['metadata']['n_rows']} rows, source={result['metadata']['source']}, bbox={result['metadata']['bbox']}")

    print("\n=== Random-split CV (LEAKY -- points are spatially autocorrelated, this score is inflated) ===")
    _print_scores(result["random_split_scores"])

    print("\n=== Spatial-block CV (HONEST -- whole 4x4-grid blocks held out; use THIS one) ===")
    _print_scores(result["spatial_block_scores"])

    print("\n=== Mean-baseline (always predict the training mean), same spatial CV ===")
    _print_scores(result["baseline_scores"])

    spatial_r2 = result["spatial_block_scores"]["r2"]
    baseline_r2 = result["baseline_scores"]["r2"]
    if spatial_r2 <= baseline_r2:
        print(f"\n*** WARNING: under honest spatial CV the model does NOT beat the mean-baseline ({spatial_r2:.3f} <= {baseline_r2:.3f}). ***")
    else:
        print(f"\nModel beats the mean-baseline under spatial CV: R2 {spatial_r2:.3f} vs {baseline_r2:.3f}")

    print("\n=== Permutation importances (held-out spatial fold, NOT impurity-based) ===")
    for name, importance, std in result["permutation_importances"]:
        print(f"  {name:35s} {importance:+.4f} +/- {std:.4f}")

    print(f"\nmodel saved: {args.output}")


if __name__ == "__main__":
    main()
