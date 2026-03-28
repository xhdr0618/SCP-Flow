"""
Evaluate SCP-Flow outputs.

This script summarizes:
1. image generation metrics using the existing official image metric script
2. next-visit interval metrics from flow_predictions.csv
3. uncertainty-error alignment statistics
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.calculate_metric import ResultsDataset, img_metrics
from torch.utils.data import DataLoader


def evaluate_interval_and_uncertainty(pred_csv: Path):
    df = pd.read_csv(pred_csv)
    interval_error = df["pred_interval"] - df["target_interval"]
    abs_error = np.abs(interval_error)

    metrics = {
        "interval_mae": float(abs_error.mean()),
        "interval_rmse": float(np.sqrt(np.mean(interval_error ** 2))),
        "uncertainty_mean": float(df["pred_uncertainty"].mean()),
    }

    if len(df) > 1:
        metrics["uncertainty_abs_error_corr"] = float(df["pred_uncertainty"].corr(abs_error, method="spearman"))
    else:
        metrics["uncertainty_abs_error_corr"] = None

    if "missing_count" in df.columns:
        metrics["missing_count_mean"] = float(df["missing_count"].mean())
        metrics["missing_count_max"] = float(df["missing_count"].max())
    if "missing_strategy" in df.columns and len(df) > 0:
        metrics["missing_strategy"] = str(df["missing_strategy"].iloc[0])

    return metrics


def main():
    parser = argparse.ArgumentParser(description="evaluate SCP-Flow outputs")
    parser.add_argument("--eval_path", type=str, required=True, help="path containing image/ and flow_predictions.csv")
    parser.add_argument("--suffix_gt", type=str, default="_gt.png")
    parser.add_argument("--suffix_gene", type=str, default="_gen.png")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--skip_image_metrics", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    summary = {}

    pred_csv = eval_path / "flow_predictions.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing {pred_csv}")

    summary["interval_uncertainty"] = evaluate_interval_and_uncertainty(pred_csv)

    if not args.skip_image_metrics:
        dataset = ResultsDataset(
            root_dir=str(eval_path),
            suffix_gt=args.suffix_gt,
            suffix_gene=args.suffix_gene,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_worker,
            drop_last=False,
        )
        metrics_df = img_metrics(args, dataloader)
        summary["image_metrics"] = {
            row["Metric"]: {
                "mean": float(row["Mean"]),
                "std": float(row["Std"]),
            }
            for _, row in metrics_df.iterrows()
        }

    out_path = eval_path / "scpflow_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    os.environ["TORCH_HOME"] = "./pre-trained"
    main()
