import argparse
import json
from pathlib import Path

import pandas as pd


def collect_eval_files(root: Path):
    return sorted(root.glob("**/scpflow_eval.json"))


def parse_eval(eval_path: Path):
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    parent_name = eval_path.parent.name
    strategy = None
    count = None
    if "_count" in parent_name:
        strategy, count_part = parent_name.split("_count", 1)
        try:
            count = int(count_part)
        except ValueError:
            count = None

    interval = data.get("interval_uncertainty", {})
    image = data.get("image_metrics", {})
    return {
        "source_dir": str(eval_path.parent.resolve()),
        "strategy": strategy if strategy is not None else interval.get("missing_strategy"),
        "count": count,
        "interval_mae": interval.get("interval_mae"),
        "interval_rmse": interval.get("interval_rmse"),
        "uncertainty_mean": interval.get("uncertainty_mean"),
        "uncertainty_abs_error_corr": interval.get("uncertainty_abs_error_corr"),
        "missing_count_mean": interval.get("missing_count_mean"),
        "missing_count_max": interval.get("missing_count_max"),
        "psnr": image.get("PSNR", {}).get("mean"),
        "ssim": image.get("SSIM", {}).get("mean"),
        "mse": image.get("MSE", {}).get("mean"),
        "fid": image.get("FID", {}).get("mean"),
        "lpips": image.get("LPIPS", {}).get("mean"),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate SCP-Flow missing-visit results")
    parser.add_argument("--root", type=str, required=True, help="directory containing *_count*/scpflow_eval.json")
    parser.add_argument("--extra_eval", nargs="*", default=[], help="extra scpflow_eval.json files to merge")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = [parse_eval(path) for path in collect_eval_files(root)]
    for extra in args.extra_eval:
        rows.append(parse_eval(Path(extra)))

    if not rows:
        raise FileNotFoundError(f"No scpflow_eval.json found under {root}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["strategy", "count"], keep="last")
    strategy_order = {"none": 0, "uniform": 1, "random": 2, "tail": 3}
    df = df.sort_values(
        by=["count", "strategy"],
        key=lambda col: col.map(strategy_order).fillna(col) if col.name == "strategy" else col,
    ).reset_index(drop=True)

    output_csv = Path(args.output_csv) if args.output_csv else root / "missing_visit_all_summary.csv"
    output_md = Path(args.output_md) if args.output_md else root / "missing_visit_all_summary.md"
    df.to_csv(output_csv, index=False)

    show = df[
        [
            "strategy",
            "count",
            "psnr",
            "ssim",
            "fid",
            "lpips",
            "interval_mae",
            "interval_rmse",
            "uncertainty_abs_error_corr",
        ]
    ]
    md = "# SCP-Flow Missing-Visit Robustness (All Counts)\n\n"
    md += show.to_markdown(index=False)
    md += f"\n\nCSV: `{output_csv.resolve()}`\n"
    output_md.write_text(md, encoding="utf-8")

    print(df.to_string(index=False))
    print(f"saved csv: {output_csv}")
    print(f"saved md: {output_md}")


if __name__ == "__main__":
    main()
