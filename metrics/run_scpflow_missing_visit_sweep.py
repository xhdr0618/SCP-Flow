import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_command(command, cwd):
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def load_eval_metrics(eval_json_path: Path):
    data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    interval = data.get("interval_uncertainty", {})
    image = data.get("image_metrics", {})
    return {
        "interval_mae": interval.get("interval_mae"),
        "interval_rmse": interval.get("interval_rmse"),
        "uncertainty_mean": interval.get("uncertainty_mean"),
        "uncertainty_abs_error_corr": interval.get("uncertainty_abs_error_corr"),
        "missing_count_mean": interval.get("missing_count_mean"),
        "missing_count_max": interval.get("missing_count_max"),
        "missing_strategy": interval.get("missing_strategy"),
        "psnr": image.get("PSNR", {}).get("mean"),
        "ssim": image.get("SSIM", {}).get("mean"),
        "mse": image.get("MSE", {}).get("mean"),
        "fid": image.get("FID", {}).get("mean"),
        "lpips": image.get("LPIPS", {}).get("mean"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run SCP-Flow missing-visit robustness sweep")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/SIGF_make")
    parser.add_argument("--first_stage_ckpt", type=str, default="pre-trained/VQGAN/vqgan.ckpt")
    parser.add_argument("--output_root", type=str, default="results/scpflow_missing_sweep")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--structure_loss_weight", type=float, default=0.05)
    parser.add_argument("--count", type=int, default=1, help="number of historical visits to hide")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["none", "tail", "uniform", "random"],
    )
    parser.add_argument("--missing_seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy in args.strategies:
        missing_count = 0 if strategy == "none" else args.count
        run_dir = output_root / f"{strategy}_count{missing_count}"
        eval_json_path = run_dir / "scpflow_eval.json"

        if not (args.skip_existing and eval_json_path.exists()):
            test_cmd = [
                sys.executable,
                "train_vqldm_PMN.py",
                "--predictor_type", "scpflow",
                "--command", "test",
                "--data_root", args.data_root,
                "--first_stage_ckpt", args.first_stage_ckpt,
                "--resume", args.checkpoint,
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--devices", args.devices,
                "--accelerator", args.accelerator,
                "--structure_loss_weight", str(args.structure_loss_weight),
                "--missing_visit_count", str(missing_count),
                "--missing_visit_strategy", strategy,
                "--missing_seed", str(args.missing_seed),
                "--test_type", "test",
                "--image_save_dir", str(run_dir),
            ]
            run_command(test_cmd, cwd=str(repo_root))

            eval_cmd = [
                sys.executable,
                "metrics/evaluate_scpflow.py",
                "--eval_path", str(run_dir),
                "--batch_size", "1",
                "--num_worker", "0",
            ]
            run_command(eval_cmd, cwd=str(repo_root))

        metrics = load_eval_metrics(eval_json_path)
        metrics["run_dir"] = str(run_dir)
        metrics["strategy"] = strategy
        metrics["count"] = missing_count
        rows.append(metrics)

    df = pd.DataFrame(rows)
    csv_path = output_root / "missing_visit_summary.csv"
    md_path = output_root / "missing_visit_summary.md"
    df.to_csv(csv_path, index=False)

    display_df = df[
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
    ].copy()
    md_lines = [
        "# SCP-Flow Missing-Visit Robustness",
        "",
        display_df.to_markdown(index=False),
        "",
        f"CSV: `{csv_path}`",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"saved csv: {csv_path}")
    print(f"saved md: {md_path}")


if __name__ == "__main__":
    main()
