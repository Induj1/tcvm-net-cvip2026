"""Benchmark clean, attacked, and image-only YOLOv8 robustness with mAP and ASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd
import yaml
from ultralytics import YOLO

from advtraffic.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 robustness benchmark over generated attack splits.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--clean-data", required=True, help="Clean YOLO dataset YAML.")
    parser.add_argument("--attacks-root", default="outputs/attacks")
    parser.add_argument("--split", default="test")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "patch", "sticker", "reflective", "motion_blur", "occlusion", "low_light"])
    parser.add_argument("--output-dir", default="outputs/results")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--method-name", default="YOLOv8", help="Method label written to the result CSV.")
    return parser.parse_args()


def load_names(clean_yaml: str | Path):
    data = yaml.safe_load(Path(clean_yaml).read_text(encoding="utf-8"))
    return data.get("names", {})


def make_eval_yaml(root: Path, names, split_name: str) -> Path:
    config = {"path": str(root.resolve()), "train": "images", "val": "images", "test": "images", "names": names}
    path = root / f"{split_name}_eval.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def val_model(
    model: YOLO,
    data_yaml: str | Path,
    imgsz: int,
    batch: int,
    device: str | None,
    project: Path,
    name: str,
    split: str,
) -> dict[str, float]:
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    names = load_names(args.clean_data)
    rows = []

    clean = val_model(model, args.clean_data, args.imgsz, args.batch, args.device, output_dir, "clean_yolo", args.split)
    rows.append({"attack": "clean", "method": args.method_name, "attack_success_rate": 0.0, "robust_accuracy": 1.0, **clean})

    for attack in args.attacks:
        attack_root = Path(args.attacks_root) / attack / args.split
        if not (attack_root / "images").exists():
            continue
        eval_yaml = make_eval_yaml(attack_root, names, split_name=attack)
        metrics = val_model(model, eval_yaml, args.imgsz, args.batch, args.device, output_dir, f"{attack}_yolo", args.split)
        summary_path = attack_root / "attack_summary.json"
        asr = read_json(summary_path).get("attack_success_rate", None) if summary_path.exists() else None
        rows.append(
            {
                "attack": attack,
                "method": args.method_name,
                "attack_success_rate": asr,
                "robust_accuracy": None if asr is None else 1.0 - asr,
                **metrics,
            }
        )

        image_def_root = Path(args.attacks_root) / f"{attack}_image_defense" / args.split
        if (image_def_root / "images").exists():
            image_def_yaml = make_eval_yaml(image_def_root, names, split_name=f"{attack}_image_defense")
            image_def_metrics = val_model(
                model,
                image_def_yaml,
                args.imgsz,
                args.batch,
                args.device,
                output_dir,
                f"{attack}_image_defense",
                args.split,
            )
            rows.append({"attack": attack, "method": "Image-only defense", "attack_success_rate": asr, "robust_accuracy": None if asr is None else 1.0 - asr, **image_def_metrics})

    df = pd.DataFrame(rows)
    csv_path = output_dir / "robustness_summary.csv"
    json_path = output_dir / "robustness_summary.json"
    df.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    print(df.to_string(index=False))
    print(f"Wrote {csv_path.resolve()}")


if __name__ == "__main__":
    main()
