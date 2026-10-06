# Public Artifact Handoff

This repository is the public, self-contained implementation handoff for TCVM-Net. It preserves the runnable source code, experiment configurations, selected generated results and figures, and the three validated model checkpoints. The anonymous submission manuscript and author metadata are deliberately excluded.

## Immutable release state

- Repository: <https://github.com/Induj1/tcvm-net-cvip2026>
- Reference branch: `main`
- Python tests at release: 11 passed
- Release checkpoints: `weights/`

## Checkpoint integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `weights/helmet_yolov8n_best.pt` | 6,249,827 | `E43D687D7E1E150ECE418FBCE1AFF39B69DF8782F5967EF1AE05B29AB35E7023` |
| `weights/helmet_yolov8s_best.pt` | 22,519,587 | `E6DDD5A23E97A169B4A4CF4A06A1F077B8B8F6E0478824A2D4A15324F30C38A5` |
| `weights/helmet_yolov8n_advtrain_best.pt` | 6,249,187 | `F1795072E091DC0A46DC4FA3FC527E5DA6510146A61B24F1331CB18A4EACD878` |

Verify a checkpoint on PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 weights/helmet_yolov8n_best.pt
```

## Public data sources

The executed datasets are not redistributed. Restore them with the Kaggle CLI after accepting the source terms:

```powershell
python -m kaggle datasets download -d andrewmvd/hard-hat-detection -p data/source/helmet --unzip
python -m kaggle datasets download -d a7madmostafa/bdd100k-yolo -p data/source/bdd100k-yolo-kaggle --unzip
python -m kaggle datasets download -d ayushraj2349/sample-videos-for-helmet-detection-on-yolov8 -p data/source/helmet_videos --unzip
```

Dataset conversion, attacks, threshold calibration, benchmarking, ablation, and plotting commands are documented in the root [`README.md`](../README.md). Never place Kaggle credentials in this repository.

## Deliberate exclusions

- manuscript and submission source;
- author and institutional metadata;
- raw third-party datasets and videos;
- redundant training caches and intermediate detector outputs.

These exclusions keep the public artifact legally redistributable and prevent accidental deanonymization while retaining everything required to inspect, run, retrain, and evaluate the method.
