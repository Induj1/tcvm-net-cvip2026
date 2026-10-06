# Validated Checkpoints

These checkpoints are preserved to make the reported comparisons reproducible without retraining:

| File | Training role | SHA-256 |
|---|---|---|
| `helmet_yolov8n_best.pt` | Clean YOLOv8n baseline | `E43D687D7E1E150ECE418FBCE1AFF39B69DF8782F5967EF1AE05B29AB35E7023` |
| `helmet_yolov8s_best.pt` | Clean YOLOv8s baseline | `E6DDD5A23E97A169B4A4CF4A06A1F077B8B8F6E0478824A2D4A15324F30C38A5` |
| `helmet_yolov8n_advtrain_best.pt` | YOLOv8n adversarial-augmentation baseline | `F1795072E091DC0A46DC4FA3FC527E5DA6510146A61B24F1331CB18A4EACD878` |

The files are standard Ultralytics/PyTorch checkpoints. Only load model files from sources you trust because PyTorch checkpoint deserialization may execute code.
