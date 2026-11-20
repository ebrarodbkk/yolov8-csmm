# SEAMHeadCSMM for YOLOv8s (Small-Object Aerial Tracking)

Multi-scale patch-based **CSMM** (`1x1`, `3x3`, `5x5` depthwise-separable branches) combined with **Squeeze-and-Excitation (SE)** channel attention (`SEAMHeadCSMM`) integrated into **YOLOv8s** at the **P3/8** and **P4/16** feature levels for small aerial object detection and tracking.

## Architecture Evolution & Ablation Summary

| Variant | Heads & Patches | Epochs | Params / GFLOPs | Val SO-HOTA | Public Test SO-HOTA | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline `YOLOv8s` | Standard P3–P5 | 20 | ~11.2M / ~28.8 | 0.2430 | 0.2769 | COCO pretrained baseline |
| `CSMM v1` (Initial) | P3–P5 (`6x6, 7x7, 8x8`) | 20 | 63.5M / 124.5 | 0.2200 | — | Heavy compute & even/large patches underperformed |
| **`CSMM P3–P4` (Final)** | **P3–P4 (`1x1, 3x3, 5x5`)** | **50** | **11.66M / 31.3** | **0.2502** | **0.2890** | **Lightweight, focused on small targets (+0.0121 vs baseline)** |

### Key Engineering Takeaways
- **Targeted Feature Placement:** Moving from `P3–P5` to `P3–P4` preserves spatial resolution for small objects while cutting parameter count from `63.5M` down to `11.66M` (`31.3 GFLOPs`).
- **Odd Multi-Scale Kernels:** Replacing `6x6, 7x7, 8x8` branches with `1x1, 3x3, 5x5` improves feature fusion stability and training efficiency.
- **Dynamic Channel Initialization:** `SEAMHeadCSMM` registers with Ultralytics (`register_csmm()`) and lazily initializes channel-dependent branches cleanly across feature maps.

## Repository Structure
- `src/csmm_module.py`: `PatchBranch`, `SEBlock`, `SEAMHeadCSMM`, and `register_csmm()` hook.
- `configs/yolov8s_csmm_p3p4.yaml`: YOLOv8s model configuration with `SEAMHeadCSMM` on P3 (`15`) and P4 (`18`).
- `configs/csmm_p3p4_135_ep50.yaml`: 50-epoch training (`imgsz: 1280`, `batch: 32`, `AdamW`) and evaluation config.
- `scripts/verify_forward.py`: Standalone model build and forward-pass check.

## Artifacts & Weights
Model checkpoints (`best.pt`, epoch checkpoints) and evaluation outputs are archived in Google Drive (`SMOTP4SB_Checkpoints/csmm_p3p4_135_ep50`) rather than committed to Git.
