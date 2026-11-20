import torch
from ultralytics import YOLO
from src.csmm_module import SEAMHeadCSMM, register_csmm


def main():
    register_csmm()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("configs/yolov8s_csmm_p3p4.yaml")
    model.model.to(device).eval()

    x = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        out = model.model(x)

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"Device: {device} | Parameters: {n_params:,} | Output heads: {len(out)}")


if __name__ == "__main__":
    main()
