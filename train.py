import argparse
from ultralytics import YOLO
import os

# --- Training Configuration ---
EPOCHS = 5
MOSAIC = 0.1
OPTIMIZER = 'AdamW'
MOMENTUM = 0.2
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False

# --- Path Configuration ---
D_DRIVE_ROOT = "D:/HACKATHON_DATASET/HackByte_Dataset"
MODEL_PATH = "yolov8s.pt"  # Assumes pretrained weights are already in Ultralytics default cache
DATA_PATH = os.path.join(D_DRIVE_ROOT, "yolo_params.yaml")
PROJECT_PATH = os.path.join(D_DRIVE_ROOT, "runs")  # Where training logs will go

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--mosaic', type=float, default=MOSAIC)
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--lr0', type=float, default=LR0)
    parser.add_argument('--lrf', type=float, default=LRF)
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS)
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(PROJECT_PATH, exist_ok=True)

    # Load YOLOv8 pretrained model
    model = YOLO(MODEL_PATH)

    # Train the model
    results = model.train(
        data=DATA_PATH,
        epochs=args.epochs,
        device="cpu",  # or 'cuda' if you have GPU support
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        project=PROJECT_PATH,
        name="space_train",
        exist_ok=True
    )
