from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml

# Function to predict and save images + labels
def predict_and_save(model, image_path, output_img_path, output_txt_path):
    results = model.predict(image_path, conf=0.5)
    result = results[0]

    # Save image with bounding boxes
    annotated_img = result.plot()
    cv2.imwrite(str(output_img_path), annotated_img)

    # Save bounding boxes in YOLO format
    with open(output_txt_path, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywh[0].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

if __name__ == '__main__':
    # Set root dir based on your structure
    project_root = Path("D:/Hackathon_Dataset/HackByte_Dataset")
    os.chdir(project_root)

    # Load YAML
    yaml_path = project_root / "yolo_params.yaml"
    if not yaml_path.exists():
        print("❌ data.yaml not found")
        exit()

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        test_dir = Path(data['test']) / "images"

    if not test_dir.exists():
        print(f"❌ Test images directory not found: {test_dir}")
        exit()

    # Load trained model
    model_path = project_root / "runs" / "space_train" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        exit()

    model = YOLO(model_path)

    # Create output folders
    pred_dir = project_root / "runs" / "space_predict"
    img_out_dir = pred_dir / "images"
    label_out_dir = pred_dir / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # Predict on each test image
    for img_file in test_dir.glob("*"):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        out_img = img_out_dir / img_file.name
        out_txt = label_out_dir / img_file.with_suffix('.txt').name

        predict_and_save(model, img_file, out_img, out_txt)

    print(f"\n✅ Predictions saved to {pred_dir}")

    # Optional: evaluate on test set if labels exist
    print("\n📊 Running evaluation...")
    model.val(data=str(yaml_path), split="test")
