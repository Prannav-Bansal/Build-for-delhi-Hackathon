from ultralytics import YOLO
import argparse

def evaluate_model(weights_path, data_yaml, save_dir):
    model = YOLO(weights_path)

    metrics = model.val(
        data=data_yaml,            # path to your data.yaml
        split='test',              # evaluate on test set
        project='runs/space_eval',
        name=save_dir,
        exist_ok=True
    )

    print("Evaluation complete.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on test data")
    parser.add_argument('--weights', type=str, default='runs/space_train/weights/best.pt', help='Path to trained weights')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml with test set specified')
    parser.add_argument('--output', type=str, default='eval_results', help='Subdirectory inside space_eval to save results')
    
    args = parser.parse_args()
    evaluate_model(args.weights, args.data, args.output)
