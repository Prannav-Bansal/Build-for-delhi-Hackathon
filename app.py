import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLOv8 model using the correct file path
model = YOLO("runs/space_train/weights/best.pt")

def detect_objects(image):
    results = model(image)  # Run inference on the uploaded image
    annotated = results[0].plot()  # Annotated image with bounding boxes
    return Image.fromarray(annotated)  # Convert to PIL image for display

# Launch Gradio interface
gr.Interface(
    fn=detect_objects,
    inputs="image",
    outputs="image",
    title="Space Station Object Detector",
    description="Upload an image of the space station interior to detect ToolBox, FireExtinguisher, and OxygenTank."
).launch()
