from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np

IMAGE_PATH = "bigbus.jpg"  # apni image ka naam

print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

print("Loading Depth model...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Step 1: YOLO detection
print("Running object detection...")
results = yolo_model(IMAGE_PATH)

# Step 2: Depth estimation
print("Running depth estimation...")
image = Image.open(IMAGE_PATH)
depth_result = depth_pipe(image)
depth_map = np.array(depth_result["depth"])  # depth values as numpy array


print("\n--- FUSION RESULTS ---")
for box in results[0].boxes:
    cls = int(box.cls[0])
    label = yolo_model.names[cls]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    depth_h, depth_w = depth_map.shape
    img_w, img_h = image.size

    
    scaled_x = int(center_x * depth_w / img_w)
    scaled_y = int(center_y * depth_h / img_h)

    depth_value = depth_map[scaled_y, scaled_x]

    print(f"{label} (confidence: {conf:.2f}) -> relative depth value: {depth_value:.1f}")

print("\nDone.")