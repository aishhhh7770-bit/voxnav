print("SCRIPT STARTED")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print("MODEL LOADED")

results = model("bigbus.jpg")
print("INFERENCE DONE")

results[0].save("output.jpg")
print("IMAGE SAVED")

for box in results[0].boxes:
    cls = int(box.cls[0])
    label = model.names[cls]
    conf = float(box.conf[0])
    print(f"Detected: {label}, Confidence: {conf:.2f}")