import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)


prev_sizes = {}

print("Chalu ho gaya. 'q' se band karo.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        
        area = (x2 - x1) * (y2 - y1)

    
        if label in prev_sizes:
            prev_area = prev_sizes[label]
            change = area - prev_area

            if change > 2000:  
                status = "APPROACHING"
                color = (0, 0, 255)  # red warning
            elif change < -2000:
                status = "MOVING AWAY"
                color = (0, 255, 0)
            else:
                status = "STATIC"
                color = (255, 255, 0)

            cv2.putText(annotated_frame, f"{label}: {status}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        prev_sizes[label] = area

    cv2.imshow("VoxNav - Approach Safety", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()