import cv2
from ultralytics import YOLO
import mediapipe as mp

yolo_model = YOLO("yolov8n.pt")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
print("Chalu ho gaya. 'q' se band karo.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # YOLO detection
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    hand_points = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Index finger tip (landmark 8) ko reference point banate hain
            tip = hand_landmarks.landmark[8]
            hand_points.append((int(tip.x * w), int(tip.y * h)))

    # Check: kya koi hand point kissi object box ke andar hai
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = yolo_model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        for (hx, hy) in hand_points:
            if x1 <= hx <= x2 and y1 <= hy <= y2:
                cv2.putText(annotated_frame, f"HAND NEAR: {label}",
                            (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Hand-Object Interaction", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()