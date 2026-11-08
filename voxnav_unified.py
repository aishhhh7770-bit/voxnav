import cv2
import time
import threading
import queue
import pythoncom
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import mediapipe as mp
import numpy as np
import pyttsx3

from vlm_finder import VLMFinder, describe_position, describe_distance
from clip_finder import CLIPDetector

FIND_BACKEND = "clip"   # "clip" (CLIP crops, good for colours/attributes) or "owlvit" (tighter boxes)

# ----------------------------
# Load models
# ----------------------------
print("Loading models... (thoda time lagega)")
yolo_model = YOLO("yolov8n.pt")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
# VLM for "find <anything>" (first run downloads ~600 MB)
finder = VLMFinder(detector=CLIPDetector()) if FIND_BACKEND == "clip" else VLMFinder()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# ----------------------------
# TTS setup (queue-based, ek hi background thread, COM init)
# ----------------------------
tts_queue = queue.Queue()

def tts_worker():
    pythoncom.CoInitialize()
    engine = pyttsx3.init()
    while True:
        text = tts_queue.get()
        if text is None:
            break
        print(f"[TTS] Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    print(f"[TTS] Queued: {text}")
    tts_queue.put(text)

last_announced = {}
ANNOUNCE_COOLDOWN = 4

# ----------------------------
# Settings / thresholds
# ----------------------------
APPROACH_THRESHOLD_PERCENT = 15
DEPTH_INTERVAL = 3
FIND_COOLDOWN = 5        # seconds between spoken "found" updates
RESULT_MAX_AGE = 5       # hide VLM boxes older than this many seconds

# ----------------------------
# Camera + state
# ----------------------------
cap = cv2.VideoCapture(0)
prev_sizes = {}
last_depth_time = 0
depth_map = None

typing = False           # True while the user is typing a find-query
typed = ""
current_query = ""       # what the VLM is currently searching for


def draw_bar(img, text, color=(255, 255, 255)):
    ih, iw = img.shape[:2]
    cv2.rectangle(img, (0, ih - 36), (iw, ih), (0, 0, 0), -1)
    cv2.putText(img, text, (10, ih - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


print("VoxNav Unified chalu ho gaya.")
print("Camera window mein: F = kuch dhoondo (type karke Enter), C = clear, Q = band.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # --- YOLO detection ---
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # --- Hand frame + YOLO boxes to the finder (it only takes them when it's free) ---
    finder.submit_frame(frame, boxes=[b.xyxy[0].tolist() for b in results[0].boxes])

    # Debug: terminal mein saare detected labels print karo
    detected_labels = [yolo_model.names[int(b.cls[0])] for b in results[0].boxes]
    if detected_labels:
        print(f"[DETECTED] {detected_labels}")

    # --- Depth estimation (periodic) ---
    current_time = time.time()
    if current_time - last_depth_time > DEPTH_INTERVAL:
        frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_result = depth_pipe(frame_rgb_pil)
        depth_map = np.array(depth_result["depth"])
        last_depth_time = current_time

    # --- Hand detection ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)
    hand_points = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            tip = hand_landmarks.landmark[8]
            hand_points.append((int(tip.x * w), int(tip.y * h)))

    # --- Process each detected object ---
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = yolo_model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Approach safety (percentage-based)
        status_text = ""
        if label in prev_sizes and prev_sizes[label] > 0:
            prev_area = prev_sizes[label]
            percent_change = ((area - prev_area) / prev_area) * 100

            if percent_change > APPROACH_THRESHOLD_PERCENT:
                status_text = " [APPROACHING]"
            elif percent_change < -APPROACH_THRESHOLD_PERCENT:
                status_text = " [MOVING AWAY]"

        prev_sizes[label] = area

        # Depth value
        depth_text = ""
        if depth_map is not None:
            dh, dw = depth_map.shape
            scaled_x = int(center_x * dw / w)
            scaled_y = int(center_y * dh / h)
            scaled_x = min(max(scaled_x, 0), dw - 1)
            scaled_y = min(max(scaled_y, 0), dh - 1)
            depth_val = depth_map[scaled_y, scaled_x]
            depth_text = f" depth:{depth_val:.0f}"

        # Hand-object check
        hand_text = ""
        for (hx, hy) in hand_points:
            if x1 <= hx <= x2 and y1 <= hy <= y2:
                hand_text = " [HAND NEAR]"

        # Draw label on screen
        cv2.putText(annotated_frame, f"{label}{depth_text}{status_text}{hand_text}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Voice announcement logic ---
        current_time_now = time.time()
        should_announce = False
        announce_text = ""

        if "APPROACHING" in status_text:
            announce_text = f"{label} approaching"
            should_announce = True
        elif hand_text:
            announce_text = f"hand near {label}"
            should_announce = True

        if should_announce:
            key = label + announce_text
            last_time = last_announced.get(key, 0)
            if current_time_now - last_time > ANNOUNCE_COOLDOWN:
                speak(announce_text)
                last_announced[key] = current_time_now

    # --- VLM "find" results ---
    bar_text, bar_color = "F = find something | C = clear | Q = quit", (200, 200, 200)
    if current_query:
        vlm_results, age = finder.get_results()
        if age > RESULT_MAX_AGE:
            vlm_results = []

        if vlm_results:
            found_parts = []
            for r in vlm_results:
                fx1, fy1, fx2, fy2 = map(int, r["box"])
                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 3)
                cv2.putText(annotated_frame, f'FIND: {r["query"]} {r["score"]:.2f}',
                            (fx1, max(fy1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                pos = describe_position(r["box"], w)
                dist = describe_distance(depth_map, r["box"], w, h)
                text = f'{r["query"]} {pos}' + (f", {dist}" if dist else "")
                found_parts.append(text)

                find_key = "find:" + r["query"]
                if time.time() - last_announced.get(find_key, 0) > FIND_COOLDOWN:
                    speak(text)
                    last_announced[find_key] = time.time()

            bar_text, bar_color = "FOUND: " + " | ".join(found_parts), (0, 255, 0)
        else:
            bar_text, bar_color = f"Searching for: {current_query} ...", (0, 200, 255)

    if typing:
        bar_text, bar_color = f"Find: {typed}_   (Enter = search, Esc = cancel)", (255, 255, 255)

    draw_bar(annotated_frame, bar_text, bar_color)
    cv2.imshow("VoxNav Unified", annotated_frame)

    # --- Keyboard (click the camera window first) ---
    k = cv2.waitKey(1) & 0xFF
    if typing:
        if k == 13:                    # Enter
            current_query = finder.set_query(typed)
            typing = False
            if current_query:
                speak(f"Looking for {current_query}")
        elif k == 27:                  # Esc
            typing = False
        elif k == 8:                   # Backspace
            typed = typed[:-1]
        elif 32 <= k <= 126:
            typed += chr(k)
    else:
        if k == ord('q'):
            break
        elif k in (ord('f'), ord('/')):
            typing, typed = True, ""
        elif k == ord('c'):
            current_query = finder.set_query("")

finder.stop()
cap.release()
cv2.destroyAllWindows()
