"""
vlm_finder.py - "Find anything" for VoxNav using a vision-language model.

Type what you want (e.g. "water bottle" or "red bag, keys") and the model
looks for it in the live camera feed. Inference runs in a background thread,
so the webcam loop never freezes.

Model: OWL-ViT (CLIP-based open-vocabulary detector, ~600 MB, works on CPU).
Queries must be in English.
"""
import re
import threading
import time

import numpy as np
from PIL import Image

MODEL_ID = "google/owlvit-base-patch32"
# More accurate but slower on CPU: "google/owlv2-base-patch16-ensemble"
MIN_SCORE = 0.10  # lower = finds more (and more false positives)

_FILLER = re.compile(r"^(please\s+)?(find|look for|search for|locate|where is|where are)\s+", re.I)
_ARTICLE = re.compile(r"^(my|the|a|an)\s+", re.I)


def parse_queries(text):
    """'find my water bottle, keys' -> ['water bottle', 'keys']"""
    queries = []
    for part in text.split(","):
        part = _FILLER.sub("", part.strip())
        part = _ARTICLE.sub("", part).strip().lower()
        if part:
            queries.append(part)
    return queries


def describe_position(box, frame_w):
    """Left / center / right of the frame."""
    cx = (box[0] + box[2]) / 2
    if cx < frame_w / 3:
        return "on the left"
    if cx > 2 * frame_w / 3:
        return "on the right"
    return "in the center"


def describe_distance(depth_map, box, frame_w, frame_h):
    """Rough relative distance from the depth map (higher value = closer)."""
    if depth_map is None:
        return ""
    dh, dw = depth_map.shape[:2]
    cx = int((box[0] + box[2]) / 2 * dw / frame_w)
    cy = int((box[1] + box[3]) / 2 * dh / frame_h)
    cx = min(max(cx, 0), dw - 1)
    cy = min(max(cy, 0), dh - 1)
    r = max(2, dw // 40)
    patch = depth_map[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
    val = float(np.median(patch)) / 255.0
    if val > 0.7:
        return "very close"
    if val > 0.45:
        return "close"
    return "farther away"


class VLMFinder:
    def __init__(self, detector=None):
        if detector is None:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            detector = pipeline("zero-shot-object-detection", model=MODEL_ID, device=device)
        self.detector = detector

        self._lock = threading.Lock()
        self._queries = []
        self._version = 0        # bumps on every new query so stale results get dropped
        self._pending = None     # latest frame waiting for the model
        self._pending_boxes = None
        self._busy = False
        self._results = []
        self._result_time = 0.0
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set_query(self, text):
        """Set what to look for. Empty text clears the search. Returns the cleaned query."""
        queries = parse_queries(text)
        with self._lock:
            self._queries = queries
            self._version += 1
            self._pending = None
            self._results = []
        return ", ".join(queries)

    def submit_frame(self, frame_bgr, boxes=None):
        """Call every frame; it only keeps a frame when the model is free.
        boxes = YOLO boxes [(x1,y1,x2,y2), ...] (only used by the CLIP backend)."""
        with self._lock:
            if not self._queries or self._busy or self._pending is not None:
                return
            self._pending = frame_bgr.copy()
            self._pending_boxes = list(boxes) if boxes else []

    def get_results(self):
        """-> (list of {query, score, box=(x1,y1,x2,y2)}, seconds since last update)"""
        with self._lock:
            return list(self._results), time.time() - self._result_time

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            with self._lock:
                frame = self._pending
                if frame is not None:
                    self._pending = None
                    boxes = self._pending_boxes
                    self._busy = True
                    queries = list(self._queries)
                    version = self._version
            if frame is None:
                time.sleep(0.03)
                continue

            results, failed = [], False
            try:
                image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR -> RGB
                extra = {"boxes": boxes} if getattr(self.detector, "uses_boxes", False) else {}
                raw = self.detector(image, candidate_labels=queries, threshold=MIN_SCORE, **extra)
                best = {}
                for r in raw:
                    if r["label"] not in best or r["score"] > best[r["label"]]["score"]:
                        best[r["label"]] = r
                for q, r in best.items():
                    b = r["box"]
                    results.append({
                        "query": q,
                        "score": float(r["score"]),
                        "box": (b["xmin"], b["ymin"], b["xmax"], b["ymax"]),
                    })
            except Exception as e:
                failed = True
                print(f"[VLM] error: {e}")

            with self._lock:
                self._busy = False
                if version == self._version and not failed:
                    self._results = results
                    self._result_time = time.time()
            if failed:
                time.sleep(1)
