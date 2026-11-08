"""
clip_finder.py - CLIP backend for VoxNav's "find <anything>" feature.

CLIP is not a detector: it scores how well an image crop matches a text prompt.
So we cut the camera frame into candidate crops (YOLO boxes + a grid of
windows + the full frame), score every crop against your prompt, and return
the best-matching region. Good at attributes ("red bottle", "black bag").
Boxes are coarser than OWL-ViT unless YOLO already boxes the object.

Use:  VLMFinder(detector=CLIPDetector())
"""
import numpy as np

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"   # ~600 MB
MIN_PROB = 0.5        # a crop must beat the "background" prompts by this much
MAX_YOLO_BOXES = 6
GRIDS = (2, 3)        # 2x2 and 3x3 overlapping windows

NEGATIVES = ["a wall", "the floor", "the ceiling", "a table", "a person",
             "a window", "a door", "a blurry background", "a room"]


def _core(text):
    for art in ("a ", "an ", "the "):
        if text.startswith(art):
            return text[len(art):]
    return text


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def make_proposals(w, h, yolo_boxes=()):
    """Candidate regions (x1, y1, x2, y2): YOLO boxes, grid windows, full frame."""
    props = []
    big = sorted(yolo_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for x1, y1, x2, y2 in big[:MAX_YOLO_BOXES]:
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        props.append((max(x1 - mx, 0), max(y1 - my, 0), min(x2 + mx, w), min(y2 + my, h)))
    for n in GRIDS:
        ww, wh = w * 1.3 / n, h * 1.3 / n
        for i in range(n):
            for j in range(n):
                x0 = i * (w - ww) / (n - 1)
                y0 = j * (h - wh) / (n - 1)
                props.append((x0, y0, x0 + ww, y0 + wh))
    props.append((0, 0, w, h))
    return [tuple(int(v) for v in p) for p in props]


class CLIPDetector:
    """Same call signature as the HF zero-shot-object-detection pipeline."""
    uses_boxes = True   # tells VLMFinder to pass the YOLO boxes along

    def __init__(self, score_fn=None):
        if score_fn is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
            proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

            @torch.no_grad()
            def score_fn(images, texts):
                inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(device)
                return model(**inputs).logits_per_image.cpu().numpy()   # (n_images, n_texts)
        self.score_fn = score_fn

    def __call__(self, image, candidate_labels, threshold=None, boxes=None):
        w, h = image.size
        props = make_proposals(w, h, boxes or [])
        crops = [image.crop(p) for p in props]

        queries = list(candidate_labels)
        cores = {_core(q) for q in queries}
        negatives = [n for n in NEGATIVES if _core(n) not in cores]
        texts = [f"a photo of a {q}" for q in queries] + [f"a photo of {n}" for n in negatives]

        probs = _softmax(self.score_fn(crops, texts))
        areas = np.array([(p[2] - p[0]) * (p[3] - p[1]) for p in props])

        out = []
        for qi, q in enumerate(queries):
            p = probs[:, qi]
            best = p.max()
            if best < MIN_PROB:
                continue
            close = np.where(p >= 0.9 * best)[0]          # near-best crops...
            idx = close[np.argmin(areas[close])]          # ...take the tightest one
            x1, y1, x2, y2 = props[idx]
            out.append({"label": q, "score": float(p[idx]),
                        "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}})
        return out
