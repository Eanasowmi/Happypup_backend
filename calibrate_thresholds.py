"""
Mixed Breed Threshold Calibration Tool  (Auto-label from filename)
===================================================================
Steps:
  1. Drop your dog images into:
       calibration_images/

     Name each file after the breed(s), e.g.:
       golden_retriever.jpg           ← purebred
       labrador_husky.jpg             ← mixed
       beagle_poodle_mix.jpg          ← mixed
       border_collie_german_shepherd.jpg

  2. Run:
       python calibrate_thresholds.py

  3. The script will:
       - Parse breed labels automatically from the filename
       - Run every image through the breed model
       - Grid-search thresholds to find the best combination
       - Print the optimal values to paste into api_server.py
       - Save full results to calibration_results.csv
"""

import os, sys, json, csv, itertools
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "dog_breed_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_labels.json")
IMG_DIR     = os.path.join(BASE_DIR, "calibration_images")
OUT_CSV     = os.path.join(BASE_DIR, "calibration_results.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading breed model …")
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
print("  ✓ Model ready\n")

with open(LABELS_PATH) as f:
    raw = json.load(f)
BREED_LABELS = [raw[str(i)] for i in range(len(raw))] if isinstance(raw, dict) else raw
NUM_CLASSES  = len(BREED_LABELS)

# ── Pre-processing ─────────────────────────────────────────────────────────
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess(img_path):
    img     = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224))
    arr     = mobile_preprocess(img_to_array(img_res))
    return np.expand_dims(arr, axis=0)

# ── Auto-parse breed labels from filename ─────────────────────────────────
def parse_breeds_from_filename(fname):
    """
    Parses breed names from filename, supporting formats like:
      (chihuahua + dachshund).jpeg
      Beagle + Pug-puggle.jpeg
      boxer x labrador retriever.jpg
      German shepherd x siberian husky.jpg
      golden retriever.jpg
    Returns a list of matched breed names (1 = purebred, 2+ = mixed).
    """
    stem = os.path.splitext(fname)[0]   # strip extension

    # Remove parentheses and common punctuation
    for ch in ["(", ")", "[", "]"]:
        stem = stem.replace(ch, " ")

    # Replace separators with space
    stem = stem.replace("-", " ").replace("_", " ").replace("+", " ")

    stem = stem.lower()

    # Remove noise words (but keep 'x' handling carefully — it's a breed separator)
    noise = ["mix", "mixed", "breed", "dog", "puppy", "cross", "and", "&",
             "retriever", "double", "doodle"]
    # Remove standalone 'x' used as cross separator
    import re as _re
    stem = _re.sub(r'\bx\b', ' ', stem)

    for n in noise:
        stem = _re.sub(rf'\b{n}\b', ' ', stem)

    tokens = stem.split()

    # Try to greedily match known breeds (longest match first)
    breeds_lower = sorted(
        [b.lower() for b in BREED_LABELS],
        key=lambda x: -len(x)          # prefer longer matches
    )

    found = []
    used_indices = set()

    for breed_lc in breeds_lower:
        breed_tokens = breed_lc.split()
        blen = len(breed_tokens)
        for start in range(len(tokens) - blen + 1):
            span = tokens[start:start + blen]
            if span == breed_tokens and not any(
                i in used_indices for i in range(start, start + blen)
            ):
                # Find the original-case label
                orig = next(b for b in BREED_LABELS if b.lower() == breed_lc)
                found.append(orig)
                for i in range(start, start + blen):
                    used_indices.add(i)
                break

    return found if found else None

# ── Hybrid breed decoder ───────────────────────────────────────────────────
HYBRID_DECODER = {
    "Cockapoo":    ["Cocker", "Poodle"],
    "Labradoodle": ["Labrador", "Poodle"],
}

# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(preds_array):
    top_n    = min(10, len(preds_array))
    top_idx  = np.argsort(preds_array)[-top_n:][::-1]
    top10    = [(BREED_LABELS[i], float(preds_array[i])) for i in top_idx]

    conf     = top10[0][1]
    conf_gap = conf - top10[1][1]

    probs    = np.clip(preds_array, 1e-9, 1.0)
    entropy  = float(-np.sum(probs * np.log(probs)))
    norm_ent = entropy / float(np.log(NUM_CLASSES))

    top5_sum = sum(c for _, c in top10[:5])
    comp = [(b, round(c / top5_sum * 100, 1) if top5_sum > 0 else 0.0)
            for b, c in top10[:5]]

    return {
        "top1_breed":   top10[0][0], "top1_conf":  round(conf, 4),
        "top2_breed":   top10[1][0], "top2_conf":  round(top10[1][1], 4),
        "top3_breed":   top10[2][0], "top3_conf":  round(top10[2][1], 4),
        "conf_gap":     round(conf_gap, 4),
        "norm_entropy": round(norm_ent, 4),
        "composition":  comp,
        "top10":        top10,
    }

def breed_hit(metrics, true_breeds):
    """
    Check whether each true breed appears in top-10,
    also decoding any known hybrid breed names into their constituents.
    """
    # Expand top-10: if a breed is a known hybrid, add its constituents too
    expanded = set()
    for b, _ in metrics["top10"]:
        expanded.add(b.lower())
        if b in HYBRID_DECODER:
            for constituent in HYBRID_DECODER[b]:
                expanded.add(constituent.lower())

    hits = sum(
        1 for tb in true_breeds
        if any(tb.lower() in exp or exp in tb.lower() for exp in expanded)
    )
    return hits / len(true_breeds)

# ── Scan images from root + subfolders ────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(folder, forced_label=None):
    """Collect (filepath, forced_label_or_None) pairs from a folder."""
    pairs = []
    if not os.path.exists(folder):
        return pairs
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in IMG_EXTS:
            pairs.append((os.path.join(folder, f), f, forced_label))
    return pairs

image_queue = []
# Root calibration_images/ — infer label from filename
image_queue += collect_images(IMG_DIR, forced_label=None)
# mixed/ subfolder — force is_mixed=True
image_queue += collect_images(os.path.join(IMG_DIR, "mixed"),  forced_label="mixed")
# purebred/ subfolder — force is_mixed=False
image_queue += collect_images(os.path.join(IMG_DIR, "purebred"), forced_label="purebred")

if not image_queue:
    print(f"⚠  No images found in:  {IMG_DIR}")
    print("   Drop images there (or in mixed/ and purebred/ sub-folders) and re-run.\n")
    sys.exit(0)

print(f"Found {len(image_queue)} images\n")


results       = []
skipped       = []

for path, fname, folder_label in image_queue:
    true_breeds = parse_breeds_from_filename(fname)

    # Determine if mixed from folder label or filename
    if folder_label == "mixed":
        is_mixed = True
    elif folder_label == "purebred":
        is_mixed = False
    else:
        is_mixed = len(true_breeds) > 1 if true_breeds else False

    if not true_breeds:
        print(f"  ⚠  {fname}: could not match any known breed in filename — skipped")
        skipped.append(fname)
        continue

    try:
        inp      = preprocess(path)
        preds    = model.predict(inp, verbose=0)[0]
        m        = compute_metrics(preds)
        hit      = breed_hit(m, true_breeds)

        m["filename"]    = fname
        m["true_breeds"] = ", ".join(true_breeds)
        m["is_mixed"]    = is_mixed
        m["hit_score"]   = round(hit, 2)
        results.append(m)

        status   = "✓" if hit > 0.5 else "✗"
        mix_tag  = "[MIXED]" if is_mixed else "[PURE] "
        print(f"  {status} {mix_tag}  {fname}")
        print(f"       Truth : {', '.join(true_breeds)}")
        print(f"       Top-1 : {m['top1_breed']} ({m['top1_conf']:.3f})"
              f"   Top-2: {m['top2_breed']} ({m['top2_conf']:.3f})")
        print(f"       Entropy={m['norm_entropy']:.4f}  Gap={m['conf_gap']:.4f}  Hit={hit:.2f}\n")

    except Exception as e:
        print(f"  ✗ {fname}: {e}\n")

if not results:
    print("No processable images found.")
    sys.exit(0)

mixed_rows = [r for r in results if r["is_mixed"]]
pure_rows  = [r for r in results if not r["is_mixed"]]

# ── Distribution stats ─────────────────────────────────────────────────────
def stats(lst, key):
    vals = [r[key] for r in lst]
    return min(vals), max(vals), round(float(np.mean(vals)), 4)

print("="*65)
print("  DATA STATISTICS")
print("="*65)
print(f"  Purebred images : {len(pure_rows)}")
print(f"  Mixed images    : {len(mixed_rows)}\n")
if pure_rows:
    mn,mx,avg = stats(pure_rows, "norm_entropy")
    print(f"  Purebred entropy : min={mn:.4f}  max={mx:.4f}  avg={avg:.4f}")
    mn,mx,avg = stats(pure_rows, "conf_gap")
    print(f"  Purebred gap     : min={mn:.4f}  max={mx:.4f}  avg={avg:.4f}")
if mixed_rows:
    mn,mx,avg = stats(mixed_rows, "norm_entropy")
    print(f"  Mixed    entropy : min={mn:.4f}  max={mx:.4f}  avg={avg:.4f}")
    mn,mx,avg = stats(mixed_rows, "conf_gap")
    print(f"  Mixed    gap     : min={mn:.4f}  max={mx:.4f}  avg={avg:.4f}")

# ── Grid search for best thresholds ───────────────────────────────────────
print(f"\n{'='*65}")
print("  THRESHOLD GRID SEARCH …")
print("="*65)

entropy_cands = [round(x, 2) for x in np.arange(0.15, 0.80, 0.05)]
gap_cands     = [round(x, 2) for x in np.arange(0.05, 0.65, 0.05)]

best_acc   = -1
best_combo = (0.25, 0.50)

for ent_t, gap_t in itertools.product(entropy_cands, gap_cands):
    correct = sum(
        1 for r in results
        if (not (r["top1_conf"] > 0.80 and r["conf_gap"] > gap_t and r["norm_entropy"] < ent_t))
           == r["is_mixed"]
    )
    acc = correct / len(results)
    if acc > best_acc:
        best_acc   = acc
        best_combo = (ent_t, gap_t)

ent_t, gap_t = best_combo
mild_ent = round(ent_t * 1.6, 2)
mild_gap = round(gap_t * 0.5, 2)
comp_ent = round(ent_t * 2.2, 2)

print(f"\n  ✓ BEST ACCURACY: {best_acc*100:.1f}%  over {len(results)} images")
print(f"    entropy_threshold = {ent_t}")
print(f"    gap_threshold     = {gap_t}")
print(f"""
{'─'*65}
  Paste these constants into api_server.py  →  predict_breed()
{'─'*65}
  # ── CALIBRATED THRESHOLDS ────────────────────────────────
  PURE_ENTROPY_MAX = {ent_t}   # norm_entropy must be BELOW this for purebred
  PURE_GAP_MIN     = {gap_t}   # conf_gap must be ABOVE this for purebred

  if conf > 0.80 and conf_gap > PURE_GAP_MIN and norm_entropy < PURE_ENTROPY_MAX:
      mixed_result = f"Likely Purebred: {{breed}} ({{int(conf * 100)}}% confidence)"

  elif conf > 0.55 and conf_gap > {mild_gap} and norm_entropy < {mild_ent}:
      second_breed, second_pct = composition[1]
      mixed_result = (
          f"Predominantly {{breed}} ({{composition[0][1]}}%). "
          f"Some {{second_breed}} ({{second_pct}}%) traits detected — possibly a mild mix."
      )

  elif conf > 0.35 and norm_entropy < {comp_ent}:
      parts = [f\"{{b}} ({{p}}%)\" for b, p in composition[:2] if p > 5.0]
      mixed_result = f"Mixed Breed: Combination of {{' and '.join(parts)}}."

  elif norm_entropy < {round(comp_ent*1.3, 2)}:
      parts = [f\"{{b}} ({{p}}%)\" for b, p in composition[:3] if p > 5.0]
      mixed_result = f"Complex Mix: Traits of {{', '.join(parts)}} detected."

  else:
      parts = [f\"{{b}} ({{p}}%)\" for b, p in composition[:3]]
      mixed_result = (
          f"Highly Mixed or Uncommon Breed. "
          f"Closest matches: {{', '.join(parts)}}. "
          f"Consider a DNA test for accurate results."
      )
{'─'*65}
""")

# ── Save CSV ──────────────────────────────────────────────────────────────
csv_cols = ["filename","true_breeds","is_mixed","top1_breed","top1_conf",
            "top2_breed","top2_conf","top3_breed","top3_conf",
            "conf_gap","norm_entropy","hit_score"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(results)

print(f"  Full results saved to  {OUT_CSV}\n")
if skipped:
    print(f"  Skipped {len(skipped)} file(s) — breed not recognized in filename:")
    for s in skipped:
        print(f"    {s}")
    print()
