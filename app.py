import os
import json
import hashlib
import hmac
import random
import copy
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Matplotlib backend configuration for server-side execution
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# New imports from your script
import cv2
from skimage import filters, morphology
from scipy import ndimage as ndi

# ==========================================================
# CONFIGURATION
# ==========================================================
DATABASE_FILE = "working/hybrid_biohash_records.json"
UPLOAD_FOLDER = "uploads/"
LEAF_VISUALS_DIR = "static/leaf_visuals"
ANALYSIS_VISUALS_DIR = "static/analysis_visuals"

# --- NEW: Filename prefixes ---
# Use a prefix for registered files to protect them
REGISTERED_PREFIX = "reg_"
VERIFY_PREFIX = "verify_"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LEAF_VISUALS_DIR'] = LEAF_VISUALS_DIR
app.config['ANALYSIS_VISUALS_DIR'] = ANALYSIS_VISUALS_DIR

# Create necessary directories
Path("working").mkdir(parents=True, exist_ok=True)
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['LEAF_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)
Path(app.config['ANALYSIS_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)


# ==========================================================
# FEATURE STATS & CONSTANTS
# ==========================================================
feature_stats = {
    "junctions":   {"mean": 1989.9, "sd": 1165.210869, "min": 460, "max": 6365},
    "endpoints":   {"mean": 3533.83333, "sd": 1237.4215, "min": 1876, "max": 8642},
    "avg_distance":{"mean": 1.1943633, "sd": 0.009445798, "min": 1.1752, "max": 1.2114},
    "avg_angle":   {"mean": 18.2992633, "sd": 6.816824697, "min": 0.4588, "max": 31.9622},
    "fft_mean":    {"mean": 5.504186667, "sd": 0.11093634, "min": 5.2416, "max": 5.7633},
    "fft_energy":  {"mean": 178733376.8811, "sd": 121685961.7000, "min": 93514400.2723, "max": 379925649.9890},
    "fractal_dim": {"mean": 1.559893333, "sd": 0.131922736, "min": 1.2950, "max": 1.6802}
}
_EPS = 1e-9
TAMPER_CODE_MAP = {
    "invert": 101, "pixelate": 202, "noise": 303, "rectangle": 404, "watermark": 505,
    "combined": 606, "cropping": 707, "rotation": 808, "overlay": 909, "blur": 111,
    "contrast_change": 222, "brightness_change": 333, "erase/obliteration": 444,
    "scaling/resize": 555, "splice/paste": 999
}
TAMPER_CODE_TO_NAME = {v: k for k, v in TAMPER_CODE_MAP.items()}

# ==========================================================
# HYBRID BIOHASH & FEATURE UTILITIES
# ==========================================================
def file_seed_from_path(file_path):
    h = hashlib.sha256()
    # h.update(file_path.encode('utf-8')) # <-- Removed this line, seed is content-only
    try:
        with open(file_path, "rb") as f:
            h.update(f.read(65536))  # first 64 KB
    except Exception:
        pass
    digest = h.digest()
    return int.from_bytes(digest[:8], 'big', signed=False)

def generate_synthetic_features(stats, seed):
    rng = np.random.default_rng(seed)
    features = {}
    for feature, vals in stats.items():
        val = rng.normal(vals["mean"], vals["sd"])
        val = float(np.clip(val, vals["min"], vals["max"]))
        features[feature] = round(val, 5)
    return features

def clamp_features(features):
    f = {}
    for k, v in features.items():
        if k == "tamper_code":
            f[k] = int(v)
            continue
        if k in feature_stats:
            mn = feature_stats[k]["min"]
            mx = feature_stats[k]["max"]
            f[k] = float(np.clip(v, mn, mx))
        else:
            f[k] = float(v)
    return f

def generate_bio_key(features):
    ordered_keys = sorted(features.keys())
    feature_string = ",".join([f"{k}:{features[k]:.5f}" if k != "tamper_code" else f"{k}:{int(features[k])}" for k in ordered_keys])
    bio_key = hashlib.sha256(feature_string.encode()).hexdigest()
    return bio_key, feature_string

def generate_hybrid_hash(bio_key, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    hybrid_input = (salt + bio_key).encode('utf-8')
    hybrid_hash = hashlib.sha256(hybrid_input).hexdigest()
    return hybrid_hash, salt

def compute_hmac_file(file_path, hybrid_hash):
    h = hmac.new(hybrid_hash.encode('utf-8'), digestmod=hashlib.sha256)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ==========================================================
# DATABASE HELPERS
# ==========================================================
def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_database(data):
    with open(DATABASE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ==========================================================
# TAMPER INFERENCE ENGINE
# ==========================================================
def _relative_change(orig, new):
    return (new - orig) / (abs(orig) + _EPS)

def infer_tamper_type(original_features, new_features, top_k=3):
    tc = new_features.get("tamper_code", None)
    if tc is not None:
        tamper_name = TAMPER_CODE_TO_NAME.get(int(tc), None)
        if tamper_name:
            return {
                "ratios": {}, "scores": {tamper_name: 1.0}, "top": [(tamper_name, 1.0)],
                "final": tamper_name, "best": {"type": tamper_name, "score": 1.0},
                "tamper_code": int(tc), "note": "Decoded from tamper_code"
            }
    ratios = {}
    for k in original_features.keys():
        orig = float(original_features.get(k, 0.0))
        new = float(new_features.get(k, orig))
        ratios[k] = _relative_change(orig, new)
    tamper_types = {
        "invert": {"fft_mean": 0.6, "fft_energy": 0.6, "fractal_dim": 0.2},
        "noise": {"fft_energy": 1.0, "fft_mean": 0.6, "fractal_dim": 0.4},
        "pixelation": {"fft_energy": -0.8, "fft_mean": -0.4, "avg_distance": -0.3},
        "rectangle_overlay": {"junctions": -0.9, "endpoints": -0.6},
        "watermark/text_overlay": {"endpoints": 0.9, "fft_mean": 0.2},
        "combined_hybrid": {"fft_energy": 0.8, "junctions": -0.6, "endpoints": 0.6, "fractal_dim": 0.3},
        "cropping": {"junctions": -1.0, "endpoints": -0.6, "avg_distance": -0.2},
        "erase/obliteration": {"junctions": -1.0, "endpoints": -0.9},
        "drawing/annotation": {"endpoints": 1.0, "junctions": 0.2},
        "rotation/distortion": {"avg_angle": 1.0},
        "scaling/resize": {"avg_distance": 1.0},
        "jpeg_compression": {"fft_energy": -0.6, "fft_mean": -0.3},
        "blur/smoothing": {"fft_energy": -0.7, "fft_mean": -0.4},
        "contrast_change": {"fft_mean": 0.5},
        "brightness_change": {"fft_mean": 0.3},
        "splice/paste": {"junctions": -0.8, "fft_energy": 0.2},
    }
    def _map_to_confidence(sum_positive, sum_abs_weights, scale=0.25):
        if sum_abs_weights <= 0: return 0.0
        raw = sum_positive / (sum_abs_weights + _EPS)
        conf = float(min(1.0, raw / scale))
        return conf
    scores = {}
    for tname, fmap in tamper_types.items():
        sum_positive, sum_abs_weights = 0.0, 0.0
        for feat, weight in fmap.items():
            sum_abs_weights += abs(weight)
            ratio = ratios.get(feat, 0.0)
            contrib_raw = ratio * weight
            if contrib_raw > 0:
                sum_positive += abs(contrib_raw)
        conf = _map_to_confidence(sum_positive, sum_abs_weights, scale=0.25)
        scores[tname] = conf
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_scores[:top_k]
    best_name, best_score = top[0] if top else (None, 0.0)
    if best_score < 0.15:
        final = "Unknown (no significant feature change)"
    else:
        if len(top) > 1 and top[1][1] >= 0.65 * best_score and top[1][1] > 0.15:
            final = f"Combined: {top[0][0]} + {top[1][0]}"
        else:
            final = best_name
    return {"ratios": ratios, "scores": scores, "top": top, "final": final, 
            "best": {"type": best_name, "score": best_score}, "note": "Heuristic feature-only inference"}

# ==========================================================
# LEAF VISUALIZATION & ANALYSIS
# ==========================================================
def visualize_leaf_features(features, size=(512, 512)):
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    rng = np.random.default_rng(int(features.get('junctions', 1000) + features.get('endpoints', 1000) * 100))
    mid_x = w // 2
    draw.line([(mid_x, int(h * 0.05)), (mid_x, int(h * 0.95))], fill=(34, 139, 34), width=4)
    num_main_veins = max(6, int(features.get('junctions', 1200) / 200))
    for i in range(num_main_veins):
        y_start = int(h * (0.1 + 0.8 * (i / num_main_veins)))
        angle_deg = float(rng.normal(35, features.get('avg_angle', 18) / 3))
        length = float(rng.normal(h * 0.25, features.get('avg_distance', 1.19) * 50))
        for side in [-1, 1]:
            x_end = int(mid_x + side * length * np.cos(np.radians(angle_deg)))
            y_end = int(y_start - length * np.sin(np.radians(angle_deg)))
            draw.line([(mid_x, y_start), (x_end, y_end)], fill=(34, 139, 34), width=2)
            sub_count = int(rng.integers(2, 5))
            for _ in range(sub_count):
                frac = float(rng.uniform(0.2, 0.8))
                sx = mid_x + side * length * frac * np.cos(np.radians(angle_deg))
                sy = y_start - length * frac * np.sin(np.radians(angle_deg))
                sub_angle = angle_deg + float(rng.normal(20, 10)) * (1 if rng.choice([True, False]) else -1)
                sub_len = float(rng.normal(length * 0.4, 10))
                ex = int(sx + side * sub_len * np.cos(np.radians(sub_angle)))
                ey = int(sy - sub_len * np.sin(np.radians(sub_angle)))
                draw.line([(sx, sy), (ex, ey)], fill=(34, 139, 34), width=1)
    num_branches = max(10, int(features.get('endpoints', 1200) / 300))
    for _ in range(num_branches):
        x = int(rng.integers(0, w)); y = int(rng.integers(0, h)); r = int(rng.integers(1, 3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(34, 139, 34))
    return img

def analyze_leaf_visual(leaf_img, paths):
    img_rgb = np.array(leaf_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    thresh = filters.threshold_otsu(gray); binary = gray < thresh
    binary = morphology.remove_small_objects(binary, 60); binary = morphology.remove_small_holes(binary, 60)
    def compute_box_counts(binary, sizes):
        counts = []
        for k in sizes:
            k = int(k); H = (binary.shape[0] // k) * k; W = (binary.shape[1] // k) * k
            if H == 0 or W == 0: continue
            Z = binary[:H, :W].astype(int)
            S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, H, k), axis=0), np.arange(0, W, k), axis=1)
            counts.append(float(np.count_nonzero(S)))
        return np.array(counts)
    min_side = min(binary.shape); sizes = [max(1, min_side // (2 ** i)) for i in range(1, 5)]
    counts = compute_box_counts(binary, sizes); logs = np.log(np.array(sizes) + 1e-9); logn = np.log(counts + 1e-9)
    D = -np.polyfit(logs, logn, 1)[0] if len(logs) >= 2 and len(logn) >= 2 else float('nan')
    fig, axes = plt.subplots(1, max(1, len(sizes)), figsize=(12, 3))
    if not isinstance(axes, np.ndarray): axes = [axes]
    for ax, s, c in zip(axes, sizes, counts):
        H, W = (binary.shape[0] // s) * s, (binary.shape[1] // s) * s
        img = img_rgb.copy().astype(np.float32) / 255.0
        for y in range(0, H, s): cv2.line(img, (0, y), (W, y), (0, 0, 0), 1)
        for x in range(0, W, s): cv2.line(img, (x, 0), (x, H), (0, 0, 0), 1)
        ax.imshow(img); ax.set_title(f"Box={s}px\nN={int(c)}"); ax.axis('off')
    fig.suptitle(f"Fractal Box-Counting (D ≈ {D:.3f})")
    plt.tight_layout(); plt.savefig(paths['fractal'], dpi=100); plt.close(fig)
    skel = morphology.skeletonize(binary); kernel = np.ones((3, 3))
    neighbors = ndi.convolve(skel.astype(int), kernel, mode='constant') - skel
    endpoints = (skel & (neighbors == 1)); junctions = (skel & (neighbors >= 3))
    yj, xj = np.nonzero(junctions); ye, xe = np.nonzero(endpoints)
    fig2, ax2 = plt.subplots(figsize=(6, 6)); ax2.imshow(img_rgb)
    if len(xj) > 0: ax2.scatter(xj, yj, c='red', s=15, label='junction')
    if len(xe) > 0: ax2.scatter(xe, ye, c='cyan', s=10, label='endpoint')
    if len(xj) > 0 or len(xe) > 0: ax2.legend()
    ax2.set_title("Skeleton + Nodes"); ax2.axis('off')
    plt.tight_layout(); plt.savefig(paths['skeleton'], dpi=100); plt.close(fig2)
    F = np.fft.fftshift(np.fft.fft2(gray)); mag = np.log1p(np.abs(F))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
    plt.figure(figsize=(5, 4)); plt.imshow(mag, cmap='inferno'); plt.title("2D DFT (log-magnitude)"); plt.axis('off')
    plt.tight_layout(); plt.savefig(paths['dft'], dpi=100); plt.close()
    return {"fractal_dim": D}

def generate_all_visuals(features, filename, suffix=""):
    base_filename = os.path.splitext(filename)[0]
    leaf_file = f"{base_filename}_leaf{suffix}.png"
    fractal_file = f"{base_filename}_fractal{suffix}.png"
    skeleton_file = f"{base_filename}_skeleton{suffix}.png"
    dft_file = f"{base_filename}_dft{suffix}.png"
    leaf_save_path = os.path.join(app.config['LEAF_VISUALS_DIR'], leaf_file)
    analysis_save_paths = {
        "fractal": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], fractal_file),
        "skeleton": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], skeleton_file),
        "dft": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], dft_file),
    }
    leaf_img = visualize_leaf_features(features)
    leaf_img.save(leaf_save_path)
    analyze_leaf_visual(leaf_img, analysis_save_paths)
    return {
        "leaf": f"leaf_visuals/{leaf_file}",
        "fractal": f"analysis_visuals/{fractal_file}",
        "skeleton": f"analysis_visuals/{skeleton_file}",
        "dft": f"analysis_visuals/{dft_file}",
    }

# ==========================================================
# FLASK ROUTES (Updated)
# ==========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_image():
    if 'file' not in request.files:
        return render_template('result.html', message="No file selected.", status="danger")
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', message="No file selected.", status="danger")

    if file:
        # 1. Get the original filename for lookup
        original_filename = secure_filename(file.filename)

        assignments = load_database()
        record = next((r for r in assignments if r["file_name"] == original_filename), None)

        if record is not None:
            return render_template('result.html', 
                                   message=f"Error: File '{original_filename}' is already registered.", 
                                   status="warning", 
                                   record=record,
                                   stored_visuals=record.get("visuals"))
        
        # 2. Create a new, permanent "storage" filename for the copy
        storage_filename = f"{REGISTERED_PREFIX}{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], storage_filename)
        
        # 3. Save the file to its new permanent path
        file.save(file_path)

        # 4. Run all logic on the new, saved copy
        seed = file_seed_from_path(file_path)
        features_raw = generate_synthetic_features(feature_stats, seed)
        features = clamp_features(features_raw)
        bio_key, _ = generate_bio_key(features)
        hybrid_hash, salt = generate_hybrid_hash(bio_key)
        file_hmac = compute_hmac_file(file_path, hybrid_hash)
        
        # 5. Generate visuals using the *storage_filename* base
        visual_paths = generate_all_visuals(features, storage_filename, suffix="_stored")

        # 6. Create the record
        record = {
            "file_name": original_filename,  # The original name is the "key"
            "storage_filename": storage_filename, # The new, protected filename
            "file_path": file_path, # Full path for server use (optional)
            "features": features,
            "bio_key": bio_key,
            "salt": salt,
            "hybrid_hash": hybrid_hash,
            "hmac": file_hmac,
            "visuals": visual_paths
        }
        assignments.append(record)
        save_database(assignments)
        
        return render_template('result.html', 
                               message=f"File '{original_filename}' registered successfully!", 
                               status="success", 
                               record=record,
                               stored_visuals=visual_paths)
    
    return redirect(url_for('index'))


@app.route('/verify', methods=['POST'])
def verify_image():
    if 'file' not in request.files:
        return render_template('result.html', message="No file selected.", status="danger")
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', message="No file selected.", status="danger")

    if file:
        # 1. Get the original filename for lookup
        original_filename = secure_filename(file.filename)

        assignments = load_database()
        record = next((r for r in assignments if r["file_name"] == original_filename), None)

        if record is None:
            return render_template('result.html', 
                                   message=f"Error: File '{original_filename}' not found in database.", 
                                   status="danger")
        
        # 2. Get the "golden" stored filename (e.g., "reg_image.jpg")
        # Fallback to file_name for any old records before this change
        stored_filename = record.get("storage_filename", record["file_name"])

        # 3. Create a temporary "duplicate" filename for the uploaded file
        temp_filename = f"{VERIFY_PREFIX}{original_filename}"
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_file_path)
        
        # === VERIFICATION LOGIC ===
        
        stored_bio_key = record["bio_key"]
        stored_hybrid_hash = record["hybrid_hash"]
        stored_hmac = record["hmac"]
        stored_salt = record["salt"] # <-- ADD THIS LINE
        stored_visuals = record.get("visuals", {})
        stored_features = record.get("features", {})
        
        # --- Check 1: HMAC Integrity (uses temp_file_path) ---
        new_hmac_for_integrity = compute_hmac_file(temp_file_path, stored_hybrid_hash)
        is_hmac_match = (new_hmac_for_integrity == stored_hmac)
        
        # --- Check 2: Biometric Key (uses temp_file_path) ---
        new_seed = file_seed_from_path(temp_file_path)
        new_features_raw = generate_synthetic_features(feature_stats, new_seed)
        new_features = clamp_features(new_features_raw)
        new_bio_key, _ = generate_bio_key(new_features)
        is_biokey_match = (new_bio_key == stored_bio_key)

        # --- Check 3: Hybrid Hash Check (derived from Bio-Key) ---
        new_hybrid_hash, _ = generate_hybrid_hash(new_bio_key, stored_salt)
        is_hybrid_hash_match = (new_hybrid_hash == stored_hybrid_hash)

        # --- Tamper Inference ---
        tamper_analysis = infer_tamper_type(stored_features, new_features)

        # --- Generate visuals for the UPLOADED file (uses temp_filename) ---
        uploaded_visuals = generate_all_visuals(new_features, temp_filename, suffix="_uploaded")

        # --- Set final status ---
        if is_hmac_match and is_biokey_match:
            message = f"VERIFIED: HMAC and Biometric Key match."
            status = "success"
        else:
            message = f"TAMPERED! File data does not match the registered record."
            status = "danger"
            
        return render_template('result.html', 
                               message=message, 
                               status=status, 
                               record=record,
                               check_results={
                                   "new_hmac": new_hmac_for_integrity,
                                   "is_hmac_match": is_hmac_match,
                                   "new_bio_key": new_bio_key,
                                   "is_biokey_match": is_biokey_match,
                                   "new_features": new_features,
                                   "new_hybrid_hash": new_hybrid_hash,
                                   "is_hybrid_hash_match": is_hybrid_hash_match
                               },
                               # --- NEW/MODIFIED CONTEXT VARIABLES ---
                               stored_visuals=stored_visuals,
                               uploaded_visuals=uploaded_visuals,
                               tamper_analysis=tamper_analysis,
                               # Pass the filenames for the side-by-side image comparison
                               stored_image_filename=stored_filename,
                               uploaded_image_filename=temp_filename
                               )
    
    return redirect(url_for('index'))

# ==========================================================
# ROUTE FOR LISTING ALL ITEMS
# ==========================================================
@app.route('/list_items')
def list_items():
    """Serves the page that lists all registered items."""
    assignments = load_database()
    return render_template('list_items.html', items=assignments)

# ==========================================================
# ROUTE TO SERVE UPLOADED FILES
# ==========================================================
@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serves a file from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)