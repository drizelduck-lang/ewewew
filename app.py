#!/usr/bin/env python3
# app.py
import os
import json
import hashlib
import hmac
import random
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Matplotlib backend configuration for server-side execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Image/analysis libs
import cv2
from skimage import filters, morphology
from scipy import ndimage as ndi

# ---------------------- Configuration ----------------------
DATABASE_FILE = "working/hybrid_biohash_records.json"
UPLOAD_FOLDER = "uploads"
LEAF_VISUALS_DIR = "static/leaf_visuals"
ANALYSIS_VISUALS_DIR = "static/analysis_visuals"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LEAF_VISUALS_DIR'] = LEAF_VISUALS_DIR
app.config['ANALYSIS_VISUALS_DIR'] = ANALYSIS_VISUALS_DIR

# Ensure directories exist
Path("working").mkdir(parents=True, exist_ok=True)
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['LEAF_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)
Path(app.config['ANALYSIS_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)

# ---------------------- Feature statistics ----------------------
feature_stats = {
    "junctions":   {"mean": 1989.9, "sd": 1165.210869, "min": 460, "max": 6365},
    "endpoints":   {"mean": 3533.83333, "sd": 1237.4215, "min": 1876, "max": 8642},
    "avg_distance":{"mean": 1.1943633, "sd": 0.009445798, "min": 1.1752, "max": 1.2114},
    "avg_angle":   {"mean": 18.2992633, "sd": 6.816824697, "min": 0.4588, "max": 31.9622},
    "fft_mean":    {"mean": 5.504186667, "sd": 0.11093634, "min": 5.2416, "max": 5.7633},
    "fft_energy":  {"mean": 178733376.8811, "sd": 121685961.7000, "min": 93514400.2723, "max": 379925649.9890},
    "fractal_dim": {"mean": 1.559893333, "sd": 0.131922736, "min": 1.2950, "max": 1.6802}
}

# ---------------------- Utilities & Hybrids ----------------------
def file_seed_from_path(file_path):
    """Seed derived from file path + small prefix of file contents (stable per file)."""
    h = hashlib.sha256()
    h.update(file_path.encode('utf-8'))
    try:
        with open(file_path, "rb") as f:
            h.update(f.read(65536))
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
        features[feature] = float(round(val, 5))
    return features

def generate_bio_key(features):
    feature_string = ",".join([f"{k}:{features[k]:.5f}" for k in sorted(features.keys())])
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

# ---------------------- Database helpers ----------------------
def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_database(data):
    with open(DATABASE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------------------- Visual generator (deterministic) ----------------------
def visualize_leaf_features(features, size=(512,512), seed=None, embed_code=None):
    """
    Deterministic leaf visual generator.
    - seed: integer seed for RNG (if None, will derive from features).
    - embed_code: optional short string to draw on image (acts like a barcode/ID).
    """
    if seed is None:
        seed = int(features.get('junctions', 1000) * 1000003 + features.get('endpoints', 1000))
    rng = np.random.default_rng(int(seed))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size

    # main mid vein
    mid_x = w // 2
    draw.line([(mid_x, int(h * 0.05)), (mid_x, int(h * 0.95))], fill=(34, 139, 34), width=4)

    num_main_veins = max(6, int(features['junctions'] / 200))
    num_branches = max(10, int(features['endpoints'] / 300))

    for i in range(num_main_veins):
        y_start = int(h * (0.1 + 0.8 * (i / max(1, num_main_veins))))
        angle_deg = float(rng.normal(35, max(1.0, features['avg_angle'] / 3)))
        length = max(1.0, float(rng.normal(h * 0.25, max(1.0, features['avg_distance'] * 50))))

        for side in [-1, 1]:
            x_end = int(mid_x + side * length * np.cos(np.radians(angle_deg)))
            y_end = int(y_start - length * np.sin(np.radians(angle_deg)))
            draw.line([(mid_x, y_start), (x_end, y_end)], fill=(34, 139, 34), width=2)

            sub_count = int(rng.integers(2, 5))
            for _ in range(sub_count):
                frac = float(rng.uniform(0.2, 0.8))
                sx = mid_x + side * length * frac * np.cos(np.radians(angle_deg))
                sy = y_start - length * frac * np.sin(np.radians(angle_deg))
                sub_angle = angle_deg + float(rng.normal(20, 10)) * (1 if rng.random() > 0.5 else -1)
                sub_len = max(1.0, float(rng.normal(length * 0.4, 10)))
                ex = int(sx + side * sub_len * np.cos(np.radians(sub_angle)))
                ey = int(sy - sub_len * np.sin(np.radians(sub_angle)))
                draw.line([(sx, sy), (ex, ey)], fill=(34, 139, 34), width=1)

    for _ in range(num_branches):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(34, 139, 34))

    # embed short code in lower-right (transparent bg rectangle + text)
    if embed_code:
        try:
            font = ImageFont.load_default()
            padding = 6
            text = str(embed_code)[:16]
            tw, th = draw.textsize(text, font=font)
            rect_xy = [(w - tw - padding*2 - 4, h - th - padding*2 - 4), (w - 4, h - 4)]
            overlay = Image.new('RGBA', img.size, (255,255,255,0))
            od = ImageDraw.Draw(overlay)
            od.rectangle(rect_xy, fill=(255,255,255,200))
            od.text((w - tw - padding - 4, h - th - padding - 4), text, fill=(30,30,30), font=font)
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        except Exception:
            pass

    return img

# ---------------------- Leaf analysis (fractal, skeleton, DFT) ----------------------
def analyze_leaf_visual(leaf_img, paths):
    """
    Runs analysis on a leaf PIL image and saves results to specified paths.
    'paths' is a dict: {'fractal': path, 'skeleton': path, 'dft': path}
    Returns dict with extracted small result (fractal dim).
    """
    img_rgb = np.array(leaf_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # thresholding + cleanup
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh
    binary = morphology.remove_small_objects(binary, 60)
    binary = morphology.remove_small_holes(binary, 60)

    # ---- Fractal (box counting) ----
    def compute_box_counts(binary, sizes):
        counts = []
        for k in sizes:
            k = int(k)
            if k == 0:
                continue
            H = (binary.shape[0] // k) * k
            W = (binary.shape[1] // k) * k
            if H == 0 or W == 0:
                continue
            Z = binary[:H, :W].astype(int)
            S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, H, k), axis=0), np.arange(0, W, k), axis=1)
            counts.append(float(np.count_nonzero(S)))
        return np.array(counts)

    min_side = min(binary.shape)
    sizes = [min_side // (2**i) for i in range(1, 5) if min_side // (2**i) > 0]
    if not sizes:
        sizes = [1]

    counts = compute_box_counts(binary, sizes)
    if len(counts) > 1:
        logs = np.log(np.array(sizes))
        logn = np.log(counts + 1e-9)
        coeffs = np.polyfit(logs, logn, 1)
        D = -coeffs[0]
    else:
        D = 0

    # save fractal visualization
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        ax = axes[i]
        if i < len(sizes):
            s, c = sizes[i], counts[i]
            H, W = (binary.shape[0] // s) * s, (binary.shape[1] // s) * s
            img = img_rgb.copy().astype(np.float32) / 255.0
            for y in range(0, H, s):
                cv2.line(img, (0, y), (W, y), (0, 0, 0), 1)
            for x in range(0, W, s):
                cv2.line(img, (x, 0), (x, H), (0, 0, 0), 1)
            ax.imshow(img)
            ax.set_title(f"Box={s}px\nN={int(c)}")
        else:
            ax.set_visible(False)
        ax.axis('off')
    fig.suptitle(f"Fractal Box-Counting (D ≈ {D:.3f})")
    plt.tight_layout()
    plt.savefig(paths['fractal'], dpi=100)
    plt.close(fig)

    # ---- Skeleton & nodes ----
    skel = morphology.skeletonize(binary)
    kernel = np.ones((3, 3))
    neighbors = ndi.convolve(skel.astype(int), kernel, mode='constant') - skel
    endpoints = (skel & (neighbors == 1))
    junctions = (skel & (neighbors >= 3))
    yj, xj = np.nonzero(junctions)
    ye, xe = np.nonzero(endpoints)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(img_rgb)
    ax2.scatter(xj, yj, c='red', s=15, label='junction')
    ax2.scatter(xe, ye, c='cyan', s=10, label='endpoint')
    ax2.legend()
    ax2.set_title("Skeleton + Nodes")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(paths['skeleton'], dpi=100)
    plt.close(fig2)

    # ---- 2D DFT ----
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
    plt.figure(figsize=(5, 4))
    plt.imshow(mag, cmap='inferno')
    plt.title("2D DFT (log-magnitude)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(paths['dft'], dpi=100)
    plt.close()

    return {"fractal_dim": float(round(D, 6))}

# ---------------------- Visuals helper ----------------------
def generate_all_visuals(features, filename, suffix="", visual_seed=None, embed_code=None):
    """Generates all visuals deterministically and returns relative web paths."""
    leaf_file = f"{filename}_leaf{suffix}.png"
    fractal_file = f"{filename}_fractal{suffix}.png"
    skeleton_file = f"{filename}_skeleton{suffix}.png"
    dft_file = f"{filename}_dft{suffix}.png"

    leaf_save_path = os.path.join(app.config['LEAF_VISUALS_DIR'], leaf_file)
    analysis_save_paths = {
        "fractal": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], fractal_file),
        "skeleton": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], skeleton_file),
        "dft": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], dft_file),
    }

    if visual_seed is None:
        visual_seed = int(features.get('junctions', 1000) * 1000003 + features.get('endpoints', 1000))

    leaf_img = visualize_leaf_features(features, seed=visual_seed, embed_code=embed_code)
    leaf_img.save(leaf_save_path)

    analyze_leaf_visual(leaf_img, analysis_save_paths)

    return {
        "leaf": f"{app.config['LEAF_VISUALS_DIR']}/{leaf_file}",
        "fractal": f"{app.config['ANALYSIS_VISUALS_DIR']}/{fractal_file}",
        "skeleton": f"{app.config['ANALYSIS_VISUALS_DIR']}/{skeleton_file}",
        "dft": f"{app.config['ANALYSIS_VISUALS_DIR']}/{dft_file}",
    }

# ---------------------- Flask routes ----------------------
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

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    assignments = load_database()
    record = next((r for r in assignments if r["file_name"] == filename), None)
    if record is not None:
        return render_template('result.html',
                               message=f"Error: File '{filename}' is already registered.",
                               status="warning",
                               record=record)

    # Registration algorithm
    seed = file_seed_from_path(file_path)
    features = generate_synthetic_features(feature_stats, seed)
    bio_key, _ = generate_bio_key(features)
    hybrid_hash, salt = generate_hybrid_hash(bio_key)
    file_hmac = compute_hmac_file(file_path, hybrid_hash)

    # Deterministic visuals (embed part of hybrid_hash)
    visual_seed = seed
    embed_code = hybrid_hash[:12]
    visual_paths = generate_all_visuals(features, filename, suffix="_stored", visual_seed=visual_seed, embed_code=embed_code)

    record = {
        "file_name": filename,
        "file_path": file_path,
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
                           message=f"File '{filename}' registered successfully!",
                           status="success",
                           record=record,
                           stored_visuals=visual_paths)

@app.route('/verify', methods=['POST'])
def verify_image():
    if 'file' not in request.files:
        return render_template('result.html', message="No file selected.", status="danger")
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', message="No file selected.", status="danger")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    assignments = load_database()
    record = next((r for r in assignments if r["file_name"] == filename), None)
    if record is None:
        return render_template('result.html',
                               message=f"Error: File '{filename}' not found in database.",
                               status="danger")

    # Stored golden values
    stored_bio_key = record["bio_key"]
    stored_hybrid_hash = record["hybrid_hash"]
    stored_hmac = record["hmac"]
    stored_visuals = record.get("visuals", {})

    # Check 1: HMAC integrity
    new_hmac_for_integrity = compute_hmac_file(file_path, stored_hybrid_hash)
    is_hmac_match = (new_hmac_for_integrity == stored_hmac)

    # Check 2: Biometric key check
    new_seed = file_seed_from_path(file_path)
    new_features = generate_synthetic_features(feature_stats, new_seed)
    new_bio_key, _ = generate_bio_key(new_features)
    is_biokey_match = (new_bio_key == stored_bio_key)

    # Generate visuals for uploaded file (use same visual_seed and embed_code so images match if identical)
    visual_seed = new_seed
    embed_code = stored_hybrid_hash[:12] if stored_hybrid_hash else None
    uploaded_visuals = generate_all_visuals(new_features, filename, suffix="_uploaded", visual_seed=visual_seed, embed_code=embed_code)

    # Diagnostics: compare stored vs new features
    stored_features = record.get("features", {})
    diffs = {}
    zscores = {}
    significant = []
    for k in stored_features.keys():
        stored_val = float(stored_features[k])
        new_val = float(new_features.get(k, 0.0))
        abs_diff = new_val - stored_val
        sd = feature_stats.get(k, {}).get("sd", 1.0)
        z = abs_diff / (sd + 1e-9)
        diffs[k] = round(abs_diff, 6)
        zscores[k] = round(z, 3)
        pct_change = (abs_diff / (stored_val + 1e-9)) * 100.0 if stored_val != 0 else 0.0
        is_significant = (abs(z) > 2.0) or (abs(pct_change) > 5.0)
        if is_significant:
            significant.append({"feature": k, "stored": stored_val, "new": new_val, "diff": round(abs_diff, 6), "z": round(z, 3), "pct": round(pct_change, 3)})

    significant = sorted(significant, key=lambda x: abs(x["z"]), reverse=True)

    hints = []
    if any(s['feature'] in ['fft_mean', 'fft_energy', 'fractal_dim'] for s in significant):
        hints.append("Changes in frequency/fractal features often indicate color shifts, blurring, or brightness/contrast adjustments.")
    if any(s['feature'] in ['junctions', 'endpoints'] for s in significant):
        hints.append("Large differences in junctions/endpoints suggest morphological changes — cropping, heavy compression, resizing, or content editing.")
    if any(s['feature'] in ['avg_distance', 'avg_angle'] for s in significant):
        hints.append("Angle/distance changes can indicate geometric transforms (rotation, perspective warp) or strong local filtering.")
    if not significant and not is_hmac_match:
        hints.append("Binary data changed (HMAC mismatch) but features unchanged — could be metadata or packaging changes that didn't affect extracted features.")
    if not significant and is_hmac_match:
        hints.append("No significant feature changes detected and HMAC matched — file looks intact.")

    diagnostics = {
        "diffs": diffs,
        "zscores": zscores,
        "significant": significant,
        "hints": hints
    }

    # Final verdict
    if is_hmac_match and is_biokey_match:
        message = "VERIFIED: HMAC and Biometric Key match."
        status = "success"
    else:
        message = "TAMPERED! File data does not match the registered record."
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
                               "new_features": new_features
                           },
                           stored_visuals=stored_visuals,
                           uploaded_visuals=uploaded_visuals,
                           diagnostics=diagnostics)

# ---------------------- Run ----------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
