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

# New imports from your script
import cv2
from skimage import filters, morphology
from scipy import ndimage as ndi

# ==========================================================
# CONFIGURATION
# ==========================================================
DATABASE_FILE = "working/hybrid_biohash_records.json"
UPLOAD_FOLDER = "uploads"
LEAF_VISUALS_DIR = "static/leaf_visuals"
ANALYSIS_VISUALS_DIR = "static/analysis_visuals"
REPORTS_DIR = "working/reports"

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LEAF_VISUALS_DIR'] = LEAF_VISUALS_DIR
app.config['ANALYSIS_VISUALS_DIR'] = ANALYSIS_VISUALS_DIR
app.config['REPORTS_DIR'] = REPORTS_DIR

# Create necessary directories
Path("working").mkdir(parents=True, exist_ok=True)
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['LEAF_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)
Path(app.config['ANALYSIS_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)
Path(app.config['REPORTS_DIR']).mkdir(parents=True, exist_ok=True)

# ==========================================================
# FEATURE STATS (From your script)
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

# ==========================================================
# HYBRID BIOHASH FUNCTIONS (From your script)
# ==========================================================
def file_seed_from_path(file_path):
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
        features[feature] = round(val, 5)
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

# ==========================================================
# DATABASE HELPER FUNCTIONS
# ==========================================================
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

# ==========================================================
# VISUALIZATION & ANALYSIS HELPERS (enhancements added)
# ==========================================================
def compute_radial_profile(mag):
    """
    Compute a simple radial average of the 2D FFT magnitude (centered).
    Returns radius bins and averaged magnitude per radius.
    """
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.indices((h, w))
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    R = R.astype(np.int32)
    maxr = np.max(R)
    tbin = np.bincount(R.ravel(), mag.ravel())
    nr = np.bincount(R.ravel())
    radialprofile = tbin / (nr + 1e-9)
    radii = np.arange(len(radialprofile))
    return radii, radialprofile

# ==========================================================
# NEW: LEAF VEIN VISUALIZATION FUNCTION (From your script)
# ==========================================================
def visualize_leaf_features(features, size=(512,512)):
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size

    rng = np.random.default_rng(int(features['junctions'] + features['endpoints'] * 100))

    mid_x = w // 2
    draw.line([(mid_x, h * 0.05), (mid_x, h * 0.95)], fill=(34, 139, 34), width=4)

    num_main_veins = max(6, int(features['junctions'] / 200))
    num_branches = max(10, int(features['endpoints'] / 300))

    for i in range(num_main_veins):
        y_start = int(h * (0.1 + 0.8 * (i / num_main_veins)))
        angle_deg = rng.normal(35, features['avg_angle'] / 3) 
        length = rng.normal(h * 0.25, features['avg_distance'] * 50)

        for side in [-1, 1]: 
            x_end = int(mid_x + side * length * np.cos(np.radians(angle_deg)))
            y_end = int(y_start - length * np.sin(np.radians(angle_deg)))
            draw.line([(mid_x, y_start), (x_end, y_end)], fill=(34, 139, 34), width=2)
            
            sub_count = rng.integers(2, 5)
            for _ in range(sub_count):
                frac = rng.uniform(0.2, 0.8)
                sx = mid_x + side * length * frac * np.cos(np.radians(angle_deg))
                sy = y_start - length * frac * np.sin(np.radians(angle_deg))
                sub_angle = angle_deg + rng.normal(20, 10) * rng.choice([-1, 1])
                sub_len = rng.normal(length * 0.4, 10)
                ex = int(sx + side * sub_len * np.cos(np.radians(sub_angle)))
                ey = int(sy - sub_len * np.sin(np.radians(sub_angle)))
                draw.line([(sx, sy), (ex, ey)], fill=(34, 139, 34), width=1)

    for _ in range(num_branches):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        r = rng.integers(1, 3)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(34, 139, 34))
    return img

# ==========================================================
# NEW: LEAF ANALYSIS FUNCTION (Adapted from your script)
# ==========================================================
def analyze_leaf_visual(leaf_img, paths):
    """
    Runs analysis on a leaf PIL image and saves results to specified paths.
    'paths' is a dict: {'fractal': path, 'skeleton': path, 'dft': path, 'hist': path, 'radial': path}
    Returns dict of computed numeric results.
    """
    img_rgb = np.array(leaf_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Basic thresholding and cleanup
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh
    binary = morphology.remove_small_objects(binary, 60)
    binary = morphology.remove_small_holes(binary, 60)

    # ---- Fractal (box counting) ----
    def compute_box_counts(binary, sizes):
        counts = []
        for k in sizes:
            k = int(k)
            if k == 0: continue
            H = (binary.shape[0] // k) * k
            W = (binary.shape[1] // k) * k
            if H == 0 or W == 0: continue
            Z = binary[:H,:W].astype(int)
            S = np.add.reduceat(np.add.reduceat(Z, np.arange(0,H,k), axis=0), np.arange(0,W,k), axis=1)
            counts.append(float(np.count_nonzero(S)))
        return np.array(counts)

    min_side = min(binary.shape)
    sizes = [max(1, min_side//(2**i)) for i in range(1,6) if min_side//(2**i) > 0]
    if not sizes: sizes = [1]
        
    counts = compute_box_counts(binary, sizes)
    if len(counts) > 1:
        logs = np.log(np.array(sizes))
        logn = np.log(counts + 1e-9)
        coeffs = np.polyfit(logs, logn, 1)
        D = -coeffs[0]
    else:
        D = 0 # Not enough data for fit

    # Save fractal visualization (box overlays)
    fig, axes = plt.subplots(1,4, figsize=(12,3))
    for i in range(4):
        if i < len(sizes):
            s, c = sizes[i], counts[i]
            ax = axes[i]
            H, W = (binary.shape[0] // s) * s, (binary.shape[1] // s) * s
            img = img_rgb.copy().astype(np.float32)/255.0
            for y in range(0, H, s): cv2.line(img, (0,y), (W,y), (0,0,0), 1)
            for x in range(0, W, s): cv2.line(img, (x,0), (x,H), (0,0,0), 1)
            ax.imshow(img)
            ax.set_title(f"Box={s}px\nN={int(c)}")
            ax.axis('off')
        else:
            axes[i].set_visible(False)
    fig.suptitle(f"Fractal Box-Counting (D ≈ {D:.3f})")
    plt.tight_layout()
    plt.savefig(paths['fractal'], dpi=100)
    plt.close(fig)

    # ---- Skeleton & nodes ----
    skel = morphology.skeletonize(binary)
    kernel = np.ones((3,3))
    neighbors = ndi.convolve(skel.astype(int), kernel, mode='constant') - skel
    endpoints = (skel & (neighbors==1))
    junctions = (skel & (neighbors>=3))
    yj,xj = np.nonzero(junctions)
    ye,xe = np.nonzero(endpoints)
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.imshow(img_rgb)
    ax2.scatter(xj,yj, c='red', s=15, label='junction')
    ax2.scatter(xe,ye, c='cyan', s=10, label='endpoint')
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
    plt.figure(figsize=(5,4))
    plt.imshow(mag, cmap='inferno')
    plt.title("2D DFT (log-magnitude)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(paths['dft'], dpi=100)
    plt.close()

    # ---- Histogram visualization ----
    fig3, ax3 = plt.subplots(figsize=(5,3))
    ax3.hist((gray*255).ravel(), bins=64)
    ax3.set_title("Grayscale Histogram")
    ax3.set_xlabel("Pixel value")
    ax3.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(paths['hist'], dpi=100)
    plt.close(fig3)

    # ---- Radial profile of FFT magnitude ----
    mag_abs = np.abs(F)
    mag_log = np.log1p(mag_abs)
    mag_norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-9)
    radii, radial = compute_radial_profile(mag_norm)
    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.plot(radii, radial)
    ax4.set_title("FFT Radial Profile")
    ax4.set_xlabel("Radius (px)")
    ax4.set_ylabel("Average magnitude")
    plt.tight_layout()
    plt.savefig(paths['radial'], dpi=100)
    plt.close(fig4)

    # Numeric features to return (we keep fractal_dim for biohash; others are optional)
    results = {
        "fractal_dim": float(D),
        "junctions_count": int(len(yj)),
        "endpoints_count": int(len(ye)),
        "fft_mean": float(np.mean(np.abs(F))),
        "fft_energy": float(np.sum(np.abs(F)))
    }
    return results

# ==========================================================
# NEW: HELPER FOR GENERATING VISUALS
# ==========================================================
def generate_all_visuals(features, filename, suffix=""):
    """
    Generates all visuals for a set of features and saves them.
    Returns a dict of their web-accessible paths.
    """
    # Create filenames
    leaf_file = f"{filename}_leaf{suffix}.png"
    fractal_file = f"{filename}_fractal{suffix}.png"
    skeleton_file = f"{filename}_skeleton{suffix}.png"
    dft_file = f"{filename}_dft{suffix}.png"
    hist_file = f"{filename}_hist{suffix}.png"
    radial_file = f"{filename}_radial{suffix}.png"
    
    # Define full disk paths for saving
    leaf_save_path = os.path.join(app.config['LEAF_VISUALS_DIR'], leaf_file)
    analysis_save_paths = {
        "fractal": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], fractal_file),
        "skeleton": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], skeleton_file),
        "dft": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], dft_file),
        "hist": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], hist_file),
        "radial": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], radial_file),
    }
    
    # Generate and save leaf image visualization
    leaf_img = visualize_leaf_features(features)
    leaf_img.save(leaf_save_path)
    
    # Generate and save analysis images (this also returns numeric analysis results)
    analysis_results = analyze_leaf_visual(leaf_img, analysis_save_paths)
    
    # Return web paths (relative to repo root)
    return {
        "leaf": f"{app.config['LEAF_VISUALS_DIR']}/{leaf_file}",
        "fractal": f"{app.config['ANALYSIS_VISUALS_DIR']}/{fractal_file}",
        "skeleton": f"{app.config['ANALYSIS_VISUALS_DIR']}/{skeleton_file}",
        "dft": f"{app.config['ANALYSIS_VISUALS_DIR']}/{dft_file}",
        "hist": f"{app.config['ANALYSIS_VISUALS_DIR']}/{hist_file}",
        "radial": f"{app.config['ANALYSIS_VISUALS_DIR']}/{radial_file}",
        "analysis_results": analysis_results
    }

# ==========================================================
# NEW: TAMPERING TYPE DETECTION
# ==========================================================
def detect_tampering_type(stored_features, new_features):
    """
    Infer likely tampering types based on relative changes between stored and new features.
    Returns a list of human-readable strings (one or more).
    """
    tampering = []
    # compute diffs and relative diffs safely
    diffs = {}
    rel = {}
    for k in stored_features:
        try:
            diffs[k] = new_features.get(k, 0) - stored_features[k]
            denom = max(abs(stored_features[k]), 1e-9)
            rel[k] = diffs[k] / denom
        except Exception:
            diffs[k] = 0
            rel[k] = 0

    # Heuristics (tunable thresholds)
    # Cropping: large negative drop in junctions/endpoints
    if rel.get("junctions", 0) < -0.20 or rel.get("endpoints", 0) < -0.20:
        tampering.append("Cropping likely — large reduction in junctions/endpoints.")

    # Drawing / added strokes: endpoints increased significantly
    if rel.get("endpoints", 0) > 0.20:
        tampering.append("Drawing or added lines likely — endpoints increased.")

    # Blur / smoothing: fractal_dim drops and fft_energy drops
    if rel.get("fractal_dim", 0) < -0.03 or rel.get("fft_energy", 0) < -0.10:
        tampering.append("Blurring/smoothing likely — fractal dimension or FFT energy decreased.")

    # Noise injection: FFT energy increase
    if rel.get("fft_energy", 0) > 0.20 or abs(rel.get("fft_mean", 0)) > 0.15:
        tampering.append("Noise injection likely — FFT energy/mean increased.")

    # Geometric transform: avg_distance or avg_angle changes
    if abs(rel.get("avg_distance", 0)) > 0.05 or abs(diffs.get("avg_angle", 0)) > 3:
        tampering.append("Geometric transform (resize/rotate) possible — avg_distance/avg_angle changed.")

    # If nothing matched but there is any mismatch, return generic
    any_mismatch = any(abs(rel[k]) > 0.02 for k in rel)
    if not tampering and any_mismatch:
        tampering.append("Tampering detected but type uncertain — feature deviations present.")

    if not any_mismatch:
        tampering.append("No significant tampering detected by feature heuristics.")

    return tampering

# ==========================================================
# FLASK ROUTES
# ==========================================================
@app.route('/')
def index():
    # simple index page to upload and register/verify
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

    # --- Run registration algorithm ---
    seed = file_seed_from_path(file_path)
    features = generate_synthetic_features(feature_stats, seed)
    bio_key, _ = generate_bio_key(features)
    hybrid_hash, salt = generate_hybrid_hash(bio_key)
    file_hmac = compute_hmac_file(file_path, hybrid_hash)

    # --- Generate and save visuals ---
    visual_paths = generate_all_visuals(features, filename, suffix="_stored")

    record = {
        "file_name": filename,
        "file_path": file_path,
        "features": features,
        "bio_key": bio_key,
        "salt": salt,
        "hybrid_hash": hybrid_hash,
        "hmac": file_hmac,
        "visuals": visual_paths  # Store paths in the database
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

    # === VERIFICATION LOGIC ===

    # 1. Get stored "golden record" values
    stored_bio_key = record["bio_key"]
    stored_hybrid_hash = record["hybrid_hash"]
    stored_hmac = record["hmac"]
    stored_visuals = record.get("visuals", {})

    # --- Check 1: HMAC Integrity Check ---
    new_hmac_for_integrity = compute_hmac_file(file_path, stored_hybrid_hash)
    is_hmac_match = (new_hmac_for_integrity == stored_hmac)

    # --- Check 2: Biometric Key Check (and generate new visuals) ---
    new_seed = file_seed_from_path(file_path)
    new_features = generate_synthetic_features(feature_stats, new_seed)
    new_bio_key, _ = generate_bio_key(new_features)
    is_biokey_match = (new_bio_key == stored_bio_key)

    # --- Generate visuals for the UPLOADED file ---
    uploaded_visuals = generate_all_visuals(new_features, filename, suffix="_uploaded")

    # --- Tampering detection (if mismatch) ---
    tampering_types = []
    if not (is_hmac_match and is_biokey_match):
        # Combine stored features (the synthetic features saved) and the new ones
        tampering_types = detect_tampering_type(record["features"], new_features)

    # --- Generate JSON report (saved to working/reports) ---
    report = {
        "file_name": filename,
        "stored": {
            "features": record["features"],
            "bio_key": record["bio_key"],
            "salt": record["salt"],
            "hmac": record["hmac"]
        },
        "uploaded": {
            "features": new_features,
            "bio_key": new_bio_key,
            "hmac": new_hmac_for_integrity
        },
        "checks": {
            "is_hmac_match": is_hmac_match,
            "is_biokey_match": is_biokey_match,
            "tampering_types": tampering_types
        },
        "visuals": {
            "stored": stored_visuals,
            "uploaded": uploaded_visuals
        }
    }
    report_filename = os.path.splitext(filename)[0] + "_report.json"
    report_path = os.path.join(app.config['REPORTS_DIR'], report_filename)
    with open(report_path, "w") as rf:
        json.dump(report, rf, indent=2)

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
                               "tampering_types": tampering_types
                           },
                           stored_visuals=stored_visuals,
                           uploaded_visuals=uploaded_visuals,
                           report_file=os.path.basename(report_path))

@app.route('/download_report/<fname>')
def download_report(fname):
    safe = secure_filename(fname)
    full = os.path.join(app.config['REPORTS_DIR'], safe)
    if os.path.exists(full):
        return send_from_directory(app.config['REPORTS_DIR'], safe, as_attachment=True)
    return "Report not found", 404

# ==========================================================
# RUN SERVER
# ==========================================================
if _name_ == '_main_':
    # ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['LEAF_VISUALS_DIR'], exist_ok=True)
    os.makedirs(app.config['ANALYSIS_VISUALS_DIR'], exist_ok=True)
    os.makedirs(app.config['REPORTS_DIR'], exist_ok=True)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
