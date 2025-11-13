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
UPLOAD_FOLDER = "uploads/"
LEAF_VISUALS_DIR = "static/leaf_visuals"
ANALYSIS_VISUALS_DIR = "static/analysis_visuals"

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

def generate_file_features(file_path, stats):
    """Generate deterministic feature values derived from file content and metadata."""
    with open(file_path, "rb") as f:
        data = f.read()
    
    # Create a stable hash digest for reproducibility
    digest = hashlib.sha256(data).digest()
    digest_nums = [int.from_bytes(digest[i:i+4], "big") for i in range(0, 32, 4)]
    
    # Derive file attributes (entropy, color mean, etc.) as signal-like features
    entropy = -sum((b/255) * np.log2((b/255) + 1e-9) for b in data[:10000]) / 10000
    mean_byte = sum(data[:10000]) / len(data[:10000]) if len(data) > 0 else 128
    
    features = {}
    for i, (feature, vals) in enumerate(stats.items()):
        base_val = vals["mean"]
        scale = vals["sd"] * 0.5
        mod = ((digest_nums[i % len(digest_nums)] % 1000) / 1000) - 0.5
        adjustment = mod * scale + (entropy * 0.01) + ((mean_byte - 128) / 512)
        val = np.clip(base_val + adjustment, vals["min"], vals["max"])
        features[feature] = round(float(val), 5)
    
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
# NEW: LEAF VEIN VISUALIZATION FUNCTION (From your script)
# ==========================================================
def visualize_leaf_features(features, size=(512,512)):
    """
    Draws a realistic, feature-driven synthetic leaf from numeric vectors.
    Deterministic: same features => identical leaf.
    """
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    mid_x = w // 2
    mid_y = h // 2
    
    # Colors affected by FFT stats (frequency = tone intensity)
    base_green = int(120 + (features["fft_mean"] - 5.4) * 80)
    color = (max(0, min(60, base_green)), min(180, base_green + 60), max(0, base_green - 20))
    
    # Draw leaf outline
    outline_pts = []
    for y in range(h // 6, h - h // 6, 4):
        rel = (y - h/2) / (h/2)
        width = (1 - abs(rel)**1.8) * (w * 0.35 + features["avg_distance"] * 30)
        left = mid_x - width
        right = mid_x + width
        outline_pts.append((left, y))
    for y in reversed(range(h // 6, h - h // 6, 4)):
        rel = (y - h/2) / (h/2)
        width = (1 - abs(rel)**1.8) * (w * 0.35 + features["avg_distance"] * 30)
        right = mid_x + width
        outline_pts.append((right, y))
    draw.polygon(outline_pts, fill=color, outline=(20,80,20))
    
    # Midrib (main vein)
    draw.line([(mid_x, h*0.1), (mid_x, h*0.9)], fill=(30,90,30), width=5)
    
    # Primary veins: count and angle from features
    n_veins = int(5 + (features["junctions"] / 800))
    spread = int(10 + features["avg_angle"])
    
    for i in range(n_veins):
        y_start = int(h * (0.15 + (i / n_veins) * 0.7))
        angle = np.radians(spread * ((i % 2) * 2 - 1))
        length = w * 0.25 + (features["avg_distance"] * 100)
        x_end = int(mid_x + np.sign(angle) * length * np.cos(abs(angle)))
        y_end = int(y_start - length * np.sin(abs(angle)))
        draw.line([(mid_x, y_start), (x_end, y_end)], fill=(40,100,40), width=2)
        
        # Sub-veins (based on endpoints)
        sub_count = int(features["endpoints"] / 1500)
        for j in range(sub_count):
            frac = 0.2 + 0.6 * (j / sub_count)
            sx = mid_x + np.sign(angle) * length * frac * np.cos(abs(angle))
            sy = y_start - length * frac * np.sin(abs(angle))
            sub_angle = angle * 0.5 + (j - sub_count/2) * 0.05
            sub_len = length * 0.3
            ex = sx + np.sign(angle) * sub_len * np.cos(sub_angle)
            ey = sy - sub_len * np.sin(sub_angle)
            draw.line([(sx, sy), (ex, ey)], fill=(50,120,50), width=1)
    
    # Texture speckles (related to fractal dimension)
    np.random.seed(int(features["fractal_dim"] * 1000))
    for _ in range(int(200 * (features["fractal_dim"] - 1.2))):
        x = np.random.randint(mid_x - w//4, mid_x + w//4)
        y = np.random.randint(h//6, h - h//6)
        draw.point((x, y), fill=(25,90,25))
    
    return img



# ==========================================================
# NEW: LEAF ANALYSIS FUNCTION (Adapted from your script)
# ==========================================================
def analyze_leaf_visual(leaf_img, paths):
    """
    Runs analysis on a leaf PIL image and saves results to specified paths.
    'paths' is a dict: {'fractal': path, 'skeleton': path, 'dft': path}
    """
    img_rgb = np.array(leaf_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

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
    sizes = [min_side//(2**i) for i in range(1,5) if min_side//(2**i) > 0]
    if not sizes: sizes = [1]
        
    counts = compute_box_counts(binary, sizes)
    if len(counts) > 1:
        logs = np.log(np.array(sizes))
        logn = np.log(counts + 1e-9)
        coeffs = np.polyfit(logs, logn, 1)
        D = -coeffs[0]
    else:
        D = 0 # Not enough data for fit

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
        else:
            axes[i].set_visible(False)
        ax.axis('off')
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

    return {"fractal_dim": D}

# ==========================================================
# NEW: HELPER FOR GENERATING VISUALS
# ==========================================================
def generate_all_visuals(features, filename, suffix=""):
    """
    Generates all visuals for a set of features and saves them.
    Returns a dict of their *web-accessible* paths.
    """
    # Create filenames
    leaf_file = f"{filename}_leaf{suffix}.png"
    fractal_file = f"{filename}_fractal{suffix}.png"
    skeleton_file = f"{filename}_skeleton{suffix}.png"
    dft_file = f"{filename}_dft{suffix}.png"
    
    # Define *full disk paths* for saving
    leaf_save_path = os.path.join(app.config['LEAF_VISUALS_DIR'], leaf_file)
    analysis_save_paths = {
        "fractal": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], fractal_file),
        "skeleton": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], skeleton_file),
        "dft": os.path.join(app.config['ANALYSIS_VISUALS_DIR'], dft_file),
    }
    
    # Generate and save leaf
    leaf_img = visualize_leaf_features(features)
    leaf_img.save(leaf_save_path)
    
    # Generate and save analysis images
    analyze_leaf_visual(leaf_img, analysis_save_paths)
    
    # Return *web paths* (relative to 'static' folder)
    return {
        "leaf": f"{app.config['LEAF_VISUALS_DIR']}/{leaf_file}",
        "fractal": f"{app.config['ANALYSIS_VISUALS_DIR']}/{fractal_file}",
        "skeleton": f"{app.config['ANALYSIS_VISUALS_DIR']}/{skeleton_file}",
        "dft": f"{app.config['ANALYSIS_VISUALS_DIR']}/{dft_file}",
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

        # --- NEW: Generate and save visuals ---
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
    
    return redirect(url_for('index'))


@app.route('/verify', methods=['POST'])
def verify_image():
    if 'file' not in request.files:
        return render_template('result.html', message="No file selected.", status="danger")
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', message="No file selected.", status="danger")

    if file:
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
        stored_visuals = record.get("visuals", {}) # Get stored visual paths
        
        # --- Check 1: HMAC Integrity Check ---
        new_hmac_for_integrity = compute_hmac_file(file_path, stored_hybrid_hash)
        is_hmac_match = (new_hmac_for_integrity == stored_hmac)
        
        # --- Check 2: Biometric Key Check (and generate new visuals) ---
        new_seed = file_seed_from_path(file_path)
        new_features = generate_synthetic_features(feature_stats, new_seed)
        new_bio_key, _ = generate_bio_key(new_features)
        is_biokey_match = (new_bio_key == stored_bio_key)

        # --- NEW: Generate visuals for the UPLOADED file ---
        uploaded_visuals = generate_all_visuals(new_features, filename, suffix="_uploaded")

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
                                   "new_features": new_features
                               },
                               stored_visuals=stored_visuals,
                               uploaded_visuals=uploaded_visuals
                               )
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
