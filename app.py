import os
import json
import hashlib
import hmac
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import cv2
from skimage import filters, morphology
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
for d in ["working", UPLOAD_FOLDER, LEAF_VISUALS_DIR, ANALYSIS_VISUALS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ==========================================================
# FEATURE STATS
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
# HYBRID BIOHASH FUNCTIONS
# ==========================================================
def file_seed_from_path(file_path):
    """Generate deterministic seed based on path + file content hash."""
    h = hashlib.sha256()
    h.update(file_path.encode('utf-8'))
    with open(file_path, "rb") as f:
        h.update(f.read(65536))
    digest = h.digest()
    return int.from_bytes(digest[:8], 'big', signed=False)

def generate_synthetic_features(stats, seed):
    """Generate stable features per file seed."""
    rng = np.random.default_rng(seed)
    features = {}
    for feature, vals in stats.items():
        val = rng.normal(vals["mean"], vals["sd"])
        val = float(np.clip(val, vals["min"], vals["max"]))
        features[feature] = round(val, 5)
    return features

def generate_bio_key(features):
    """Make a consistent hash from feature set."""
    feature_string = ",".join([f"{k}:{features[k]:.5f}" for k in sorted(features.keys())])
    return hashlib.sha256(feature_string.encode()).hexdigest(), feature_string

def generate_hybrid_hash(bio_key, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    hybrid_input = (salt + bio_key).encode('utf-8')
    return hashlib.sha256(hybrid_input).hexdigest(), salt

def compute_hmac_file(file_path, hybrid_hash):
    """Compute file-level HMAC for tamper detection."""
    h = hmac.new(hybrid_hash.encode('utf-8'), digestmod=hashlib.sha256)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ==========================================================
# DATABASE FUNCTIONS
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
# VISUAL GENERATORS
# ==========================================================
def visualize_leaf_features(features, size=(512,512)):
    """Generate deterministic leaf-like visual as a barcode-like ID."""
    rng = np.random.default_rng(int(sum(features.values()) * 100))
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    for i in range(200):
        x = int(rng.integers(0, w))
        y1, y2 = int(rng.integers(0, h//2)), int(rng.integers(h//2, h))
        color = (int(30 + 200 * rng.random()), int(100 + 100 * rng.random()), int(30 + 150 * rng.random()))
        draw.line([(x, y1), (x, y2)], fill=color, width=2)
    return img

# ==========================================================
# ANALYSIS & DIFFERENCE CHECK
# ==========================================================
def compare_features(old, new):
    """Return which features changed and by how much."""
    diffs = {}
    for k in old.keys():
        diff = round(new[k] - old[k], 5)
        diffs[k] = diff
    return diffs

# ==========================================================
# FLASK ROUTES
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

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    assignments = load_database()
    if any(r["file_name"] == filename for r in assignments):
        return render_template('result.html', message=f"'{filename}' already registered.", status="warning")

    seed = file_seed_from_path(file_path)
    features = generate_synthetic_features(feature_stats, seed)
    bio_key, _ = generate_bio_key(features)
    hybrid_hash, salt = generate_hybrid_hash(bio_key)
    file_hmac = compute_hmac_file(file_path, hybrid_hash)

    leaf_path = os.path.join(LEAF_VISUALS_DIR, f"{filename}_stored.png")
    visualize_leaf_features(features).save(leaf_path)

    record = {
        "file_name": filename,
        "file_path": file_path,
        "features": features,
        "bio_key": bio_key,
        "salt": salt,
        "hybrid_hash": hybrid_hash,
        "hmac": file_hmac,
        "leaf_visual": leaf_path
    }

    assignments.append(record)
    save_database(assignments)
    return render_template('result.html', message=f"'{filename}' registered successfully!", status="success", record=record)

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

    db = load_database()
    record = next((r for r in db if r["file_name"] == filename), None)
    if not record:
        return render_template('result.html', message=f"No record found for '{filename}'.", status="danger")

    new_seed = file_seed_from_path(file_path)
    new_features = generate_synthetic_features(feature_stats, new_seed)
    new_bio_key, _ = generate_bio_key(new_features)
    new_hmac = compute_hmac_file(file_path, record["hybrid_hash"])

    diffs = compare_features(record["features"], new_features)
    leaf_path_uploaded = os.path.join(LEAF_VISUALS_DIR, f"{filename}_uploaded.png")
    visualize_leaf_features(new_features).save(leaf_path_uploaded)

    is_valid = (new_bio_key == record["bio_key"] and new_hmac == record["hmac"])
    status = "success" if is_valid else "danger"
    msg = "VERIFIED" if is_valid else "TAMPERED! Changes detected."

    return render_template('result.html',
                           message=msg,
                           status=status,
                           record=record,
                           diffs=diffs,
                           new_features=new_features,
                           uploaded_leaf=leaf_path_uploaded)

# ==========================================================
# RUN
# ==========================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
