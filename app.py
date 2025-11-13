# ==========================================================
# 1. Imports
# ==========================================================
import os, json, hashlib, hmac, random, numpy as np
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from skimage import filters, morphology
from scipy import ndimage as ndi

# ==========================================================
# 2. Config / Paths
# ==========================================================
DATABASE_FILE = "working/hybrid_biohash_records.json"
UPLOAD_FOLDER = "uploads/"
LEAF_VISUALS_DIR = "static/leaf_visuals"
ANALYSIS_VISUALS_DIR = "static/analysis_visuals"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LEAF_VISUALS_DIR'] = LEAF_VISUALS_DIR
app.config['ANALYSIS_VISUALS_DIR'] = ANALYSIS_VISUALS_DIR

Path("working").mkdir(parents=True, exist_ok=True)
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['LEAF_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)
Path(app.config['ANALYSIS_VISUALS_DIR']).mkdir(parents=True, exist_ok=True)

# ==========================================================
# 3. Feature stats
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
# 4. Helper functions
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
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    features = {}
    for key, vals in stats.items():
        val = np.random.normal(vals["mean"], vals["sd"] * 0.1)
        val = np.clip(val, vals["min"], vals["max"])
        features[key] = round(float(val), 5)
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

def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_database(data):
    with open(DATABASE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def visualize_leaf_features(features, size=(512,512)):
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    mid_x = w // 2
    mid_y = h // 2

    base_green = int(120 + (features["fft_mean"] - 5.4) * 80)
    color = (max(0, min(60, base_green)), min(180, base_green + 60), max(0, base_green - 20))

    outline_pts = []
    for y in range(h // 6, h - h // 6, 4):
        rel = (y - h/2) / (h/2)
        width = (1 - abs(rel)**1.8) * (w * 0.35 + features["avg_distance"] * 30)
        outline_pts.append((mid_x - width, y))
    for y in reversed(range(h // 6, h - h // 6, 4)):
        rel = (y - h/2) / (h/2)
        width = (1 - abs(rel)**1.8) * (w * 0.35 + features["avg_distance"] * 30)
        outline_pts.append((mid_x + width, y))
    draw.polygon(outline_pts, fill=color, outline=(20,80,20))

    draw.line([(mid_x, h*0.1), (mid_x, h*0.9)], fill=(30,90,30), width=5)

    n_veins = int(5 + (features["junctions"] / 800))
    spread = int(10 + features["avg_angle"])
    for i in range(n_veins):
        y_start = int(h * (0.15 + (i / n_veins) * 0.7))
        angle = np.radians(spread * ((i % 2) * 2 - 1))
        length = w * 0.25 + (features["avg_distance"] * 100)
        x_end = int(mid_x + np.sign(angle) * length * np.cos(abs(angle)))
        y_end = int(y_start - length * np.sin(abs(angle)))
        draw.line([(mid_x, y_start), (x_end, y_end)], fill=(40,100,40), width=2)

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

    np.random.seed(int(features["fractal_dim"] * 1000))
    for _ in range(int(200 * (features["fractal_dim"] - 1.2))):
        x = np.random.randint(mid_x - w//4, mid_x + w//4)
        y = np.random.randint(h//6, h - h//6)
        draw.point((x, y), fill=(25,90,25))

    return img

def analyze_leaf_visual(leaf_img, paths):
    img_rgb = np.array(leaf_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh
    binary = morphology.remove_small_objects(binary, 60)
    binary = morphology.remove_small_holes(binary, 60)

    # Fractal box counting
    min_side = min(binary.shape)
    sizes = [min_side//(2**i) for i in range(1,5) if min_side//(2**i) > 0]
    if not sizes: sizes = [1]

    counts = []
    for k in sizes:
        H, W = (binary.shape[0]//k)*k, (binary.shape[1]//k)*k
        Z = binary[:H,:W].astype(int)
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0,H,k), axis=0), np.arange(0,W,k), axis=1)
        counts.append(float(np.count_nonzero(S)))

    if len(counts) > 1:
        logs = np.log(np.array(sizes))
        logn = np.log(np.array(counts) + 1e-9)
        coeffs = np.polyfit(logs, logn, 1)
        D = -coeffs[0]
    else:
        D = 0

    return {"fractal_dim": D}

def generate_all_visuals(features, filename, suffix=""):
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

    leaf_img = visualize_leaf_features(features)
    leaf_img.save(leaf_save_path)
    analyze_leaf_visual(leaf_img, analysis_save_paths)

    return {
        "leaf": f"{app.config['LEAF_VISUALS_DIR']}/{leaf_file}",
        "fractal": f"{app.config['ANALYSIS_VISUALS_DIR']}/{fractal_file}",
        "skeleton": f"{app.config['ANALYSIS_VISUALS_DIR']}/{skeleton_file}",
        "dft": f"{app.config['ANALYSIS_VISUALS_DIR']}/{dft_file}",
    }

# ==========================================================
# 5. Flask routes
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('result.html', message="No file selected.", status="danger")

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    db = load_database()
    if any(r["file_name"] == filename for r in db):
        return render_template('result.html', message=f"File '{filename}' already registered.", status="warning")

    seed = file_seed_from_path(file_path)
    features = generate_synthetic_features(feature_stats, seed)
    bio_key, _ = generate_bio_key(features)
    hybrid_hash, salt = generate_hybrid_hash(bio_key)
    file_hmac = compute_hmac_file(file_path, hybrid_hash)
    visuals = generate_all_visuals(features, filename, suffix="_stored")

    record = {
        "file_name": filename,
        "file_path": file_path,
        "features": features,
        "bio_key": bio_key,
        "salt": salt,
        "hybrid_hash": hybrid_hash,
        "hmac": file_hmac,
        "visuals": visuals
    }
    db.append(record)
    save_database(db)

    return render_template('result.html', message=f"File '{filename}' registered successfully!", status="success", record=record, stored_visuals=visuals)

@app.route('/verify', methods=['POST'])
def verify_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('result.html', message="No file selected.", status="danger")

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    db = load_database()
    record = next((r for r in db if r["file_name"] == filename), None)
    if record is None:
        return render_template('result.html', message=f"File '{filename}' not found.", status="danger")

    # HMAC and bio_key checks
    new_hmac = compute_hmac_file(file_path, record["hybrid_hash"])
    new_seed = file_seed_from_path(file_path)
    new_features = generate_synthetic_features(feature_stats, new_seed)
    new_bio_key, _ = generate_bio_key(new_features)
    visuals_uploaded = generate_all_visuals(new_features, filename, suffix="_uploaded")

    is_hmac_match = (new_hmac == record["hmac"])
    is_biokey_match = (new_bio_key == record["bio_key"])

    status = "success" if is_hmac_match and is_biokey_match else "danger"
    message = "VERIFIED: HMAC and Biometric Key match." if status=="success" else "TAMPERED! File data does not match the registered record."

    return render_template('result.html', message=message, status=status, record=record, check_results={
        "new_hmac": new_hmac,
        "is_hmac_match": is_hmac_match,
        "new_bio_key": new_bio_key,
        "is_biokey_match": is_biokey_match,
        "new_features": new_features
    }, stored_visuals=record.get("visuals", {}), uploaded_visuals=visuals_uploaded)

# ==========================================================
# 6. Run server
# ==========================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
