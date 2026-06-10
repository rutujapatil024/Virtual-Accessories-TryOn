"""
app.py — Flask Web Application for Accessories Virtual Try-On

Serves the web UI, streams webcam feed with real-time accessory
overlays, handles accessory selection/upload, snapshots, and
face shape analysis.
"""

import os
import cv2
import json
import time
import threading
from flask import (
    Flask, render_template, Response, request,
    jsonify, send_file
)
from overlay_engine import OverlayEngine

# ── Flask app setup ───────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCESSORIES_DIR = os.path.join(BASE_DIR, 'static', 'accessories')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

# Ensure uploads directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────
engine = OverlayEngine()
camera = None
camera_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()


def get_camera():
    """Get or initialize the webcam capture object."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            time.sleep(0.5)  # Give camera time to warm up
    return camera


def generate_frames():
    """
    Generator that yields MJPEG frames for streaming.
    Each frame is processed through the overlay engine
    before being encoded and sent.
    """
    global latest_frame

    while True:
        cap = get_camera()
        ret, frame = cap.read()

        if not ret:
            # If frame capture fails, try reopening camera
            with camera_lock:
                if camera is not None:
                    camera.release()
                    camera = None
            time.sleep(0.1)
            continue

        # Process frame through the overlay engine
        processed = engine.process_frame(frame)

        # Store the latest processed frame for snapshots
        with frame_lock:
            latest_frame = processed.copy()

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        # Yield as multipart frame
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Live webcam stream with accessory overlay.
    Uses multipart/x-mixed-replace for MJPEG streaming.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/accessories', methods=['GET'])
def list_accessories():
    """
    Return a JSON list of all available accessories,
    organized by category.
    """
    catalogue = {}
    for category in ['glasses', 'hats', 'earrings']:
        cat_dir = os.path.join(ACCESSORIES_DIR, category)
        if os.path.exists(cat_dir):
            files = [
                f for f in os.listdir(cat_dir)
                if f.lower().endswith('.png')
            ]
            catalogue[category] = sorted(files)
        else:
            catalogue[category] = []

    return jsonify(catalogue)


@app.route('/select', methods=['POST'])
def select_accessory():
    """
    Select an accessory from the built-in catalogue.
    Expects JSON: { "category": "glasses", "filename": "aviator.png" }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    category = data.get('category', '').strip()
    filename = data.get('filename', '').strip()

    if category not in ['glasses', 'hats', 'earrings']:
        return jsonify({"error": "Invalid category"}), 400

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    # Build the full path to the accessory
    filepath = os.path.join(ACCESSORIES_DIR, category, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": f"Accessory not found: {filename}"}), 404

    # Load into the overlay engine
    success = engine.load_accessory(filepath, category)
    if success:
        return jsonify({"status": "ok", "category": category, "filename": filename})
    else:
        return jsonify({"error": "Failed to load accessory"}), 500


@app.route('/clear', methods=['POST'])
def clear_accessory():
    """Remove the current accessory overlay."""
    engine.clear_accessory()
    return jsonify({"status": "ok"})


@app.route('/upload', methods=['POST'])
def upload_accessory():
    """
    Upload a custom PNG accessory.
    Expects a multipart form with:
      - 'file': the PNG image file
      - 'category': one of 'glasses', 'hats', 'earrings'
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    category = request.form.get('category', 'glasses').strip()

    if category not in ['glasses', 'hats', 'earrings']:
        return jsonify({"error": "Invalid category"}), 400

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    if not file.filename.lower().endswith('.png'):
        return jsonify({"error": "Only PNG files are accepted"}), 400

    # Save to uploads directory
    filename = f"upload_{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    file.save(filepath)

    # Attempt background removal with rembg
    try:
        from rembg import remove
        from PIL import Image
        import io

        input_img = Image.open(filepath)
        output_img = remove(input_img)
        output_img.save(filepath)
        print(f"[Upload] Background removed: {filepath}")
    except ImportError:
        print("[Upload] rembg not installed — skipping background removal")
    except Exception as e:
        print(f"[Upload] Background removal failed: {e} — using original")

    # Load into the overlay engine
    success = engine.load_accessory(filepath, category)
    if success:
        return jsonify({
            "status": "ok",
            "category": category,
            "filename": filename,
        })
    else:
        return jsonify({"error": "Failed to load uploaded accessory"}), 500


@app.route('/snapshot', methods=['GET'])
def snapshot():
    """
    Capture the current processed frame and return it
    as a downloadable PNG image.
    """
    with frame_lock:
        if latest_frame is None:
            return jsonify({"error": "No frame available"}), 503

        frame = latest_frame.copy()

    # Encode as PNG for high quality
    ret, buffer = cv2.imencode('.png', frame)
    if not ret:
        return jsonify({"error": "Failed to encode snapshot"}), 500

    # Save temporarily
    snapshot_path = os.path.join(UPLOADS_DIR, f"snapshot_{int(time.time())}.png")
    with open(snapshot_path, 'wb') as f:
        f.write(buffer.tobytes())

    return send_file(
        snapshot_path,
        mimetype='image/png',
        as_attachment=True,
        download_name='tryon_snapshot.png'
    )


@app.route('/face_shape', methods=['POST'])
def face_shape():
    """
    Analyze the current frame to detect face shape
    and return personalized accessory suggestions.
    """
    with frame_lock:
        if latest_frame is None:
            return jsonify({"error": "No frame available"}), 503

        frame = latest_frame.copy()

    result = engine.analyze_face_shape(frame)

    if result is None:
        return jsonify({"error": "No face detected. Please face the camera."}), 400

    return jsonify(result)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Accessories Virtual Try-On")
    print("  Open in browser: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
