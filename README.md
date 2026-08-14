# Virtual Accessories Try-On

A real-time web application that lets you **virtually try on accessories** (glasses, hats, and earrings) using your webcam — right inside a browser.

---

## What Does This Project Do?

1. Opens your webcam inside a browser.
2. Detects your face using AI (MediaPipe).
3. Places accessories (glasses / hats / earrings) on your face in real time.
4. Lets you choose from a built-in catalogue or upload your own accessory image.
5. Can analyse your **face shape** and suggest the best accessories for you.
6. Lets you take a **snapshot** (download the photo with the accessory on).

---

## Project Structure

```
Virtual Try-on/
└── accessories-tryon/
    ├── app.py                  ← Main server (Flask)
    ├── overlay_engine.py       ← Face detection + accessory overlay (AI logic)
    ├── face_landmarker.task    ← Pre-trained AI model file (MediaPipe)
    ├── requirements.txt        ← Python libraries needed
    ├── static/
    │   ├── accessories/
    │   │   ├── glasses/        ← Built-in glasses PNG images
    │   │   ├── hats/           ← Built-in hat PNG images
    │   │   └── earrings/       ← Built-in earring PNG images
    │   ├── css/                ← Styling for the web page
    │   └── js/                 ← Frontend logic (camera feed, buttons)
    ├── templates/
    │   └── index.html          ← The web page shown in the browser
    └── uploads/                ← Stores user-uploaded accessories & snapshots
```

---

## Key Files Explained

### `app.py` — The Web Server
- Built with **Flask** (a Python web framework).
- Starts the web server at `http://localhost:5000`.
- Continuously reads frames from the webcam and sends them to the browser as a live video stream (MJPEG).
- Handles all API routes:
  | Route | What it does |
  |---|---|
  | `/` | Serves the main web page |
  | `/video_feed` | Streams the live webcam video |
  | `/accessories` | Returns the list of available accessories |
  | `/select` | Loads a selected accessory onto the stream |
  | `/clear` | Removes the current accessory |
  | `/upload` | Accepts a user-uploaded PNG accessory |
  | `/snapshot` | Downloads the current frame as a PNG photo |
  | `/face_shape` | Analyses face shape and gives suggestions |

---

### `overlay_engine.py` — The AI Brain
This is the most important file. It does all the computer vision work.

- Uses **MediaPipe FaceLandmarker** to detect 478 face landmark points (eyes, ears, forehead, jaw, etc.).
- Once landmarks are found, it:
  - **Glasses** → placed between the two eye points, scaled to eye distance, rotated to match head tilt.
  - **Hat** → placed above the forehead point, scaled to face width.
  - **Earrings** → placed at both ear points, mirrored left-right automatically.
- Uses **alpha blending** so the transparent background of PNG accessories blends naturally into the video.
- The `analyze_face_shape()` method measures face proportions (height, width, jawline, forehead) and classifies face shape as: **Oval, Round, Square, Heart, Oblong, or Diamond** — and gives accessory suggestions.

---

### `face_landmarker.task` — The AI Model
- A pre-trained model file provided by **Google MediaPipe**.
- It detects 478 precise points on a face from any image frame.
- This file is downloaded once and used locally — **no internet needed** to run.

---

### `requirements.txt` — Dependencies
Lists the Python packages needed to run the project:

| Package | Purpose |
|---|---|
| `flask` | Web server |
| `opencv-python` | Webcam capture & image processing |
| `mediapipe` | Face landmark detection |
| `numpy` | Array/pixel math |
| `Pillow` | Image handling |
| `rembg` | Auto background removal for uploaded images |
| `onnxruntime` | Required by rembg to run the AI model |

---

## 🔄 How It Works — Full Flow

```
User opens browser
       ↓
Flask serves index.html
       ↓
Browser displays webcam stream (/video_feed)
       ↓
Flask reads webcam frame (OpenCV)
       ↓
Sends frame to OverlayEngine.process_frame()
       ↓
MediaPipe detects face landmarks
       ↓
Accessory PNG is scaled, rotated, alpha-blended onto frame
       ↓
Processed frame is streamed back to browser
       ↓
User selects accessory → /select API called
User uploads image   → /upload API called (background removed by rembg)
User clicks Snapshot → /snapshot saves and downloads the photo
User clicks Analyse  → /face_shape returns face shape + suggestions
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
cd accessories-tryon
python app.py

# 3. Open in browser
http://localhost:5000
```

---

## 🧠 Technologies Used

- **Python + Flask** — Backend web server
- **OpenCV** — Webcam access and image processing
- **MediaPipe** — Real-time face landmark detection (Google AI)
- **NumPy** — Pixel-level blending math
- **rembg** — AI-based background removal for uploaded PNGs
- **HTML / CSS / JavaScript** — Frontend UI
