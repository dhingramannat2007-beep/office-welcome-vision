# Design Document — Office Welcome Vision

Real-time people detection, counting, and face-recognition system with a live
"welcome screen" displayed on a TV (via a casted Chrome tab).

> **Use case:** a camera watches an entrance (office / store). It counts how many
> people pass through, and when it recognises a known person it shows a big
> *"Hello &lt;name&gt;, welcome!"* on a TV. When nobody known is around, the TV shows
> a default *"Welcome to the office"*.

---

## 1. What the app does (in plain words)

1. Grabs video from a camera (laptop webcam **or** a WiFi/RTSP security camera).
2. Uses an AI model (**YOLOv8**) to find every **person** in each frame and draw a
   box around them.
3. Gives each person a **stable tracking ID** (**ByteTrack**) so the same person
   keeps the same ID frame to frame — this is how we count *unique* people.
4. Every few frames, runs **face recognition** (**InsightFace**) to check if any
   visible face matches a **known person** (from the `known_faces/` folder).
5. When a known person is seen, it pushes *"Hello &lt;name&gt;, welcome!"* to a
   **web page** served by a small built-in **Flask** server.
6. That web page is opened in Chrome and **cast to a TV** — it updates live, with
   a fade animation, and falls back to a default idle message.

---

## 2. Architecture overview

```
                          ┌───────────────────────────────────────────────┐
                          │              human_and_face_detection.py        │
                          │                                                 │
   ┌──────────┐  frames   │  ┌───────────────┐   ┌────────────────────┐    │
   │  Camera  │──────────►│  │ FreshestFrame │──►│  YOLOv8 + ByteTrack │    │
   │ webcam / │  (RTSP or │  │ (bg thread:   │   │  detect + track     │    │
   │  WiFi    │   USB)    │  │ newest frame) │   │  each PERSON        │    │
   └──────────┘           │  └───────────────┘   └─────────┬──────────┘    │
        ▲                 │                                 │ person boxes   │
        │ found by MAC    │                       every Nth │ + track IDs    │
        │ (find_camera_ip)│                         frame   ▼                │
        │                 │                       ┌────────────────────┐    │
        │                 │                       │  InsightFace        │    │
        │                 │                       │  recognise faces    │    │
        │                 │                       │  vs known_faces/    │    │
        │                 │                       └─────────┬──────────┘    │
        │                 │                  known person?  │ name           │
        │                 │                                 ▼                │
        │                 │   ┌──────────────┐    ┌────────────────────┐    │
        │                 │   │ OpenCV window│◄───│  state + overlays  │    │
        │                 │   │ (debug view) │    │  people count etc. │    │
        │                 │   └──────────────┘    └─────────┬──────────┘    │
        │                 │                       web_greet()│ "Hello X"     │
        │                 │                                 ▼                │
        │                 │                       ┌────────────────────┐    │
        │                 │                       │  Flask web server   │    │
        │                 │                       │  (bg thread) + SSE  │    │
        │                 │                       └─────────┬──────────┘    │
        └─────────────────┴─────────────────────────────────┼──────────────┘
                                                             │ Server-Sent Events
                                                             ▼
                                                   ┌────────────────────┐
                                                   │ Chrome tab (cast)  │
                                                   │  → TV "Hello X!"   │
                                                   └────────────────────┘
```

### Threads (the app runs three at once)
| Thread | Job |
|--------|-----|
| **Main thread** | The detection loop: read frame → YOLO → face recog → draw → repeat. |
| **FreshestFrame thread** | Continuously pulls frames from the camera, keeps only the latest (prevents lag build-up). |
| **Flask server thread** | Serves the welcome web page and streams live messages to the browser. |

---

## 3. Components in detail

### 3.1 Camera input
- **Laptop webcam:** `cv2.VideoCapture(0)`.
- **WiFi camera:** connected over **RTSP** (`rtsp://user:pass@ip:554/stream1`).
- **MAC-based discovery (`find_camera_ip`)** — the camera's IP changes with DHCP,
  which kept breaking the connection. We instead find the camera by its permanent
  **MAC address**: read the OS **ARP table** (`arp -a`), and if it's not cached,
  ping-sweep the subnet to populate it, then map MAC → current IP.

### 3.2 Person detection + tracking
- **YOLOv8** (`ultralytics`), model `yolov8m.pt` (medium — good accuracy/speed
  balance on CPU). `classes=[0]` restricts detection to **people**.
- **ByteTrack** (`bytetrack.yaml`) assigns a **persistent track ID** per person.
  `persist=True` keeps IDs stable between frames.
- **Counting:** number of distinct track IDs currently visible = "in frame";
  the running set `all_seen_tids` = total unique people seen.

### 3.3 Face recognition
- **InsightFace** `buffalo_sc` model (small/fast; `buffalo_l` available for higher
  accuracy). Runs on **CPU** (`ctx_id=-1`).
- A **face embedding** is a 512-number vector summarising a face. Same person →
  similar vectors.
- Comparison uses **cosine distance** on **L2-normalised** vectors
  (`0` = identical). Two thresholds:
  - `KNOWN_MATCH_THRESHOLD` (0.65): face matches a named person in `known_faces/`.
  - `NEW_HUMAN_THRESHOLD` (0.75): face is the same person we already tracked this run.
- To save CPU, recognition runs only **1 in every `RECOG_EVERY_N_FRAMES` frames**.
- A face is linked to a tracked person via `face_center_in_box` (face centre
  inside a person's box).

### 3.4 Welcome screen (web)
- A tiny **Flask** server serves one **HTML page** (`/`) and a **Server-Sent
  Events** stream (`/events`).
- The detection loop calls `web_greet(name)`, which records the greeting with an
  expiry time. `current_web_message()` computes what to show (idle / one / many).
- The browser holds an open `EventSource` connection; the server pushes the new
  message **only when it changes**, and the page **fades** between messages.
- Opened at `http://localhost:8080` in Chrome and **cast to a TV**.
  (Port 8080, not 5000 — macOS AirPlay Receiver occupies 5000.)

---

## 4. Configuration reference (top of the file)

| Setting | Default | Meaning |
|---------|---------|---------|
| `CONF_THRESH` | `0.35` | Min YOLO confidence to count a detection as a person. |
| `TRACKER` | `bytetrack.yaml` | Tracking algorithm config. |
| `RECOG_EVERY_N_FRAMES` | `5` | Run face recognition once per N frames. |
| `DET_SIZE` | `(640, 640)` | Input size for InsightFace. |
| `NEW_HUMAN_THRESHOLD` | `0.75` | Same-person (this session) distance cutoff. |
| `KNOWN_MATCH_THRESHOLD` | `0.65` | Known-named-person distance cutoff. |
| `KNOWN_DIR` | `known_faces` | Folder of known people: `known_faces/<Name>/*.jpg`. |
| `USE_WEBCAM` | `True` | `True` = laptop webcam, `False` = WiFi camera. |
| `CAMERA_MAC` | `98:25:4a:28:ad:b3` | WiFi camera's permanent MAC address. |
| `RTSP_USER` / `RTSP_PASS` | from `.env` | RTSP credentials — loaded from the gitignored `.env` file, never hard-coded. |
| `RTSP_PORT` / `RTSP_PATH` | `554` / `stream1` | RTSP port and stream path. |
| `LAN_SUBNET` | `192.168.1` | Network prefix for the MAC sweep. |
| `WEB_PORT` | `8080` | Port for the welcome web page. |
| `IDLE_MESSAGE` | `Welcome to the office` | Shown when nobody is being greeted. |
| `GREET_DURATION` | `8` | Seconds a greeting stays before fading to idle. |

---

## 5. Setup & run

### Prerequisites
- Python 3.11+ (tested on the Anaconda `base` interpreter at `/opt/anaconda3`).
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- For the WiFi camera, provide credentials via a local `.env` file (never committed):
  ```bash
  cp .env.example .env     # then edit .env with your camera's RTSP username/password
  ```
  (Not needed if you only use the laptop webcam, i.e. `USE_WEBCAM = True`.)

### Add known people (for greetings)
```
known_faces/
  Mannat/  photo1.jpg  photo2.jpg
  Vishal/  vishal.jpg
```
(JPG/JPEG/PNG only — HEIC is ignored.)

### Run
```bash
python human_and_face_detection.py
```
- Press **Q** in the OpenCV window to quit.
- Open `http://localhost:8080` in Chrome and cast the tab to your TV.

---

## 6. Deployment notes (retail / office at scale)

- **Real cameras:** set `USE_WEBCAM = False` and connect to the store's IP cameras
  over RTSP. MAC-based discovery keeps it stable across DHCP changes.
- **Headless servers:** if running without a display, remove/guard the
  `cv2.imshow` calls and log results instead.
- **Performance:** YOLOv8m + InsightFace run on CPU here. For multiple cameras or
  higher FPS, use a GPU or a smaller model (`yolov8n`, `buffalo_sc`).
- **Cheapest path:** an old laptop / mini-PC on-site reading one RTSP camera; no
  cloud required.

---

## 7. Known limitations / future work

- Counting is per-session (restarting resets counts). A database would persist it.
- Single camera at a time. Multi-camera would need one capture/loop per camera.
- Face recognition accuracy depends on lighting and photo quality in `known_faces/`.
- Possible extensions: zone/dwell-time analytics, crowd alerts, a dashboard,
  persistence to a database.

### Security
- Camera RTSP credentials are loaded from a local **`.env`** file (gitignored), not
  hard-coded — so they never reach GitHub. `.env.example` documents the format.
- For production, consider a secrets manager (e.g. environment injection from the
  deployment platform) rather than a file on disk.

---

## 8. File map

| File | Purpose |
|------|---------|
| `human_and_face_detection.py` | The whole application (camera → detect → recognise → web screen). |
| `known_faces/` | Known people's photos (not committed — privacy). |
| `requirements.txt` | Python dependencies. |
| `.env` | Camera RTSP credentials (gitignored — never committed). |
| `.env.example` | Template showing which credentials to set. |
| `DESIGN.md` | This document. |
| `INTERVIEW.md` | Interview prep Q&A on the tech used. |
| `yolov8m.pt` | YOLO model weights (auto-downloaded on first run; not committed). |

---

*Keep this document updated whenever the architecture or behaviour changes.*
