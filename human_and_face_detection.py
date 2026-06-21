"""
human_and_face_detection.py
===========================

Real-time people detection + counting + (optional) face recognition, with a
live "welcome screen" that is served as a web page to a casted Chrome tab/TV.

High-level pipeline (see DESIGN.md for the full architecture):

    Camera (webcam or WiFi/RTSP)
        -> FreshestFrame   (background thread: always hand us the LATEST frame)
        -> YOLOv8 + ByteTrack   (detect each person, give them a stable track ID)
        -> InsightFace          (every Nth frame: recognise known faces)
        -> overlays + people count   (shown in an OpenCV debug window)
        -> Flask web server          (pushes "Hello <name>" to the browser via SSE)

Run it:   python human_and_face_detection.py
Quit it:  press Q in the OpenCV window.
"""

# ---------- IMPORTS ----------
import os            # filesystem checks (does the known_faces folder exist?)
import glob          # find image files inside each known-person folder
import time          # timestamps, greeting expiry, small sleeps
import json          # encode the message we push to the browser (SSE payload)
import logging       # used to silence Flask's noisy per-request logs
import threading     # background threads: camera grabber + web server
import subprocess    # run `arp`/`ping` to find the camera's IP from its MAC
import re            # parse IP/MAC out of the `arp -a` output
import numpy as np   # vector maths for face embeddings (normalise, dot product)
import cv2           # OpenCV: camera capture, drawing, displaying windows
from ultralytics import YOLO            # YOLOv8 person detector + tracker
from insightface.app import FaceAnalysis  # face detection + embedding model
from flask import Flask, Response       # tiny web server for the welcome screen


# ---------- SECRETS (loaded from a local, gitignored .env file) ----------
def _load_dotenv(path=None):
    """
    Load KEY=VALUE lines from a local `.env` file into environment variables.
    This keeps secrets (the camera's RTSP username/password) OUT of the source
    code and out of git. The `.env` file is listed in `.gitignore`, so it never
    reaches GitHub. See `.env.example` for the format.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            # setdefault: a real environment variable always wins over the file
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# ---------- CONFIG ----------
# All the knobs you might want to tune live here so you never hunt through code.

CONF_THRESH = 0.35            # min confidence for YOLO to count something as a person
TRACKER = "bytetrack.yaml"    # the tracking algorithm config (ByteTrack)
RECOG_EVERY_N_FRAMES = 5      # run the heavy face recognition only 1 in every 5 frames
DET_SIZE = (640, 640)         # input size InsightFace resizes faces to before analysing

# Face thresholds (cosine distance = 1 - cosine_similarity; smaller = more similar)
NEW_HUMAN_THRESHOLD = 0.75    # if distance <= this, it's the SAME person we already saw
KNOWN_MATCH_THRESHOLD = 0.65  # if distance <= this, it matches a known named person

KNOWN_DIR = "known_faces"     # folder of known people: known_faces/<Name>/*.jpg

# Video source: True = laptop webcam, False = WiFi camera (found by MAC below).
USE_WEBCAM = True

# WiFi camera — located automatically by its (fixed) MAC address, so a changing
# DHCP IP never breaks the connection again.
CAMERA_MAC = "98:25:4a:28:ad:b3"          # TP-Link (Tapo) camera's permanent hardware address
RTSP_USER = os.environ.get("RTSP_USER", "")  # set in your local .env (never committed)
RTSP_PASS = os.environ.get("RTSP_PASS", "")  # set in your local .env (never committed)
RTSP_PORT = 554                    # standard RTSP port
RTSP_PATH = "stream1"              # Tapo high-res stream path ("stream2" = low-res)
LAN_SUBNET = "192.168.1"           # your network prefix (first three numbers of your IP)

# Welcome TV screen (served as a web page to a casted Chrome tab)
WEB_PORT = 8080                    # open http://localhost:8080 in Chrome, then cast that tab
                                   # (avoid 5000 — macOS AirPlay Receiver uses it)
IDLE_MESSAGE = "Welcome to the office"   # shown when nobody is being greeted
GREET_DURATION = 8                 # seconds a "Hello <name>" stays before fading to idle


# ---------- CAMERA DISCOVERY HELPERS ----------
# These let us find the WiFi camera by its permanent MAC address instead of a
# hard-coded IP (which the router keeps reassigning).

def _normalize_mac(mac: str) -> str:
    """
    macOS `arp` prints MACs WITHOUT leading zeros (e.g. 'a0:91:a2:55:d0:f').
    Convert every octet to a standard two-digit hex form so two MACs that mean
    the same thing compare as equal.
    """
    try:
        return ":".join(f"{int(p, 16):02x}" for p in mac.split(":"))
    except ValueError:
        return mac.lower()


def find_camera_ip(mac: str, subnet: str = LAN_SUBNET, do_sweep: bool = True):
    """
    Return the current IP of the device with this MAC address, or None.

    The camera's IP can change (DHCP), but its MAC never does — so we look it
    up fresh every run instead of hard-coding an IP that keeps breaking.
    """
    target = _normalize_mac(mac)

    # Inner helper: read the OS ARP table (IP <-> MAC map) and find our MAC.
    def lookup():
        out = subprocess.run(["arp", "-a"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            # Each line looks like: "? (192.168.1.50) at 98:25:4a:28:ad:b3 on en0 ..."
            m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\) at ([0-9a-fA-F:]+)", line)
            if m and _normalize_mac(m.group(2)) == target:
                return m.group(1)   # group(1) = the IP address
        return None

    # First try: the MAC may already be in the ARP cache.
    ip = lookup()
    if ip:
        return ip

    # Not cached yet — ping every host on the subnet to force them to announce
    # themselves (which populates the ARP table), then look again.
    if do_sweep:
        procs = [subprocess.Popen(["ping", "-c", "1", "-W", "1", f"{subnet}.{i}"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 for i in range(1, 255)]
        for p in procs:
            p.wait()
        ip = lookup()
    return ip


def build_rtsp_url(ip: str) -> str:
    """Assemble the full RTSP stream URL from the configured credentials + IP."""
    return f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:{RTSP_PORT}/{RTSP_PATH}"


# ---------- FACE-MATH HELPERS ----------
# A "face embedding" is a 512-number vector describing a face. Comparing two
# faces = comparing their vectors. These helpers do that comparison.

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    Scale a vector so its length becomes exactly 1 (unit vector).
    With unit vectors, the dot product equals the cosine similarity, which
    makes face comparison fast and stable. The tiny 1e-12 avoids divide-by-zero.
    """
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance between two normalised face vectors: 0 = identical, ~2 = opposite.
    (Since both are unit vectors, dot product = cosine similarity, and
    distance = 1 - similarity.)
    """
    return float(1.0 - np.dot(a, b))  # assumes both inputs are already normalized


def face_center_in_box(face_bbox, person_bbox) -> bool:
    """
    Decide which tracked person a detected face belongs to: True if the face
    box's CENTRE point falls inside the person's bounding box.
    """
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    cx = (fx1 + fx2) / 2.0   # face centre X
    cy = (fy1 + fy2) / 2.0   # face centre Y
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)


def load_known_faces(face_app: FaceAnalysis, known_dir: str):
    """
    Build the "known people" database from the known_faces/ folder.

    Layout expected:   known_faces/Mannat/photo1.jpg, known_faces/Vishal/a.png ...
    For each person we average the embeddings of all their photos into one
    representative vector (more photos = more robust recognition).

    Returns: dict { name -> averaged_normalized_embedding }.
    """
    if not os.path.isdir(known_dir):
        return {}

    known = {}
    # Each sub-folder name is a person's name.
    person_dirs = [d for d in os.listdir(
        known_dir) if os.path.isdir(os.path.join(known_dir, d))]
    if not person_dirs:
        return {}

    for person in sorted(person_dirs):
        # Collect every jpg/jpeg/png inside this person's folder.
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs.extend(glob.glob(os.path.join(known_dir, person, ext)))

        # Turn each photo into one face embedding.
        embs = []
        for p in sorted(imgs):
            img = cv2.imread(p)
            if img is None:        # unreadable file (e.g. HEIC) — skip it
                continue
            faces = face_app.get(img)
            if not faces:          # no face found in this photo — skip
                continue
            best = max(faces, key=lambda f: f.det_score)  # most confident face
            embs.append(l2_normalize(best.embedding.astype(np.float32)))

        # Average all that person's embeddings into a single reference vector.
        if embs:
            avg = l2_normalize(np.mean(np.stack(embs), axis=0))
            known[person] = avg
            print(f"[OK] Known identity loaded: {person} ({len(embs)} images)")
        else:
            print(f"[WARN] No usable faces for: {person}")

    return known


def match_known_name(face_emb_norm: np.ndarray, known_db: dict):
    """
    Compare one face embedding against every known person.
    Returns (name, distance) for the closest match if it's close enough,
    otherwise (None, distance).
    """
    if not known_db:
        return None, 999.0

    # Find the known person with the smallest distance to this face.
    best_name = None
    best_dist = 999.0
    for name, kemb in known_db.items():
        d = cosine_distance(face_emb_norm, kemb)
        if d < best_dist:
            best_dist = d
            best_name = name

    # Only accept the match if it's within the threshold (else it's a stranger).
    if best_dist <= KNOWN_MATCH_THRESHOLD:
        return best_name, best_dist
    return None, best_dist


# ---------- CAMERA FRAME GRABBER ----------

class FreshestFrame(threading.Thread):
    """
    Continuously reads frames from a capture in the background and keeps ONLY
    the latest one.

    Why: our processing (YOLO + face recognition) is slower than the camera's
    frame rate. Without this, frames pile up in a buffer and the displayed feed
    drifts further and further behind real time. By always discarding old frames
    and handing back the newest, the feed stays "live".
    """

    def __init__(self, capture: cv2.VideoCapture):
        super().__init__(daemon=True)      # daemon = dies automatically with the program
        self.capture = capture
        self.lock = threading.Lock()       # protects self.frame across threads
        self.frame = None                  # the most recent frame
        self.running = True
        self.start()                       # begin grabbing immediately

    def run(self):
        # Background loop: grab frames as fast as the camera produces them.
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                continue
            with self.lock:
                self.frame = frame         # overwrite — we only keep the latest

    def read(self):
        # Hand the main loop a private copy of the newest frame (or None yet).
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        # Stop the thread cleanly and release the camera.
        self.running = False
        self.join()
        self.capture.release()


# ---------- WELCOME WEB SCREEN ----------
# A tiny Flask server serves one HTML page. The detection loop pushes greetings
# into `_active_greets`; the page live-updates via Server-Sent Events (SSE) — so
# the TV updates instantly with no page reloads or flicker.

_greet_lock = threading.Lock()   # protects _active_greets (touched by 2 threads)
_active_greets = {}              # name -> expiry timestamp (when its greeting ends)


def web_greet(name: str):
    """Show 'Hello <name>, welcome!' on the TV for GREET_DURATION seconds."""
    with _greet_lock:
        _active_greets[name] = time.time() + GREET_DURATION


def current_web_message() -> str:
    """
    Work out what the TV should show right now:
      - nobody recently greeted  -> the idle message
      - one person               -> "Hello <name>, welcome!"
      - several at once          -> "Hello <A> & <B>, welcome!"
    """
    now = time.time()
    with _greet_lock:
        # Keep only greetings that haven't expired yet.
        names = [n for n, exp in _active_greets.items() if exp > now]
    if not names:
        return IDLE_MESSAGE
    if len(names) == 1:
        return f"Hello {names[0]}, welcome!"
    return "Hello " + " & ".join(names) + ", welcome!"


# The single HTML page shown on the TV: white background, big centred black text,
# with a JavaScript EventSource that listens for live message updates and fades
# between them.
WELCOME_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Welcome</title>
<style>
  html, body { margin: 0; height: 100%; background: #ffffff; }
  #wrap { height: 100vh; display: flex; align-items: center; justify-content: center; }
  #msg {
    color: #000000;
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-weight: 700;
    font-size: 8vw;             /* scales with screen width = big on a TV */
    line-height: 1.1;
    text-align: center;
    padding: 0 5vw;
    opacity: 1;
    transition: opacity 0.6s ease;   /* the fade animation */
  }
  #msg.fade { opacity: 0; }
</style>
</head>
<body>
  <div id="wrap"><div id="msg">Welcome to the office</div></div>
<script>
  const el = document.getElementById('msg');
  // Fade out, swap the text, fade back in.
  function setMessage(text) {
    if (text === el.textContent) return;   // no change, do nothing
    el.classList.add('fade');
    setTimeout(() => { el.textContent = text; el.classList.remove('fade'); }, 600);
  }
  // Open a live stream to the server; each event carries the latest message.
  const es = new EventSource('/events');
  es.onmessage = (e) => {
    try { setMessage(JSON.parse(e.data).message); } catch (err) {}
  };
</script>
</body>
</html>"""


_web_app = Flask(__name__)   # the Flask application object


@_web_app.route("/")
def _index():
    """Serve the welcome HTML page itself."""
    return WELCOME_PAGE


@_web_app.route("/events")
def _events():
    """
    Server-Sent Events endpoint. Keeps the connection open and streams the
    current message to the browser whenever it CHANGES (not on a fixed timer),
    so the TV updates the instant someone is recognised.
    """
    def stream():
        last = None
        while True:
            msg = current_web_message()
            if msg != last:                 # only send when the text actually changes
                last = msg
                yield f"data: {json.dumps({'message': msg})}\n\n"
            time.sleep(0.3)                 # check ~3x/second
    return Response(stream(), mimetype="text/event-stream")


def start_web_server():
    """Launch the Flask server in a background thread so it runs alongside detection."""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # quiet the per-request logs
    threading.Thread(
        target=lambda: _web_app.run(
            host="0.0.0.0", port=WEB_PORT,   # 0.0.0.0 = reachable from other devices too
            threaded=True, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    print(f"[INFO] Welcome screen live at http://localhost:{WEB_PORT}  "
          f"(open in Chrome and cast that tab to your TV)")


# ---------- MAIN ----------

def main():
    # 1) Start the welcome web server first so the TV page is ready immediately.
    start_web_server()

    # 2) Load the YOLOv8 model used to detect + track people.
    print("[INFO] Loading YOLO model...")
    yolo = YOLO("yolov8m.pt")
    print("[INFO] YOLO ready.")

    # 3) Load the face-recognition model — but ONLY if there are known faces to
    #    match against (it's heavy, so skip it when not needed).
    face_app = None
    known_db = {}
    has_known_faces = (
        os.path.isdir(KNOWN_DIR) and
        any(os.path.isdir(os.path.join(KNOWN_DIR, d)) for d in os.listdir(KNOWN_DIR))
    )
    if has_known_faces:
        print("[INFO] Loading InsightFace model (known_faces found)...")
        # face_app = FaceAnalysis(name="buffalo_l")  # more accurate, slower to load
        face_app = FaceAnalysis(name="buffalo_sc")     # smaller/faster model
        face_app.prepare(ctx_id=-1, det_size=DET_SIZE)  # ctx_id=-1 means run on CPU
        known_db = load_known_faces(face_app, KNOWN_DIR)
        print("[INFO] InsightFace ready.")
    else:
        print("[INFO] No known_faces folder — skipping face recognition. Will count by tracker IDs only.")

    # 4) Open the video source — laptop webcam or the WiFi camera.
    if USE_WEBCAM:
        # --- Laptop webcam ---
        print("[INFO] Using laptop webcam...")
        cap = cv2.VideoCapture(0)            # 0 = default built-in camera
        if not cap.isOpened():
            raise RuntimeError("Could not open the laptop webcam.")
    else:
        # --- WiFi camera (found automatically by MAC, so a changing IP never breaks it) ---
        # Credentials come from the local .env file (kept out of the code/git).
        if not RTSP_USER or not RTSP_PASS:
            raise RuntimeError(
                "Missing camera credentials. Copy .env.example to .env and fill in "
                "RTSP_USER and RTSP_PASS (or set USE_WEBCAM = True to use the laptop camera).")
        print(f"[INFO] Looking up camera by MAC {CAMERA_MAC} ...")
        camera_ip = find_camera_ip(CAMERA_MAC)
        if not camera_ip:
            raise RuntimeError(
                f"Could not find a device with MAC {CAMERA_MAC} on the network. "
                "Is the camera powered on and connected to WiFi?")
        rtsp_url = build_rtsp_url(camera_ip)
        print(f"[INFO] Camera found at {camera_ip}. Connecting...")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(
                f"Found the camera at {camera_ip} but could not open its RTSP stream. "
                f"Make sure RTSP is ENABLED in the camera app (needs port {RTSP_PORT} open).")

    # 5) Wrap the capture so we always process the freshest frame (see class above).
    print("[INFO] Camera connected. Starting live feed...")
    cap = FreshestFrame(cap)

    # ----- State we keep across frames -----
    # Each known/seen human: {"human_id": int, "embedding": vector, "name": str|None}
    humans = []
    next_human_id = 1            # counter for assigning new human IDs
    tracker_to_human = {}        # maps YOLO's track id -> our human_id
    start_time = time.time()     # for the on-screen elapsed timer
    frame_idx = 0                # frame counter (drives "every Nth frame" logic)
    all_seen_tids = set()        # every unique track id ever seen = total unique people

    # ----- Main loop: runs once per frame until you press Q -----
    while True:
        # Get the newest frame; if none ready yet, wait briefly and retry.
        frame = cap.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # --- (a) Detect + track every person in the frame ---
        results = yolo.track(
            source=frame,
            persist=True,        # remember tracks between frames (stable IDs)
            tracker=TRACKER,     # ByteTrack
            conf=CONF_THRESH,    # confidence threshold
            classes=[0],         # class 0 = "person" only (ignore other objects)
            verbose=False,
            imgsz=640            # resize frame to 640px for inference
        )
        r = results[0]
        annotated = r.plot()     # frame with YOLO's boxes/IDs drawn on it

        # --- (b) Collect this frame's people: track id -> bounding box ---
        person_boxes = {}
        current_tids = []
        if r.boxes is not None and r.boxes.id is not None:
            tids = r.boxes.id.cpu().numpy().astype(int).tolist()       # track IDs
            boxes = r.boxes.xyxy.cpu().numpy().tolist()                # box corners
            for tid, box in zip(tids, boxes):
                person_boxes[int(tid)] = tuple(map(float, box))
            current_tids = sorted(set(tids))

        # --- (c) Face recognition (only every Nth frame, and only if enabled) ---
        if face_app and person_boxes and (frame_idx % RECOG_EVERY_N_FRAMES == 0):
            faces = face_app.get(frame)      # detect + embed every face in the frame
            for face in faces:
                fb = face.bbox.astype(float)
                emb = l2_normalize(face.embedding.astype(np.float32))  # this face's vector

                # Which tracked person does this face sit inside?
                assigned_tid = None
                for tid, pb in person_boxes.items():
                    if face_center_in_box(fb, pb):
                        assigned_tid = tid
                        break
                if assigned_tid is None:
                    continue                 # face not inside any person box — ignore

                # Is this face someone we've already seen this session?
                best_human = None
                best_dist = 999.0
                for h in humans:
                    d = cosine_distance(emb, h["embedding"])
                    if d < best_dist:
                        best_dist = d
                        best_human = h

                if best_human is not None and best_dist <= NEW_HUMAN_THRESHOLD:
                    # --- Existing person ---
                    human_id = best_human["human_id"]
                    tracker_to_human[assigned_tid] = human_id

                    # Nudge the stored embedding toward the new one (90/10) so it
                    # stays accurate as lighting/angle changes.
                    best_human["embedding"] = l2_normalize(
                        0.9 * best_human["embedding"] + 0.1 * emb)

                    # If we hadn't named them yet, try now.
                    if best_human["name"] is None:
                        name, _ = match_known_name(
                            best_human["embedding"], known_db)
                        if name:
                            best_human["name"] = name
                    # Keep the welcome message alive while a known person is on camera.
                    if best_human["name"]:
                        web_greet(best_human["name"])
                else:
                    # --- New person ---
                    human_id = next_human_id
                    next_human_id += 1
                    name, _ = match_known_name(emb, known_db)   # known name? or None
                    humans.append({
                        "human_id": human_id,
                        "embedding": emb,
                        "name": name
                    })
                    tracker_to_human[assigned_tid] = human_id
                    if name:                 # if they're a known person, greet them
                        web_greet(name)

        # --- (d) Which of our humans are currently visible (by track id) ---
        current_humans_in_frame = set()
        for tid in current_tids:
            hid = tracker_to_human.get(tid)
            if hid is not None:
                current_humans_in_frame.add(hid)

        # --- (e) Draw the stats overlay on the debug window ---
        elapsed = int(time.time() - start_time)
        all_seen_tids.update(current_tids)   # accumulate every unique track id ever seen
        cv2.putText(
            annotated,
            f"Current in frame: {len(current_tids)}  |  Unique seen: {len(all_seen_tids)}  |  Time: {elapsed}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.putText(
            annotated,
            f"Current tracker IDs: {current_tids}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # --- (f) List each visible tracker -> person/name (up to 10 lines) ---
        y = 120
        for tid in current_tids[:10]:
            hid = tracker_to_human.get(tid, None)
            if hid is None:
                label = f"Tracker {tid} -> (unassigned)"
            else:
                h = next((x for x in humans if x["human_id"] == hid), None)
                nm = h["name"] if (h and h["name"]) else f"Human {hid}"
                label = f"Tracker {tid} -> {nm}"
            cv2.putText(
                annotated, label, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            y += 26

        # --- (g) Show the debug window; quit on Q ---
        cv2.imshow("People Counter + Face Identity (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

        frame_idx += 1

    # ----- Cleanup + final summary -----
    cap.release()
    cv2.destroyAllWindows()

    print("\n[RESULT]")
    print(f"Unique humans seen: {len(humans)}")
    print()

    recognized = [h for h in humans if h["name"] is not None]
    unrecognized = [h for h in humans if h["name"] is None]

    if recognized:
        print("=== Recognized People ===")
    for h in recognized:
        print(
            f"  Hi {h['name']}! 👋  (appeared as tracker Human {h['human_id']})")

    if unrecognized:
        print("\n=== Unrecognized People ===")
    for h in unrecognized:
        print(f"  Unknown person (Human {h['human_id']})")

    if not recognized and not unrecognized:
        print("  No one was detected.")


# Standard Python entry point: only run main() when this file is executed directly.
if __name__ == "__main__":
    main()
