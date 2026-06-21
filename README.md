# Office Welcome Vision 👋

Real-time people detection, counting, and face recognition with a live
**welcome screen** for a TV. A camera watches an entrance; when it recognises a
known person it shows *"Hello &lt;name&gt;, welcome!"* on a casted Chrome tab, and
otherwise shows a default *"Welcome to the office"*.

Built with **YOLOv8** (detection + ByteTrack tracking), **InsightFace** (face
recognition), **OpenCV** (video), and **Flask + Server-Sent Events** (the live TV
screen).

---

## Quick start

```bash
# 1) install dependencies
pip install -r requirements.txt

# 2) (optional) add known people for greetings
#    known_faces/Mannat/photo.jpg, known_faces/Vishal/vishal.jpg, ...

# 3) run
python human_and_face_detection.py
```

- Press **Q** in the OpenCV window to quit.
- Open **http://localhost:8080** in Chrome and **cast that tab** to your TV.
- Toggle the camera at the top of the file: `USE_WEBCAM = True` (laptop) or
  `False` (WiFi/RTSP camera, found automatically by its MAC address).

---

## Documentation

- **[DESIGN.md](DESIGN.md)** — full architecture, components, config reference,
  deployment notes.
- **[INTERVIEW.md](INTERVIEW.md)** — interview-style Q&A covering all the tech used.

---

## How it works (one paragraph)

Each camera frame is grabbed by a background thread that always keeps the newest
frame (so the feed never lags). **YOLOv8** detects every person and **ByteTrack**
gives each a stable ID for counting. Every few frames, **InsightFace** turns
visible faces into embeddings and compares them (cosine distance) against a
known-faces database. When a known person appears, the app pushes a greeting to a
small **Flask** server, which streams it to a browser page via **Server-Sent
Events** — shown big and centred on a TV with a fade animation.

See [DESIGN.md](DESIGN.md) for the details.
