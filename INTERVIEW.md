# Interview Prep — Office Welcome Vision

Questions an interviewer might ask about this project, with answers you can say
in your own words. Grouped by topic, easy → harder.

---

## A. Project overview

**Q: Describe this project in a few sentences.**
> It's a real-time computer-vision app that watches a camera feed, detects and
> counts people, and recognises known faces. When it sees a known person it
> shows a personalised *"Hello &lt;name&gt;, welcome!"* on a TV via a web page; when
> nobody known is around it shows a default welcome message. It's aimed at office
> or retail entrances.

**Q: Walk me through the data flow / pipeline.**
> Camera frame → a background thread that always keeps the newest frame →
> YOLOv8 detects each person and ByteTrack gives them a stable ID → every few
> frames InsightFace checks visible faces against a known-faces database → if a
> known person is found, the app pushes a greeting to a small Flask web server →
> that page is cast to a TV and updates live.

**Q: Why these components and not something else?**
> YOLOv8 is fast, accurate, and easy to use for person detection. ByteTrack is a
> strong, lightweight tracker bundled with it. InsightFace gives high-quality
> face embeddings and runs on CPU. Flask + Server-Sent Events is the simplest way
> to get a live-updating browser screen without a heavy frontend framework.

---

## B. Object detection (YOLO)

**Q: What is YOLO and why is it called that?**
> YOLO = "You Only Look Once". It's a single-stage object detector: one forward
> pass through the network produces all bounding boxes and class scores at once,
> which makes it very fast — suitable for real-time video.

**Q: How does it differ from older detectors like R-CNN?**
> Two-stage detectors (R-CNN family) first propose regions, then classify each —
> accurate but slow. YOLO does detection in a single pass over the whole image,
> trading a little accuracy for big speed gains.

**Q: What do `conf`, `classes`, and `imgsz` do in your code?**
> `conf` is the confidence threshold — detections below it are discarded.
> `classes=[0]` restricts output to the "person" class (COCO class 0).
> `imgsz=640` resizes the frame to 640px before inference — bigger sees more
> detail but is slower.

**Q: What's the difference between the model sizes (n/s/m/l/x)?**
> They trade accuracy for speed/size. `n` (nano) is fastest and least accurate;
> `x` (xlarge) is most accurate and slowest. I use `m` (medium) as a balance for
> CPU. (Said another way: more parameters = better detection, more compute.)

**Q: How do you count people accurately when detections flicker?**
> I rely on **track IDs**, not raw detections. Counting unique track IDs avoids
> double-counting the same person across frames, and ByteTrack keeps the ID
> stable even if detection briefly drops.

---

## C. Tracking (ByteTrack)

**Q: Why do you need tracking on top of detection?**
> Detection alone tells you "there's a person here" each frame but not whether
> it's the *same* person as last frame. Tracking links detections over time into
> consistent identities, which is what lets me count unique people and keep a
> greeting attached to the right person.

**Q: How does ByteTrack work at a high level?**
> It associates detections between frames using motion prediction (a Kalman
> filter) and box overlap (IoU). Its key idea is to also use *low-confidence*
> detections during association, which recovers people who are briefly occluded —
> reducing ID switches.

**Q: What is an "ID switch" and why does it matter?**
> When the tracker accidentally gives an existing person a new ID (or swaps two
> people's IDs). It inflates the unique count and breaks identity continuity,
> so minimizing ID switches is a key quality metric for a tracker.

---

## D. Face recognition (InsightFace, embeddings)

**Q: What is a face embedding?**
> A fixed-length vector (512 numbers here) that numerically summarises a face.
> The model is trained so that the *same* person's photos produce nearby vectors
> and *different* people produce far-apart vectors. Recognition becomes a
> distance comparison between vectors.

**Q: How do you compare two faces?**
> I L2-normalise both embeddings (scale to length 1) and compute **cosine
> distance** = `1 - dot(a, b)`. 0 means identical direction; larger means more
> different. If the distance to a known person is below a threshold, it's a match.

**Q: Why cosine distance and not Euclidean?**
> Face embeddings encode identity mainly in their *direction*, not magnitude.
> Cosine similarity compares direction and is the metric these models are trained
> with. After L2-normalisation, cosine and Euclidean are monotonically related,
> so cosine is the natural, scale-invariant choice.

**Q: Why L2-normalise?**
> It removes magnitude differences so comparisons depend only on direction, and
> it makes the dot product equal the cosine similarity — simpler and more stable.

**Q: How do you decide the thresholds (0.65 / 0.75)?**
> Empirically — by trying values and watching false matches vs missed matches.
> Lower threshold = stricter (fewer false matches, more misses); higher = looser.
> `0.65` for "is this a known named person", a looser `0.75` for "is this the
> same person I already tracked this session".

**Q: Why average multiple photos per person?**
> A single photo captures one angle/lighting. Averaging several embeddings gives
> a more robust, representative vector, improving recognition reliability.

**Q: How do you link a face to the right person box?**
> I check whether the face box's centre falls inside a tracked person's bounding
> box (`face_center_in_box`). Simple and effective for entrance scenarios.

**Q: What are the privacy implications of face recognition?**
> It's biometric data, so it needs consent, secure storage, retention limits, and
> compliance with laws (e.g. GDPR). In this project, known-face photos are kept
> local and excluded from version control.

---

## E. Concurrency / threading

**Q: Why do you use threads here?**
> Three things must happen at once: grabbing frames, running detection, and
> serving the web page. I use a background thread for frame grabbing and another
> for the Flask server, leaving the main thread for the detection loop.

**Q: What problem does `FreshestFrame` solve?**
> Processing is slower than the camera's frame rate. If I read frames in order,
> they queue up and the displayed feed drifts seconds behind reality. The grabber
> thread constantly overwrites a single "latest frame" buffer, so the main loop
> always processes the *current* moment and just skips the backlog.

**Q: How do you avoid race conditions on shared data?**
> A `threading.Lock` guards the shared frame in `FreshestFrame`, and another lock
> guards the `_active_greets` dictionary shared between the detection loop and the
> web server.

**Q: Isn't Python limited by the GIL?**
> Yes, the GIL prevents two threads running Python bytecode simultaneously. But
> threading still helps here because the heavy work (OpenCV, YOLO/Torch,
> InsightFace, socket I/O) releases the GIL during native/C calls and I/O waits,
> so the threads genuinely overlap.

---

## F. The web screen (Flask + SSE)

**Q: How does the TV screen update in real time?**
> The browser opens a **Server-Sent Events** connection to `/events`. The server
> keeps it open and pushes the new message whenever it changes. The page swaps
> the text with a CSS fade. No polling, no reloads.

**Q: Why Server-Sent Events instead of WebSockets or polling?**
> The data only flows one way (server → browser) and is simple text, which is
> exactly what SSE is for. It's lighter than WebSockets and auto-reconnects.
> Polling would either lag or hammer the server; SSE pushes instantly.

**Q: Why port 8080 and not 5000?**
> On macOS, port 5000 is taken by the AirPlay Receiver (Control Center), which
> returned an "access denied" page. 8080 is free, so I moved the server there.

**Q: Is this production-grade web serving?**
> No — Flask's built-in server is for development/single-screen use, which is fine
> here. For production I'd put it behind a proper WSGI server (gunicorn) and a
> reverse proxy.

---

## G. Networking (RTSP, MAC discovery)

**Q: What is RTSP?**
> Real-Time Streaming Protocol — the standard way IP cameras expose a video
> stream. You connect with a URL like `rtsp://user:pass@ip:554/stream1`.

**Q: Your camera kept "disconnecting." How did you diagnose and fix it?**
> The error was "connection refused". I scanned the network and found the
> camera's IP had changed (DHCP) and a different device had taken the old IP. The
> robust fix: look the camera up by its **MAC address** (which never changes)
> using the ARP table, then build the RTSP URL with whatever IP it currently has.

**Q: What's the difference between an IP and a MAC address?**
> A MAC is a permanent hardware address baked into the network interface
> (layer 2, local network only). An IP is a logical, routable address (layer 3)
> that can change — e.g. when assigned by DHCP. You can't open a TCP connection
> to a MAC directly, but you can use it to discover the current IP locally.

**Q: How does the MAC→IP lookup actually work?**
> I read the OS ARP cache (`arp -a`), which maps IPs to MACs for devices the
> machine has talked to. If the camera isn't cached, I ping every host on the
> subnet to force ARP entries, then search for my MAC and return its IP.

---

## H. Performance & scaling

**Q: What were your performance bottlenecks?**
> Two: (1) face recognition is heavy, so I only run it every Nth frame; (2) on a
> slow disk/iCloud-synced folder, importing the big ML libraries was extremely
> slow — moving the project to a local folder fixed startup time.

**Q: How would you scale this to many cameras / a whole store?**
> One capture+detection pipeline per camera (process or thread), ideally on a GPU;
> push results to a central service/database; and separate the analytics/dashboard
> from the capture nodes. A smaller model (`yolov8n`) per stream if CPU-bound.

**Q: How would you reduce false greetings?**
> Require the same identity across a few consecutive recognitions before greeting,
> tune thresholds, use the larger `buffalo_l` model, and add more/better
> reference photos per person.

---

## I. General / design

**Q: How do you handle secrets (the camera password)?**
> The RTSP username/password are loaded from a local `.env` file that's listed in
> `.gitignore`, so they never get committed. The code reads them via
> `os.environ.get(...)`, and a committed `.env.example` documents the format
> without exposing real values. (Lesson learned: I'd originally hard-coded them
> and they reached the public repo — so I moved them to `.env`, scrubbed the
> history, and rotated the credential.)

**Q: What would you improve with more time?**
> Add persistence (DB) for counts and visits; support multiple cameras; add a real
> dashboard; add unit tests; and put the web server behind a production WSGI stack
> (gunicorn) instead of Flask's dev server.

**Q: How is the code organised?**
> A single module with clear sections: config, camera-discovery helpers,
> face-math helpers, the frame-grabber class, the Flask web screen, and `main()`
> with the detection loop. Configuration is centralised at the top so behaviour
> can be tuned without touching logic.

**Q: What did you learn building this?**
> How detection, tracking, and recognition fit together; why threading matters
> for real-time video; how to debug real networking issues (DHCP/RTSP); and the
> value of centralised config and clear documentation.

---

*Keep this document updated as the project's tech evolves.*
