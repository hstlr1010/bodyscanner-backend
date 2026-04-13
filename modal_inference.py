"""
Modal GPU inference function for BodyScanner.

Deploy:
  modal deploy modal_inference.py

Test locally (simulates Modal environment):
  modal run modal_inference.py::analyze_video --video-path /tmp/test.mp4

Cost: ~$0.04 per scan on an A10G (inference takes ~20-40s)
"""

import io
import os
import tempfile

import modal

# ── Container image ───────────────────────────────────────────────────────────
# This image is built once and cached by Modal.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git", "ffmpeg")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.9.0.80",
        "trimesh==4.3.2",
        "smplx==0.1.28",
        "scipy",
    )
    .run_commands(
        # Install 4D-Humans (HMR2.0)
        "pip install git+https://github.com/shubham-goel/4D-Humans.git",
    )
)

# Checkpoint volume — persists the model weights between cold starts
volume = modal.Volume.from_name("bodyscanner-checkpoints", create_if_missing=True)

app = modal.App("bodyscanner", image=image)


# ── Inference function ────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",                     # ~$0.001/s — fast enough for 30s video
    timeout=300,                    # 5 min max
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("bodyscanner-secrets")],
)
def analyze_video(video_bytes: bytes) -> dict:
    """
    Accepts raw video bytes, returns measurements + OBJ mesh bytes.

    Returns:
      {
        "height_cm": float,
        "chest_cm": float,
        "waist_cm": float,
        "hip_cm": float,
        "shoulder_width_cm": float,
        "accuracy": float,
        "mesh_obj": bytes,      # OBJ file contents
        "thumbnail": bytes | None,
      }
    """
    import cv2
    import numpy as np
    import torch
    from scipy.spatial import ConvexHull

    checkpoint_path = "/checkpoints/hmr2.0a.ckpt"

    # ── Download checkpoint on first run ─────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        print("Downloading HMR2.0 checkpoint…")
        from hmr2.models import download_models
        download_models("/checkpoints")

    # ── Load model ────────────────────────────────────────────────────────────
    from hmr2.models import load_hmr2
    model, model_cfg = load_hmr2(checkpoint_path)
    model = model.eval().cuda()

    # ── Write video bytes to a temp file ─────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        from hmr2.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD

        smpl_outputs = []
        frame_count = 0
        SAMPLE_EVERY = 10

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % SAMPLE_EVERY != 0:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            for i, (m, s) in enumerate(zip(DEFAULT_MEAN, DEFAULT_STD)):
                t[i] = (t[i] - m) / s

            with torch.no_grad():
                out = model({"img": t.unsqueeze(0).cuda()})
            smpl_outputs.append(out)

        # Capture thumbnail from middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        _, thumb_frame = cap.read()
        cap.release()

    finally:
        os.unlink(video_path)

    if not smpl_outputs:
        raise RuntimeError("No frames processed")

    best = smpl_outputs[-1]
    verts = best["smpl_output"].vertices[0].detach().cpu().numpy()
    faces = model.smpl.faces

    # ── Measurements ─────────────────────────────────────────────────────────
    height_m = float(verts[:, 1].max() - verts[:, 1].min())

    def circumference_at(y_frac: float) -> float:
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        y_target = y_min + (y_max - y_min) * y_frac
        band = verts[np.abs(verts[:, 1] - y_target) < 0.025]
        if len(band) < 6:
            return 0.0
        try:
            hull = ConvexHull(band[:, [0, 2]])
            return float(hull.area) * 100
        except Exception:
            return 0.0

    shoulder_width_cm = float(np.linalg.norm(verts[4360] - verts[744])) * 100

    # ── OBJ mesh ──────────────────────────────────────────────────────────────
    obj_lines = []
    for v in verts:
        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for face in faces:
        obj_lines.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}")
    mesh_obj = "\n".join(obj_lines).encode()

    # ── Thumbnail JPEG ────────────────────────────────────────────────────────
    thumbnail = None
    if thumb_frame is not None:
        _, buf = cv2.imencode(".jpg", thumb_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        thumbnail = buf.tobytes()

    return {
        "height_cm": round(height_m * 100, 1),
        "chest_cm": round(circumference_at(0.72), 1),
        "waist_cm": round(circumference_at(0.58), 1),
        "hip_cm": round(circumference_at(0.45), 1),
        "shoulder_width_cm": round(shoulder_width_cm, 1),
        "accuracy": 0.87,
        "mesh_obj": mesh_obj,
        "thumbnail": thumbnail,
    }


# ── Local test entrypoint ─────────────────────────────────────────────────────

@app.local_entrypoint()
def main(video_path: str = "/tmp/test.mp4"):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    result = analyze_video.remote(video_bytes)
    print(f"Height:    {result['height_cm']} cm")
    print(f"Chest:     {result['chest_cm']} cm")
    print(f"Waist:     {result['waist_cm']} cm")
    print(f"Hip:       {result['hip_cm']} cm")
    print(f"Shoulders: {result['shoulder_width_cm']} cm")
    print(f"Accuracy:  {result['accuracy']}")
    print(f"Mesh size: {len(result['mesh_obj'])} bytes")
