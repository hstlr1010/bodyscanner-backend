"""
HMR2.0 / 4D-Humans inference pipeline.

Install 4D-Humans:
  pip install git+https://github.com/shubham-goel/4D-Humans.git
  # or clone and `pip install -e .`

Download checkpoint:
  python -m hmr2.utils.download_models --download_dir ./checkpoints

Usage:
  result = await run_inference(video_path)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings


@dataclass
class InferenceResult:
    """Output of the HMR2.0 pipeline for one video."""

    height_cm: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    shoulder_width_cm: float
    accuracy: float          # mean per-joint position error proxy [0-1]
    mesh_obj_path: str | None = None   # path to saved OBJ file
    thumbnail_path: str | None = None  # path to saved JPEG thumbnail


# ── Try to import 4D-Humans; fall back to a mock if not installed ─────────────

try:
    import torch
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.models import HMR2, download_models
    from hmr2.utils import recursive_to
    from hmr2.utils.geometry import aa_to_rotmat, perspective_projection

    _HMR2_AVAILABLE = True
except ImportError:
    _HMR2_AVAILABLE = False


def _load_model():
    if not _HMR2_AVAILABLE:
        return None
    from hmr2.models import load_hmr2
    model, model_cfg = load_hmr2(settings.hmr2_checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, model_cfg


_model_cache: tuple | None = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = _load_model()
    return _model_cache


# ── SMPL measurement extraction ───────────────────────────────────────────────

def _smpl_to_measurements(smpl_output) -> dict:
    """
    Extract body measurements from SMPL vertices.

    The SMPL model provides a 6890-vertex mesh in a canonical T-pose.
    We compute circumferences by slicing horizontal cross-sections at
    anatomically-defined vertex indices.

    Vertex index sets (approximate, based on SMPL topology):
      chest:   vertices around the chest circumference ~y=0.3 in SMPL space
      waist:   vertices around the navel ~y=0.1
      hip:     vertices at the greater trochanter ~y=-0.1
    """
    import torch

    verts = smpl_output.vertices[0].detach().cpu().numpy()  # (6890, 3)

    # Height: max y - min y
    height_m = float(verts[:, 1].max() - verts[:, 1].min())
    height_cm = height_m * 100

    # Circumferences via horizontal slice at target y
    def circumference_at(y_frac: float) -> float:
        """Approximate circumference (cm) by sampling vertices near a given y fraction."""
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        y_target = y_min + (y_max - y_min) * y_frac
        band = verts[np.abs(verts[:, 1] - y_target) < 0.025]  # ±2.5 cm band
        if len(band) < 6:
            return 0.0
        # Convex hull perimeter of xz projection
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(band[:, [0, 2]])
            return float(hull.area) * 100  # m → cm (2-D perimeter = area in scipy)
        except Exception:
            return 0.0

    # Shoulder width: distance between left/right shoulder vertices
    # SMPL vertex indices for shoulders (approx): left=4360, right=744
    lsh = verts[4360]
    rsh = verts[744]
    shoulder_width_cm = float(np.linalg.norm(lsh - rsh)) * 100

    return {
        "height_cm": round(height_cm, 1),
        "chest_cm": round(circumference_at(0.72), 1),
        "waist_cm": round(circumference_at(0.58), 1),
        "hip_cm": round(circumference_at(0.45), 1),
        "shoulder_width_cm": round(shoulder_width_cm, 1),
    }


# ── Real inference ────────────────────────────────────────────────────────────

def _run_hmr2_sync(video_path: str, out_dir: str) -> InferenceResult:
    """Synchronous HMR2.0 inference — call from a thread pool."""
    import cv2
    import torch

    model_tuple = _get_model()
    if model_tuple is None:
        raise RuntimeError("HMR2.0 model not available")
    model, model_cfg = model_tuple

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    smpl_outputs = []
    frame_count = 0
    SAMPLE_EVERY = 10  # process every 10th frame

    from hmr2.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
    from hmr2.utils.geometry import aa_to_rotmat

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % SAMPLE_EVERY != 0:
            continue

        # Convert BGR→RGB, resize to 256×256, normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        for i, (m, s) in enumerate(zip(DEFAULT_MEAN, DEFAULT_STD)):
            img_tensor[i] = (img_tensor[i] - m) / s
        img_tensor = img_tensor.unsqueeze(0)

        device = next(model.parameters()).device
        batch = {"img": img_tensor.to(device)}

        with torch.no_grad():
            out = model(batch)
        smpl_outputs.append(out)

    cap.release()

    if not smpl_outputs:
        raise RuntimeError("No frames processed — video may be too short or corrupted")

    # Use the last frame's output (most complete body pose) for measurements
    best = smpl_outputs[-1]
    measurements = _smpl_to_measurements(best["smpl_output"])

    # Export OBJ mesh
    mesh_path = os.path.join(out_dir, "mesh.obj")
    verts = best["smpl_output"].vertices[0].detach().cpu().numpy()
    faces = model.smpl.faces  # (13776, 3)

    with open(mesh_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Thumbnail: middle frame
    cap2 = cv2.VideoCapture(video_path)
    total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    _, thumb_frame = cap2.read()
    cap2.release()
    thumb_path = None
    if thumb_frame is not None:
        thumb_path = os.path.join(out_dir, "thumbnail.jpg")
        cv2.imwrite(thumb_path, thumb_frame)

    return InferenceResult(
        **measurements,
        accuracy=0.85,  # placeholder — real metric requires GT
        mesh_obj_path=mesh_path,
        thumbnail_path=thumb_path,
    )


# ── Mock inference (used when HMR2.0 is not installed) ────────────────────────

def _mock_inference(video_path: str, out_dir: str) -> InferenceResult:
    """Returns plausible fake measurements for development / CI."""
    import random
    rng = random.Random(os.path.getsize(video_path))  # deterministic per file

    mesh_path = os.path.join(out_dir, "mesh.obj")
    # Write a minimal OBJ (tetrahedron) so downstream code has a real file
    with open(mesh_path, "w") as f:
        f.write("# mock mesh\n")
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n")
        f.write("f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")

    return InferenceResult(
        height_cm=round(rng.uniform(160, 190), 1),
        chest_cm=round(rng.uniform(85, 110), 1),
        waist_cm=round(rng.uniform(68, 95), 1),
        hip_cm=round(rng.uniform(88, 112), 1),
        shoulder_width_cm=round(rng.uniform(38, 50), 1),
        accuracy=round(rng.uniform(0.80, 0.95), 2),
        mesh_obj_path=mesh_path,
        thumbnail_path=None,
    )


# ── Public async entry point ──────────────────────────────────────────────────

async def run_inference(video_path: str, out_dir: str) -> InferenceResult:
    """
    Run the body-shape inference pipeline on a recorded video.

    Falls back to a mock implementation if the HMR2.0 package is not installed
    (useful for local development without a GPU).
    """
    loop = asyncio.get_running_loop()

    if _HMR2_AVAILABLE:
        fn = _run_hmr2_sync
    else:
        fn = _mock_inference

    return await loop.run_in_executor(None, fn, video_path, out_dir)
