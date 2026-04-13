"""
Scan endpoints.

POST /contacts/{contactProfileId}/analyze
  — Upload a video, run inference, save scan, return measurements + mesh URL.
  — Authenticated: valid bizbozz JWT required.
  — The contactProfileId in the URL MUST belong to the JWT's businessId.
    (We trust the caller here; add a reverse-proxy bizbozz API check if needed.)

GET  /contacts/{contactProfileId}/scans
  — List all scans for a contact.
  — Only returns scans where business_id matches the JWT.

GET  /scans/{scanId}
  — Fetch a single scan (must belong to JWT's businessId).

GET  /scans/{scanId}/mesh
  — Stream the OBJ mesh file.

GET  /scans/{scanId}/thumbnail
  — Stream the JPEG thumbnail.

DELETE /scans/{scanId}
  — Delete scan + files (must belong to JWT's businessId).
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import BizBozzClaims, verify_jwt
from app.config import settings
from app.database import get_db
from app.inference.pipeline import run_inference
from app.models import Scan
from app.schemas import AnalyzeResponse, Measurements, ScanOut

router = APIRouter(prefix="/contacts", tags=["scans"])
scan_router = APIRouter(prefix="/scans", tags=["scans"])


def _mesh_dir(business_id: str) -> Path:
    p = Path(settings.mesh_storage_path) / business_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _assert_same_business(scan: Scan, claims: BizBozzClaims):
    if scan.business_id != claims.businessId:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


# ── POST /contacts/{contactProfileId}/analyze ──────────────────────────────

@router.post("/{contactProfileId}/analyze", response_model=AnalyzeResponse)
async def analyze(
    contactProfileId: str,
    video: UploadFile = File(..., description="Recorded body scan video (mp4/mov)"),
    db: AsyncSession = Depends(get_db),
    claims: BizBozzClaims = Depends(verify_jwt),
):
    # Validate video size
    max_bytes = settings.max_video_size_mb * 1024 * 1024
    contents = await video.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video exceeds {settings.max_video_size_mb} MB limit",
        )

    # Write upload to a temp file
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    scan_id = str(uuid.uuid4())
    out_dir = str(_mesh_dir(claims.businessId) / scan_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        result = await run_inference(tmp_path, out_dir)
    finally:
        os.unlink(tmp_path)

    # Persist mesh + thumbnail to final locations
    mesh_path = None
    if result.mesh_obj_path and os.path.exists(result.mesh_obj_path):
        mesh_path = str(_mesh_dir(claims.businessId) / f"{scan_id}.obj")
        shutil.move(result.mesh_obj_path, mesh_path)

    thumb_path = None
    if result.thumbnail_path and os.path.exists(result.thumbnail_path):
        thumb_path = str(_mesh_dir(claims.businessId) / f"{scan_id}_thumb.jpg")
        shutil.move(result.thumbnail_path, thumb_path)

    scan = Scan(
        id=scan_id,
        contact_profile_id=contactProfileId,
        business_id=claims.businessId,
        created_by_user_id=claims.userId,
        height_cm=result.height_cm,
        chest_cm=result.chest_cm,
        waist_cm=result.waist_cm,
        hip_cm=result.hip_cm,
        shoulder_width_cm=result.shoulder_width_cm,
        accuracy=result.accuracy,
        mesh_file_path=mesh_path,
        thumbnail_file_path=thumb_path,
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)

    base = f"/scans/{scan_id}"
    return AnalyzeResponse(
        scanId=scan_id,
        contactProfileId=contactProfileId,
        measurements=Measurements(
            height_cm=result.height_cm,
            chest_cm=result.chest_cm,
            waist_cm=result.waist_cm,
            hip_cm=result.hip_cm,
            shoulder_width_cm=result.shoulder_width_cm,
            accuracy=result.accuracy,
        ),
        meshUrl=f"{base}/mesh" if mesh_path else None,
        thumbnailUrl=f"{base}/thumbnail" if thumb_path else None,
    )


# ── GET /contacts/{contactProfileId}/scans ────────────────────────────────

@router.get("/{contactProfileId}/scans", response_model=list[ScanOut])
async def list_scans(
    contactProfileId: str,
    db: AsyncSession = Depends(get_db),
    claims: BizBozzClaims = Depends(verify_jwt),
):
    result = await db.execute(
        select(Scan)
        .where(
            Scan.contact_profile_id == contactProfileId,
            Scan.business_id == claims.businessId,
        )
        .order_by(Scan.created_at.desc())
    )
    scans = result.scalars().all()
    return [
        ScanOut(
            id=s.id,
            contactProfileId=s.contact_profile_id,
            businessId=s.business_id,
            createdAt=s.created_at,
            measurements=Measurements(
                height_cm=s.height_cm,
                chest_cm=s.chest_cm,
                waist_cm=s.waist_cm,
                hip_cm=s.hip_cm,
                shoulder_width_cm=s.shoulder_width_cm,
                accuracy=s.accuracy,
            ),
            hasMesh=s.mesh_file_path is not None,
            hasThumbnail=s.thumbnail_file_path is not None,
        )
        for s in scans
    ]


# ── GET /scans/{scanId} ───────────────────────────────────────────────────

@scan_router.get("/{scanId}", response_model=ScanOut)
async def get_scan(
    scanId: str,
    db: AsyncSession = Depends(get_db),
    claims: BizBozzClaims = Depends(verify_jwt),
):
    scan = await db.get(Scan, scanId)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    _assert_same_business(scan, claims)
    return ScanOut(
        id=scan.id,
        contactProfileId=scan.contact_profile_id,
        businessId=scan.business_id,
        createdAt=scan.created_at,
        measurements=Measurements(
            height_cm=scan.height_cm,
            chest_cm=scan.chest_cm,
            waist_cm=scan.waist_cm,
            hip_cm=scan.hip_cm,
            shoulder_width_cm=scan.shoulder_width_cm,
            accuracy=scan.accuracy,
        ),
        hasMesh=scan.mesh_file_path is not None,
        hasThumbnail=scan.thumbnail_file_path is not None,
    )


# ── GET /scans/{scanId}/mesh ──────────────────────────────────────────────

@scan_router.get("/{scanId}/mesh")
async def get_mesh(
    scanId: str,
    db: AsyncSession = Depends(get_db),
    claims: BizBozzClaims = Depends(verify_jwt),
):
    scan = await db.get(Scan, scanId)
    if not scan or not scan.mesh_file_path:
        raise HTTPException(status_code=404, detail="Mesh not found")
    _assert_same_business(scan, claims)
    return FileResponse(scan.mesh_file_path, media_type="text/plain", filename="mesh.obj")


# ── GET /scans/{scanId}/thumbnail ─────────────────────────────────────────

@scan_router.get("/{scanId}/thumbnail")
async def get_thumbnail(
    scanId: str,
    db: AsyncSession = Depends(get_db),
    claims: BizBozzClaims = Depends(verify_jwt),
):
    scan = await db.get(Scan, scanId)
    if not scan or not scan.thumbnail_file_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    _assert_same_business(scan, claims)
    return FileResponse(scan.thumbnail_file_path, media_type="image/jpeg")


# ── DELETE /scans/{scanId} ────────────────────────────────────────────────

@scan_router.delete("/{scanId}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scan(
    scanId: str,
    db: AsyncSession = Depends(get_db),
    claims: BizBozzClaims = Depends(verify_jwt),
):
    scan = await db.get(Scan, scanId)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    _assert_same_business(scan, claims)

    for path in [scan.mesh_file_path, scan.thumbnail_file_path]:
        if path and os.path.exists(path):
            os.unlink(path)

    await db.delete(scan)
    await db.commit()
