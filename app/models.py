"""
SQLAlchemy ORM models.

Key design decision:
  A scan belongs to a ContactProfile (end customer / gym member),
  NOT to the owner/staff user who performed it.

  bizbozz data model:
    User ──< business_persona >── Business ──< ContactProfile
                                                    │
                                                  Scan  ◄─── here

  We store business_id redundantly for fast filtering without a JOIN.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    # Owner references (bizbozz UUIDs — we never store these users ourselves)
    contact_profile_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    business_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    # Who triggered the scan (owner/staff user that was logged in)
    created_by_user_id: Mapped[str] = mapped_column(String(36), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # ── Measurements ─────────────────────────────────────────────────────────
    height_cm: Mapped[float | None] = mapped_column(Float)
    chest_cm: Mapped[float | None] = mapped_column(Float)
    waist_cm: Mapped[float | None] = mapped_column(Float)
    hip_cm: Mapped[float | None] = mapped_column(Float)
    shoulder_width_cm: Mapped[float | None] = mapped_column(Float)
    accuracy: Mapped[float | None] = mapped_column(Float)

    # ── 3-D mesh ─────────────────────────────────────────────────────────────
    # OBJ file stored under MESH_STORAGE_PATH/{business_id}/{id}.obj
    mesh_file_path: Mapped[str | None] = mapped_column(Text)

    # Thumbnail JPEG stored under MESH_STORAGE_PATH/{business_id}/{id}_thumb.jpg
    thumbnail_file_path: Mapped[str | None] = mapped_column(Text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "contactProfileId": self.contact_profile_id,
            "businessId": self.business_id,
            "createdAt": self.created_at.isoformat(),
            "measurements": {
                "heightCm": self.height_cm,
                "chestCm": self.chest_cm,
                "waistCm": self.waist_cm,
                "hipCm": self.hip_cm,
                "shoulderWidthCm": self.shoulder_width_cm,
                "accuracy": self.accuracy,
            },
            "hasMesh": self.mesh_file_path is not None,
            "hasThumbnail": self.thumbnail_file_path is not None,
        }
