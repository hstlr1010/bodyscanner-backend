from datetime import datetime

from pydantic import BaseModel, Field


class Measurements(BaseModel):
    height_cm: float | None = None
    chest_cm: float | None = None
    waist_cm: float | None = None
    hip_cm: float | None = None
    shoulder_width_cm: float | None = None
    accuracy: float | None = None


class ScanOut(BaseModel):
    id: str
    contact_profile_id: str = Field(alias="contactProfileId")
    business_id: str = Field(alias="businessId")
    created_at: datetime = Field(alias="createdAt")
    measurements: Measurements
    has_mesh: bool = Field(alias="hasMesh")
    has_thumbnail: bool = Field(alias="hasThumbnail")

    model_config = {"populate_by_name": True}


class AnalyzeResponse(BaseModel):
    scan_id: str = Field(alias="scanId")
    contact_profile_id: str = Field(alias="contactProfileId")
    measurements: Measurements
    mesh_url: str | None = Field(default=None, alias="meshUrl")
    thumbnail_url: str | None = Field(default=None, alias="thumbnailUrl")

    model_config = {"populate_by_name": True}
