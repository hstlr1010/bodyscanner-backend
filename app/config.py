from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    jwt_secret_business: str = "REPLACE_ME"
    jwt_algorithm: str = "HS256"
    jwt_audience: str = "business-app"
    jwt_issuer: str = "gymheros"

    database_url: str = "sqlite+aiosqlite:///./bodyscanner.db"

    mesh_storage_path: str = "./meshes"

    hmr2_checkpoint: str = "./checkpoints/hmr2.0a.ckpt"

    max_video_size_mb: int = 200


settings = Settings()
