"""Configuration settings via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env and environment variables.

    Names match SPEC.md section 14 exactly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    MODEL: str = "gemini-3.8-flash"

    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "pdf-rag"
    GITHUB_PAT: str = ""
    GITHUB_ALLOW_WRITE: bool = False
    GOOGLE_OAUTH_CREDENTIALS: str = "./gcp-oauth.keys.json"
    GOOGLE_TOKEN_PATH: str = "./token.json"
    TIMEZONE: str = "Asia/Karachi"
    DATABASE_URL: str = "sqlite:///./data/app.db"
    CHECKPOINT_DB: str = "./data/checkpoints.db"
    APPROVE_EVENT_CREATE: bool = True
    MAX_EMAIL_RECIPIENTS: int = 10
    MAX_LATE_MINUTES: int = 60
    LOG_EMAIL_BODIES: bool = False
    MOCK_MODE: bool = False
    API_KEY: str = ""
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "assistant"


settings = Settings()
