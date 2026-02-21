from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	app_name: str = "AI-Assisted Text Classification API"
	database_url: str = "sqlite:///./classifier.db"

	similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

	ollama_host: str = Field(default="http://localhost:11434")
	ollama_embedding_model: str = Field(default="qwen3-embedding:8b-fp16")
	ollama_timeout_seconds: float = Field(default=20.0, ge=1.0, le=300.0)

	model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

