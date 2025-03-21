from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Khmer OCR Api"
    version: str = "0.1.0"
