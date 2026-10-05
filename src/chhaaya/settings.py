from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    postgres_password: str
    database_url: str
    whatsapp_verify_token: str
    whatsapp_app_secret: str
    whatsapp_access_token: str
    whatsapp_phone_number_id: str
    # Read by Compose for the optional `tunnel` profile, not by the service.
    cloudflare_tunnel_token: str | None = None
