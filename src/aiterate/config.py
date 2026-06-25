from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIT_", env_file=".env", extra="ignore")

    storage_dir: Path = Field(default=Path(".aiterate"))
    git_author_name: str = "AIterate Bot"
    git_author_email: str = "bot@aiterate.dev"
    enable_local_git: bool = Field(default=False, validation_alias="AIT_ENABLE_LOCAL_GIT")
    tracker: str = "mlflow"
    environment: str = Field(default="development", validation_alias="AIT_ENV")
    database_url: str = Field(
        default="sqlite:///.aiterate/aiterate.db",
        validation_alias="AIT_DATABASE_URL",
    )
    secret_key: str | None = Field(default=None, validation_alias="AIT_SECRET_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    azure_openai_api_key: str | None = Field(default=None, validation_alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str | None = Field(default=None, validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(
        default="2024-10-21", validation_alias="AZURE_OPENAI_API_VERSION"
    )
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    aws_profile: str | None = Field(default=None, validation_alias="AWS_PROFILE")
    mlflow_tracking_uri: str | None = Field(default=None, validation_alias="MLFLOW_TRACKING_URI")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")
    github_app_id: str | None = Field(default=None, validation_alias="GITHUB_APP_ID")
    bitbucket_token: str | None = Field(default=None, validation_alias="BITBUCKET_TOKEN")
    secret_provider: str = Field(default="database", validation_alias="AIT_SECRET_PROVIDER")
    auth_enabled: bool = Field(default=False, validation_alias="AIT_AUTH_ENABLED")
    admin_api_key: str | None = Field(default=None, validation_alias="AIT_ADMIN_API_KEY")
    jwt_secret: str | None = Field(default=None, validation_alias="AIT_JWT_SECRET")
    vault_addr: str | None = Field(default=None, validation_alias="VAULT_ADDR")
    vault_token: str | None = Field(default=None, validation_alias="VAULT_TOKEN")
    vault_mount: str = Field(default="secret", validation_alias="AIT_VAULT_MOUNT")
    vault_path_prefix: str = Field(default="aiterate", validation_alias="AIT_VAULT_PATH_PREFIX")
    aws_secrets_prefix: str = Field(default="aiterate", validation_alias="AIT_AWS_SECRETS_PREFIX")
    azure_key_vault_url: str | None = Field(default=None, validation_alias="AZURE_KEY_VAULT_URL")
    gcp_project_id: str | None = Field(default=None, validation_alias="GCP_PROJECT_ID")
    bitbucket_workspace: str | None = Field(default=None, validation_alias="BITBUCKET_WORKSPACE")
    bitbucket_repo_slug: str | None = Field(default=None, validation_alias="BITBUCKET_REPO_SLUG")
    cors_origins: str = Field(
        default="http://127.0.0.1:5173,http://localhost:5173",
        validation_alias="AIT_CORS_ORIGINS",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
