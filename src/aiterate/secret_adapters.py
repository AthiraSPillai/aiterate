from __future__ import annotations

from abc import ABC, abstractmethod

from aiterate.config import settings


class ManagedSecretAdapter(ABC):
    @abstractmethod
    def put(self, name: str, value: str) -> None:
        """Store a secret value in the managed backend."""

    @abstractmethod
    def get(self, name: str) -> str | None:
        """Read a secret value from the managed backend."""

    def delete(self, name: str) -> None:
        """Delete a secret value from the managed backend when supported."""


class NoopManagedSecretAdapter(ManagedSecretAdapter):
    def put(self, name: str, value: str) -> None:
        raise RuntimeError(f"AIT_SECRET_PROVIDER={settings.secret_provider} does not support UI secret writes.")

    def get(self, name: str) -> str | None:
        return None

    def delete(self, name: str) -> None:
        return None


class VaultSecretAdapter(ManagedSecretAdapter):
    def __init__(self) -> None:
        if not settings.vault_addr or not settings.vault_token:
            raise RuntimeError("VAULT_ADDR and VAULT_TOKEN are required for Vault secret storage.")
        import hvac

        self.client = hvac.Client(url=settings.vault_addr, token=settings.vault_token)

    def put(self, name: str, value: str) -> None:
        self.client.secrets.kv.v2.create_or_update_secret(
            mount_point=settings.vault_mount,
            path=f"{settings.vault_path_prefix}/{name}",
            secret={"value": value},
        )

    def get(self, name: str) -> str | None:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                mount_point=settings.vault_mount,
                path=f"{settings.vault_path_prefix}/{name}",
            )
        except Exception:
            return None
        return response["data"]["data"].get("value")

    def delete(self, name: str) -> None:
        self.client.secrets.kv.v2.delete_metadata_and_all_versions(
            mount_point=settings.vault_mount,
            path=f"{settings.vault_path_prefix}/{name}",
        )


class AwsSecretsManagerAdapter(ManagedSecretAdapter):
    def __init__(self) -> None:
        import boto3

        session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
        self.client = session.client("secretsmanager")

    def put(self, name: str, value: str) -> None:
        secret_id = f"{settings.aws_secrets_prefix}/{name}"
        try:
            self.client.put_secret_value(SecretId=secret_id, SecretString=value)
        except self.client.exceptions.ResourceNotFoundException:
            self.client.create_secret(Name=secret_id, SecretString=value)

    def get(self, name: str) -> str | None:
        try:
            response = self.client.get_secret_value(SecretId=f"{settings.aws_secrets_prefix}/{name}")
        except Exception:
            return None
        return response.get("SecretString")

    def delete(self, name: str) -> None:
        self.client.delete_secret(
            SecretId=f"{settings.aws_secrets_prefix}/{name}",
            ForceDeleteWithoutRecovery=True,
        )


class AzureKeyVaultSecretAdapter(ManagedSecretAdapter):
    def __init__(self) -> None:
        if not settings.azure_key_vault_url:
            raise RuntimeError("AZURE_KEY_VAULT_URL is required for Azure Key Vault secret storage.")
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        self.client = SecretClient(
            vault_url=settings.azure_key_vault_url,
            credential=DefaultAzureCredential(),
        )

    def put(self, name: str, value: str) -> None:
        self.client.set_secret(name.replace("_", "-"), value)

    def get(self, name: str) -> str | None:
        try:
            return self.client.get_secret(name.replace("_", "-")).value
        except Exception:
            return None

    def delete(self, name: str) -> None:
        self.client.begin_delete_secret(name.replace("_", "-"))


class GcpSecretManagerAdapter(ManagedSecretAdapter):
    def __init__(self) -> None:
        if not settings.gcp_project_id:
            raise RuntimeError("GCP_PROJECT_ID is required for GCP Secret Manager storage.")
        from google.cloud import secretmanager

        self.client = secretmanager.SecretManagerServiceClient()

    def put(self, name: str, value: str) -> None:
        parent = f"projects/{settings.gcp_project_id}"
        secret_name = f"{parent}/secrets/{name}"
        try:
            self.client.get_secret(request={"name": secret_name})
        except Exception:
            self.client.create_secret(
                request={"parent": parent, "secret_id": name, "secret": {"replication": {"automatic": {}}}}
            )
        self.client.add_secret_version(
            request={"parent": secret_name, "payload": {"data": value.encode("utf-8")}}
        )

    def get(self, name: str) -> str | None:
        try:
            response = self.client.access_secret_version(
                request={"name": f"projects/{settings.gcp_project_id}/secrets/{name}/versions/latest"}
            )
        except Exception:
            return None
        return response.payload.data.decode("utf-8")

    def delete(self, name: str) -> None:
        self.client.delete_secret(
            request={"name": f"projects/{settings.gcp_project_id}/secrets/{name}"}
        )


def build_managed_secret_adapter() -> ManagedSecretAdapter:
    if settings.secret_provider == "vault":
        return VaultSecretAdapter()
    if settings.secret_provider == "aws":
        return AwsSecretsManagerAdapter()
    if settings.secret_provider == "azure":
        return AzureKeyVaultSecretAdapter()
    if settings.secret_provider == "gcp":
        return GcpSecretManagerAdapter()
    return NoopManagedSecretAdapter()
