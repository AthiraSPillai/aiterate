from fastapi.testclient import TestClient

from aiterate.api.main import app
from aiterate.config import settings
from aiterate.domain import OptimizationRequest, ProviderConfig, ProviderKind
from aiterate.secrets import SecretInput, SecretStore


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_providers_include_native_bedrock():
    client = TestClient(app)
    response = client.get("/v1/providers")
    assert "aws_bedrock" in response.json()["native"]
    assert "anthropic" in response.json()["native"]


def test_runs_endpoint_returns_grouped_runs():
    client = TestClient(app)
    response = client.get("/v1/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_integration_setup_endpoint_hides_secret_values():
    client = TestClient(app)
    response = client.get("/v1/integrations/setup")
    assert response.status_code == 200
    payload = response.json()
    assert payload["secret_policy"]["where"] == "server"
    assert "OPENAI_API_KEY" in payload["providers"][0]["env"]
    assert any("ANTHROPIC_API_KEY" in provider["env"] for provider in payload["providers"])


def test_approve_run_endpoint():
    client = TestClient(app)
    response = client.post(
        "/v1/runs/run_123/approve",
        json={"artifact_id": "art_123", "version_id": "ver_123", "approved_by": "tester"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"


def test_secret_roundtrip_hides_value():
    client = TestClient(app)
    response = client.post(
        "/v1/secrets",
        json={"name": "TEST_SECRET", "integration": "Test", "value": "secret-value-123"},
    )
    assert response.status_code == 200
    assert response.json()["fingerprint"] == "secr...-123"

    list_response = client.get("/v1/secrets")
    assert list_response.status_code == 200
    assert "secret-value-123" not in list_response.text


def test_secret_delete_removes_metadata():
    client = TestClient(app)
    response = client.post(
        "/v1/secrets",
        json={"name": "DELETE_ME", "integration": "Test", "value": "secret-value-456"},
    )
    assert response.status_code == 200

    delete_response = client.delete("/v1/secrets/DELETE_ME")
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"

    list_response = client.get("/v1/secrets")
    assert "DELETE_ME" not in list_response.text


def test_stored_provider_secret_is_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    store = SecretStore()
    store.upsert(SecretInput(name="OPENAI_API_KEY", integration="OpenAI", value="sk-test-value"))
    request = OptimizationRequest(
        name="demo",
        raw_data="Enough raw data to build a useful prompt.",
        provider=ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-test"),
    )

    from aiterate.api.main import _with_stored_provider_secret

    resolved = _with_stored_provider_secret(request)
    assert resolved.provider.api_key is not None
    assert resolved.provider.api_key.get_secret_value() == "sk-test-value"


def test_optimize_returns_eval_insights():
    client = TestClient(app)
    response = client.post(
        "/v1/optimize",
        json={
            "name": "insight-demo",
            "raw_data": "Answers must cite sources and escalate incomplete data.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {"kind": "mock", "model": "mock-optimizer"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["insights"]["worked"]
    assert payload["insights"]["prompt_changes_needed"]


def test_compare_models_same_prompt():
    client = TestClient(app)
    response = client.post(
        "/v1/compare-models",
        json={
            "prompt": "Always cite sources and escalate incomplete data.",
            "raw_data": "A support answer needs citations.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "model_a": {"kind": "openai", "model": "gpt-test"},
            "model_b": {"kind": "anthropic", "model": "claude-test"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["winner"]
