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


def test_provider_readiness_endpoint_uses_selected_provider():
    client = TestClient(app)
    response = client.post(
        "/v1/providers/test",
        json={"kind": "mock", "model": "mock-optimizer"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["provider"] == "mock"


def test_model_catalog_returns_provider_models():
    client = TestClient(app)
    response = client.get("/v1/model-catalog")
    assert response.status_code == 200
    payload = response.json()
    assert any(model["id"] == "gpt-5.5" for model in payload["openai"])
    assert any(model["id"] == "gpt-5.4-mini" for model in payload["openai"])
    assert any(model["id"] == "gpt-4.1" for model in payload["openai"])
    assert any(model["id"] == "gpt-4o-mini" for model in payload["openai"])
    assert all("optimizer" in model["recommended_for"] for models in payload.values() for model in models)
    assert any("claude" in model["id"] for model in payload["aws_bedrock"])


def test_model_catalog_can_merge_live_provider_models(monkeypatch):
    from aiterate.api import main

    monkeypatch.setattr(main, "_discover_live_models", lambda: {"openai": [main._model_option("live-model-1", "live")]})
    client = TestClient(app)
    response = client.get("/v1/model-catalog?include_live=true")
    assert response.status_code == 200
    payload = response.json()
    assert any(model["id"] == "live-model-1" and model["source"] == "live" for model in payload["openai"])


def test_model_prices_returns_approximate_prices():
    client = TestClient(app)
    response = client.get("/v1/model-prices")
    assert response.status_code == 200
    payload = response.json()
    assert payload["openai"][0]["currency"] == "USD"
    assert any(price["model"] == "gpt-5.5" for price in payload["openai"])
    assert any(price["model"] == "gpt-4.1" for price in payload["openai"])


def test_optimization_job_lifecycle_endpoints():
    client = TestClient(app)
    enqueue_response = client.post(
        "/v1/optimization-jobs",
        json={
            "name": "job-api-demo",
            "raw_data": "Support answers must cite policy sources and escalate incomplete cases.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {"kind": "mock", "model": "mock-optimizer"},
        },
    )
    assert enqueue_response.status_code == 200
    job = enqueue_response.json()
    assert job["status"] == "queued"

    get_response = client.get(f"/v1/jobs/{job['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == job["id"]

    run_response = client.post("/v1/jobs/run-next")
    assert run_response.status_code == 200
    completed = run_response.json()
    assert completed["id"] == job["id"]
    assert completed["status"] == "succeeded"
    assert completed["result"]["best_version"]["content"]


def test_runs_endpoint_returns_grouped_runs():
    client = TestClient(app)
    response = client.get("/v1/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_unknown_api_route_returns_json_404_not_frontend_html():
    client = TestClient(app)
    response = client.get("/v1/not-a-real-route")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == "API route not found."


def test_run_detail_endpoint_returns_persisted_run():
    client = TestClient(app)
    optimize_response = client.post(
        "/v1/optimize",
        json={
            "name": "history-detail-demo",
            "raw_data": "Support answers must cite policy sources and escalate incomplete cases.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {"kind": "mock", "model": "mock-optimizer"},
        },
    )
    assert optimize_response.status_code == 200
    run_id = optimize_response.json()["id"]

    response = client.get(f"/v1/runs/{run_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == run_id
    assert payload["name"] == "history-detail-demo"
    assert payload["best_version"]["content"]


def test_delete_run_endpoint_removes_persisted_run():
    client = TestClient(app)
    optimize_response = client.post(
        "/v1/optimize",
        json={
            "name": "delete-run-demo",
            "raw_data": "Support answers must cite policy sources and escalate incomplete cases.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {"kind": "mock", "model": "mock-optimizer"},
        },
    )
    run_id = optimize_response.json()["id"]

    delete_response = client.delete(f"/v1/runs/{run_id}")

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert client.get(f"/v1/runs/{run_id}").status_code == 404


def test_delete_run_post_fallback_removes_persisted_run():
    client = TestClient(app)
    optimize_response = client.post(
        "/v1/optimize",
        json={
            "name": "delete-run-fallback-demo",
            "raw_data": "Support answers must cite policy sources and escalate incomplete cases.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {"kind": "mock", "model": "mock-optimizer"},
        },
    )
    run_id = optimize_response.json()["id"]

    delete_response = client.post(f"/v1/runs/{run_id}/delete")

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert client.get(f"/v1/runs/{run_id}").status_code == 404


def test_delete_run_post_compat_removes_persisted_run():
    client = TestClient(app)
    optimize_response = client.post(
        "/v1/optimize",
        json={
            "name": "delete-run-compat-demo",
            "raw_data": "Support answers must cite policy sources and escalate incomplete cases.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {"kind": "mock", "model": "mock-optimizer"},
        },
    )
    run_id = optimize_response.json()["id"]

    delete_response = client.post(f"/v1/runs/{run_id}", json={"action": "delete"})

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert client.get(f"/v1/runs/{run_id}").status_code == 404


def test_delete_project_runs_removes_grouped_history():
    client = TestClient(app)
    project_name = "delete-project-demo"
    for _ in range(2):
        response = client.post(
            "/v1/optimize",
            json={
                "name": project_name,
                "raw_data": "Support answers must cite policy sources and escalate incomplete cases.",
                "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
                "provider": {"kind": "mock", "model": "mock-optimizer"},
            },
        )
        assert response.status_code == 200

    delete_response = client.post(f"/v1/projects/{project_name}/delete")

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted_runs"] == 2
    assert project_name not in client.get("/v1/runs").json()


def test_integration_setup_endpoint_hides_secret_values():
    client = TestClient(app)
    response = client.get("/v1/integrations/setup")
    assert response.status_code == 200
    payload = response.json()
    assert payload["secret_policy"]["where"] == "server"
    assert "OPENAI_API_KEY" in payload["providers"][0]["env"]
    assert any("ANTHROPIC_API_KEY" in provider["env"] for provider in payload["providers"])
    assert any("MLFLOW_TRACKING_TOKEN" in tracker["env"] for tracker in payload["tracking"])
    assert any("LANGSMITH_API_KEY" in tracker["env"] for tracker in payload["tracking"])


def test_approve_run_endpoint():
    client = TestClient(app)
    response = client.post(
        "/v1/runs/run_123/approve",
        json={"artifact_id": "art_123", "version_id": "ver_123", "approved_by": "tester"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"


def test_approved_run_is_marked_in_history():
    client = TestClient(app)
    optimize_response = client.post(
        "/v1/optimize",
        json={
            "name": "approved-history-demo",
            "raw_data": "Support answers must cite policy sources and escalate incomplete cases.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {"kind": "mock", "model": "mock-optimizer"},
        },
    )
    run = optimize_response.json()

    approval_response = client.post(
        f"/v1/runs/{run['id']}/approve",
        json={
            "artifact_id": run["artifact_id"],
            "version_id": run["best_version"]["id"],
            "approved_by": "tester",
        },
    )

    assert approval_response.status_code == 200
    history = client.get("/v1/runs").json()
    saved = next(item for item in history["approved-history-demo"] if item["id"] == run["id"])
    assert saved["approved"] is True
    assert saved["approved_version_id"] == run["best_version"]["id"]


def test_git_browser_auth_start_reports_missing_oauth_config(monkeypatch):
    monkeypatch.setattr(settings, "github_oauth_client_id", None)
    monkeypatch.setattr(settings, "github_oauth_client_secret", None)
    client = TestClient(app)
    response = client.get("/v1/git/auth/github/start")
    assert response.status_code == 200
    assert response.json()["status"] == "not_configured"


def test_git_browser_auth_start_returns_authorize_url(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "github_oauth_client_id", "client-id")
    monkeypatch.setattr(settings, "github_oauth_client_secret", "client-secret")
    monkeypatch.setattr(settings, "public_base_url", "http://localhost:5173")
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    client = TestClient(app)
    response = client.get("/v1/git/auth/github/start")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "github.com/login/oauth/authorize" in payload["auth_url"]
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A5173%2Fv1%2Fgit%2Fauth%2Fgithub%2Fcallback" in payload["auth_url"]


def test_secret_roundtrip_hides_value():
    client = TestClient(app)
    response = client.post(
        "/v1/secrets",
        json={"name": "TEST_SECRET", "integration": "Test", "value": "secret-value-123"},
    )
    assert response.status_code == 200
    assert response.json()["fingerprint"] == "****...-123"

    list_response = client.get("/v1/secrets")
    assert list_response.status_code == 200
    assert "secret-value-123" not in list_response.text


def test_secret_rejects_placeholder_api_key():
    client = TestClient(app)
    response = client.post(
        "/v1/secrets",
        json={"name": "OPENAI_API_KEY", "integration": "OpenAI", "value": "OPENAI_API_KEY "},
    )
    assert response.status_code == 400
    assert "actual secret value" in response.json()["detail"]["message"]


def test_secret_save_trims_wrapping_whitespace():
    client = TestClient(app)
    response = client.post(
        "/v1/secrets",
        json={"name": "OPENAI_API_KEY", "integration": "OpenAI", "value": " sk-test-value \n"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "OPENAI_API_KEY"


def test_secret_save_allows_provider_to_validate_non_placeholder_value():
    client = TestClient(app)
    response = client.post(
        "/v1/secrets",
        json={"name": "OPENAI_API_KEY", "integration": "OpenAI", "value": "sk-test value"},
    )
    assert response.status_code == 200


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
    assert payload["cost_estimate"]["total_cost"] >= 0
    assert payload["cost_estimate"]["approximate"] is True


def test_optimize_returns_provider_error_for_placeholder_key():
    client = TestClient(app)
    response = client.post(
        "/v1/optimize",
        json={
            "name": "bad-key-demo",
            "raw_data": "Answers must cite sources and escalate incomplete data.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "provider": {
                "kind": "openai",
                "model": "gpt-4.1",
                "api_key": "OPENAI_API_KEY ",
            },
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["section"] == "models"
    assert response.json()["detail"]["secret_name"] == "OPENAI_API_KEY"
    assert "actual API key value" in response.json()["detail"]["message"]


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


def test_compare_models_uses_model_specific_fit_for_offline_scores():
    client = TestClient(app)
    response = client.post(
        "/v1/compare-models",
        json={
            "prompt": "Always cite sources and escalate incomplete data.",
            "raw_data": "A support answer needs citations.",
            "policies": [{"id": "cite", "text": "Always cite sources.", "weight": 1}],
            "model_a": {"kind": "openai", "model": "gpt-5.5"},
            "model_b": {"kind": "openai", "model": "gpt-4o-mini-2024-07-18"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    scores = [result["score"] for result in payload["results"]]
    assert scores[0] > scores[1]
    assert "Offline comparison" in payload["results"][0]["risks"][0]


def test_project_settings_persist_git_defaults():
    client = TestClient(app)
    response = client.put(
        "/v1/projects/settings-demo/settings",
        json={
            "project_name": "ignored-client-name",
            "enable_git_tracking": True,
            "git_remote": "https://github.com/org/repo.git",
            "artifact_branch": "artifacts",
            "enable_promotion_pr_workflow": True,
            "pr_base_branch": "main",
        },
    )
    assert response.status_code == 200
    assert response.json()["project_name"] == "settings-demo"

    loaded = client.get("/v1/projects/settings-demo/settings")
    assert loaded.status_code == 200
    payload = loaded.json()
    assert payload["git_remote"] == "https://github.com/org/repo.git"
    assert payload["artifact_branch"] == "artifacts"
    assert payload["enable_promotion_pr_workflow"] is True
