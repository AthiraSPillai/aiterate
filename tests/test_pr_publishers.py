import json
import hashlib

from aiterate.pr_publishers import _friendly_provider_error, parse_git_remote, spec_from_payload


def test_parse_github_https_remote():
    parsed = parse_git_remote("https://github.com/AthiraSPillai/aiterate.git")

    assert parsed == {"owner": "AthiraSPillai", "repo": "aiterate"}


def test_parse_github_ssh_remote():
    parsed = parse_git_remote("git@github.com:AthiraSPillai/aiterate.git")

    assert parsed == {"owner": "AthiraSPillai", "repo": "aiterate"}


def test_spec_from_payload_maps_ui_fields_and_avoids_main_to_main_pr():
    provider, spec = spec_from_payload(
        {
            "remote": "https://github.com/AthiraSPillai/aiterate.git",
            "branch": "main",
            "base": "main",
            "artifact_id": "art_123",
            "version_id": "ver_123",
            "artifact_content": "approved prompt",
            "run_id": "run_123",
            "run_json": json.dumps(
                {
                    "id": "run_123",
                    "name": "support-agent",
                    "artifact_id": "art_123",
                    "dataset": {
                        "id": "data_123",
                        "name": "support-agent",
                        "content_hash": "hash-data",
                        "raw_text": "raw tickets",
                        "normalized_cases": ["case one"],
                    },
                    "policy_set": {
                        "id": "pol_123",
                        "rules": [{"id": "cite", "text": "Cite sources.", "weight": 1}],
                    },
                    "optimizer": {
                        "framework": "skillopt",
                        "policy_context_hash": "hash-policy",
                        "knowledge_base_hash": "hash-kb",
                        "policy_context": "policy source text",
                        "knowledge_base_context": "kb source text",
                    },
                    "best_version": {
                        "id": "ver_123",
                        "version": 2,
                        "kind": "prompt",
                        "score": 0.91,
                        "change_summary": "SkillOpt accept: applied bounded edit operations.",
                        "policy_hash": "hash-policy-set",
                    },
                    "accepted_versions": [],
                    "rejected_versions": [],
                }
            ),
        }
    )

    assert provider == "github"
    assert spec.owner == "AthiraSPillai"
    assert spec.repo == "aiterate"
    assert spec.source_branch == "aiterate/promote-art-123"
    assert spec.target_branch == "main"
    assert spec.files["aiterate/artifacts/art_123/versions/ver_123/artifact.md"] == "approved prompt"
    assert spec.files["aiterate/sources/run_123/data/raw_data.txt"] == "raw tickets"
    assert spec.files["aiterate/sources/run_123/policies/policy_context.txt"] == "policy source text"
    assert spec.files["aiterate/sources/run_123/knowledge/knowledge_base.txt"] == "kb source text"
    raw_sha = hashlib.sha256("raw tickets".encode("utf-8")).hexdigest()
    policy_sha = hashlib.sha256("policy source text".encode("utf-8")).hexdigest()
    kb_sha = hashlib.sha256("kb source text".encode("utf-8")).hexdigest()
    assert spec.files[f"aiterate/immutable/sources/data/{raw_sha}/raw_data.txt"] == "raw tickets"
    assert (
        spec.files[f"aiterate/immutable/sources/policies/{policy_sha}/policy_context.txt"]
        == "policy source text"
    )
    assert (
        spec.files[f"aiterate/immutable/sources/knowledge/{kb_sha}/knowledge_base.txt"]
        == "kb source text"
    )
    assert "aiterate/dvc/run_123/data/raw_data.txt.dvc" in spec.files
    assert f"sha256: {raw_sha}" in spec.files["aiterate/dvc/run_123/data/raw_data.txt.dvc"]
    assert "aiterate/artifacts/art_123/versions/ver_123/run.redacted.json" in spec.files
    assert "aiterate/runs/run_123.redacted.json" in spec.files
    assert "aiterate/sources/** filter=lfs" in spec.files[".gitattributes"]
    assert "aiterate/immutable/sources/** filter=lfs" in spec.files[".gitattributes"]
    metadata = json.loads(spec.files["aiterate/artifacts/art_123/versions/ver_123/metadata.json"])
    assert metadata["artifact"]["version_id"] == "ver_123"
    assert metadata["data_sources"]["dataset_hash"] == "hash-data"
    assert metadata["data_sources"]["raw_data_path"] == "aiterate/sources/run_123/data/raw_data.txt"
    assert (
        metadata["data_sources"]["immutable_raw_data_path"]
        == f"aiterate/immutable/sources/data/{raw_sha}/raw_data.txt"
    )
    assert metadata["data_sources"]["dvc_pointer_path"] == "aiterate/dvc/run_123/data/raw_data.txt.dvc"
    assert metadata["policy_sources"]["policy_context_hash"] == "hash-policy"
    assert metadata["policy_sources"]["policy_context_path"] == "aiterate/sources/run_123/policies/policy_context.txt"
    assert (
        metadata["policy_sources"]["immutable_policy_context_path"]
        == f"aiterate/immutable/sources/policies/{policy_sha}/policy_context.txt"
    )
    assert metadata["knowledge_sources"]["knowledge_base_hash"] == "hash-kb"
    assert metadata["knowledge_sources"]["knowledge_base_path"] == "aiterate/sources/run_123/knowledge/knowledge_base.txt"
    assert (
        metadata["knowledge_sources"]["immutable_knowledge_base_path"]
        == f"aiterate/immutable/sources/knowledge/{kb_sha}/knowledge_base.txt"
    )
    assert metadata["artifact"]["prompt_change_meaning"]
    manifest = json.loads(spec.files["aiterate/artifacts/art_123/versions/ver_123/source_manifest.json"])
    assert manifest["data_examples"]["path"] == "aiterate/sources/run_123/data/raw_data.txt"
    assert manifest["data_examples"]["immutable"]["sha256"] == raw_sha
    assert (
        manifest["data_examples"]["immutable"]["immutable_path"]
        == f"aiterate/immutable/sources/data/{raw_sha}/raw_data.txt"
    )
    assert manifest["data_examples"]["immutable"]["dvc_pointer_path"] == "aiterate/dvc/run_123/data/raw_data.txt.dvc"
    assert "Git LFS" in manifest["storage_modes"]["git_lfs"]
    assert "DVC" in manifest["storage_modes"]["dvc"]


def test_github_token_permission_error_is_actionable():
    message = _friendly_provider_error({"message": "Resource not accessible by personal access token"})

    assert "Contents read/write" in message
    assert "Pull requests read/write" in message
    assert "Commit statuses permission is not enough" in message
