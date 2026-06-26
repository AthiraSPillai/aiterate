from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import requests

from aiterate.config import settings
from aiterate.secrets import SecretStore


@dataclass(frozen=True)
class PullRequestSpec:
    title: str
    body: str
    source_branch: str
    target_branch: str = "main"
    owner: str | None = None
    repo: str | None = None
    workspace: str | None = None
    repo_slug: str | None = None
    files: dict[str, str] | None = None


class PullRequestPublisher:
    def publish(self, spec: PullRequestSpec) -> dict[str, Any]:
        raise NotImplementedError


class GitHubPullRequestPublisher(PullRequestPublisher):
    def publish(self, spec: PullRequestSpec) -> dict[str, Any]:
        token = settings.github_token or SecretStore().get_value("GITHUB_TOKEN")
        if not token:
            return {"status": "not_configured", "message": "GITHUB_TOKEN is required to publish GitHub PRs."}
        if not spec.owner or not spec.repo:
            return {
                "status": "invalid_request",
                "message": "GitHub owner and repo are required. Enter a GitHub remote like https://github.com/owner/repo.git.",
            }
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if spec.files:
            setup = _prepare_github_branch_and_files(spec, headers)
            if setup["status"] == "failed":
                return setup

        response = requests.post(
            f"https://api.github.com/repos/{spec.owner}/{spec.repo}/pulls",
            headers=headers,
            json={
                "title": spec.title,
                "body": spec.body,
                "head": spec.source_branch,
                "base": spec.target_branch,
                "maintainer_can_modify": True,
            },
            timeout=30,
        )
        return _response_payload(response)


class BitbucketPullRequestPublisher(PullRequestPublisher):
    def publish(self, spec: PullRequestSpec) -> dict[str, Any]:
        token = settings.bitbucket_token or SecretStore().get_value("BITBUCKET_TOKEN")
        workspace = spec.workspace or settings.bitbucket_workspace
        repo_slug = spec.repo_slug or settings.bitbucket_repo_slug
        if not token:
            return {"status": "not_configured", "message": "BITBUCKET_TOKEN is required to publish Bitbucket PRs."}
        if not workspace or not repo_slug:
            return {"status": "invalid_request", "message": "Bitbucket workspace and repo slug are required."}

        response = requests.post(
            f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/pullrequests",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            json={
                "title": spec.title,
                "description": spec.body,
                "source": {"branch": {"name": spec.source_branch}},
                "destination": {"branch": {"name": spec.target_branch}},
                "close_source_branch": False,
            },
            timeout=30,
        )
        return _response_payload(response)


def build_pull_request_publisher(kind: str) -> PullRequestPublisher:
    if kind == "bitbucket":
        return BitbucketPullRequestPublisher()
    return GitHubPullRequestPublisher()


def infer_provider_from_remote(remote: str | None) -> str:
    if not remote:
        return "github"
    remote_lower = remote.lower()
    if "bitbucket.org" in remote_lower:
        return "bitbucket"
    return "github"


def spec_from_payload(payload: dict[str, Any]) -> tuple[str, PullRequestSpec]:
    remote = str(payload.get("remote") or "").strip()
    provider = str(payload.get("provider") or infer_provider_from_remote(remote)).lower()
    parsed = parse_git_remote(remote)
    source_branch = str(
        payload.get("source_branch")
        or payload.get("head")
        or payload.get("branch")
        or "aiterate-promotion"
    )
    target_branch = str(payload.get("target_branch") or payload.get("base") or "main")
    artifact_id = str(payload.get("artifact_id") or "artifact")
    if source_branch == target_branch:
        source_branch = f"aiterate/promote-{artifact_id.replace('_', '-')}"
    files = _files_from_payload(payload)
    spec = PullRequestSpec(
        title=payload.get("title") or "Promote Aiterate artifact",
        body=payload.get("body") or "Promote approved Aiterate artifact version.",
        source_branch=source_branch,
        target_branch=target_branch,
        owner=payload.get("owner") or parsed.get("owner"),
        repo=payload.get("repo") or parsed.get("repo"),
        workspace=payload.get("workspace") or parsed.get("workspace"),
        repo_slug=payload.get("repo_slug") or parsed.get("repo_slug"),
        files=files,
    )
    return provider, spec


def parse_git_remote(remote: str | None) -> dict[str, str]:
    if not remote:
        return {}
    value = remote.strip()
    patterns = [
        r"github\.com[:/](?P<owner>[^/\s:]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$",
        r"bitbucket\.org[:/](?P<workspace>[^/\s:]+)/(?P<repo_slug>[^/\s]+?)(?:\.git)?/?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if not match:
            continue
        parsed = {key: item for key, item in match.groupdict().items() if item}
        if parsed.get("repo"):
            parsed["repo"] = parsed["repo"].removesuffix(".git")
        if parsed.get("repo_slug"):
            parsed["repo_slug"] = parsed["repo_slug"].removesuffix(".git")
        return parsed
    return {}


def _files_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    files: dict[str, str] = {}
    artifact_content = payload.get("artifact_content")
    artifact_id = str(payload.get("artifact_id") or "artifact")
    run_json = payload.get("run_json")
    run_payload = _parse_json_object(run_json)
    best_version = run_payload.get("best_version") if isinstance(run_payload.get("best_version"), dict) else {}
    version_id = str(payload.get("version_id") or best_version.get("id") or "version")
    run_id = str(payload.get("run_id") or run_payload.get("id") or "run")
    package_dir = f"aiterate/artifacts/{artifact_id}/versions/{version_id}"
    source_dir = f"aiterate/sources/{run_id}"
    immutable_sources = _immutable_source_entries(run_payload)
    if artifact_content:
        files[f"{package_dir}/artifact.md"] = str(artifact_content)
    files.update(_raw_source_files(run_payload, source_dir))
    files.update(_immutable_source_files(immutable_sources))
    files.update(_dvc_pointer_files(run_id, immutable_sources))
    source_manifest = _source_manifest(run_payload)
    files[f"{package_dir}/metadata.json"] = json.dumps(
        _promotion_metadata(payload, run_payload),
        indent=2,
        sort_keys=True,
    )
    files[f"{package_dir}/source_manifest.json"] = json.dumps(
        source_manifest,
        indent=2,
        sort_keys=True,
    )
    if run_json:
        redacted_run = _redacted_run(run_payload)
        files[f"{package_dir}/run.redacted.json"] = json.dumps(redacted_run, indent=2, sort_keys=True)
        files[f"aiterate/runs/{run_id}.redacted.json"] = json.dumps(redacted_run, indent=2, sort_keys=True)
    files.setdefault(
        ".gitattributes",
        (
            "aiterate/sources/** filter=lfs diff=lfs merge=lfs -text\n"
            "aiterate/immutable/sources/** filter=lfs diff=lfs merge=lfs -text\n"
        ),
    )
    return files


def _promotion_metadata(payload: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    optimizer = run.get("optimizer") if isinstance(run.get("optimizer"), dict) else {}
    policy_set = run.get("policy_set") if isinstance(run.get("policy_set"), dict) else {}
    best_version = run.get("best_version") if isinstance(run.get("best_version"), dict) else {}
    evaluation_report = run.get("evaluation_report") if isinstance(run.get("evaluation_report"), dict) else {}
    behavior_report = run.get("behavior_report") if isinstance(run.get("behavior_report"), dict) else None
    accepted_versions = run.get("accepted_versions") if isinstance(run.get("accepted_versions"), list) else []
    rejected_versions = run.get("rejected_versions") if isinstance(run.get("rejected_versions"), list) else []
    immutable_sources = _immutable_source_entries(run)
    return {
        "schema_version": "aiterate.promotion.metadata.v1",
        "product": "Aiterate",
        "purpose": "Traceable promotion package for an approved prompt or agent skill artifact.",
        "promotion": {
            "status": payload.get("approval_status") or "approved",
            "approved_by": payload.get("approved_by") or "local-user",
            "target_branch": payload.get("base") or payload.get("target_branch") or "main",
            "source_branch": payload.get("source_branch") or payload.get("branch") or "aiterate-promotion",
        },
        "artifact": {
            "artifact_id": payload.get("artifact_id") or run.get("artifact_id"),
            "version_id": payload.get("version_id") or best_version.get("id"),
            "version_number": best_version.get("version"),
            "kind": best_version.get("kind"),
            "score": best_version.get("score"),
            "change_summary": best_version.get("change_summary"),
            "prompt_change_meaning": (
                "This change summary describes the bounded prompt/skill edit that survived "
                "validation scoring and approval. Rejected edits are retained below for audit."
            ),
            "content_sha_source": "artifact.md in this same folder",
        },
        "run": {
            "run_id": payload.get("run_id") or run.get("id"),
            "name": run.get("name"),
            "created_at": run.get("created_at"),
        },
        "data_sources": {
            "dataset_id": dataset.get("id"),
            "dataset_name": dataset.get("name"),
            "dataset_hash": dataset.get("content_hash"),
            "raw_data_path": f"aiterate/sources/{payload.get('run_id') or run.get('id') or 'run'}/data/raw_data.txt",
            "immutable_raw_data_path": immutable_sources.get("data_examples", {}).get("immutable_path"),
            "dvc_pointer_path": immutable_sources.get("data_examples", {}).get("dvc_pointer_path"),
            "normalized_case_count": len(dataset.get("normalized_cases") or []),
            "normalized_case_examples": (dataset.get("normalized_cases") or [])[:5],
            "raw_data_excerpt": _excerpt(dataset.get("raw_text")),
        },
        "policy_sources": {
            "policy_set_id": policy_set.get("id"),
            "policy_hash": best_version.get("policy_hash"),
            "policy_context_hash": optimizer.get("policy_context_hash"),
            "policy_context_path": f"aiterate/sources/{payload.get('run_id') or run.get('id') or 'run'}/policies/policy_context.txt",
            "immutable_policy_context_path": immutable_sources.get("policy_sources", {}).get("immutable_path"),
            "dvc_pointer_path": immutable_sources.get("policy_sources", {}).get("dvc_pointer_path"),
            "policy_context_excerpt": _excerpt(optimizer.get("policy_context")),
            "weighted_rules": policy_set.get("rules") or [],
        },
        "knowledge_sources": {
            "knowledge_base_hash": optimizer.get("knowledge_base_hash"),
            "knowledge_base_path": f"aiterate/sources/{payload.get('run_id') or run.get('id') or 'run'}/knowledge/knowledge_base.txt",
            "immutable_knowledge_base_path": immutable_sources.get("knowledge_sources", {}).get("immutable_path"),
            "dvc_pointer_path": immutable_sources.get("knowledge_sources", {}).get("dvc_pointer_path"),
            "knowledge_base_excerpt": _excerpt(optimizer.get("knowledge_base_context")),
            "role": "Grounding and source-reference context for generated artifact behavior.",
        },
        "models_and_providers": {
            "optimizer_provider": run.get("provider"),
            "target_model": optimizer.get("target_model"),
            "target_validation_enabled": optimizer.get("run_target_validation"),
        },
        "optimization": {
            "framework": optimizer.get("framework"),
            "wrapper": optimizer.get("wrapper"),
            "loop": optimizer.get("loop"),
            "seed": optimizer.get("seed"),
            "iterations_requested": optimizer.get("iterations") or optimizer.get("requested_iterations"),
            "validation_split": optimizer.get("validation_split"),
            "train_case_count": optimizer.get("train_case_count"),
            "validation_case_count": optimizer.get("validation_case_count"),
            "promotion_threshold": optimizer.get("promotion_threshold"),
            "max_budget_usd": optimizer.get("max_budget_usd"),
        },
        "evaluation": {
            "report": evaluation_report,
            "target_model_behavior_report": behavior_report,
            "cost_estimate": run.get("cost_estimate"),
            "insights": run.get("insights"),
        },
        "lineage": {
            "accepted_versions": [_version_summary(version) for version in accepted_versions],
            "rejected_candidates": [_version_summary(version) for version in rejected_versions],
        },
        "limitations": [
            "Scores are evaluation signals, not a guarantee of production safety.",
            "Approximate cost uses configured model prices and may differ from provider billing.",
            "Manual approval should include domain review for regulated or high-risk use cases.",
            "Raw source snapshots are linked by run path, immutable hash-addressed path, and hash. For large files, configure Git LFS or DVC in the repository/CI workflow.",
        ],
    }


def _raw_source_files(run: dict[str, Any], source_dir: str) -> dict[str, str]:
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    optimizer = run.get("optimizer") if isinstance(run.get("optimizer"), dict) else {}
    files: dict[str, str] = {}
    raw_data = str(dataset.get("raw_text") or "")
    policy_context = str(optimizer.get("policy_context") or "")
    knowledge_base = str(optimizer.get("knowledge_base_context") or "")
    if raw_data:
        files[f"{source_dir}/data/raw_data.txt"] = raw_data
    if policy_context:
        files[f"{source_dir}/policies/policy_context.txt"] = policy_context
    if knowledge_base:
        files[f"{source_dir}/knowledge/knowledge_base.txt"] = knowledge_base
    return files


def _immutable_source_entries(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    optimizer = run.get("optimizer") if isinstance(run.get("optimizer"), dict) else {}
    run_id = str(run.get("id") or "run")
    sources = {
        "data_examples": {
            "text": str(dataset.get("raw_text") or ""),
            "declared_hash": dataset.get("content_hash"),
            "filename": "raw_data.txt",
            "kind": "data",
        },
        "policy_sources": {
            "text": str(optimizer.get("policy_context") or ""),
            "declared_hash": optimizer.get("policy_context_hash"),
            "filename": "policy_context.txt",
            "kind": "policies",
        },
        "knowledge_sources": {
            "text": str(optimizer.get("knowledge_base_context") or ""),
            "declared_hash": optimizer.get("knowledge_base_hash"),
            "filename": "knowledge_base.txt",
            "kind": "knowledge",
        },
    }
    entries: dict[str, dict[str, Any]] = {}
    for name, source in sources.items():
        text = source["text"]
        if not text:
            continue
        sha256 = _sha256(text)
        md5 = _md5(text)
        kind = source["kind"]
        filename = source["filename"]
        entries[name] = {
            "hash_algorithm": "sha256",
            "sha256": sha256,
            "md5": md5,
            "declared_hash": source["declared_hash"],
            "byte_size": len(text.encode("utf-8")),
            "immutable_path": f"aiterate/immutable/sources/{kind}/{sha256}/{filename}",
            "run_snapshot_path": f"aiterate/sources/{run_id}/{kind}/{filename}",
            "dvc_pointer_path": f"aiterate/dvc/{run_id}/{kind}/{filename}.dvc",
            "text": text,
        }
    return entries


def _immutable_source_files(entries: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        str(entry["immutable_path"]): str(entry["text"])
        for entry in entries.values()
        if entry.get("immutable_path") and entry.get("text")
    }


def _dvc_pointer_files(run_id: str, entries: dict[str, dict[str, Any]]) -> dict[str, str]:
    files: dict[str, str] = {}
    for entry in entries.values():
        pointer_path = entry.get("dvc_pointer_path")
        immutable_path = entry.get("immutable_path")
        if not pointer_path or not immutable_path:
            continue
        files[str(pointer_path)] = "\n".join(
            [
                "outs:",
                f"- md5: {entry['md5']}",
                f"  size: {entry['byte_size']}",
                f"  path: ../../../../{immutable_path}",
                f"  meta: {{aiterate_run_id: {run_id}, sha256: {entry['sha256']}}}",
                "",
            ]
        )
    return files


def _source_manifest(run: dict[str, Any]) -> dict[str, Any]:
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    optimizer = run.get("optimizer") if isinstance(run.get("optimizer"), dict) else {}
    run_id = run.get("id") or "run"
    immutable_sources = _immutable_source_entries(run)
    return {
        "schema_version": "aiterate.source_manifest.v1",
        "strategy": (
            "Promotion packages include raw source snapshots under aiterate/sources/<run_id>/ "
            "for review and immutable hash-addressed copies under aiterate/immutable/sources/. "
            "Metadata links both paths by SHA-256. DVC pointer files are emitted under "
            "aiterate/dvc/<run_id>/ for teams that use DVC remotes."
        ),
        "storage_modes": {
            "github_contents_api": "writes raw source snapshots as regular Git blobs",
            "immutable_git": "review immutable copies under aiterate/immutable/sources/<kind>/<sha256>/",
            "git_lfs": "configure .gitattributes for aiterate/sources/** and aiterate/immutable/sources/** through a Git LFS-aware client or CI",
            "dvc": "use emitted .dvc pointer files under aiterate/dvc/<run_id>/ and configure your own DVC remote/object store",
        },
        "data_examples": {
            "hash": dataset.get("content_hash"),
            "path": f"aiterate/sources/{run_id}/data/raw_data.txt",
            "immutable": _manifest_source_entry(immutable_sources.get("data_examples")),
            "case_count": len(dataset.get("normalized_cases") or []),
            "sample_cases": (dataset.get("normalized_cases") or [])[:5],
            "raw_text_length": len(str(dataset.get("raw_text") or "")),
            "raw_text_excerpt": _excerpt(dataset.get("raw_text")),
        },
        "policy_sources": {
            "hash": optimizer.get("policy_context_hash"),
            "path": f"aiterate/sources/{run_id}/policies/policy_context.txt",
            "immutable": _manifest_source_entry(immutable_sources.get("policy_sources")),
            "text_length": len(str(optimizer.get("policy_context") or "")),
            "excerpt": _excerpt(optimizer.get("policy_context")),
        },
        "knowledge_sources": {
            "hash": optimizer.get("knowledge_base_hash"),
            "path": f"aiterate/sources/{run_id}/knowledge/knowledge_base.txt",
            "immutable": _manifest_source_entry(immutable_sources.get("knowledge_sources")),
            "text_length": len(str(optimizer.get("knowledge_base_context") or "")),
            "excerpt": _excerpt(optimizer.get("knowledge_base_context")),
        },
    }


def _manifest_source_entry(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry:
        return None
    return {
        "hash_algorithm": entry.get("hash_algorithm"),
        "sha256": entry.get("sha256"),
        "md5_for_dvc_pointer": entry.get("md5"),
        "declared_hash": entry.get("declared_hash"),
        "byte_size": entry.get("byte_size"),
        "immutable_path": entry.get("immutable_path"),
        "dvc_pointer_path": entry.get("dvc_pointer_path"),
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def _redacted_run(run: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(run))
    dataset = redacted.get("dataset")
    if isinstance(dataset, dict):
        dataset["raw_text"] = _redaction_marker(dataset.get("raw_text"))
    optimizer = redacted.get("optimizer")
    if isinstance(optimizer, dict):
        optimizer["policy_context"] = _redaction_marker(optimizer.get("policy_context"))
        optimizer["knowledge_base_context"] = _redaction_marker(optimizer.get("knowledge_base_context"))
    return redacted


def _redaction_marker(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return f"[omitted from Git package: {len(text)} characters; see source_manifest.json for hash and excerpt]"


def _version_summary(version: Any) -> dict[str, Any]:
    if not isinstance(version, dict):
        return {}
    return {
        "id": version.get("id"),
        "version": version.get("version"),
        "accepted": version.get("accepted"),
        "score": version.get("score"),
        "parent_version_id": version.get("parent_version_id"),
        "change_summary": version.get("change_summary"),
        "created_at": version.get("created_at"),
        "metadata": version.get("metadata"),
    }


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _excerpt(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _prepare_github_branch_and_files(spec: PullRequestSpec, headers: dict[str, str]) -> dict[str, Any]:
    base_url = f"https://api.github.com/repos/{spec.owner}/{spec.repo}"
    base_ref = requests.get(
        f"{base_url}/git/ref/heads/{spec.target_branch}",
        headers=headers,
        timeout=30,
    )
    if not base_ref.ok:
        return _response_payload(base_ref)
    base_sha = base_ref.json().get("object", {}).get("sha")
    branch_ref = requests.get(
        f"{base_url}/git/ref/heads/{spec.source_branch}",
        headers=headers,
        timeout=30,
    )
    if branch_ref.status_code == 404:
        created = requests.post(
            f"{base_url}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{spec.source_branch}", "sha": base_sha},
            timeout=30,
        )
        if not created.ok:
            return _response_payload(created)
    elif not branch_ref.ok:
        return _response_payload(branch_ref)

    for path, content in (spec.files or {}).items():
        existing = requests.get(
            f"{base_url}/contents/{path}",
            headers=headers,
            params={"ref": spec.source_branch},
            timeout=30,
        )
        payload = {
            "message": f"Promote Aiterate artifact: {path}",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": spec.source_branch,
        }
        if existing.ok:
            payload["sha"] = existing.json().get("sha")
        elif existing.status_code != 404:
            return _response_payload(existing)
        updated = requests.put(
            f"{base_url}/contents/{path}",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if not updated.ok:
            return _response_payload(updated)
    return {"status": "ready"}


def _response_payload(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text}
    if response.ok:
        return {
            "status": "published",
            "url": payload.get("html_url") or payload.get("links", {}).get("html", {}).get("href"),
            "message": "Pull request created.",
            "provider_response": payload,
        }
    message = _friendly_provider_error(payload)
    return {
        "status": "failed",
        "status_code": response.status_code,
        "message": message or "Pull request publishing failed.",
        "provider_response": payload,
    }


def _friendly_provider_error(payload: dict[str, Any]) -> str | None:
    message = payload.get("message") if isinstance(payload, dict) else None
    if not message:
        return None
    if "Resource not accessible by personal access token" in message:
        return (
            "GitHub token cannot publish this PR. For a fine-grained token, select this repository "
            "and grant Repository permissions: Contents read/write, Pull requests read/write, "
            "and Metadata read-only. Commit statuses permission is not enough because Aiterate "
            "creates a promotion branch and writes artifact metadata files before opening the PR."
        )
    return message
