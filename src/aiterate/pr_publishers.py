from __future__ import annotations

from dataclasses import dataclass
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


class PullRequestPublisher:
    def publish(self, spec: PullRequestSpec) -> dict[str, Any]:
        raise NotImplementedError


class GitHubPullRequestPublisher(PullRequestPublisher):
    def publish(self, spec: PullRequestSpec) -> dict[str, Any]:
        token = settings.github_token or SecretStore().get_value("GITHUB_TOKEN")
        if not token:
            return {"status": "not_configured", "message": "GITHUB_TOKEN is required to publish GitHub PRs."}
        if not spec.owner or not spec.repo:
            return {"status": "invalid_request", "message": "GitHub owner and repo are required."}

        response = requests.post(
            f"https://api.github.com/repos/{spec.owner}/{spec.repo}/pulls",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
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


def _response_payload(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text}
    if response.ok:
        return {
            "status": "published",
            "url": payload.get("html_url") or payload.get("links", {}).get("html", {}).get("href"),
            "provider_response": payload,
        }
    return {"status": "failed", "status_code": response.status_code, "provider_response": payload}
