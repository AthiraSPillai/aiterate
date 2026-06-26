from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, field_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ArtifactKind(StrEnum):
    PROMPT = "prompt"
    SKILL = "skill"
    POLICY = "policy"
    EVAL_RUBRIC = "eval_rubric"


class ProviderKind(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    LITELLM = "litellm"
    MOCK = "mock"


class TrackerKind(StrEnum):
    MLFLOW = "mlflow"
    LANGSMITH = "langsmith"
    NOOP = "noop"


class AssertionKind(StrEnum):
    EQUALS = "equals"
    CONTAINS = "contains"
    CONTAINS_ANY = "contains_any"
    CONTAINS_ALL = "contains_all"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    CONTAINS_JSON = "contains_json"
    JSON_SCHEMA = "json_schema"
    REGEX = "regex"
    MAX_LENGTH = "max_length"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    POLICY_RUBRIC = "policy_rubric"
    SOURCE_GROUNDED = "source_grounded"
    UNCERTAINTY_HANDLING = "uncertainty_handling"
    REFUSAL_SAFETY = "refusal_safety"
    PROMPT_INJECTION_SAFETY = "prompt_injection_safety"
    PII_SAFETY = "pii_safety"
    TOOL_USE = "tool_use"


class PriorityRule(BaseModel):
    id: str
    text: str
    weight: float = Field(ge=0, le=1)


class EvalAssertion(BaseModel):
    id: str = Field(default_factory=lambda: new_id("assert"))
    type: AssertionKind
    value: str | None = None
    threshold: float | None = None
    weight: float = Field(default=1, ge=0)
    metric: str | None = None
    description: str = ""


class EvalCase(BaseModel):
    id: str = Field(default_factory=lambda: new_id("case"))
    input: str
    expected: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetExecutionResult(BaseModel):
    case_id: str
    input: str
    expected: str | None = None
    output: str
    score: float
    failed_metrics: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BehaviorEvaluationReport(BaseModel):
    score: float
    pass_rate: float
    case_count: int
    executions: list[TargetExecutionResult] = Field(default_factory=list)
    failed_metrics: list[str] = Field(default_factory=list)


class EvalCheckResult(BaseModel):
    assertion_id: str
    type: AssertionKind
    metric: str
    passed: bool
    score: float
    weight: float
    message: str


class EvaluationReport(BaseModel):
    score: float
    pass_rate: float
    passed: int
    failed: int
    checks: list[EvalCheckResult] = Field(default_factory=list)
    failed_metrics: list[str] = Field(default_factory=list)


class PolicySet(BaseModel):
    id: str = Field(default_factory=lambda: new_id("pol"))
    name: str = "default"
    rules: list[PriorityRule] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("rules")
    @classmethod
    def normalize_weights(cls, rules: list[PriorityRule]) -> list[PriorityRule]:
        total = sum(rule.weight for rule in rules)
        if rules and total <= 0:
            raise ValueError("at least one rule weight must be positive")
        if total > 1.001:
            return [rule.model_copy(update={"weight": rule.weight / total}) for rule in rules]
        return rules


class DatasetSnapshot(BaseModel):
    id: str = Field(default_factory=lambda: new_id("data"))
    name: str
    raw_text: str
    normalized_cases: list[str] = Field(default_factory=list)
    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProviderConfig(BaseModel):
    id: str = Field(default_factory=lambda: new_id("prov"))
    kind: ProviderKind = ProviderKind.MOCK
    model: str = "mock-optimizer"
    deployment: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    api_key: SecretStr | None = None
    region: str | None = None
    profile: str | None = None
    timeout_seconds: float = 60
    max_retries: int = 2
    rate_limit_rpm: int | None = None
    cost_metadata: dict[str, Any] = Field(default_factory=dict)

    def redacted(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if data.get("api_key"):
            data["api_key"] = "***"
        return data


class OptimizationRequest(BaseModel):
    name: str
    artifact_kind: ArtifactKind = ArtifactKind.PROMPT
    raw_data: str
    policy_context: str | None = None
    knowledge_base_context: str | None = None
    baseline_artifact: str | None = None
    policies: list[PriorityRule] = Field(default_factory=list)
    eval_assertions: list[EvalAssertion] = Field(default_factory=list)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    target_provider: ProviderConfig | None = None
    target_model: str | None = None
    enable_git_tracking: bool = True
    create_pull_request: bool = False
    tracker_uri: str | None = None
    tracker_project: str | None = None
    tracker_api_key: SecretStr | None = None
    tracker: TrackerKind = TrackerKind.NOOP
    iterations: int = Field(default=3, ge=1, le=20)
    validation_split: float = Field(default=0.25, ge=0.05, le=0.8)
    promotion_threshold: float = Field(default=0.8, ge=0, le=1)
    max_budget_usd: float | None = Field(default=None, ge=0)
    run_target_validation: bool = False
    seed: int = 7


class ProjectSettings(BaseModel):
    project_name: str
    enable_git_tracking: bool = True
    git_remote: str = ""
    artifact_branch: str = "main"
    enable_promotion_pr_workflow: bool = False
    pr_base_branch: str = "main"


class ArtifactVersion(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ver"))
    artifact_id: str
    kind: ArtifactKind
    version: int
    content: str
    score: float
    parent_version_id: str | None = None
    accepted: bool = True
    change_summary: str
    dataset_hash: str
    policy_hash: str
    provider: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    behavior_report: BehaviorEvaluationReport | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationInsight(BaseModel):
    worked: list[str] = Field(default_factory=list)
    went_wrong: list[str] = Field(default_factory=list)
    prompt_changes_needed: list[str] = Field(default_factory=list)
    policy_coverage: list[str] = Field(default_factory=list)
    data_risks: list[str] = Field(default_factory=list)


class CostEstimate(BaseModel):
    provider: str
    model: str
    currency: str = "USD"
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0
    output_cost: float = 0
    total_cost: float = 0
    input_per_1m_tokens: float = 0
    output_per_1m_tokens: float = 0
    pricing_source: str = ""
    approximate: bool = True
    notes: str = ""


class OptimizationRun(BaseModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    name: str
    artifact_id: str
    dataset: DatasetSnapshot
    policy_set: PolicySet
    provider: dict[str, Any]
    optimizer: dict[str, Any] = Field(default_factory=dict)
    accepted_versions: list[ArtifactVersion] = Field(default_factory=list)
    rejected_versions: list[ArtifactVersion] = Field(default_factory=list)
    best_version: ArtifactVersion | None = None
    insights: EvaluationInsight | None = None
    evaluation_report: EvaluationReport | None = None
    behavior_report: BehaviorEvaluationReport | None = None
    cost_estimate: CostEstimate | None = None
    approval: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelComparisonRequest(BaseModel):
    prompt: str
    raw_data: str = ""
    policies: list[PriorityRule] = Field(default_factory=list)
    model_a: ProviderConfig
    model_b: ProviderConfig
    execute_live: bool = False


class ModelComparisonResult(BaseModel):
    model: dict[str, Any]
    score: float
    sample_output: str | None = None
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendation: str


class ModelComparisonResponse(BaseModel):
    prompt_hash: str
    results: list[ModelComparisonResult]
    winner: str
    summary: str
