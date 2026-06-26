import {
  AlertCircle,
  BarChart3,
  Check,
  DatabaseZap,
  Eye,
  EyeOff,
  FileUp,
  GitBranch,
  KeyRound,
  Plus,
  Play,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sparkles,
  Trash2,
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || (window.location.port === "5173" ? "http://127.0.0.1:8000" : "");
const defaultMlflowTrackingUri = "http://host.docker.internal:5000";
const supportedTypes = [".txt", ".csv", ".json", ".yaml", ".yml", ".md"];
const maxUploadBytes = 2 * 1024 * 1024;
const maxTotalUploadBytes = 10 * 1024 * 1024;
const providerOptions = [
  ["openai", "OpenAI"],
  ["anthropic", "Anthropic"],
  ["azure_openai", "Azure OpenAI"],
  ["aws_bedrock", "AWS Bedrock"],
  ["litellm", "LiteLLM"],
];
const defaultPolicies = [
  { id: "accuracy", text: "Use the provided data and avoid unsupported claims.", weight: 0.25 },
  { id: "citations", text: "Cite source data or policy sections when possible.", weight: 0.25 },
  { id: "uncertainty", text: "Escalate when source data is incomplete or contradictory.", weight: 0.25 },
  { id: "tone", text: "Keep the answer concise, professional, and easy to verify.", weight: 0.25 },
];
const defaultEvalAssertions = [
  { id: "source_grounded", type: "source_grounded", metric: "source_grounded", description: "Require citation/source/evidence behavior.", weight: 1, enabled: true },
  { id: "uncertainty_handling", type: "uncertainty_handling", metric: "uncertainty_handling", description: "Require escalation for incomplete or contradictory data.", weight: 1, enabled: true },
  { id: "refusal_safety", type: "refusal_safety", metric: "refusal_safety", description: "Avoid unsafe override, jailbreak, or secret-exfiltration behavior.", weight: 1, enabled: true },
  { id: "prompt_injection_safety", type: "prompt_injection_safety", metric: "prompt_injection_safety", description: "Flag policy-bypass, jailbreak, or instruction-override behavior.", weight: 1, enabled: true },
  { id: "pii_safety", type: "pii_safety", metric: "pii_safety", description: "Flag obvious email, phone, credit-card, or SSN leakage patterns.", weight: 1, enabled: true },
  { id: "max_length", type: "max_length", metric: "brevity", description: "Keep artifact concise enough for operational review.", threshold: 450, weight: 0.5, enabled: false },
  { id: "contains_json", type: "contains_json", metric: "json_shape", description: "Require parseable JSON when the target workflow needs structured output.", weight: 0.5, enabled: false },
];
const sampleRawData = `Ticket 1421: Customer asks why the invoice doubled after plan upgrade. Billing policy says cite the invoice line item and explain prorated charges.

Ticket 1450: Customer asks for a refund after missing trial cancellation. Refund policy allows one courtesy refund if requested within 7 days.

Ticket 1512: Customer reports contradictory account status between dashboard and email. Support policy says do not guess; escalate to account operations.

Policy: Answers must cite the relevant policy or source note, avoid unsupported claims, and clearly state next steps.`;
const sampleBaselineArtifact = `You are a support assistant. Answer customer billing questions using only the supplied context.`;
const samplePolicies = [
  { id: "source_grounded", text: "Cite the source ticket, policy, or invoice detail used in the answer.", weight: 0.35 },
  { id: "refund_safety", text: "Do not promise refunds unless the policy allows it.", weight: 0.25 },
  { id: "escalation", text: "Escalate contradictory or incomplete account data instead of guessing.", weight: 0.25 },
  { id: "clarity", text: "Give concise next steps in a professional support tone.", weight: 0.15 },
];

function modelOption(id, label) {
  return { id, label, recommended_for: ["optimizer", "target"], source: "fallback" };
}

function mergeModelCatalogs(fallbackCatalog, apiCatalog) {
  const merged = {};
  const providers = new Set([...Object.keys(fallbackCatalog || {}), ...Object.keys(apiCatalog || {})]);
  for (const provider of providers) {
    const models = [];
    const seen = new Set();
    for (const model of [...(apiCatalog?.[provider] || []), ...(fallbackCatalog?.[provider] || [])]) {
      if (!model?.id || seen.has(model.id)) continue;
      seen.add(model.id);
      models.push(model);
    }
    merged[provider] = models;
  }
  return merged;
}

const fallbackModelCatalog = {
  openai: [
    modelOption("gpt-5.5", "GPT-5.5"),
    modelOption("gpt-5.4", "GPT-5.4"),
    modelOption("gpt-5.4-mini", "GPT-5.4 mini"),
    modelOption("gpt-4.1", "GPT-4.1"),
    modelOption("gpt-4.1-mini", "GPT-4.1 mini"),
    modelOption("gpt-4o", "GPT-4o"),
    modelOption("gpt-4o-mini", "GPT-4o mini"),
  ],
  anthropic: [
    modelOption("claude-opus-4-1-20250805", "Claude Opus 4.1"),
    modelOption("claude-sonnet-4-20250514", "Claude Sonnet 4"),
    modelOption("claude-3-7-sonnet-latest", "Claude 3.7 Sonnet"),
    modelOption("claude-3-5-sonnet-latest", "Claude 3.5 Sonnet"),
    modelOption("claude-3-5-haiku-latest", "Claude 3.5 Haiku"),
  ],
  azure_openai: [
    modelOption("gpt-5.5", "GPT-5.5 deployment"),
    modelOption("gpt-5.4", "GPT-5.4 deployment"),
    modelOption("gpt-4.1", "GPT-4.1 deployment"),
    modelOption("gpt-4o", "GPT-4o deployment"),
  ],
  aws_bedrock: [
    modelOption("anthropic.claude-opus-4-1-20250805-v1:0", "Claude Opus 4.1"),
    modelOption("anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4"),
    modelOption("anthropic.claude-3-5-sonnet-20240620-v1:0", "Claude 3.5 Sonnet"),
    modelOption("amazon.nova-pro-v1:0", "Amazon Nova Pro"),
    modelOption("amazon.nova-lite-v1:0", "Amazon Nova Lite"),
    modelOption("meta.llama3-1-70b-instruct-v1:0", "Llama 3.1 70B Instruct"),
  ],
  litellm: [
    modelOption("openai/gpt-5.5", "OpenAI GPT-5.5 via LiteLLM"),
    modelOption("openai/gpt-4.1", "OpenAI GPT-4.1 via LiteLLM"),
    modelOption("anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4 via LiteLLM"),
    modelOption("anthropic/claude-3-5-sonnet-latest", "Claude 3.5 Sonnet via LiteLLM"),
    modelOption("gemini/gemini-2.5-pro", "Gemini 2.5 Pro via LiteLLM"),
    modelOption("bedrock/amazon.nova-pro-v1:0", "Bedrock Nova Pro via LiteLLM"),
    modelOption("ollama/llama3.1", "Ollama Llama 3.1 via LiteLLM"),
  ],
};

function App() {
  const [projectName, setProjectName] = useState("support-agent");
  const [rawData, setRawData] = useState("");
  const [policyContext, setPolicyContext] = useState("");
  const [knowledgeBaseContext, setKnowledgeBaseContext] = useState("");
  const [baselineArtifact, setBaselineArtifact] = useState("");
  const [baselineSourceNotice, setBaselineSourceNotice] = useState("");
  const [validationSplit, setValidationSplit] = useState(0.25);
  const [iterations, setIterations] = useState(3);
  const [promotionThreshold, setPromotionThreshold] = useState(0.8);
  const [maxBudgetUsd, setMaxBudgetUsd] = useState("");
  const [runTargetValidation, setRunTargetValidation] = useState(false);
  const [seed, setSeed] = useState(7);
  const [fileInfo, setFileInfo] = useState({ data: [], policy: [], knowledge: [] });
  const [policies, setPolicies] = useState(defaultPolicies);
  const [evalAssertions, setEvalAssertions] = useState(defaultEvalAssertions);
  const [optimizerProvider, setOptimizerProvider] = useState("openai");
  const [optimizerModel, setOptimizerModel] = useState("gpt-5.5");
  const [customOptimizerModel, setCustomOptimizerModel] = useState("");
  const [targetProvider, setTargetProvider] = useState("openai");
  const [targetModel, setTargetModel] = useState("gpt-5.5");
  const [customTargetModel, setCustomTargetModel] = useState("");
  const [optimizerKey, setOptimizerKey] = useState("");
  const [targetKey, setTargetKey] = useState("");
  const [reuseOptimizerCredential, setReuseOptimizerCredential] = useState(true);
  const [reuseTargetCredential, setReuseTargetCredential] = useState(true);
  const [tracker, setTracker] = useState("noop");
  const [mlflowUri, setMlflowUri] = useState(defaultMlflowTrackingUri);
  const [langsmithProject, setLangsmithProject] = useState("");
  const [langsmithUrl, setLangsmithUrl] = useState("https://api.smith.langchain.com");
  const [reuseTrackingCredential, setReuseTrackingCredential] = useState(false);
  const [trackingSecret, setTrackingSecret] = useState("");
  const [gitRemote, setGitRemote] = useState("");
  const [prBase, setPrBase] = useState("main");
  const [enableGit, setEnableGit] = useState(true);
  const [createPr, setCreatePr] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [integrationStatus, setIntegrationStatus] = useState(null);
  const [integrationSetup, setIntegrationSetup] = useState(null);
  const [modelCatalog, setModelCatalog] = useState(fallbackModelCatalog);
  const [providerTestStatus, setProviderTestStatus] = useState("");
  const [testingProviderRole, setTestingProviderRole] = useState("");
  const [secretDraft, setSecretDraft] = useState(null);
  const [secretDeleteDraft, setSecretDeleteDraft] = useState(null);
  const [resetConfirmOpen, setResetConfirmOpen] = useState(false);
  const [secretValue, setSecretValue] = useState("");
  const [showSecretValue, setShowSecretValue] = useState(false);
  const [secretSaveError, setSecretSaveError] = useState("");
  const [runHistory, setRunHistory] = useState({});
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [openingRunId, setOpeningRunId] = useState("");
  const [runDeleteDraft, setRunDeleteDraft] = useState(null);
  const [projectDeleteDraft, setProjectDeleteDraft] = useState(null);
  const [run, setRun] = useState(null);
  const [approval, setApproval] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [comparingModels, setComparingModels] = useState(false);
  const [compareModelAProvider, setCompareModelAProvider] = useState("openai");
  const [compareModelAModel, setCompareModelAModel] = useState("gpt-5.5");
  const [compareCustomModelA, setCompareCustomModelA] = useState("");
  const [compareModelBProvider, setCompareModelBProvider] = useState("openai");
  const [compareModelBModel, setCompareModelBModel] = useState("gpt-5.5");
  const [compareCustomModelB, setCompareCustomModelB] = useState("");
  const [compareExecuteLive, setCompareExecuteLive] = useState(false);
  const [compareSourceRunId, setCompareSourceRunId] = useState("");
  const [prStatus, setPrStatus] = useState("");
  const [gitAuthStatus, setGitAuthStatus] = useState("");
  const [projectSettingsSaveStatus, setProjectSettingsSaveStatus] = useState("");
  const [projectSettingsLoadedFor, setProjectSettingsLoadedFor] = useState("");
  const [error, setError] = useState("");
  const [errorAction, setErrorAction] = useState(null);
  const [fileError, setFileError] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState("history");
  const runAbortRef = useRef(null);
  const projectSettingsSaveTimer = useRef(null);
  const projectSettingsDirtyRef = useRef(false);
  const projectSettingsLoadSeq = useRef(0);
  const dataRef = useRef(null);
  const policyRef = useRef(null);
  const providerRef = useRef(null);
  const trackingRef = useRef(null);
  const gitPrRef = useRef(null);
  const comparisonRef = useRef(null);
  const resultsRef = useRef(null);
  const approvalRef = useRef(null);
  const approveButtonRef = useRef(null);

  const categories = useMemo(() => inferCategories(rawData), [rawData]);
  const splitPreview = useMemo(() => previewDataSplit(rawData, validationSplit), [rawData, validationSplit]);
  const totalWeight = useMemo(() => policies.reduce((sum, rule) => sum + Number(rule.weight), 0), [policies]);
  const validationIssues = useMemo(
    () =>
      buildValidationIssues({
        backendStatus,
        rawData,
        policyContext,
        knowledgeBaseContext,
        policies,
        optimizerProvider,
        optimizerModel,
        customOptimizerModel,
        targetProvider,
        targetModel,
        customTargetModel,
        tracker,
        mlflowUri,
        langsmithProject,
        langsmithUrl,
        reuseTrackingCredential,
        trackingSecret,
        enableGit,
        createPr,
        gitRemote,
        prBase,
        reuseOptimizerCredential,
        reuseTargetCredential,
        optimizerKey,
        targetKey,
        integrationStatus,
        iterations,
        promotionThreshold,
        maxBudgetUsd,
      }),
    [
      backendStatus,
      rawData,
      policyContext,
      knowledgeBaseContext,
      policies,
      optimizerProvider,
      optimizerModel,
      customOptimizerModel,
      targetProvider,
      targetModel,
      customTargetModel,
      tracker,
      mlflowUri,
      langsmithProject,
      langsmithUrl,
      reuseTrackingCredential,
      trackingSecret,
      enableGit,
      createPr,
      gitRemote,
      prBase,
      reuseOptimizerCredential,
      reuseTargetCredential,
      optimizerKey,
      targetKey,
      integrationStatus,
      iterations,
      promotionThreshold,
      maxBudgetUsd,
    ]
  );
  const modelSetupReady = modelsAndCredentialsReady({
    optimizerProvider,
    optimizerModel,
    customOptimizerModel,
    targetProvider,
    targetModel,
    customTargetModel,
    reuseOptimizerCredential,
    reuseTargetCredential,
    optimizerKey,
    targetKey,
    integrationStatus,
  });
  const trackingEnabled = tracker !== "noop";
  const trackingCredentialReady = tracker === "langsmith"
    ? reuseTrackingCredential
      ? integrationStatus?.tracking?.langsmith?.credential_configured === true
      : Boolean(trackingSecret.trim())
    : true;
  const trackingReady = trackingEnabled
    ? tracker === "mlflow"
      ? Boolean(mlflowUri)
      : tracker === "langsmith"
        ? Boolean(langsmithProject && langsmithUrl && trackingCredentialReady)
        : false
    : false;
  const steps = [
    { id: "history", number: "H", title: "Run History", complete: Object.keys(runHistory || {}).length > 0 },
    { id: "data", number: "1", title: "Import context", complete: rawData.trim().length > 20, ref: dataRef },
    {
      id: "policy",
      number: "2",
      title: "Weight policies",
      complete: policies.length > 0 && policies.every((rule) => rule.id.trim() && rule.text.trim()),
      ref: policyRef,
    },
    {
      id: "models",
      number: "3",
      title: "Configure models",
      complete: modelSetupReady,
      ref: providerRef,
    },
    {
      id: "tracking",
      number: "4",
      title: "Tracking optional",
      complete: trackingReady,
      optional: !trackingEnabled,
      ref: trackingRef,
    },
    { id: "run", number: "5", title: "Run optimizer", complete: Boolean(run?.best_version), running: loading, ref: resultsRef },
    { id: "review", number: "6", title: "Review and approve", complete: Boolean(approval?.status === "approved"), ref: approvalRef },
    { id: "git", number: "7", title: "Create Git PR", complete: Boolean(prStatus), ref: gitPrRef },
    { id: "compare", number: "8", title: "Compare models", complete: Boolean(modelComparison), ref: comparisonRef },
  ];
  const requiredStepsComplete = steps.filter((step) => ["data", "policy", "models"].includes(step.id)).every((step) => step.complete);
  const canRun = requiredStepsComplete && validationIssues.length === 0;
  const canCreatePr = Boolean(run && approval?.status === "approved" && createPr && gitRemote.trim() && prBase.trim());
  const approvedRunOptions = useMemo(() => approvedRunsFromHistory(runHistory), [runHistory]);
  const currentRunApproved = Boolean(approval?.status === "approved" || run?.approval?.status === "approved");
  const compareSourceAvailable = currentRunApproved || approvedRunOptions.length > 0;
  const projectGitSettings = {
    project_name: projectName,
    enable_git_tracking: enableGit,
    git_remote: gitRemote,
    enable_promotion_pr_workflow: createPr,
    pr_base_branch: prBase,
  };

  useEffect(() => {
    checkBackend();
    refreshRunHistory();
  }, []);

  useEffect(() => {
    if (currentRunApproved && run?.id) {
      setCompareSourceRunId(run.id);
    } else if (!compareSourceRunId && approvedRunOptions.length > 0) {
      setCompareSourceRunId(approvedRunOptions[0].id);
    }
  }, [approvedRunOptions, compareSourceRunId, currentRunApproved, run?.id]);

  useEffect(() => {
    const name = projectName.trim();
    if (!name) return;
    projectSettingsDirtyRef.current = false;
    const loadSeq = projectSettingsLoadSeq.current + 1;
    projectSettingsLoadSeq.current = loadSeq;
    const controller = new AbortController();
    setProjectSettingsSaveStatus("");
    fetch(`${apiBaseUrl}/v1/projects/${encodeURIComponent(name)}/settings`, { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : null))
      .then((settings) => {
        if (!settings || controller.signal.aborted || projectSettingsLoadSeq.current !== loadSeq) return;
        setProjectSettingsLoadedFor(name);
        if (projectSettingsDirtyRef.current) {
          setProjectSettingsSaveStatus("Unsaved Git settings. Save to keep them for this project.");
          return;
        }
        setEnableGit(settings.enable_git_tracking);
        setGitRemote(settings.git_remote || "");
        setCreatePr(settings.enable_promotion_pr_workflow);
        setPrBase(settings.pr_base_branch || "main");
      })
      .catch((caught) => {
        if (caught?.name !== "AbortError") {
          setProjectSettingsSaveStatus("Project Git settings will save after the backend is available.");
        }
      });
    return () => controller.abort();
  }, [projectName]);

  useEffect(() => {
    const name = projectName.trim();
    if (!name || projectSettingsLoadedFor !== name || !projectSettingsDirtyRef.current) return;
    window.clearTimeout(projectSettingsSaveTimer.current);
    projectSettingsSaveTimer.current = window.setTimeout(() => {
      saveProjectGitSettings();
    }, 500);
    return () => window.clearTimeout(projectSettingsSaveTimer.current);
  }, [projectName, projectSettingsLoadedFor, enableGit, gitRemote, createPr, prBase]);

  function checkBackend() {
    setBackendStatus("checking");
    fetch(`${apiBaseUrl}/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Backend health check failed");
        setBackendStatus("online");
      })
      .catch(() => setBackendStatus("offline"));
    fetch(`${apiBaseUrl}/v1/integrations/status`)
    .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        setIntegrationStatus(payload);
        syncCredentialReuse(payload);
      })
      .catch(() => setIntegrationStatus(null));
    fetch(`${apiBaseUrl}/v1/integrations/setup`)
      .then((response) => (response.ok ? response.json() : null))
      .then(setIntegrationSetup)
      .catch(() => setIntegrationSetup(null));
    fetch(`${apiBaseUrl}/v1/model-catalog?include_live=true`)
      .then((response) => (response.ok ? response.json() : {}))
      .then((catalog) => setModelCatalog(mergeModelCatalogs(fallbackModelCatalog, catalog)))
      .catch(() => setModelCatalog(fallbackModelCatalog));
  }

  function syncCredentialReuse(status) {
    if (!status?.providers?.[optimizerProvider]?.configured) {
      setReuseOptimizerCredential(false);
    }
    if (!status?.providers?.[targetProvider]?.configured) {
      setReuseTargetCredential(false);
    }
    if (tracker !== "noop" && !status?.tracking?.[tracker]?.configured) {
      setReuseTrackingCredential(false);
    }
  }

  function updateProjectGitSetting(setter, value) {
    projectSettingsDirtyRef.current = true;
    setProjectSettingsSaveStatus("Unsaved Git settings. Save to keep them for this project.");
    setter(value);
  }

  async function saveProjectGitSettings() {
    const name = projectName.trim();
    if (!name) return false;
    if (projectSettingsLoadedFor !== name) {
      setProjectSettingsSaveStatus("Project settings are still loading. Try again in a moment.");
      return false;
    }
    setProjectSettingsSaveStatus("Saving project Git settings...");
    try {
      const response = await fetch(`${apiBaseUrl}/v1/projects/${encodeURIComponent(name)}/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_name: name,
          enable_git_tracking: enableGit,
          git_remote: gitRemote.trim(),
          artifact_branch: "auto",
          enable_promotion_pr_workflow: createPr,
          pr_base_branch: prBase.trim() || "main",
        }),
      });
      if (!response.ok) throw new Error(await apiErrorMessage(response, "Could not save project Git settings."));
      const saved = await response.json();
      projectSettingsDirtyRef.current = false;
      setGitRemote(saved.git_remote || "");
      setPrBase(saved.pr_base_branch || "main");
      setProjectSettingsSaveStatus("Project Git settings saved for this project.");
      return true;
    } catch (caught) {
      projectSettingsDirtyRef.current = true;
      setProjectSettingsSaveStatus(caught?.message || "Could not save project Git settings yet.");
      return false;
    }
  }

  async function saveSecret() {
    if (!secretDraft || !secretValue.trim()) return;
    setSecretSaveError("");
    const response = await fetch(`${apiBaseUrl}/v1/secrets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: secretDraft.secret_name,
        integration: secretDraft.name,
        value: secretValue.trim(),
      }),
    });
    if (response.ok) {
      setSecretValue("");
      setShowSecretValue(false);
      setSecretSaveError("");
      setSecretDraft(null);
      checkBackend();
    } else {
      setSecretSaveError(await apiErrorMessage(response, "Could not save credential."));
    }
  }

  async function deleteSecret() {
    if (!secretDeleteDraft?.name) return;
    const response = await fetch(`${apiBaseUrl}/v1/secrets/${encodeURIComponent(secretDeleteDraft.name)}`, {
      method: "DELETE",
    });
    if (response.ok) {
      setSecretDeleteDraft(null);
      checkBackend();
    } else {
      setError(await apiErrorMessage(response, "Could not delete credential."));
    }
  }

  function refreshRunHistory() {
    setHistoryLoading(true);
    setHistoryError("");
    fetch(`${apiBaseUrl}/v1/runs`)
      .then((response) => (response.ok ? response.json() : {}))
      .then(setRunHistory)
      .catch(() => {
        setHistoryError("Run history is unavailable until the backend is online.");
        setRunHistory({});
      })
      .finally(() => setHistoryLoading(false));
  }

  async function openPreviousRun(runId) {
    setOpeningRunId(runId);
    setHistoryError("");
    setError("");
    setErrorAction(null);
    try {
      const response = await fetch(`${apiBaseUrl}/v1/runs/${encodeURIComponent(runId)}`);
      if (!response.ok) throw new Error(await apiErrorMessage(response, "Could not open this run."));
      const loadedRun = await readJsonResponse(response, "Could not open this run.");
      setRun(loadedRun);
      setApproval(loadedRun.approval || null);
      setModelComparison(null);
      setPrStatus("");
      setProjectName(loadedRun.name || projectName);
      setRawData(loadedRun.dataset?.raw_text || rawData);
      setPolicyContext(loadedRun.optimizer?.policy_context || "");
      setKnowledgeBaseContext(loadedRun.optimizer?.knowledge_base_context || "");
      setBaselineArtifact(loadedRun.best_version?.content || baselineArtifact);
      if (loadedRun.best_version?.content) {
        setBaselineSourceNotice(
          `Baseline updated from ${loadedRun.name || "selected project"} run ${loadedRun.id}, best version v${loadedRun.best_version.version}.`
        );
      }
      setPolicies(loadedRun.policy_set?.rules?.length ? loadedRun.policy_set.rules : policies);
      setValidationSplit(loadedRun.optimizer?.validation_split ?? validationSplit);
      setIterations(loadedRun.optimizer?.iterations ?? loadedRun.optimizer?.requested_iterations ?? iterations);
      setPromotionThreshold(loadedRun.optimizer?.promotion_threshold ?? promotionThreshold);
      setMaxBudgetUsd(
        loadedRun.optimizer?.max_budget_usd === null || loadedRun.optimizer?.max_budget_usd === undefined
          ? ""
          : String(loadedRun.optimizer.max_budget_usd)
      );
      setRunTargetValidation(Boolean(loadedRun.optimizer?.run_target_validation));
      setSeed(loadedRun.optimizer?.seed ?? seed);
      if (loadedRun.provider?.kind) setOptimizerProvider(loadedRun.provider.kind);
      if (loadedRun.provider?.model) {
        const modelInCatalog = modelCatalog?.[loadedRun.provider.kind]?.some(
          (model) => model.id === loadedRun.provider.model
        );
        setOptimizerModel(modelInCatalog ? loadedRun.provider.model : "__custom");
        setCustomOptimizerModel(modelInCatalog ? "" : loadedRun.provider.model);
        setCompareModelAProvider(loadedRun.provider.kind);
        setCompareModelAModel(modelInCatalog ? loadedRun.provider.model : "__custom");
        setCompareCustomModelA(modelInCatalog ? "" : loadedRun.provider.model);
      }
      if (loadedRun.target_provider?.kind) {
        setTargetProvider(loadedRun.target_provider.kind);
        setCompareModelBProvider(loadedRun.target_provider.kind);
      }
      if (loadedRun.target_provider?.model) {
        const targetModelInCatalog = modelCatalog?.[loadedRun.target_provider.kind]?.some(
          (model) => model.id === loadedRun.target_provider.model
        );
        setTargetModel(targetModelInCatalog ? loadedRun.target_provider.model : "__custom");
        setCustomTargetModel(targetModelInCatalog ? "" : loadedRun.target_provider.model);
        setCompareModelBModel(targetModelInCatalog ? loadedRun.target_provider.model : "__custom");
        setCompareCustomModelB(targetModelInCatalog ? "" : loadedRun.target_provider.model);
      }
      setActiveView("review");
    } catch (caught) {
      setHistoryError(caught instanceof Error ? caught.message : "Could not open this run.");
    } finally {
      setOpeningRunId("");
    }
  }

  async function deletePreviousRun() {
    if (!runDeleteDraft?.id) return;
    setHistoryError("");
    try {
      await deleteRunRequest(runDeleteDraft.id);
      if (run?.id === runDeleteDraft.id) {
        setRun(null);
        setApproval(null);
        setModelComparison(null);
        setPrStatus("");
      }
      setRunDeleteDraft(null);
      refreshRunHistory();
    } catch (caught) {
      setHistoryError(caught instanceof Error ? caught.message : "Could not delete this run.");
    }
  }

  async function deleteRunRequest(runId) {
    const encoded = encodeURIComponent(runId);
    const attempts = [
      { url: `${apiBaseUrl}/v1/runs/${encoded}/delete`, options: { method: "POST" } },
      {
        url: `${apiBaseUrl}/v1/runs/${encoded}`,
        options: {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "delete" }),
        },
      },
      { url: `${apiBaseUrl}/v1/runs/${encoded}`, options: { method: "DELETE" } },
    ];
    let lastError = "Could not delete this run.";
    for (const attempt of attempts) {
      const response = await fetch(attempt.url, attempt.options);
      if (response.ok) return;
      lastError = await apiErrorMessage(response, "Could not delete this run.");
      if (![404, 405].includes(response.status)) break;
    }
    throw new Error(lastError);
  }

  async function deleteProjectRuns() {
    if (!projectDeleteDraft?.name) return;
    setHistoryError("");
    try {
      const response = await fetch(`${apiBaseUrl}/v1/projects/${encodeURIComponent(projectDeleteDraft.name)}/delete`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(await apiErrorMessage(response, "Could not delete this project."));
      if (run?.name === projectDeleteDraft.name) {
        setRun(null);
        setApproval(null);
        setModelComparison(null);
        setPrStatus("");
      }
      setProjectDeleteDraft(null);
      refreshRunHistory();
    } catch (caught) {
      setHistoryError(caught instanceof Error ? caught.message : "Could not delete this project.");
    }
  }

  async function uploadFile(event, role = "data") {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    setFileError("");
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    const invalid = files.find((file) => !supportedTypes.includes(fileExtension(file.name)));
    const tooLarge = files.find((file) => file.size > maxUploadBytes);
    if (invalid) {
      setFileError(`Unsupported file type ${fileExtension(invalid.name) || "unknown"} in ${invalid.name}. Use ${supportedTypes.join(", ")}.`);
      event.target.value = "";
      return;
    }
    if (tooLarge || totalBytes > maxTotalUploadBytes) {
      setFileError("One or more files are too large for the browser workflow. Use the CLI or SDK for larger datasets.");
      event.target.value = "";
      return;
    }
    try {
      const loaded = await Promise.all(files.map(async (file) => ({
        name: file.name,
        size: file.size,
        extension: fileExtension(file.name),
        text: await file.text(),
      })));
      const combined = loaded
        .map((file) => `## Source file: ${file.name}\n\n${file.text.trim()}`)
        .join("\n\n---\n\n");
      setFileInfo((current) => ({ ...current, [role]: loaded.map(({ text, ...file }) => file) }));
      if (role === "policy") {
        setPolicyContext(combined);
        setPolicies(equalizeWeights(policyRulesFromText(combined)));
      } else if (role === "knowledge") {
        setKnowledgeBaseContext(combined);
      } else {
        setRawData(combined);
        const inferred = inferCategories(combined);
        setPolicies((current) => current.length ? current : equalizeWeights(inferred.map((category) => categoryToPolicy(category))));
      }
    } catch {
      setFileError("Could not read one or more files. Check that every upload is a text-based data file.");
    } finally {
      event.target.value = "";
    }
  }

  function updatePolicy(index, field, value) {
    const next = [...policies];
    next[index] = { ...next[index], [field]: field === "weight" ? Number(value) : value };
    setPolicies(next);
  }

  function addPolicy() {
    setPolicies([...policies, { id: `policy_${policies.length + 1}`, text: "", weight: 0 }]);
  }

  function removePolicy(index) {
    setPolicies(policies.filter((_, currentIndex) => currentIndex !== index));
  }

  function equalizePolicyWeights() {
    setPolicies(equalizeWeights(policies));
  }

  function resetWorkspace() {
    setProjectName("support-agent");
    setRawData("");
    setPolicyContext("");
    setKnowledgeBaseContext("");
    setBaselineArtifact("");
    setBaselineSourceNotice("");
    setValidationSplit(0.25);
    setIterations(3);
    setPromotionThreshold(0.8);
    setMaxBudgetUsd("");
    setRunTargetValidation(false);
    setSeed(7);
    setFileInfo({ data: [], policy: [], knowledge: [] });
    setPolicies(defaultPolicies);
    setEvalAssertions(defaultEvalAssertions);
    setRun(null);
    setApproval(null);
    setModelComparison(null);
    setError("");
    setErrorAction(null);
    setFileError("");
  }

  function loadSampleProject() {
    setProjectName("sample-support-agent");
    setRawData(sampleRawData);
    setPolicyContext(samplePolicies.map((policy) => `- ${policy.text}`).join("\n"));
    setKnowledgeBaseContext("Billing KB: Prorated plan upgrades should cite invoice line items. Refund KB: Courtesy refunds may be offered within 7 days when policy criteria are met. Account status conflicts should be escalated to account operations.");
    setBaselineArtifact(sampleBaselineArtifact);
    setBaselineSourceNotice("");
    setValidationSplit(0.25);
    setIterations(3);
    setPromotionThreshold(0.8);
    setMaxBudgetUsd("");
    setRunTargetValidation(false);
    setSeed(7);
    setPolicies(samplePolicies);
    setFileInfo({ data: [], policy: [], knowledge: [] });
    setRun(null);
    setApproval(null);
    setModelComparison(null);
    setError("");
    setErrorAction(null);
    setFileError("");
    setProviderTestStatus("Sample loaded. Review policies, choose models, then run optimization.");
  }

  function handleTargetProviderChange(provider) {
    setTargetProvider(provider);
    setTargetModel(defaultModelForProvider(provider, modelCatalog));
    setCustomTargetModel("");
    if (provider === optimizerProvider) {
      setReuseTargetCredential(true);
      setTargetKey("");
    }
  }

  async function testSelectedProvider(role) {
    const isOptimizer = role === "optimizer";
    const provider = isOptimizer ? optimizerProvider : targetProvider;
    const model = isOptimizer
      ? selectedModelValue(optimizerModel, customOptimizerModel)
      : selectedModelValue(targetModel, customTargetModel);
    const apiKey = isOptimizer
      ? reuseOptimizerCredential ? null : optimizerKey || null
      : optimizerProvider === targetProvider
        ? reuseOptimizerCredential ? null : optimizerKey || null
        : reuseTargetCredential ? null : targetKey || null;
    setTestingProviderRole(role);
    setProviderTestStatus("");
    try {
      const response = await fetch(`${apiBaseUrl}/v1/providers/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: provider, model, api_key: apiKey }),
      });
      if (!response.ok) throw await apiError(response, "Provider test failed.");
      const payload = await response.json();
      setProviderTestStatus(`${providerLabel(provider)} ${model} is ready. ${payload.message}`);
    } catch (caught) {
      setProviderTestStatus(caught?.message || "Provider test failed.");
    } finally {
      setTestingProviderRole("");
    }
  }

  async function optimize() {
    if (!canRun) {
      setError(`Complete setup before running: ${validationIssues[0] || "review required fields"}.`);
      return;
    }
    const controller = new AbortController();
    runAbortRef.current = controller;
    const budget = maxBudgetUsd.trim() ? Number(maxBudgetUsd) : null;
    setActiveView("run");
    setLoading(true);
    setRun(null);
    setApproval(null);
    setModelComparison(null);
    setPrStatus("");
    setError("");
    setErrorAction(null);
    try {
      const response = await fetch(`${apiBaseUrl}/v1/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          name: projectName,
          raw_data: rawData,
          policy_context: policyContext || null,
          knowledge_base_context: knowledgeBaseContext || null,
          baseline_artifact: baselineArtifact.trim() || null,
          validation_split: validationSplit,
          policies,
          eval_assertions: evalAssertions
            .filter((assertion) => assertion.enabled)
            .map(({ enabled, ...assertion }) => assertion),
          provider: {
            kind: optimizerProvider,
            model: selectedModelValue(optimizerModel, customOptimizerModel) || defaultModelForProvider(optimizerProvider, modelCatalog),
            api_key: reuseOptimizerCredential ? null : optimizerKey || null,
          },
          target_provider: {
            kind: targetProvider,
            model: selectedModelValue(targetModel, customTargetModel),
            api_key: optimizerProvider === targetProvider
              ? reuseOptimizerCredential ? null : optimizerKey || null
              : reuseTargetCredential ? null : targetKey || null,
          },
          target_model: selectedModelValue(targetModel, customTargetModel),
          tracker,
          tracker_uri: tracker === "noop" ? null : tracker === "mlflow" ? mlflowUri || null : langsmithUrl || null,
          tracker_project: tracker === "langsmith" ? langsmithProject || null : null,
          tracker_api_key: tracker === "noop" || reuseTrackingCredential ? null : trackingSecret || null,
          enable_git_tracking: enableGit,
          create_pull_request: false,
          iterations,
          promotion_threshold: promotionThreshold,
          max_budget_usd: budget,
          run_target_validation: runTargetValidation,
          seed,
        }),
      });
      if (!response.ok) throw await apiError(response, "Optimization failed.");
      const result = await response.json();
      setRun(result);
      setCompareModelAProvider(optimizerProvider);
      setCompareModelAModel(optimizerModel);
      setCompareCustomModelA(customOptimizerModel);
      setCompareModelBProvider(targetProvider);
      setCompareModelBModel(targetModel);
      setCompareCustomModelB(customTargetModel);
      refreshRunHistory();
      setActiveView("review");
      setTimeout(() => approveButtonRef.current?.focus(), 0);
    } catch (caught) {
      if (caught?.name === "AbortError") {
        setError("Run stopped in this browser. If the server had already started work, check Run History before starting another run.");
        setErrorAction(null);
      } else {
        setError(caught?.message || "Optimization failed");
        setErrorAction(caught?.action || null);
      }
    } finally {
      setLoading(false);
      runAbortRef.current = null;
      setOptimizerKey("");
      setTargetKey("");
      setTrackingSecret("");
    }
  }

  function stopOptimization() {
    runAbortRef.current?.abort();
  }

  async function requestPullRequest(result) {
    if (!approval) {
      setPrStatus("Approve the best version before creating a promotion PR.");
      return;
    }
    setPrStatus("Creating promotion PR...");
    try {
      if (projectSettingsDirtyRef.current) {
        const saved = await saveProjectGitSettings();
        if (!saved) {
          setPrStatus("Save Git settings before creating a promotion PR.");
          return;
        }
      }
      const response = await fetch(`${apiBaseUrl}/v1/git/pull-request`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: `Promote ${projectName} ${result.best_version?.id || ""}`.trim(),
          body: promotionPrBody(result),
          run_id: result.id,
          artifact_id: result.artifact_id,
          version_id: result.best_version?.id,
          approval_status: approval?.status || "approved",
          approved_by: approval?.approved_by || "local-user",
          artifact_content: result.best_version?.content || "",
          run_json: JSON.stringify(result, null, 2),
          remote: gitRemote,
          branch: promotionBranchName(result),
          source_branch: promotionBranchName(result),
          base: prBase,
        }),
      });
      if (!response.ok) throw new Error(await apiErrorMessage(response, "Create PR failed."));
      const payload = await readJsonResponse(response, "Create PR failed.");
      if (payload.status === "published") {
        setPrStatus(payload.url ? `Pull request created: ${payload.url}` : "Pull request created.");
      } else {
        setPrStatus(payload.message || `Create PR returned status: ${payload.status}`);
      }
    } catch (caught) {
      setPrStatus(caught?.message || "Create PR failed.");
    }
  }

  async function startGitBrowserAuth(provider) {
    setGitAuthStatus("");
    const response = await fetch(`${apiBaseUrl}/v1/git/auth/${provider}/start`);
    const payload = await response.json();
    if (payload.status === "ready" && payload.auth_url) {
      window.location.assign(payload.auth_url);
      return;
    }
    setGitAuthStatus(payload.message || "Browser auth is not configured for this Git provider.");
  }

  async function compareCurrentModels() {
    const sourceRunId = compareSourceRunId || (currentRunApproved ? run?.id : "");
    if (!sourceRunId) {
      setError("Approve a run first, or select an approved run from history.");
      return;
    }
    setComparingModels(true);
    setModelComparison(null);
    setError("");
    setErrorAction(null);
    setActiveView("compare");
    try {
      const sourceRun = run?.id === sourceRunId
        ? run
        : await fetchRunById(sourceRunId, "Could not open the approved run for comparison.");
      if (!sourceRun?.best_version) throw new Error("Selected approved run has no best version to compare.");
      const response = await fetch(`${apiBaseUrl}/v1/compare-models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: sourceRun.best_version.content,
          raw_data: sourceRun.dataset?.raw_text || rawData,
          policies: sourceRun.policy_set?.rules?.length ? sourceRun.policy_set.rules : policies,
          model_a: {
            kind: compareModelAProvider,
            model: selectedModelValue(compareModelAModel, compareCustomModelA) || defaultModelForProvider(compareModelAProvider, modelCatalog),
          },
          model_b: {
            kind: compareModelBProvider,
            model: selectedModelValue(compareModelBModel, compareCustomModelB) || defaultModelForProvider(compareModelBProvider, modelCatalog),
          },
          execute_live: compareExecuteLive,
        }),
      });
      if (!response.ok) throw new Error(await apiErrorMessage(response, "Model comparison failed."));
      setModelComparison(await response.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Model comparison failed");
    } finally {
      setComparingModels(false);
    }
  }

  async function approveBestVersion(result) {
    const response = await fetch(`${apiBaseUrl}/v1/runs/${result.id}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        artifact_id: result.artifact_id,
        version_id: result.best_version?.id,
        approved_by: "local-user",
      }),
    });
    if (!response.ok) {
      setError(await apiErrorMessage(response, "Approval failed."));
      return;
    }
    const payload = await response.json();
    setApproval(payload);
    setRun((current) => current?.id === result.id ? { ...current, approval: payload } : current);
    setCompareSourceRunId(result.id);
    refreshRunHistory();
    setPrStatus("");
    setActiveView("review");
  }

  function handleErrorAction(action) {
    if (!action) return;
    if (action.type === "add_provider_secret") {
      const item = providerSetupItem(action.provider || optimizerProvider, integrationSetup);
      if (item) {
        setSecretDraft(item);
        setSecretValue("");
        setShowSecretValue(false);
        setSecretSaveError("");
        setActiveView("models");
      }
    }
  }

  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <div>
            <div className="eyebrow"><Sparkles size={14} /> AI Artifact Lifecycle Management</div>
            <h1>Aiterate</h1>
            <p>Run governed AI artifact lifecycle management: create, optimize, regression-test, approve, version, and promote prompts and skills from raw data and policies.</p>
          </div>
          <button className="secondary reset-action" onClick={() => setResetConfirmOpen(true)}>Reset workspace</button>
        </div>
      </section>

      <section className="workspace">
        <aside className="rail">
          <div className="rail-title">Workspace</div>
          {steps.map((step) => (
            <Step
              key={step.id}
              number={step.number}
              title={step.title}
              complete={step.complete}
              optional={step.optional}
              selected={activeView === step.id}
              running={step.running}
              onClick={() => {
                setActiveView(step.id);
                if (step.id === "review") {
                  setTimeout(() => approveButtonRef.current?.focus(), 0);
                }
              }}
            />
          ))}
        </aside>

        <section className="content">
          {activeView === "history" && (
          <div className="panel wide dashboard">
            <div className="section-heading">
              <h2><BarChart3 size={18} /> Run History</h2>
              <button className="secondary" onClick={() => { checkBackend(); refreshRunHistory(); }}>
                <RefreshCw size={16} />
                Refresh
              </button>
            </div>
            <p className="section-help">Open past optimization runs, compare what changed, or delete old run records when a project is no longer useful.</p>
            <RunDashboard
              runHistory={runHistory}
              loading={historyLoading}
              error={historyError}
              openingRunId={openingRunId}
              onOpenRun={openPreviousRun}
              onDeleteRun={setRunDeleteDraft}
              onDeleteProject={setProjectDeleteDraft}
            />
          </div>
          )}

          {activeView === "data" && (
          <div className="panel wide" ref={dataRef}>
            <h2><FileUp size={18} /> Import Context</h2>
            <p className="section-help">Start with messy notes, examples, tickets, policy text, CSV, JSON, YAML, or Markdown, then optionally add the current prompt or skill baseline. This gives the optimizer both the source context and the artifact starting point.</p>
            <label>Project name</label>
            <input value={projectName} onChange={(event) => setProjectName(event.target.value)} />
            <div className="upload-row context-upload-header">
              <button className="secondary" onClick={loadSampleProject}>Load sample project</button>
              <div className="hint">Supported: {supportedTypes.join(", ")}. Limit: 2 MB per file, 10 MB total.</div>
            </div>
            {fileError && <div className="error">{fileError}</div>}
            <div className="context-upload-grid">
              <ContextUpload
                title="Data / Examples"
                badge="Train + test"
                icon={<DatabaseZap size={18} />}
                help="Tickets, conversations, eval cases, CSV/JSON rows used for training and regression testing."
                files={fileInfo.data}
                onUpload={(event) => uploadFile(event, "data")}
              />
              <ContextUpload
                title="Policies"
                badge="Score rules"
                icon={<ShieldCheck size={18} />}
                help="Rules, compliance guidance, tone, safety constraints, and acceptance criteria."
                files={fileInfo.policy}
                onUpload={(event) => uploadFile(event, "policy")}
              />
              <ContextUpload
                title="Knowledge Base / References"
                badge="Grounding"
                icon={<FileUp size={18} />}
                help="Product docs, SOPs, support articles, manuals, or source material the artifact should stay grounded in."
                files={fileInfo.knowledge}
                onUpload={(event) => uploadFile(event, "knowledge")}
              />
            </div>
            <label>Data / Examples text</label>
            <textarea value={rawData} onChange={(event) => setRawData(event.target.value)} placeholder="Paste tickets, conversations, eval cases, CSV/JSON rows, or examples here." />
            <label>Policy context</label>
            <textarea className="context-textarea" value={policyContext} onChange={(event) => setPolicyContext(event.target.value)} placeholder="Paste policy, compliance, safety, tone, or acceptance rules here." />
            <label>Knowledge Base / References</label>
            <textarea className="context-textarea" value={knowledgeBaseContext} onChange={(event) => setKnowledgeBaseContext(event.target.value)} placeholder="Paste source docs, SOPs, support articles, product docs, or reference material here." />
            <div className="split-control">
              <div>
                <label>Holdout test split</label>
                <p className="section-help compact-help">
                  Aiterate uses training examples to propose prompt/skill changes and keeps this holdout set for regression scoring. Increase it when you have many examples and want stricter validation.
                </p>
              </div>
              <div className="split-slider">
                <input
                  type="range"
                  min="5"
                  max="80"
                  step="5"
                  value={Math.round(validationSplit * 100)}
                  onChange={(event) => setValidationSplit(Number(event.target.value) / 100)}
                />
                <strong>{Math.round(validationSplit * 100)}% test</strong>
              </div>
            </div>
            <div className="split-preview">
              <div><strong>{splitPreview.train}</strong><span>training example(s)</span></div>
              <div><strong>{splitPreview.test}</strong><span>holdout test example(s)</span></div>
              <div><strong>{splitPreview.total}</strong><span>estimated total</span></div>
            </div>
            <div className="chips">
              {categories.map((category) => (
                <span key={category}>{category}</span>
              ))}
            </div>
            <label>Current prompt or skill baseline (optional)</label>
            <p className="section-help compact-help">If you already have a production prompt or agent skill, paste it here. The optimizer will treat it as the starting skill and run validation-gated edits against it.</p>
            <textarea
              className="baseline-textarea"
              value={baselineArtifact}
              onChange={(event) => {
                setBaselineArtifact(event.target.value);
                setBaselineSourceNotice("");
              }}
              placeholder="Paste an existing prompt or skill to optimize. Leave blank to generate the baseline from raw data."
            />
            {baselineSourceNotice && <div className="baseline-source-note">{baselineSourceNotice}</div>}
          </div>
          )}

          {activeView === "policy" && (
          <div className="panel" ref={policyRef}>
            <div className="section-heading">
              <h2><ShieldCheck size={18} /> Policy Priorities</h2>
              <div className="inline-actions">
                <button className="secondary" onClick={addPolicy}><Plus size={16} /> Add</button>
                <button className="secondary" onClick={equalizePolicyWeights}>Equalize</button>
              </div>
            </div>
            <p className="section-help">Define what “good” means so prompt changes are tested instead of guessed. Equal weights are fine to start; raise weights for rules that should block promotion when they regress.</p>
            {policies.map((rule, index) => (
              <div className="policy" key={`${rule.id}-${index}`}>
                <input value={rule.id} onChange={(event) => updatePolicy(index, "id", event.target.value)} />
                <input value={rule.text} onChange={(event) => updatePolicy(index, "text", event.target.value)} />
                <input type="number" min="0" max="1" step="0.05" value={rule.weight} onChange={(event) => updatePolicy(index, "weight", event.target.value)} />
                <button className="icon-button" onClick={() => removePolicy(index)} aria-label="Remove policy">
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
            <div className={Math.abs(totalWeight - 1) > 0.01 ? "warn" : "muted"}>
              Total weight: {totalWeight.toFixed(2)} {Math.abs(totalWeight - 1) > 0.01 ? "- normalized by backend if needed" : ""}
            </div>
            <div className="eval-checks">
              <div className="history-title">Eval checks</div>
              <p className="section-help">These checks turn policy intent into measurable acceptance criteria. They are used for optimization scoring, regression signals, and review insights.</p>
              {evalAssertions.map((assertion, index) => (
                <label className="eval-check" key={assertion.id}>
                  <input
                    type="checkbox"
                    checked={assertion.enabled}
                    onChange={(event) => {
                      const next = [...evalAssertions];
                      next[index] = { ...assertion, enabled: event.target.checked };
                      setEvalAssertions(next);
                    }}
                  />
                  <span>
                    <strong>{assertion.metric}</strong>
                    {assertion.description}
                  </span>
                </label>
              ))}
            </div>
          </div>
          )}

          {activeView === "models" && (
          <div className="panel" ref={providerRef}>
            <h2><Settings2 size={18} /> Models & Credentials</h2>
            <p className="section-help">Choose the model that improves the artifact, then choose where the final prompt or skill will run. If providers differ, Aiterate asks for both credentials.</p>
            <label>Optimizer provider</label>
            <select value={optimizerProvider} onChange={(event) => {
              setOptimizerProvider(event.target.value);
              setOptimizerModel(defaultModelForProvider(event.target.value, modelCatalog));
              setCustomOptimizerModel("");
            }}>
              {providerOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
            <label>Optimizer model</label>
            <ModelSelect
              provider={optimizerProvider}
              value={optimizerModel}
              onChange={setOptimizerModel}
              customValue={customOptimizerModel}
              onCustomChange={setCustomOptimizerModel}
              catalog={modelCatalog}
            />
            <label>Prompt will be used with</label>
            <select value={targetProvider} onChange={(event) => handleTargetProviderChange(event.target.value)}>
              {providerOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
            <label>Target model</label>
            <ModelSelect
              provider={targetProvider}
              value={targetModel}
              onChange={setTargetModel}
              customValue={customTargetModel}
              onCustomChange={setCustomTargetModel}
              catalog={modelCatalog}
            />
            <div className="credential-grid">
              <CredentialControl
                title="Optimizer credential"
                provider={optimizerProvider}
                status={integrationStatus}
                setup={integrationSetup}
                reuseCredential={reuseOptimizerCredential}
                setReuseCredential={setReuseOptimizerCredential}
                secretValue={optimizerKey}
                setSecretValue={setOptimizerKey}
                onAddSecret={(item) => {
                  setSecretDraft(item);
                  setSecretValue("");
                  setShowSecretValue(false);
                  setSecretSaveError("");
                }}
                onDeleteSecret={(secret) => setSecretDeleteDraft(secret)}
              />
              <CredentialControl
                title="Target credential"
                provider={targetProvider}
                status={integrationStatus}
                setup={integrationSetup}
                reuseCredential={optimizerProvider === targetProvider ? reuseOptimizerCredential : reuseTargetCredential}
                setReuseCredential={optimizerProvider === targetProvider ? setReuseOptimizerCredential : setReuseTargetCredential}
                secretValue={optimizerProvider === targetProvider ? optimizerKey : targetKey}
                setSecretValue={optimizerProvider === targetProvider ? setOptimizerKey : setTargetKey}
                sharedWithOptimizer={optimizerProvider === targetProvider}
                onAddSecret={(item) => {
                  setSecretDraft(item);
                  setSecretValue("");
                  setShowSecretValue(false);
                  setSecretSaveError("");
                }}
                onDeleteSecret={(secret) => setSecretDeleteDraft(secret)}
              />
            </div>
            <div className="inline-actions provider-test-actions">
              <button className="secondary" onClick={() => testSelectedProvider("optimizer")} disabled={Boolean(testingProviderRole)}>
                {testingProviderRole === "optimizer" ? "Testing optimizer..." : "Test optimizer provider"}
              </button>
              <button className="secondary" onClick={() => testSelectedProvider("target")} disabled={Boolean(testingProviderRole)}>
                {testingProviderRole === "target" ? "Testing target..." : "Test target provider"}
              </button>
            </div>
            {providerTestStatus && <div className="integration">{providerTestStatus}</div>}
          </div>
          )}

          {activeView === "tracking" && (
          <div className="panel" ref={trackingRef}>
            <h2><DatabaseZap size={18} /> Tracking</h2>
            <p className="section-help">Optional: send run metadata, scores, traces, and artifacts to MLflow or LangSmith. You can skip this for a quick local run and enable tracking later.</p>
            <div className="infra-note">Tracking is optional for optimization. Enable it when you want external experiment history or trace review.</div>
            <label>Experiment tracking</label>
            <select value={tracker} onChange={(event) => {
              setTracker(event.target.value);
              setReuseTrackingCredential(false);
              setTrackingSecret("");
            }}>
              <option value="noop">No external tracking</option>
              <option value="mlflow">MLflow</option>
              <option value="langsmith">LangSmith</option>
            </select>
            {tracker === "mlflow" && (
              <>
                <label>MLflow tracking URI</label>
                <input placeholder={defaultMlflowTrackingUri} value={mlflowUri} onChange={(event) => setMlflowUri(event.target.value)} />
                <div className="infra-note">
                  Default for Docker Compose: <code>{defaultMlflowTrackingUri}</code>. From inside the API container, localhost means the container itself. Use this URL for MLflow running on your machine, or replace it with the MLflow service URL if MLflow runs inside Compose or Kubernetes.
                </div>
              </>
            )}
            {tracker === "langsmith" && (
              <>
                <label>LangSmith endpoint URL</label>
                <input placeholder="https://api.smith.langchain.com" value={langsmithUrl} onChange={(event) => setLangsmithUrl(event.target.value)} />
                <label>LangSmith project</label>
                <input placeholder="aiterate-prompts" value={langsmithProject} onChange={(event) => setLangsmithProject(event.target.value)} />
              </>
            )}
            {tracker !== "noop" && (
              <TrackingCredentialControl
                tracker={tracker}
                setup={integrationSetup}
                status={integrationStatus}
                reuseCredential={reuseTrackingCredential}
                setReuseCredential={setReuseTrackingCredential}
                secretValue={trackingSecret}
                setSecretValue={setTrackingSecret}
                onAddSecret={(item) => {
                  setSecretDraft(item);
                  setSecretValue("");
                  setShowSecretValue(false);
                  setSecretSaveError("");
                }}
                onDeleteSecret={(secret) => setSecretDeleteDraft(secret)}
              />
            )}
            <div className="infra-note">
              Enterprise tracking should use TLS, private network access where possible, and server-side saved tokens/keys. Endpoint URLs are configuration; tokens and API keys are stored as credentials.
            </div>
          </div>
          )}

          {activeView === "run" && (
          <div className="panel wide result" ref={resultsRef} tabIndex="-1">
            <div className="section-heading">
              <h2><BarChart3 size={18} /> Run Optimizer</h2>
              <ValidationSummary issues={validationIssues} />
            </div>
            <p className="section-help">Run after data, policies, models, and credentials are ready. Aiterate reflects on failures, applies bounded prompt/skill edits, validates candidates, accepts improvements, and retains rejected changes for review.</p>
            <div className="run-controls">
              <div className="run-control-card">
                <div>
                  <label>Optimization depth</label>
                  <p className="section-help compact-help">Standard is a good first run. Increase it when you want more attempts; lower it for a quick smoke test.</p>
                </div>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="1"
                  value={iterations}
                  onChange={(event) => setIterations(Number(event.target.value))}
                />
                <div className="control-value">
                  <strong>{optimizationDepthLabel(iterations)}</strong>
                  <span>{iterations} iteration{iterations === 1 ? "" : "s"}</span>
                </div>
              </div>
              <div className="run-control-card">
                <div>
                  <label>Promotion threshold</label>
                  <p className="section-help compact-help">The score you want before approving a version. Start at 0.80 for production-minded review.</p>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={promotionThreshold}
                  onChange={(event) => setPromotionThreshold(Number(event.target.value))}
                />
                <div className="control-value">
                  <strong>{promotionThreshold.toFixed(2)}</strong>
                  <span>approval target</span>
                </div>
              </div>
              <div className="run-control-card">
                <label>Max spend cap</label>
                <p className="section-help compact-help">Optional estimated USD cap for this run. Aiterate stops adding new optimizer attempts when the estimate reaches the cap.</p>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  placeholder="No cap"
                  value={maxBudgetUsd}
                  onChange={(event) => setMaxBudgetUsd(event.target.value)}
                />
              </div>
              <div className="run-control-card">
                <label>Reproducibility seed</label>
                <p className="section-help compact-help">Keep the same seed to make train/test split and repeatable choices stable across reruns.</p>
                <input
                  type="number"
                  step="1"
                  value={seed}
                  onChange={(event) => setSeed(Number(event.target.value || 0))}
                />
              </div>
              <label className={runTargetValidation ? "run-control-card check-card large-check-card active" : "run-control-card check-card large-check-card"}>
                <input
                  type="checkbox"
                  checked={runTargetValidation}
                  onChange={(event) => setRunTargetValidation(event.target.checked)}
                />
                <span>
                  <strong>Test with target model</strong>
                  <small>Optional: ask the model that will use the prompt to answer holdout examples, then score those answers. This can add provider cost.</small>
                </span>
              </label>
            </div>
            {runTargetValidation && (
              <div className="integration target-validation-note">
                Target-model testing is enabled. After optimization, Aiterate will send the best artifact to {providerLabel(targetProvider)} / {selectedModelValue(targetModel, customTargetModel) || "selected target model"} on the holdout examples and show a separate answer-quality report.
              </div>
            )}
            <div className="run-action-row">
              <button className="primary run-button" onClick={optimize} disabled={loading || !canRun}>
                <Play size={18} />
                {loading ? "Running optimization" : canRun ? "Run optimizer" : "Complete setup to run"}
              </button>
              {loading && (
                <button className="secondary stop-button" onClick={stopOptimization}>
                  Stop run
                </button>
              )}
            </div>
            {error ? (
              <ErrorMessage message={error} action={errorAction} onAction={handleErrorAction} />
            ) : run?.best_version ? (
              <RunResults
                run={run}
                approval={approval}
                prStatus={prStatus}
                approvalRef={approvalRef}
                approveButtonRef={approveButtonRef}
                gitSettings={projectGitSettings}
                onApprove={() => approveBestVersion(run)}
              />
            ) : (
              <div className="empty">Upload data, confirm policy weights, choose providers, then run optimization.</div>
            )}
          </div>
          )}

          {activeView === "review" && (
          <div className="panel wide result review-only" ref={approvalRef} tabIndex="-1">
            <div className="section-heading">
              <h2><Check size={18} /> Review and Approve</h2>
              <ValidationSummary issues={run?.best_version ? [] : ["Run the optimizer before reviewing an approval candidate."]} />
            </div>
            <p className="section-help">Review the best version, inspect candidate changes, compare models if needed, and approve only when the artifact is ready to promote.</p>
            {run?.best_version ? (
              <RunResults
                run={run}
                approval={approval}
                prStatus={prStatus}
                approvalRef={approvalRef}
                approveButtonRef={approveButtonRef}
                gitSettings={projectGitSettings}
                onApprove={() => approveBestVersion(run)}
              />
            ) : (
              <div className="empty">No run is ready for review yet. Go to Run optimizer after setup is complete.</div>
            )}
          </div>
          )}

          {activeView === "git" && (
          <div className="panel wide" ref={gitPrRef}>
            <h2><GitBranch size={18} /> Git PR Promotion</h2>
            <p className="section-help">After approval, create a promotion PR so teams can review the generated prompt or skill like any other production artifact.</p>
            {projectSettingsSaveStatus && <div className="infra-note">{projectSettingsSaveStatus}</div>}
            {gitBrowserAuthAvailable(integrationSetup) ? (
              <div className="inline-actions">
                {gitSetupItem("GitHub", integrationSetup)?.browser_auth && (
                  <button className="secondary" onClick={() => startGitBrowserAuth("github")}>
                    Connect GitHub
                  </button>
                )}
                {gitSetupItem("Bitbucket", integrationSetup)?.browser_auth && (
                  <button className="secondary" onClick={() => startGitBrowserAuth("bitbucket")}>
                    Connect Bitbucket
                  </button>
                )}
              </div>
            ) : (
              <div className="infra-note">
                Browser Git auth is not configured for this server. For v1, save a GitHub or Bitbucket token below, or set OAuth app credentials in the backend environment to enable browser connect.
              </div>
            )}
            {gitAuthStatus && <div className="infra-note">{gitAuthStatus}</div>}
            {integrationSetup?.git && (
              <SetupGroup title="Git credentials" items={integrationSetup.git} stored={integrationSetup.stored_secrets} onAddSecret={(item) => {
                setSecretDraft(item);
                setSecretValue("");
                setShowSecretValue(false);
                setSecretSaveError("");
              }} onDeleteSecret={(secret) => setSecretDeleteDraft(secret)} />
            )}
            <label className="check">
              <input type="checkbox" checked={enableGit} onChange={(event) => updateProjectGitSetting(setEnableGit, event.target.checked)} />
              Enable Git artifact tracking
            </label>
            {enableGit && (
              <>
                <label>Git remote</label>
                <input
                  placeholder="https://github.com/org/repo.git"
                  value={gitRemote}
                  onChange={(event) => updateProjectGitSetting(setGitRemote, event.target.value)}
                  onBlur={saveProjectGitSettings}
                />
                <div className="infra-note">
                  Promotion branch is generated automatically, for example <code>aiterate/promote-art-...</code>. Choose only the PR base branch below.
                </div>
              </>
            )}
            <label className="check">
              <input type="checkbox" checked={createPr} onChange={(event) => updateProjectGitSetting(setCreatePr, event.target.checked)} />
              Enable promotion PR workflow
            </label>
            {createPr && (
              <>
                <label>PR base branch</label>
                <input
                  value={prBase}
                  onChange={(event) => updateProjectGitSetting(setPrBase, event.target.value)}
                  onBlur={saveProjectGitSettings}
                />
                <div className="infra-note">Save a GitHub or Bitbucket token for PR publishing. Browser connect appears only when OAuth app credentials are configured on the backend.</div>
              </>
            )}
            <div className="inline-actions">
              <button className="secondary" onClick={saveProjectGitSettings}>
                Save Git settings
              </button>
            </div>
            <div className="infra-note">Create PR is available after a run is reviewed and approved.</div>
            {run?.best_version && (
              <button
                className="secondary run-button"
                onClick={() => exportText(`${run.name}-promotion-package.json`, JSON.stringify(promotionPackage(run, approval, projectGitSettings), null, 2))}
              >
                Export package without Git
              </button>
            )}
            <button className="primary run-button" onClick={() => requestPullRequest(run)} disabled={!canCreatePr}>
              <GitBranch size={16} />
              Create Git PR
            </button>
            {!canCreatePr && (
              <div className="muted">To create a PR, approve a run, enable PR workflow, enter Git remote, and set PR base branch.</div>
            )}
            {prStatus && <div className="integration">{prStatus}</div>}
          </div>
          )}

          {activeView === "compare" && (
          <div className="panel wide" ref={comparisonRef}>
            <h2><BarChart3 size={18} /> Compare Models</h2>
            <p className="section-help">
              Use the same approved prompt or skill candidate against two models. This helps teams decide whether a prompt is portable, whether the target model needs extra guardrails, or whether a cheaper model still passes the same evaluation signals.
            </p>
            {!compareSourceAvailable ? (
              <div className="empty">Approve a run first. Approved artifacts from Run History will appear here for model comparison.</div>
            ) : (
              <>
                <div className="run-control-card compare-source-card">
                  <label>Approved artifact to compare</label>
                  <p className="section-help compact-help">
                    Select any approved run. Aiterate uses that run's approved prompt or skill, source data, and weighted policies for the comparison.
                  </p>
                  <select
                    value={compareSourceRunId || (currentRunApproved ? run?.id || "" : "")}
                    onChange={(event) => setCompareSourceRunId(event.target.value)}
                  >
                    {currentRunApproved && run?.id && (
                      <option value={run.id}>
                        Current approved run - {run.name} - score {run.best_version?.score ?? "n/a"}
                      </option>
                    )}
                    {approvedRunOptions
                      .filter((item) => item.id !== run?.id)
                      .map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.project} - {item.id} - score {item.best_score ?? "n/a"}
                        </option>
                      ))}
                  </select>
                </div>
                <div className="compare-config">
                  <div className="run-control-card">
                    <div className="history-title">Model A</div>
                    <label>Provider</label>
                    <select value={compareModelAProvider} onChange={(event) => {
                      setCompareModelAProvider(event.target.value);
                      setCompareModelAModel(defaultModelForProvider(event.target.value, modelCatalog));
                      setCompareCustomModelA("");
                    }}>
                      {providerOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                    </select>
                    <label>Model</label>
                    <ModelSelect
                      provider={compareModelAProvider}
                      value={compareModelAModel}
                      onChange={setCompareModelAModel}
                      customValue={compareCustomModelA}
                      onCustomChange={setCompareCustomModelA}
                      catalog={modelCatalog}
                    />
                  </div>
                  <div className="run-control-card">
                    <div className="history-title">Model B</div>
                    <label>Provider</label>
                    <select value={compareModelBProvider} onChange={(event) => {
                      setCompareModelBProvider(event.target.value);
                      setCompareModelBModel(defaultModelForProvider(event.target.value, modelCatalog));
                      setCompareCustomModelB("");
                    }}>
                      {providerOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                    </select>
                    <label>Model</label>
                    <ModelSelect
                      provider={compareModelBProvider}
                      value={compareModelBModel}
                      onChange={setCompareModelBModel}
                      customValue={compareCustomModelB}
                      onCustomChange={setCompareCustomModelB}
                      catalog={modelCatalog}
                    />
                  </div>
                </div>
                <label className="run-control-card check-card compare-live-toggle">
                  <input
                    type="checkbox"
                    checked={compareExecuteLive}
                    onChange={(event) => setCompareExecuteLive(event.target.checked)}
                  />
                  <span>
                    <strong>Run live model eval</strong>
                    <small>Calls each selected provider on holdout examples using the same prompt. Leave off for a quick offline comparison with no provider cost.</small>
                  </span>
                </label>
                <div className="inline-actions">
                  <button className="primary run-button" onClick={compareCurrentModels} disabled={comparingModels}>
                    <BarChart3 size={16} />
                    {comparingModels ? "Running comparison" : "Compare selected models"}
                  </button>
                </div>
                {error && <ErrorMessage message={error} action={errorAction} onAction={handleErrorAction} />}
                {modelComparison ? (
                  <ModelComparison comparison={modelComparison} />
                ) : (
                  <div className="empty">Pick two models, then run a comparison for the current best version.</div>
                )}
              </>
            )}
          </div>
          )}
        </section>
      </section>
      {secretDraft && (
        <div className="modal-backdrop">
          <div className="modal">
              <h2><KeyRound size={18} /> Add Secret</h2>
            <p>{secretDraft.name}: paste {secretDraft.secret_name} once. Aiterate stores it encrypted and only shows the last 4 characters after save.</p>
            <div className="secret-input-wrap">
              <input
                type={showSecretValue ? "text" : "password"}
                autoComplete="off"
                placeholder={secretDraft.secret_name}
                value={secretValue}
              onChange={(event) => setSecretValue(event.target.value)}
              />
              <button
                className="secondary icon-button"
                type="button"
                onClick={() => setShowSecretValue(!showSecretValue)}
                aria-label={showSecretValue ? "Hide secret value" : "Show secret value"}
                title={showSecretValue ? "Hide secret value" : "Show secret value"}
              >
                {showSecretValue ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {secretSaveError && <div className="error modal-error">{secretSaveError}</div>}
            <div className="modal-actions">
              <button className="secondary" onClick={() => {
                setSecretDraft(null);
                setShowSecretValue(false);
                setSecretSaveError("");
              }}>Cancel</button>
              <button className="primary" onClick={saveSecret} disabled={!secretValue.trim()}>Save encrypted secret</button>
            </div>
          </div>
        </div>
      )}
      {resetConfirmOpen && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2><RefreshCw size={18} /> Reset Workspace</h2>
            <p>This clears the current browser workspace and starts a new draft.</p>
            <ul className="confirm-list">
              <li>Clears pasted/uploaded raw data and file details.</li>
              <li>Restores default project name and policy priorities.</li>
              <li>Clears current optimization results, model comparison, approval, PR status, and errors.</li>
              <li>Does not delete saved credentials, previous backend runs, Git history, MLflow, or LangSmith records.</li>
            </ul>
            <div className="modal-actions">
              <button className="secondary" onClick={() => setResetConfirmOpen(false)}>Cancel</button>
              <button
                className="danger-button"
                onClick={() => {
                  resetWorkspace();
                  setResetConfirmOpen(false);
                }}
              >
                Reset workspace
              </button>
            </div>
          </div>
        </div>
      )}
      {secretDeleteDraft && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2><Trash2 size={18} /> Delete Credential</h2>
            <p>
              Remove saved credential <strong>{secretDeleteDraft.name}</strong>
              {secretDeleteDraft.integration ? ` for ${secretDeleteDraft.integration}` : ""}? This cannot be undone.
            </p>
            <div className="modal-actions">
              <button className="secondary" onClick={() => setSecretDeleteDraft(null)}>Cancel</button>
              <button className="danger-button" onClick={deleteSecret}>Delete credential</button>
            </div>
          </div>
        </div>
      )}
      {runDeleteDraft && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2><Trash2 size={18} /> Delete Previous Run</h2>
            <p>
              Delete run <strong>{runDeleteDraft.id}</strong>
              {runDeleteDraft.model ? ` for ${runDeleteDraft.model}` : ""}? This removes the saved run record from Aiterate history.
            </p>
            <ul className="confirm-list">
              <li>Removes this run from Run History.</li>
              <li>Clears the current review if this run is open.</li>
              <li>Does not delete provider credentials, MLflow/LangSmith records, or Git commits.</li>
            </ul>
            <div className="modal-actions">
              <button className="secondary" onClick={() => setRunDeleteDraft(null)}>Cancel</button>
              <button className="danger-button" onClick={deletePreviousRun}>Delete run</button>
            </div>
          </div>
        </div>
      )}
      {projectDeleteDraft && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2><Trash2 size={18} /> Delete Project History</h2>
            <p>
              Delete all saved runs for <strong>{projectDeleteDraft.name}</strong>? When the runs are removed,
              this project will disappear from Run History.
            </p>
            <ul className="confirm-list">
              <li>Deletes {projectDeleteDraft.runs?.length || 0} saved run record(s) for this project.</li>
              <li>Clears the current review if it belongs to this project.</li>
              <li>Does not delete credentials, MLflow/LangSmith records, Git commits, or exported files.</li>
            </ul>
            <div className="modal-actions">
              <button className="secondary" onClick={() => setProjectDeleteDraft(null)}>Cancel</button>
              <button className="danger-button" onClick={deleteProjectRuns}>Delete project history</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function Step({ number, title, complete, optional, selected, running, onClick }) {
  return (
    <button className={`step ${complete ? "complete" : ""} ${optional ? "optional" : ""} ${selected ? "selected" : ""} ${running ? "running" : ""}`} onClick={onClick}>
      <span>{complete ? <Check size={16} /> : running ? <RefreshCw size={15} /> : optional ? "!" : number}</span>
      <strong>{title}</strong>
    </button>
  );
}

function ContextUpload({ title, badge, icon, help, files, onUpload }) {
  return (
    <div className="context-upload-card">
      <div className="context-upload-top">
        <div className="context-upload-icon">{icon}</div>
        <div>
          <div className="context-upload-title-row">
            <strong>{title}</strong>
            <span>{badge}</span>
          </div>
          <p>{help}</p>
        </div>
      </div>
      <label className="upload">
        <input type="file" accept={supportedTypes.join(",")} onChange={onUpload} multiple />
        Upload files
      </label>
      {files?.length > 0 && (
        <div className="file-list">
          {files.map((file) => (
            <div className="file-pill" key={`${file.name}-${file.size}`}>
              {file.name} - {file.extension} - {(file.size / 1024).toFixed(1)} KB
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CredentialControl({
  title,
  provider,
  status,
  setup,
  reuseCredential,
  setReuseCredential,
  secretValue,
  setSecretValue,
  sharedWithOptimizer = false,
  onAddSecret,
  onDeleteSecret,
}) {
  const item = providerSetupItem(provider, setup);
  const providerStatus = status?.providers?.[provider];
  const stored = setup?.stored_secrets?.find((secret) => secret.name === item?.secret_name);
  const hasUsableStoredCredential = stored?.valid !== false && Boolean(stored);
  const canReuse = Boolean(providerStatus?.configured || hasUsableStoredCredential);
  return (
    <div className="credential-card">
      <div className="credential-title">
        <strong>{title}</strong>
        <span>{providerLabel(provider)}</span>
      </div>
      <div className={providerStatus?.configured || hasUsableStoredCredential ? "configured" : "not-configured"}>
        {sharedWithOptimizer
          ? "Same provider as optimizer; using the optimizer credential"
          : providerStatus?.configured || hasUsableStoredCredential ? "Server credential configured" : "Credential needed"}
      </div>
      {stored?.fingerprint && (
        <div className="stored-secret-row">
          <span>
            Stored key: {maskedSecretEnding(stored.fingerprint)}
            {stored.valid === false ? " - needs update" : ""}
          </span>
          <button className="danger-link" onClick={() => onDeleteSecret(stored)}>Delete</button>
        </div>
      )}
      {!sharedWithOptimizer && (
        <>
          <label className="check">
            <input
              type="checkbox"
              checked={canReuse && reuseCredential}
              disabled={!canReuse}
              onChange={(event) => setReuseCredential(event.target.checked)}
            />
            {canReuse ? "Reuse saved credential" : stored ? "Saved credential needs update" : "No saved credential yet"}
          </label>
          {(!canReuse || !reuseCredential) && (
            <div className="secret-row">
              <KeyRound size={16} />
              <input type="password" placeholder="Paste key for this run only" value={secretValue} onChange={(event) => setSecretValue(event.target.value)} autoComplete="off" />
            </div>
          )}
        </>
      )}
      {item?.secret_name && (
        <button className="secondary" onClick={() => onAddSecret(item)}>
          <KeyRound size={16} />
          Save server credential
        </button>
      )}
    </div>
  );
}

function TrackingCredentialControl({
  tracker,
  setup,
  status,
  reuseCredential,
  setReuseCredential,
  secretValue,
  setSecretValue,
  onAddSecret,
  onDeleteSecret,
}) {
  const item = trackingSetupItem(tracker, setup);
  const trackerStatus = status?.tracking?.[tracker];
  const stored = setup?.stored_secrets?.find((secret) => secret.name === item?.secret_name);
  const hasUsableStoredCredential = stored?.valid !== false && Boolean(stored);
  const canReuse = Boolean(trackerStatus?.credential_configured || hasUsableStoredCredential);
  const label = tracker === "mlflow" ? "MLflow access token" : "LangSmith API key";
  return (
    <div className="credential-card tracking-card">
      <div className="credential-title">
        <strong>{label}</strong>
        <span>{tracker === "mlflow" ? "Enterprise tracking credential" : "Trace API credential"}</span>
      </div>
      <div className={canReuse ? "configured" : "not-configured"}>
        {canReuse ? "Saved credential available" : "Credential not saved yet"}
      </div>
      {stored?.fingerprint && (
        <div className="stored-secret-row">
          <span>
            Stored value: {maskedSecretEnding(stored.fingerprint)}
            {stored.valid === false ? " - needs update" : ""}
          </span>
          <button className="danger-link" onClick={() => onDeleteSecret(stored)}>Delete</button>
        </div>
      )}
      <label className="check">
        <input
          type="checkbox"
          checked={canReuse && reuseCredential}
          disabled={!canReuse}
          onChange={(event) => setReuseCredential(event.target.checked)}
        />
        {canReuse ? "Reuse saved tracking credential" : stored ? "Saved tracking credential needs update" : "No saved tracking credential yet"}
      </label>
      {(!canReuse || !reuseCredential) && (
        <div className="secret-row">
          <KeyRound size={16} />
          <input
            type="password"
            placeholder={tracker === "mlflow" ? "Paste MLflow tracking token for this run" : "Paste LangSmith API key for this run"}
            value={secretValue}
            onChange={(event) => setSecretValue(event.target.value)}
            autoComplete="off"
          />
        </div>
      )}
      {item?.secret_name && (
        <button className="secondary" onClick={() => onAddSecret(item)}>
          <KeyRound size={16} />
          Save tracking credential
        </button>
      )}
    </div>
  );
}

function ModelSelect({
  provider,
  value,
  onChange,
  customValue,
  onCustomChange,
  catalog,
}) {
  const models = catalog?.[provider] || [];
  const currentValue = models.some((model) => model.id === value) ? value : "__custom";
  return (
    <>
      <select value={currentValue} onChange={(event) => {
        const next = event.target.value;
        onChange(next === "__custom" ? "__custom" : next);
        if (next !== "__custom") onCustomChange("");
      }}>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label} ({model.source === "live" ? "provider" : model.id})
          </option>
        ))}
        <option value="__custom">Custom model or deployment</option>
      </select>
      {currentValue === "__custom" && (
        <input
          className="custom-model-input"
          value={customValue}
          onChange={(event) => {
            onCustomChange(event.target.value);
            onChange("__custom");
          }}
          placeholder="Enter model, deployment, or local OpenAI-compatible model id"
        />
      )}
    </>
  );
}

function SetupGroup({ title, items, stored, onAddSecret, onDeleteSecret }) {
  return (
    <div className="setup-group">
      <div className="history-title">{title}</div>
      {items.map((item) => (
        <div className="setup-item" key={item.name}>
          <div className="setup-item-header">
            <strong>{item.name}</strong>
            {item.secret_name && <button className="secondary" onClick={() => onAddSecret(item)}>Add / update</button>}
          </div>
          <span className={item.configured ? "configured" : "not-configured"}>
            {item.configured ? "Configured" : "Needs credential"}
          </span>
          {item.browser_auth !== undefined && (
            <span className={item.browser_auth ? "configured" : "muted"}>
              Browser connect: {item.browser_auth ? "available" : "not configured"}
            </span>
          )}
          {stored?.find((secret) => secret.name === item.secret_name)?.fingerprint && (
            <div className="stored-secret-row">
              <span>
                Stored key: {maskedSecretEnding(stored.find((secret) => secret.name === item.secret_name).fingerprint)}
                {stored.find((secret) => secret.name === item.secret_name).valid === false ? " - needs update" : ""}
              </span>
              <button className="danger-link" onClick={() => onDeleteSecret(stored.find((secret) => secret.name === item.secret_name))}>Delete</button>
            </div>
          )}
          <code>{item.env.join(", ")}</code>
        </div>
      ))}
    </div>
  );
}

function RunResults({ run, approval, approvalRef, approveButtonRef, gitSettings, onApprove }) {
  const accepted = run.accepted_versions || [];
  const rejected = run.rejected_versions || [];
  const acceptedImprovements = accepted.filter((version) => version.version > 1);
  const approved = approval?.status === "approved";
  return (
    <>
      <RunSummary run={run} acceptedImprovements={acceptedImprovements} rejected={rejected} />
      <div className="result-grid">
        <div><strong>{run.best_version.score}</strong><span>Best score</span></div>
        <div><strong>{run.optimizer?.promotion_threshold ?? "0.80"}</strong><span>Promotion threshold</span></div>
        <div><strong>{formatCost(run.cost_estimate)}</strong><span>Estimated cost</span></div>
        <div><strong>{run.optimizer?.iterations ?? run.optimizer?.requested_iterations ?? "n/a"}</strong><span>Iterations requested</span></div>
        <div><strong>{acceptedImprovements.length}</strong><span>{acceptedImprovements.length === 1 ? "Candidate for approval" : "Candidates for approval"}</span></div>
        <div><strong>{rejected.length}</strong><span>Rejected candidates</span></div>
      </div>
      {run.optimizer?.budget_stop_reason && (
        <div className="integration warning-note">{run.optimizer.budget_stop_reason}</div>
      )}
      {run.optimizer?.tracking_error && (
        <div className="integration warning-note">
          Tracking did not complete, but the optimization run continued. {run.optimizer.tracking_error}
        </div>
      )}
      {run.optimizer?.target_validation_error && (
        <div className="integration warning-note">{run.optimizer.target_validation_error}</div>
      )}
      {run.cost_estimate && (
        <div className="integration">
          Cost estimate uses approximate model prices: {run.cost_estimate.input_tokens} input tokens,
          {" "}{run.cost_estimate.output_tokens} output tokens. Verify provider billing before production use.
        </div>
      )}
      {run.behavior_report && <TargetValidationReport report={run.behavior_report} />}
      {run.insights && <Insights insights={run.insights} />}
      {run.evaluation_report && <EvaluationReport report={run.evaluation_report} />}
      <div className={approved ? "approval-review approved" : "approval-review"} ref={approvalRef} tabIndex="-1">
        <div>
          <div className="history-title">Review approval candidate</div>
          <p className="section-help">
            Inspect the best version, compare changes, then manually approve it when it is ready for promotion.
          </p>
          <div className="approval-candidate">
            <div><strong>Version</strong><span>v{run.best_version.version}</span></div>
            <div><strong>Score</strong><span>{run.best_version.score}</span></div>
            <div><strong>Status</strong><span>{approved ? "Approved" : "Waiting for manual approval"}</span></div>
          </div>
        </div>
        <div className="approval-actions">
          <button className="secondary" onClick={() => navigator.clipboard?.writeText(run.best_version.content)}>
            Copy best version
          </button>
          <button className="primary" ref={approveButtonRef} onClick={onApprove} disabled={approved}>
            <Check size={16} />
            {approved ? "Approved" : "Approve best version"}
          </button>
        </div>
      </div>
      <div className="inline-actions result-actions">
        <button className="secondary" onClick={() => navigator.clipboard?.writeText(run.best_version.content)}>
          Copy prompt/skill
        </button>
        <ExportSelect run={run} approval={approval} gitSettings={gitSettings} />
      </div>
      {run.evaluation_report && <RegressionEvidence report={run.evaluation_report} optimizer={run.optimizer} />}
      <VersionReview accepted={accepted} rejected={rejected} bestVersion={run.best_version} />
      {approval?.message && <div className="integration">{approval.message}</div>}
      <div className="artifact-preview">
        <div className="history-title">Current best prompt / skill</div>
        <pre>{displayArtifactContent(run.best_version.content)}</pre>
      </div>
    </>
  );
}

function RunSummary({ run, acceptedImprovements, rejected }) {
  const firstScore = run.accepted_versions?.[0]?.score;
  const bestScore = run.best_version?.score;
  const delta = firstScore != null && bestScore != null ? Number((bestScore - firstScore).toFixed(3)) : null;
  return (
    <div className="run-summary">
      <strong>Run summary</strong>
      <span>
        Aiterate trained on {run.optimizer?.train_case_count ?? "n/a"} example(s), tested on{" "}
        {run.optimizer?.validation_case_count ?? "n/a"} holdout example(s), kept{" "}
        {acceptedImprovements.length} approval candidate(s), and skipped {rejected.length} attempt(s).
        {delta !== null ? ` Score moved from ${firstScore} to ${bestScore} (${delta >= 0 ? "+" : ""}${delta}).` : ""}
        {run.behavior_report ? ` Target-model validation pass rate was ${Math.round(run.behavior_report.pass_rate * 100)}%.` : ""}
      </span>
    </div>
  );
}

function ExportSelect({ run, approval, gitSettings }) {
  function handleExport(value) {
    if (value === "markdown") {
      exportText(`${run.name}-artifact.md`, run.best_version.content);
    } else if (value === "run") {
      exportText(`${run.name}-run.json`, JSON.stringify(run, null, 2));
    } else if (value === "package") {
      exportText(`${run.name}-promotion-package.json`, JSON.stringify(promotionPackage(run, approval, gitSettings), null, 2));
    } else if (value === "sdk") {
      navigator.clipboard?.writeText(sdkSnippet(run));
    }
  }

  return (
    <label className="export-select">
      <span>Export</span>
      <select
        defaultValue=""
        onChange={(event) => {
          handleExport(event.target.value);
          event.target.value = "";
        }}
      >
        <option value="" disabled>Choose format</option>
        <option value="markdown">Prompt/skill Markdown</option>
        <option value="package">Promotion package JSON</option>
        <option value="run">Full run JSON</option>
        <option value="sdk">Copy SDK snippet</option>
      </select>
    </label>
  );
}

function VersionReview({ accepted, rejected, bestVersion }) {
  const lastAccepted = accepted[accepted.length - 1];
  const [selectedId, setSelectedId] = useState(bestVersion?.id || lastAccepted?.id || "");
  const [selectedRejectedId, setSelectedRejectedId] = useState("");
  useEffect(() => {
    setSelectedId(bestVersion?.id || accepted[accepted.length - 1]?.id || "");
    setSelectedRejectedId("");
  }, [bestVersion?.id, accepted.length, accepted]);
  const selected = accepted.find((version) => version.id === selectedId) || bestVersion || lastAccepted;
  const selectedRejected = rejected.find((version) => version.id === selectedRejectedId);
  const previous = selected?.parent_version_id
    ? accepted.find((version) => version.id === selected.parent_version_id)
    : null;
  const scoreDelta = previous && selected
    ? Number((selected.score - previous.score).toFixed(3))
    : null;
  return (
    <div className="version-review">
      <div className="history-title">Version review</div>
      <p className="section-help">
        Review how the prompt or skill changed. The first draft is the starting point; each improvement was kept only when it scored better.
        Select a version to compare it with the version before it.
      </p>
      <VersionProgressTimeline
        versions={accepted}
        selectedId={selected?.id}
        bestVersion={bestVersion}
        onSelect={setSelectedId}
      />
      <div className="version-tabs">
        {accepted.map((version) => (
          <button
            key={version.id}
            className={version.id === selected?.id ? "version-tab active" : "version-tab"}
            onClick={() => setSelectedId(version.id)}
          >
            v{version.version}
            <span>{version.version === 1 ? "Starting draft" : version.id === bestVersion?.id ? "Best version" : "Improvement"}</span>
          </button>
        ))}
      </div>
      {selected && (
        <>
          <div className="version-summary">
            <div><strong>Selected</strong><span>v{selected.version}</span></div>
            <div><strong>Score</strong><span>{selected.score}{scoreDelta !== null ? ` (${scoreDelta >= 0 ? "+" : ""}${scoreDelta})` : ""}</span></div>
            <div><strong>What changed</strong><span>{humanizeChangeSummary(selected.change_summary)}</span></div>
          </div>
          <div className="version-compare">
            <div>
              <strong>{previous ? `Before: v${previous.version}` : "Before: none"}</strong>
              <pre>{previous?.content ? displayArtifactContent(previous.content) : "This is the starting draft."}</pre>
            </div>
            <div>
              <strong>After: v{selected.version}</strong>
              <pre>{displayArtifactContent(selected.content)}</pre>
            </div>
          </div>
          <TextDiff before={displayArtifactContent(previous?.content || "")} after={displayArtifactContent(selected.content)} />
        </>
      )}
      {rejected.length > 0 && (
        <div className="rejected-review">
          <strong>Attempts not used</strong>
          <p className="section-help">
            These candidate edits were tested but not promoted because the gate accepts only a score higher than the current best version.
            A tie is kept as evidence, not promoted.
            Open one to inspect what would have changed.
          </p>
          <div className="rejected-list">
            {rejected.map((version, index) => {
              const gateExplanation = rejectedGateExplanation(version, bestVersion);
              return (
                <button
                  key={version.id}
                  className={version.id === selectedRejected?.id ? "rejected-attempt active" : "rejected-attempt"}
                  onClick={() => setSelectedRejectedId(version.id === selectedRejected?.id ? "" : version.id)}
                >
                  <strong>Attempt {index + 1}</strong>
                  <span>{gateExplanation.summary}</span>
                  <span>{gateExplanation.reason}</span>
                  <span>{humanizeChangeSummary(version.change_summary)}</span>
                </button>
              );
            })}
          </div>
          {selectedRejected && (
            <div className="rejected-detail">
              <div className="rejected-detail-header">
                <strong>Attempt details</strong>
                <button className="secondary" onClick={() => setSelectedRejectedId("")}>Hide details</button>
              </div>
              <div className="version-summary">
                <div><strong>Promotion decision</strong><span>{humanizeGateAction(selectedRejected.metadata?.skillopt_gate_action || "rejected")}</span></div>
                <div><strong>Candidate score</strong><span>{selectedRejected.score}</span></div>
                <div><strong>Score lift</strong><span>{rejectedGateExplanation(selectedRejected, bestVersion).deltaLabel}</span></div>
                <div><strong>Why not used</strong><span>{rejectedGateExplanation(selectedRejected, bestVersion).reason}</span></div>
              </div>
              <pre>{displayArtifactContent(selectedRejected.content)}</pre>
              <TextDiff before={displayArtifactContent(accepted.find((version) => version.id === selectedRejected.parent_version_id)?.content || bestVersion?.content || "")} after={displayArtifactContent(selectedRejected.content)} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function humanizeChangeSummary(summary = "") {
  const text = String(summary);
  const operationText = text.match(/\(([^)]+)\)/)?.[1] || "";
  const operationLabels = {
    append: "added new guidance",
    insert_after: "inserted guidance into the current artifact",
    replace: "rewrote part of the artifact",
    delete: "removed text",
  };
  const operations = operationText
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => operationLabels[item] || item.replaceAll("_", " "));
  const action = text.toLowerCase().includes("reject")
    ? "Not promoted"
    : text.toLowerCase().includes("accept_new_best")
      ? "Promoted as the new best version"
      : text.toLowerCase().includes("accept")
        ? "Accepted as an improvement"
        : "";
  if (action && operations.length) return `${action}: ${dedupeStrings(operations).join(", ")}.`;
  if (text.toLowerCase().startsWith("skillopt ")) return text.replace(/^skillopt\s+/i, "").replace(/bounded edit operations/i, "candidate text edits");
  return text;
}

function humanizeGateAction(action = "") {
  return {
    accept_new_best: "Promoted as new best",
    accept: "Accepted improvement",
    reject: "Not promoted",
  }[action] || String(action).replaceAll("_", " ");
}

function displayArtifactContent(content = "") {
  return String(content)
    .replaceAll("SkillOpt Experiment Update", "Aiterate Candidate Update")
    .replaceAll("SkillOpt training start state", "optimization starting point");
}

function dedupeStrings(items) {
  return Array.from(new Set(items));
}

function rejectedGateExplanation(version, bestVersion) {
  const previousScore = Number(version.metadata?.previous_score ?? bestVersion?.score ?? 0);
  const candidateScore = Number(version.score ?? version.metadata?.candidate_eval_score ?? 0);
  const delta = Number(version.metadata?.score_delta ?? (candidateScore - previousScore));
  const deltaLabel = `${delta > 0 ? "+" : ""}${delta.toFixed(4)}`;
  const summary = `Candidate ${candidateScore.toFixed(4)} vs current best ${previousScore.toFixed(4)} (${deltaLabel})`;
  if (version.metadata?.gate_reason) {
    return { summary, deltaLabel, reason: version.metadata.gate_reason };
  }
  if (delta === 0) {
    return {
      summary,
      deltaLabel,
      reason: "Rejected because it tied the current best; the gate only promotes a measurable improvement.",
    };
  }
  if (delta < 0) {
    return {
      summary,
      deltaLabel,
      reason: "Rejected because it scored below the current best.",
    };
  }
  return {
    summary,
    deltaLabel,
    reason: "Rejected by the gate despite a small rounded lift; inspect full run metadata before promoting manually.",
  };
}

function VersionProgressTimeline({ versions, selectedId, bestVersion, onSelect }) {
  if (!versions.length) return null;
  const scores = versions.map((version) => Number(version.score || 0));
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  return (
    <div className="version-progress">
      {versions.map((version, index) => {
        const previous = versions[index - 1];
        const delta = previous ? Number((version.score - previous.score).toFixed(3)) : null;
        const range = maxScore - minScore;
        const relative = range > 0 ? (Number(version.score || 0) - minScore) / range : Number(version.score || 0);
        const width = Math.max(8, Math.min(100, Math.round(relative * 100)));
        const isSelected = version.id === selectedId;
        const isBest = version.id === bestVersion?.id;
        return (
          <button
            key={version.id}
            className={isSelected ? "version-progress-item active" : "version-progress-item"}
            onClick={() => onSelect(version.id)}
          >
            <div className="version-progress-top">
              <strong>v{version.version}</strong>
              <span>{isBest ? "Best" : index === 0 ? "Baseline" : "Accepted"}</span>
            </div>
            <div className="version-progress-score">
              <strong>{version.score}</strong>
              <span>{delta === null ? "start" : `${delta >= 0 ? "+" : ""}${delta}`}</span>
            </div>
            <div className="version-progress-bar" aria-hidden="true">
              <span style={{ width: `${width}%` }} />
            </div>
            <p>{humanizeChangeSummary(version.change_summary)}</p>
          </button>
        );
      })}
    </div>
  );
}

function TextDiff({ before, after }) {
  const rows = buildLineDiff(before, after);
  if (!before || rows.every((row) => row.type === "same")) return null;
  return (
    <div className="diff-view">
      <strong>What changed in this version</strong>
      {rows.map((row, index) => (
        <div className={`diff-line ${row.type}`} key={`${row.type}-${index}-${row.text.slice(0, 20)}`}>
          <span>{row.type === "added" ? "+" : row.type === "removed" ? "-" : " "}</span>
          <code>{row.text}</code>
        </div>
      ))}
    </div>
  );
}

function EvaluationReport({ report }) {
  return (
    <div className="eval-report">
      <div className="history-title">Evaluation report</div>
      <div className="result-grid compact-grid">
        <div><strong>{report.score}</strong><span>Eval score</span></div>
        <div><strong>{Math.round(report.pass_rate * 100)}%</strong><span>Pass rate</span></div>
        <div><strong>{report.passed}</strong><span>Passed checks</span></div>
        <div><strong>{report.failed}</strong><span>Failed checks</span></div>
      </div>
      <div className="eval-check-result-grid">
        {report.checks.map((check) => (
          <div className={check.passed ? "eval-check-result passed" : "eval-check-result failed"} key={`${check.assertion_id}-${check.metric}`}>
            <strong>{check.metric}</strong>
            <span>{check.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RegressionEvidence({ report, optimizer }) {
  return (
    <div className="eval-report">
      <div className="history-title">Regression evidence</div>
      <p className="section-help">
        These checks ran on the holdout test split after optimization. Use failures as the next prompt/policy worklist before approving.
      </p>
      <div className="result-grid compact-grid">
        <div><strong>{optimizer?.validation_case_count ?? "n/a"}</strong><span>Holdout examples</span></div>
        <div><strong>{Math.round(report.pass_rate * 100)}%</strong><span>Pass rate</span></div>
        <div><strong>{report.passed}</strong><span>Passed</span></div>
        <div><strong>{report.failed}</strong><span>Failed</span></div>
      </div>
    </div>
  );
}

function TargetValidationReport({ report }) {
  return (
    <div className="eval-report target-validation-report">
      <div className="history-title">Target model test</div>
      <p className="section-help">
        The selected target model answered holdout examples using the best prompt or skill. These scores show how the artifact behaved with the model it is meant to run on.
      </p>
      <div className="result-grid compact-grid">
        <div><strong>{report.score}</strong><span>Answer score</span></div>
        <div><strong>{Math.round(report.pass_rate * 100)}%</strong><span>Pass rate</span></div>
        <div><strong>{report.case_count}</strong><span>Examples tested</span></div>
        <div><strong>{report.failed_metrics?.length || 0}</strong><span>Metrics to review</span></div>
      </div>
      {report.failed_metrics?.length > 0 && (
        <div className="integration warning-note">
          Review: {report.failed_metrics.slice(0, 6).join(", ")}
        </div>
      )}
    </div>
  );
}

function Insights({ insights }) {
  return (
    <div className="insights-grid">
      <InsightCard title="What worked" items={insights.worked} />
      <InsightCard title="What went wrong" items={insights.went_wrong} />
      <InsightCard title="Prompt changes needed" items={insights.prompt_changes_needed} />
      <InsightCard title="Policy coverage" items={insights.policy_coverage} />
      <InsightCard title="Data risks" items={insights.data_risks} />
    </div>
  );
}

function InsightCard({ title, items }) {
  return (
    <div className="insight-card">
      <strong>{title}</strong>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function ModelComparison({ comparison }) {
  return (
    <div className="comparison">
      <div className="history-title">Model comparison using the same prompt</div>
      <div className="comparison-grid">
        {comparison.results.map((result) => (
          <div className="comparison-card" key={`${result.model.kind}-${result.model.model}`}>
            <strong>{result.model.kind}: {result.model.model}</strong>
            <span>{result.sample_output ? "Live eval score" : "Estimated compatibility score"} {result.score}</span>
            <span>{result.recommendation}</span>
            <ul>
              {result.strengths.map((strength) => <li key={strength}>{strength}</li>)}
            </ul>
            <ul>
              {result.risks.map((risk) => <li key={risk}>{risk}</li>)}
            </ul>
            {result.sample_output && <pre>{result.sample_output}</pre>}
          </div>
        ))}
      </div>
      <div className="integration">{comparison.summary}</div>
    </div>
  );
}

function RunDashboard({ runHistory, loading, error, openingRunId, onOpenRun, onDeleteRun, onDeleteProject }) {
  if (loading) return <div className="empty">Loading run history...</div>;
  if (error) return <div className="error">{error}</div>;
  const names = Object.keys(runHistory || {});
  if (!names.length) {
    return <div className="empty">No run history yet. Completed optimizations will appear here grouped by name.</div>;
  }
  return (
    <div className="history-grid">
      {names.map((name) => (
        <div className="history-group" key={name}>
          <div className="history-group-header">
            <div>
              <div className="history-title">{name}</div>
              <span>{runHistory[name].length} saved run{runHistory[name].length === 1 ? "" : "s"}</span>
            </div>
            <button
              className="history-project-delete"
              onClick={() => onDeleteProject({ name, runs: runHistory[name] })}
              title="Delete project runs"
            >
              Delete project
            </button>
          </div>
          {runHistory[name].slice(0, 4).map((item) => (
            <div className="history-run-row" key={item.id}>
              <button
                className="history-run"
                onClick={() => onOpenRun(item.id)}
                disabled={Boolean(openingRunId)}
              >
                <span className="history-score-badge">
                  <strong>{item.best_score ?? "n/a"}</strong>
                  <small>score</small>
                </span>
                <span className="history-run-content">
                  <span className="history-run-topline">
                    <strong>{openingRunId === item.id ? "Opening run..." : item.model || "unknown model"}</strong>
                    {item.approved && <span className="history-approved">Approved</span>}
                  </span>
                  <span className="history-run-time">{new Date(item.created_at).toLocaleString()}</span>
                  <span className="history-open">Open review</span>
                </span>
                <span className="history-run-metrics">
                  <span><strong>{formatHistoryCost(item)}</strong><small>estimated</small></span>
                  <span><strong>{item.accepted_versions}</strong><small>accepted</small></span>
                  <span><strong>{item.rejected_versions}</strong><small>rejected</small></span>
                </span>
              </button>
              <button
                className="history-delete"
                onClick={() => onDeleteRun(item)}
                aria-label={`Delete run ${item.id}`}
                title="Delete run"
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function ValidationSummary({ issues }) {
  if (!issues.length) return <div className="ready"><Check size={16} /> Ready to run</div>;
  return (
    <div className="validation">
      <AlertCircle size={16} />
      {issues[0]}
    </div>
  );
}

function formatCost(costEstimate) {
  if (!costEstimate || costEstimate.total_cost == null) return "n/a";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: costEstimate.currency || "USD",
    minimumFractionDigits: costEstimate.total_cost < 0.01 ? 4 : 2,
    maximumFractionDigits: costEstimate.total_cost < 0.01 ? 4 : 2,
  }).format(costEstimate.total_cost);
}

function exportText(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function sdkSnippet(run) {
  return `from aiterate.sdk import AIterateClient

client = AIterateClient()
artifact = ${JSON.stringify(run.best_version?.content || "", null, 2)}

# Use artifact in your app, or store it in your own prompt registry.
print(artifact)
`;
}

function promotionPrBody(run) {
  return [
    "Promote approved Aiterate artifact version.",
    "",
    `Run: ${run.id}`,
    `Artifact: ${run.artifact_id}`,
    `Version: ${run.best_version?.id || "n/a"}`,
    `Score: ${run.best_version?.score ?? "n/a"}`,
    `Estimated cost: ${formatCost(run.cost_estimate)}`,
    "",
    "Generated by Aiterate after review and manual approval.",
  ].join("\n");
}

function promotionBranchName(run) {
  const artifactId = String(run?.artifact_id || "artifact").replaceAll("_", "-");
  return `aiterate/promote-${artifactId}`;
}

function promotionPackage(run, approval, gitSettings = null) {
  const dataset = run.dataset || {};
  const optimizer = run.optimizer || {};
  const policySet = run.policy_set || {};
  const bestVersion = run.best_version || {};
  return {
    schema_version: "aiterate.promotion.package.v1",
    product: "Aiterate",
    purpose: "Portable promotion package for an approved prompt or agent skill artifact.",
    promotion: {
      status: approval?.status || "not_approved",
      approved_by: approval?.approved_by || null,
      approved_run_id: approval?.run_id || null,
      approved_version_id: approval?.version_id || bestVersion.id || null,
    },
    artifact: {
      artifact_id: run.artifact_id,
      version_id: bestVersion.id,
      version_number: bestVersion.version,
      kind: bestVersion.kind,
      score: bestVersion.score,
      change_summary: bestVersion.change_summary,
      prompt_change_meaning:
        "The change summary describes the bounded prompt/skill edit that survived validation scoring and manual review. Rejected edits remain in lineage for audit.",
      content: bestVersion.content,
    },
    run: {
      run_id: run.id,
      name: run.name,
      created_at: run.created_at,
    },
    promotion_destination: {
      project_name: gitSettings?.project_name || run.name,
      git_tracking_enabled: Boolean(gitSettings?.enable_git_tracking),
      promotion_pr_enabled: Boolean(gitSettings?.enable_promotion_pr_workflow),
      git_remote: gitSettings?.git_remote || "",
      promotion_branch_strategy: "automatic",
      promotion_branch_pattern: "aiterate/promote-<artifact-id>",
      pr_base_branch: gitSettings?.pr_base_branch || "",
    },
    data_sources: {
      dataset_id: dataset.id,
      dataset_name: dataset.name,
      dataset_hash: dataset.content_hash,
      raw_data_path: `aiterate/sources/${run.id}/data/raw_data.txt`,
      normalized_case_count: dataset.normalized_cases?.length || 0,
      normalized_case_examples: (dataset.normalized_cases || []).slice(0, 5),
      raw_data_excerpt: excerpt(dataset.raw_text),
    },
    policy_sources: {
      policy_set_id: policySet.id,
      policy_hash: bestVersion.policy_hash,
      policy_context_hash: optimizer.policy_context_hash,
      policy_context_path: `aiterate/sources/${run.id}/policies/policy_context.txt`,
      policy_context_excerpt: excerpt(optimizer.policy_context),
      weighted_rules: policySet.rules || [],
    },
    knowledge_sources: {
      knowledge_base_hash: optimizer.knowledge_base_hash,
      knowledge_base_path: `aiterate/sources/${run.id}/knowledge/knowledge_base.txt`,
      knowledge_base_excerpt: excerpt(optimizer.knowledge_base_context),
      role: "Grounding and source-reference context for generated artifact behavior.",
    },
    raw_source_snapshots: {
      storage_note:
        "For Git PRs, these source snapshots are written as files under aiterate/sources/<run_id>/. For large files, configure Git LFS or DVC in the repository/CI workflow.",
      data_examples: dataset.raw_text || "",
      policy_context: optimizer.policy_context || "",
      knowledge_base_context: optimizer.knowledge_base_context || "",
    },
    models_and_providers: {
      optimizer_provider: run.provider,
      target_model: optimizer.target_model,
      target_validation_enabled: optimizer.run_target_validation,
    },
    optimization: {
      framework: optimizer.framework,
      wrapper: optimizer.wrapper,
      loop: optimizer.loop,
      seed: optimizer.seed,
      iterations_requested: optimizer.iterations || optimizer.requested_iterations,
      validation_split: optimizer.validation_split,
      train_case_count: optimizer.train_case_count,
      validation_case_count: optimizer.validation_case_count,
      promotion_threshold: optimizer.promotion_threshold,
      max_budget_usd: optimizer.max_budget_usd,
    },
    evaluation: {
      report: run.evaluation_report,
      target_model_behavior_report: run.behavior_report,
      cost_estimate: run.cost_estimate,
      insights: run.insights,
    },
    lineage: {
      accepted_versions: (run.accepted_versions || []).map(versionSummary),
      rejected_candidates: (run.rejected_versions || []).map(versionSummary),
    },
    limitations: [
      "Scores are evaluation signals, not a guarantee of production safety.",
      "Approximate cost uses configured model prices and may differ from provider billing.",
      "Manual approval should include domain review for regulated or high-risk use cases.",
      "Raw source snapshots are included for local export. For large Git PR sources, use Git LFS or DVC.",
    ],
  };
}

function versionSummary(version) {
  return {
    id: version.id,
    version: version.version,
    accepted: version.accepted,
    score: version.score,
    parent_version_id: version.parent_version_id,
    change_summary: version.change_summary,
    created_at: version.created_at,
    metadata: version.metadata,
  };
}

function excerpt(value, limit = 1200) {
  const text = String(value || "").trim();
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

function buildLineDiff(before, after) {
  const beforeLines = before.split(/\r?\n/).filter((line) => line.trim());
  const afterLines = after.split(/\r?\n/).filter((line) => line.trim());
  const beforeSet = new Set(beforeLines);
  const afterSet = new Set(afterLines);
  const removed = beforeLines.filter((line) => !afterSet.has(line)).map((text) => ({ type: "removed", text }));
  const rows = afterLines.map((text) => ({ type: beforeSet.has(text) ? "same" : "added", text }));
  return [...removed, ...rows].slice(0, 80);
}

function formatHistoryCost(item) {
  if (item.estimated_cost == null) return "n/a";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: item.estimated_cost_currency || "USD",
    minimumFractionDigits: item.estimated_cost < 0.01 ? 4 : 2,
    maximumFractionDigits: item.estimated_cost < 0.01 ? 4 : 2,
  }).format(item.estimated_cost);
}

function approvedRunsFromHistory(runHistory) {
  return Object.entries(runHistory || {}).flatMap(([project, items]) =>
    (items || [])
      .filter((item) => item.approved || item.approval?.status === "approved")
      .map((item) => ({ ...item, project }))
  );
}

function maskedSecretEnding(fingerprint) {
  if (!fingerprint) return "saved";
  const ending = fingerprint.includes("...") ? fingerprint.split("...").pop() : fingerprint.slice(-4);
  return `ending in ${ending || "****"}`;
}

function ErrorMessage({ message, action, onAction }) {
  return (
    <div className="error action-error">
      <span>{message}</span>
      {action && (
        <button className="secondary" onClick={() => onAction(action)}>
          {action.label}
        </button>
      )}
    </div>
  );
}

async function apiError(response, fallback) {
  try {
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      return new Error(nonJsonApiMessage(fallback, contentType));
    }
    const payload = await response.json();
    const detail = payload?.detail;
    if (typeof detail === "string") return new Error(detail);
    if (detail?.message) {
      const error = new Error(detail.message);
      if (detail.section === "models" || detail.section === "credentials") {
        error.action = {
          type: "add_provider_secret",
          provider: detail.provider || providerFromSecretName(detail.secret_name),
          label: `Add ${providerLabel(detail.provider || providerFromSecretName(detail.secret_name) || "openai")} credential`,
        };
      }
      return error;
    }
  } catch {
    // Keep the fallback message if the response body is not JSON.
  }
  return new Error(`${fallback} HTTP ${response.status}`);
}

async function apiErrorMessage(response, fallback) {
  return (await apiError(response, fallback)).message;
}

async function readJsonResponse(response, fallback) {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    throw new Error(nonJsonApiMessage(fallback, contentType));
  }
  return response.json();
}

async function fetchRunById(runId, fallback) {
  const response = await fetch(`${apiBaseUrl}/v1/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) throw new Error(await apiErrorMessage(response, fallback));
  return readJsonResponse(response, fallback);
}

function nonJsonApiMessage(fallback, contentType) {
  const kind = contentType || "non-JSON response";
  return `${fallback} The API returned ${kind}. Restart or rebuild the backend so /v1 routes are served by FastAPI, not the frontend page.`;
}

function providerFromSecretName(secretName) {
  return {
    OPENAI_API_KEY: "openai",
    ANTHROPIC_API_KEY: "anthropic",
    AZURE_OPENAI_API_KEY: "azure_openai",
    AWS_PROFILE: "aws_bedrock",
  }[secretName] || null;
}

function inferCategories(text) {
  const lower = text.toLowerCase();
  const categories = [];
  if (/\b(policy|rule|compliance|must|should)\b/.test(lower)) categories.push("Policy rules");
  if (/\b(customer|user|ticket|support|case)\b/.test(lower)) categories.push("User scenarios");
  if (/\b(cite|source|evidence|reference)\b/.test(lower)) categories.push("Citation requirements");
  if (/\b(escalate|uncertain|incomplete|contradictory)\b/.test(lower)) categories.push("Escalation rules");
  if (/\b(tone|concise|professional|friendly)\b/.test(lower)) categories.push("Style guidance");
  return categories.length ? categories : ["General prompt context"];
}

function fileExtension(name) {
  return name.includes(".") ? name.slice(name.lastIndexOf(".")).toLowerCase() : "";
}

function previewDataSplit(text, validationSplit) {
  const total = estimateCaseCount(text);
  if (total <= 1) return { total, train: total, test: total };
  const test = Math.min(total - 1, Math.max(1, Math.round(total * validationSplit)));
  return { total, train: total - test, test };
}

function estimateCaseCount(text) {
  const clean = text.trim();
  if (!clean) return 0;
  const jsonCount = estimateStructuredCaseCount(clean);
  if (jsonCount) return jsonCount;
  return clean.split(/\n\s*\n|(?<=[.!?])\s+/).filter((part) => part.trim()).length || 1;
}

function estimateStructuredCaseCount(text) {
  try {
    const payload = JSON.parse(text);
    if (Array.isArray(payload)) return payload.length;
    if (payload && typeof payload === "object") {
      const rows = payload.cases || payload.examples || payload.data || payload.records;
      return Array.isArray(rows) ? rows.length : 1;
    }
  } catch {
    // Text, CSV, YAML, and messy notes fall back to paragraph/sentence estimates.
  }
  if (text.includes(",") && text.includes("\n")) {
    const rows = text.split(/\r?\n/).filter((row) => row.trim());
    return rows.length > 1 ? rows.length - 1 : 0;
  }
  return 0;
}

function categoryToPolicy(category) {
  const id = category.toLowerCase().replaceAll(" ", "_");
  return { id, text: `Optimize for ${category.toLowerCase()}.`, weight: 1 };
}

function policyRulesFromText(text) {
  const structuredRules = policyRulesFromYamlLikeText(text);
  if (structuredRules.length) return structuredRules;

  const lines = text
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*#\s]+/, "").trim())
    .filter((line) => {
      const lower = line.toLowerCase();
      return line.length > 12
        && !lower.startsWith("source file:")
        && !["policies:", "rules:"].includes(lower)
        && !lower.startsWith("id:")
        && !lower.startsWith("weight:");
    })
    .filter((line, index, all) => all.findIndex((candidate) => normalizePolicyText(candidate) === normalizePolicyText(line)) === index)
    .slice(0, 8);
  if (!lines.length) return defaultPolicies;
  return lines.map((line, index) => ({
    id: `policy_${index + 1}`,
    text: line.slice(0, 500),
    weight: 1,
  }));
}

function policyRulesFromYamlLikeText(text) {
  const rules = [];
  let current = null;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    const idMatch = line.match(/^-\s*id:\s*["']?([^"']+)["']?$/) || line.match(/^id:\s*["']?([^"']+)["']?$/);
    if (idMatch) {
      if (current?.text) rules.push(current);
      current = { id: slugPolicyId(idMatch[1]), text: "", weight: 1 };
      continue;
    }
    if (!current) continue;
    const textMatch = line.match(/^text:\s*["']?(.+?)["']?$/) || line.match(/^description:\s*["']?(.+?)["']?$/);
    if (textMatch) {
      current.text = textMatch[1].trim();
      continue;
    }
    const weightMatch = line.match(/^weight:\s*([0-9.]+)$/);
    if (weightMatch) {
      current.weight = Number(weightMatch[1]);
    }
  }
  if (current?.text) rules.push(current);
  return rules
    .filter((rule, index, all) => all.findIndex((candidate) => normalizePolicyText(candidate.text) === normalizePolicyText(rule.text)) === index)
    .slice(0, 8);
}

function slugPolicyId(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "policy";
}

function normalizePolicyText(text) {
  return String(text).trim().toLowerCase().replace(/\s+/g, " ");
}

function equalizeWeights(rules) {
  if (!rules.length) return rules;
  const weight = Number((1 / rules.length).toFixed(2));
  return rules.map((rule) => ({ ...rule, weight }));
}

function optimizationDepthLabel(iterations) {
  if (iterations <= 2) return "Quick";
  if (iterations <= 4) return "Standard";
  if (iterations <= 8) return "Deep";
  return "Thorough";
}

function defaultModelForProvider(provider, catalog = {}) {
  return catalog?.[provider]?.[0]?.id || {
    openai: "gpt-5.5",
    anthropic: "claude-3-5-sonnet-latest",
    azure_openai: "gpt-5.5",
    aws_bedrock: "anthropic.claude-3-5-sonnet-20240620-v1:0",
    litellm: "openai/gpt-5.5",
  }[provider] || "gpt-5.5";
}

function selectedModelValue(selected, customValue) {
  return selected === "__custom" ? customValue.trim() : selected;
}

function providerLabel(provider) {
  return providerOptions.find(([value]) => value === provider)?.[1] || provider;
}

function providerSetupItem(provider, setup) {
  if (!setup?.providers) return null;
  const secretByProvider = {
    openai: "OPENAI_API_KEY",
    anthropic: "ANTHROPIC_API_KEY",
    azure_openai: "AZURE_OPENAI_API_KEY",
    aws_bedrock: "AWS_PROFILE",
    litellm: null,
  };
  const secretName = secretByProvider[provider];
  if (!secretName) return null;
  return setup.providers.find((item) => item.secret_name === secretName || item.env?.includes(secretName)) || null;
}

function trackingSetupItem(tracker, setup) {
  if (!setup?.tracking) return null;
  const secretName = tracker === "mlflow" ? "MLFLOW_TRACKING_TOKEN" : "LANGSMITH_API_KEY";
  return setup.tracking.find((item) => item.secret_name === secretName || item.env?.includes(secretName)) || null;
}

function gitSetupItem(name, setup) {
  return setup?.git?.find((item) => item.name === name) || null;
}

function gitBrowserAuthAvailable(setup) {
  return Boolean(setup?.git?.some((item) => item.browser_auth));
}

function modelsAndCredentialsReady({
  optimizerProvider,
  optimizerModel,
  customOptimizerModel,
  targetProvider,
  targetModel,
  customTargetModel,
  reuseOptimizerCredential,
  reuseTargetCredential,
  optimizerKey,
  targetKey,
  integrationStatus,
  iterations,
  promotionThreshold,
  maxBudgetUsd,
}) {
  const optimizerModelValue = selectedModelValue(optimizerModel, customOptimizerModel);
  const targetModelValue = selectedModelValue(targetModel, customTargetModel);
  if (!optimizerProvider || !optimizerModelValue.trim() || !targetProvider || !targetModelValue.trim()) {
    return false;
  }

  const optimizerReady = reuseOptimizerCredential
    ? integrationStatus?.providers?.[optimizerProvider]?.configured === true
    : Boolean(optimizerKey.trim());
  const targetReady = optimizerProvider === targetProvider
    ? optimizerReady
    : reuseTargetCredential
      ? integrationStatus?.providers?.[targetProvider]?.configured === true
      : Boolean(targetKey.trim());

  return optimizerReady && targetReady;
}

function buildValidationIssues({
  backendStatus,
  rawData,
  policyContext,
  knowledgeBaseContext,
  policies,
  optimizerProvider,
  optimizerModel,
  customOptimizerModel,
  targetProvider,
  targetModel,
  customTargetModel,
  tracker,
  mlflowUri,
  langsmithProject,
  langsmithUrl,
  reuseTrackingCredential,
  trackingSecret,
  enableGit,
  createPr,
  gitRemote,
  prBase,
  reuseOptimizerCredential,
  reuseTargetCredential,
  optimizerKey,
  targetKey,
  integrationStatus,
  iterations,
  promotionThreshold,
  maxBudgetUsd,
}) {
  const issues = [];
  if (backendStatus !== "online") issues.push("Backend is offline or still starting.");
  if (rawData.trim().length < 20) issues.push("Add enough raw data to create a useful prompt or skill.");
  if (policyContext.trim().length > 0 && policies.length === 0) issues.push("Review uploaded policy context and confirm at least one weighted policy.");
  if (knowledgeBaseContext.trim().length > 0 && rawData.trim().length < 20) issues.push("Add data/examples so knowledge base references can be tested.");
  if (!policies.length) issues.push("Add at least one policy or priority.");
  if (policies.some((policy) => !policy.id.trim() || !policy.text.trim())) issues.push("Every policy needs an id and text.");
  if (!Number.isFinite(iterations) || iterations < 1 || iterations > 20) {
    issues.push("Choose between 1 and 20 optimizer iterations.");
  }
  if (!Number.isFinite(promotionThreshold) || promotionThreshold < 0 || promotionThreshold > 1) {
    issues.push("Promotion threshold must be between 0 and 1.");
  }
  if (maxBudgetUsd.trim()) {
    const budget = Number(maxBudgetUsd);
    if (!Number.isFinite(budget) || budget < 0) {
      issues.push("Max spend cap must be a positive USD amount or left blank.");
    }
  }
  const optimizerModelValue = selectedModelValue(optimizerModel, customOptimizerModel);
  const targetModelValue = selectedModelValue(targetModel, customTargetModel);
  if (!optimizerProvider || !optimizerModelValue.trim()) issues.push("Choose an optimizer provider and model.");
  if (!targetProvider || !targetModelValue.trim()) issues.push("Choose the provider/model where the prompt will be used.");
  const trackerCredentialConfigured = integrationStatus?.tracking?.[tracker]?.credential_configured;
  const canReuseTracking = trackerCredentialConfigured === true;
  if (tracker !== "noop" && tracker === "mlflow" && !mlflowUri.trim()) {
    issues.push("Enter the MLflow tracking URI.");
  }
  if (tracker !== "noop" && tracker === "mlflow" && reuseTrackingCredential && !canReuseTracking) {
    issues.push("Saved MLflow access token was not detected.");
  }
  if (tracker !== "noop" && tracker === "langsmith" && !langsmithUrl.trim()) issues.push("Enter a LangSmith endpoint URL.");
  if (tracker !== "noop" && tracker === "langsmith" && !langsmithProject.trim()) issues.push("Enter a LangSmith project name.");
  if (tracker !== "noop" && tracker === "langsmith" && reuseTrackingCredential && !canReuseTracking) {
    issues.push("Saved LangSmith API key was not detected.");
  }
  if (tracker !== "noop" && tracker === "langsmith" && !reuseTrackingCredential && !trackingSecret.trim()) {
    issues.push("Paste a LangSmith API key or choose a saved LangSmith credential.");
  }
  const optimizerConfigured = integrationStatus?.providers?.[optimizerProvider]?.configured;
  const targetConfigured = integrationStatus?.providers?.[targetProvider]?.configured;
  const canReuseOptimizer = optimizerConfigured === true;
  const canReuseTarget = targetConfigured === true;
  if ((!reuseOptimizerCredential || !canReuseOptimizer) && !optimizerKey.trim()) {
    issues.push("Paste an optimizer provider key or reuse a saved optimizer credential.");
  }
  if (optimizerProvider !== targetProvider && (!reuseTargetCredential || !canReuseTarget) && !targetKey.trim()) {
    issues.push("Paste a target provider key or reuse a saved target credential.");
  }
  if (reuseOptimizerCredential && !canReuseOptimizer) {
    issues.push("Saved optimizer credential was not detected for this provider.");
  }
  if (optimizerProvider !== targetProvider && reuseTargetCredential && !canReuseTarget) {
    issues.push("Saved target credential was not detected for this provider.");
  }
  return issues;
}

createRoot(document.getElementById("root")).render(<App />);
