import {
  AlertCircle,
  BarChart3,
  Check,
  DatabaseZap,
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

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const supportedTypes = [".txt", ".csv", ".json", ".yaml", ".yml", ".md"];
const maxUploadBytes = 2 * 1024 * 1024;
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

function App() {
  const [projectName, setProjectName] = useState("support-agent");
  const [rawData, setRawData] = useState("");
  const [baselineArtifact, setBaselineArtifact] = useState("");
  const [fileInfo, setFileInfo] = useState(null);
  const [policies, setPolicies] = useState(defaultPolicies);
  const [evalAssertions, setEvalAssertions] = useState(defaultEvalAssertions);
  const [optimizerProvider, setOptimizerProvider] = useState("openai");
  const [optimizerModel, setOptimizerModel] = useState("gpt-4.1");
  const [targetProvider, setTargetProvider] = useState("openai");
  const [targetModel, setTargetModel] = useState("gpt-4.1");
  const [optimizerKey, setOptimizerKey] = useState("");
  const [targetKey, setTargetKey] = useState("");
  const [reuseOptimizerCredential, setReuseOptimizerCredential] = useState(true);
  const [reuseTargetCredential, setReuseTargetCredential] = useState(true);
  const [tracker, setTracker] = useState("mlflow");
  const [mlflowUri, setMlflowUri] = useState("");
  const [langsmithProject, setLangsmithProject] = useState("");
  const [reuseTrackingCredential, setReuseTrackingCredential] = useState(true);
  const [trackingSecret, setTrackingSecret] = useState("");
  const [gitRemote, setGitRemote] = useState("");
  const [gitBranch, setGitBranch] = useState("main");
  const [prBase, setPrBase] = useState("main");
  const [enableGit, setEnableGit] = useState(true);
  const [createPr, setCreatePr] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [integrationStatus, setIntegrationStatus] = useState(null);
  const [integrationSetup, setIntegrationSetup] = useState(null);
  const [secretDraft, setSecretDraft] = useState(null);
  const [secretDeleteDraft, setSecretDeleteDraft] = useState(null);
  const [resetConfirmOpen, setResetConfirmOpen] = useState(false);
  const [secretValue, setSecretValue] = useState("");
  const [runHistory, setRunHistory] = useState({});
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [run, setRun] = useState(null);
  const [approval, setApproval] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [comparingModels, setComparingModels] = useState(false);
  const [prStatus, setPrStatus] = useState("");
  const [error, setError] = useState("");
  const [fileError, setFileError] = useState("");
  const [loading, setLoading] = useState(false);
  const dataRef = useRef(null);
  const policyRef = useRef(null);
  const providerRef = useRef(null);
  const trackingRef = useRef(null);
  const gitPrRef = useRef(null);
  const resultsRef = useRef(null);

  const categories = useMemo(() => inferCategories(rawData), [rawData]);
  const totalWeight = useMemo(() => policies.reduce((sum, rule) => sum + Number(rule.weight), 0), [policies]);
  const validationIssues = useMemo(
    () =>
      buildValidationIssues({
        backendStatus,
        rawData,
        policies,
        optimizerProvider,
        optimizerModel,
        targetProvider,
        targetModel,
        tracker,
        mlflowUri,
        langsmithProject,
        reuseTrackingCredential,
        trackingSecret,
        enableGit,
        createPr,
        gitRemote,
        gitBranch,
        prBase,
        reuseOptimizerCredential,
        reuseTargetCredential,
        optimizerKey,
        targetKey,
        integrationStatus,
      }),
    [
      backendStatus,
      rawData,
      policies,
      optimizerProvider,
      optimizerModel,
      targetProvider,
      targetModel,
      tracker,
      mlflowUri,
      langsmithProject,
      reuseTrackingCredential,
      trackingSecret,
      enableGit,
      createPr,
      gitRemote,
      gitBranch,
      prBase,
      reuseOptimizerCredential,
      reuseTargetCredential,
      optimizerKey,
      targetKey,
      integrationStatus,
    ]
  );
  const steps = [
    { number: "1", title: "Import context", complete: rawData.trim().length > 20, ref: dataRef },
    {
      number: "2",
      title: "Weight policies",
      complete: policies.length > 0 && policies.every((rule) => rule.id.trim() && rule.text.trim()),
      ref: policyRef,
    },
    {
      number: "3",
      title: "Configure models",
      complete: Boolean(optimizerProvider && optimizerModel && targetProvider && targetModel),
      ref: providerRef,
    },
    {
      number: "4",
      title: "Tracking setup",
      complete: tracker === "mlflow" ? Boolean(mlflowUri) : tracker === "langsmith" ? Boolean(langsmithProject) : false,
      ref: trackingRef,
    },
    { number: "5", title: "Run optimizer", complete: Boolean(run?.best_version), ref: resultsRef },
    { number: "6", title: "Review and approve", complete: Boolean(approval?.status === "approved"), ref: resultsRef },
    { number: "7", title: "Create Git PR", complete: Boolean(prStatus), ref: gitPrRef },
  ];
  const requiredStepsComplete = steps.slice(0, 4).every((step) => step.complete);
  const canRun = requiredStepsComplete && validationIssues.length === 0;
  const canCreatePr = Boolean(run && approval?.status === "approved" && createPr && gitRemote.trim() && prBase.trim());

  useEffect(() => {
    checkBackend();
    refreshRunHistory();
  }, []);

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
  }

  function syncCredentialReuse(status) {
    if (!status?.providers?.[optimizerProvider]?.configured) {
      setReuseOptimizerCredential(false);
    }
    if (!status?.providers?.[targetProvider]?.configured) {
      setReuseTargetCredential(false);
    }
    if (!status?.tracking?.[tracker]?.configured) {
      setReuseTrackingCredential(false);
    }
  }

  async function saveSecret() {
    if (!secretDraft || !secretValue.trim()) return;
    const response = await fetch(`${apiBaseUrl}/v1/secrets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: secretDraft.secret_name,
        integration: secretDraft.name,
        value: secretValue,
      }),
    });
    if (response.ok) {
      setSecretValue("");
      setSecretDraft(null);
      checkBackend();
    } else {
      setError(`Could not save secret. HTTP ${response.status}`);
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
      setError(`Could not delete secret. HTTP ${response.status}`);
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

  async function uploadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setFileError("");
    const extension = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
    if (!supportedTypes.includes(extension)) {
      setFileError(`Unsupported file type ${extension || "unknown"}. Use ${supportedTypes.join(", ")}.`);
      return;
    }
    if (file.size > maxUploadBytes) {
      setFileError("File is too large for the browser workflow. Use the CLI or SDK for larger datasets.");
      return;
    }
    try {
      const text = await file.text();
      setRawData(text);
      setFileInfo({ name: file.name, size: file.size, extension });
      const inferred = inferCategories(text);
      setPolicies(equalizeWeights(inferred.map((category) => categoryToPolicy(category))));
    } catch {
      setFileError("Could not read this file. Check that it is a text-based data file.");
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
    setBaselineArtifact("");
    setFileInfo(null);
    setPolicies(defaultPolicies);
    setEvalAssertions(defaultEvalAssertions);
    setRun(null);
    setApproval(null);
    setModelComparison(null);
    setError("");
    setFileError("");
  }

  function handleTargetProviderChange(provider) {
    setTargetProvider(provider);
    setTargetModel(modelForProvider(provider));
    if (provider === optimizerProvider) {
      setReuseTargetCredential(true);
      setTargetKey("");
    }
  }

  async function optimize() {
    if (!canRun) {
      setError(`Complete setup before running: ${validationIssues[0] || "review required fields"}.`);
      return;
    }
    setLoading(true);
    setRun(null);
    setApproval(null);
    setModelComparison(null);
    setPrStatus("");
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}/v1/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: projectName,
          raw_data: rawData,
          baseline_artifact: baselineArtifact.trim() || null,
          policies,
          eval_assertions: evalAssertions
            .filter((assertion) => assertion.enabled)
            .map(({ enabled, ...assertion }) => assertion),
          provider: {
            kind: optimizerProvider,
            model: optimizerModel || modelForProvider(optimizerProvider),
            api_key: reuseOptimizerCredential ? null : optimizerKey || null,
          },
          target_provider: {
            kind: targetProvider,
            model: targetModel,
            api_key: optimizerProvider === targetProvider
              ? reuseOptimizerCredential ? null : optimizerKey || null
              : reuseTargetCredential ? null : targetKey || null,
          },
          target_model: targetModel,
          tracker,
          tracker_uri: tracker === "mlflow"
            ? reuseTrackingCredential ? mlflowUri || null : trackingSecret || mlflowUri || null
            : reuseTrackingCredential ? null : trackingSecret || null,
          enable_git_tracking: enableGit,
          create_pull_request: false,
          iterations: 3,
        }),
      });
      if (!response.ok) throw new Error(`Optimization failed with HTTP ${response.status}`);
      const result = await response.json();
      setRun(result);
      refreshRunHistory();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Optimization failed");
    } finally {
      setLoading(false);
      setOptimizerKey("");
      setTargetKey("");
      setTrackingSecret("");
    }
  }

  async function requestPullRequest(result) {
    if (!approval) {
      setPrStatus("Approve the best version before creating a promotion PR.");
      return;
    }
    const response = await fetch(`${apiBaseUrl}/v1/git/pull-request`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: `Promote ${projectName} ${result.best_version?.id || ""}`.trim(),
        artifact_id: result.artifact_id,
        remote: gitRemote,
        branch: gitBranch,
        base: prBase,
      }),
    });
    const payload = await response.json();
    setPrStatus(payload.message || payload.status);
  }

  async function compareCurrentModels() {
    if (!run?.best_version) return;
    setComparingModels(true);
    setModelComparison(null);
    try {
      const response = await fetch(`${apiBaseUrl}/v1/compare-models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: run.best_version.content,
          raw_data: rawData,
          policies,
          model_a: { kind: optimizerProvider, model: optimizerModel },
          model_b: { kind: targetProvider, model: targetModel },
        }),
      });
      if (!response.ok) throw new Error(`Model comparison failed with HTTP ${response.status}`);
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
      setError(`Approval failed with HTTP ${response.status}`);
      return;
    }
    setApproval(await response.json());
    setPrStatus("");
  }

  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <div>
            <div className="eyebrow"><Sparkles size={14} /> AI Artifact Lifecycle Management</div>
            <h1>AIterate</h1>
            <p>Run governed AI artifact lifecycle management: create, optimize, regression-test, approve, version, and promote prompts and skills from raw data and policies.</p>
          </div>
          <button className="secondary reset-action" onClick={() => setResetConfirmOpen(true)}>Reset workspace</button>
        </div>
      </section>

      <section className="panel wide dashboard">
        <div className="section-heading">
          <h2><BarChart3 size={18} /> Previous Runs</h2>
          <button className="secondary" onClick={() => { checkBackend(); refreshRunHistory(); }}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>
        <RunDashboard runHistory={runHistory} loading={historyLoading} error={historyError} />
      </section>

      <section className="workspace">
        <aside className="rail">
          <div className="rail-title">Setup progress</div>
          {steps.map((step) => (
            <Step
              key={step.number}
              number={step.number}
              title={step.title}
              complete={step.complete}
              onClick={() => step.ref.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
            />
          ))}
        </aside>

        <section className="content">
          <div className="panel wide" ref={dataRef}>
            <h2><FileUp size={18} /> Import Context</h2>
            <p className="section-help">Start with messy notes, examples, tickets, policy text, CSV, JSON, YAML, or Markdown, then optionally add the current prompt or skill baseline. This gives the optimizer both the source context and the artifact starting point.</p>
            <label>Project name</label>
            <input value={projectName} onChange={(event) => setProjectName(event.target.value)} />
            <div className="upload-row">
              <label className="upload">
                <input type="file" accept={supportedTypes.join(",")} onChange={uploadFile} />
                Upload data file
              </label>
              <div className="hint">Supported: {supportedTypes.join(", ")}. Browser upload limit: 2 MB.</div>
            </div>
            {fileError && <div className="error">{fileError}</div>}
            {fileInfo && (
              <div className="file-pill">
                {fileInfo.name} - {fileInfo.extension} - {(fileInfo.size / 1024).toFixed(1)} KB
              </div>
            )}
            <label>Raw data, examples, or policy context</label>
            <textarea value={rawData} onChange={(event) => setRawData(event.target.value)} placeholder="Paste raw data, examples, policies, or notes here." />
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
              onChange={(event) => setBaselineArtifact(event.target.value)}
              placeholder="Paste an existing prompt or skill to optimize. Leave blank to generate the baseline from raw data."
            />
          </div>

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

          <div className="panel" ref={providerRef}>
            <h2><Settings2 size={18} /> Models & Credentials</h2>
            <p className="section-help">Choose the model that improves the artifact, then choose where the final prompt or skill will run. If providers differ, AIterate asks for both credentials.</p>
            <label>Optimizer provider</label>
            <select value={optimizerProvider} onChange={(event) => {
              setOptimizerProvider(event.target.value);
              setOptimizerModel(modelForProvider(event.target.value));
            }}>
              {providerOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
            <label>Optimizer model</label>
            <input value={optimizerModel} onChange={(event) => setOptimizerModel(event.target.value)} />
            <label>Prompt will be used with</label>
            <select value={targetProvider} onChange={(event) => handleTargetProviderChange(event.target.value)}>
              {providerOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
            <label>Target model</label>
            <input value={targetModel} onChange={(event) => setTargetModel(event.target.value)} />
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
                }}
                onDeleteSecret={(secret) => setSecretDeleteDraft(secret)}
              />
            </div>
          </div>

          <div className="panel" ref={trackingRef}>
            <h2><DatabaseZap size={18} /> Tracking</h2>
            <p className="section-help">Send run metadata, scores, traces, and artifacts to MLflow or LangSmith so every optimization is reproducible and reviewable.</p>
            <div className="infra-note">Tracking optimizer run setup: choose where run metrics, traces, and artifacts should be recorded.</div>
            <label>Experiment tracking</label>
            <select value={tracker} onChange={(event) => setTracker(event.target.value)}>
              <option value="mlflow">MLflow</option>
              <option value="langsmith">LangSmith</option>
            </select>
            {tracker === "mlflow" && (
              <>
                <label>MLflow tracking URI</label>
                <input placeholder="http://localhost:5000" value={mlflowUri} onChange={(event) => setMlflowUri(event.target.value)} />
              </>
            )}
            {tracker === "langsmith" && (
              <>
                <label>LangSmith project</label>
                <input placeholder="aiterate-prompts" value={langsmithProject} onChange={(event) => setLangsmithProject(event.target.value)} />
              </>
            )}
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
              }}
              onDeleteSecret={(secret) => setSecretDeleteDraft(secret)}
            />
            <div className="infra-note">
              Production tracking should use private network access and TLS for tracking services.
            </div>
          </div>

          <div className="panel wide result" ref={resultsRef}>
            <div className="section-heading">
              <h2><BarChart3 size={18} /> Run Optimizer</h2>
              <ValidationSummary issues={validationIssues} />
            </div>
            <p className="section-help">Run after data, policies, models, credentials, and tracking are ready. AIterate runs a SkillOpt-style experiment: reflect on failures, apply bounded prompt/skill edits, validate candidates, accept improvements, and retain rejected changes for review.</p>
            <button className="primary run-button" onClick={optimize} disabled={loading || !canRun}>
              <Play size={18} />
              {loading ? "Running optimization" : canRun ? "Run optimizer" : "Complete setup to run"}
            </button>
            {error ? (
              <div className="error">{error}</div>
            ) : run?.best_version ? (
              <RunResults
                run={run}
                approval={approval}
                comparison={modelComparison}
                comparingModels={comparingModels}
                prStatus={prStatus}
                onApprove={() => approveBestVersion(run)}
                onCompare={compareCurrentModels}
              />
            ) : (
              <div className="empty">Upload data, confirm policy weights, choose providers, then run optimization.</div>
            )}
          </div>

          <div className="panel wide" ref={gitPrRef}>
            <h2><GitBranch size={18} /> Git PR Promotion</h2>
            <p className="section-help">After approval, create a promotion PR so teams can review the generated prompt or skill like any other production artifact.</p>
            {integrationSetup?.git && (
              <SetupGroup title="GitHub / Bitbucket Auth" items={integrationSetup.git} stored={integrationSetup.stored_secrets} onAddSecret={(item) => {
                setSecretDraft(item);
                setSecretValue("");
              }} onDeleteSecret={(secret) => setSecretDeleteDraft(secret)} />
            )}
            <label className="check">
              <input type="checkbox" checked={enableGit} onChange={(event) => setEnableGit(event.target.checked)} />
              Enable Git artifact tracking
            </label>
            {enableGit && (
              <>
                <label>Git remote</label>
                <input placeholder="https://github.com/org/repo.git" value={gitRemote} onChange={(event) => setGitRemote(event.target.value)} />
                <label>Artifact branch</label>
                <input value={gitBranch} onChange={(event) => setGitBranch(event.target.value)} />
              </>
            )}
            <label className="check">
              <input type="checkbox" checked={createPr} onChange={(event) => setCreatePr(event.target.checked)} />
              Enable promotion PR workflow
            </label>
            {createPr && (
              <>
                <label>PR base branch</label>
                <input value={prBase} onChange={(event) => setPrBase(event.target.value)} />
                <div className="infra-note">GitHub or Bitbucket PR publishing requires a server-side token saved below or configured in the backend environment.</div>
              </>
            )}
            <div className="infra-note">Create PR is available after a run is reviewed and approved.</div>
            <button className="primary run-button" onClick={() => requestPullRequest(run)} disabled={!canCreatePr}>
              <GitBranch size={16} />
              Create Git PR
            </button>
            {!canCreatePr && (
              <div className="muted">To create a PR, approve a run, enable PR workflow, enter Git remote, and set PR base branch.</div>
            )}
            {prStatus && <div className="integration">{prStatus}</div>}
          </div>
        </section>
      </section>
      {secretDraft && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2><KeyRound size={18} /> Add Secret</h2>
            <p>{secretDraft.name}: paste {secretDraft.secret_name} once. AIterate stores it encrypted and only shows a fingerprint later.</p>
            <input
              type="password"
              autoComplete="off"
              placeholder={secretDraft.secret_name}
              value={secretValue}
              onChange={(event) => setSecretValue(event.target.value)}
            />
            <div className="modal-actions">
              <button className="secondary" onClick={() => setSecretDraft(null)}>Cancel</button>
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
    </main>
  );
}

function Step({ number, title, complete, onClick }) {
  return (
    <button className={`step ${complete ? "complete" : ""}`} onClick={onClick}>
      <span>{complete ? <Check size={16} /> : number}</span>
      <strong>{title}</strong>
    </button>
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
  const canReuse = Boolean(providerStatus?.configured || stored);
  return (
    <div className="credential-card">
      <div className="credential-title">
        <strong>{title}</strong>
        <span>{providerLabel(provider)}</span>
      </div>
      <div className={providerStatus?.configured ? "configured" : "not-configured"}>
        {sharedWithOptimizer
          ? "Same provider as optimizer; using the optimizer credential"
          : providerStatus?.configured ? "Server credential configured" : "Credential needed"}
      </div>
      {stored?.fingerprint && (
        <div className="stored-secret-row">
          <span>Stored key: {stored.fingerprint}</span>
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
            {canReuse ? "Reuse saved credential" : "No saved credential yet"}
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
  const canReuse = Boolean(trackerStatus?.configured || stored);
  const label = tracker === "mlflow" ? "MLflow tracking URI" : "LangSmith API key";
  return (
    <div className="credential-card tracking-card">
      <div className="credential-title">
        <strong>{label}</strong>
        <span>{tracker === "mlflow" ? "Metrics and artifacts" : "LLM traces"}</span>
      </div>
      <div className={trackerStatus?.configured || stored ? "configured" : "not-configured"}>
        {trackerStatus?.configured || stored ? "Saved credential available" : "Credential not saved yet"}
      </div>
      {stored?.fingerprint && (
        <div className="stored-secret-row">
          <span>Stored value: {stored.fingerprint}</span>
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
        {canReuse ? "Reuse saved tracking credential" : "No saved tracking credential yet"}
      </label>
      {(!canReuse || !reuseCredential) && (
        <div className="secret-row">
          <KeyRound size={16} />
          <input
            type={tracker === "mlflow" ? "text" : "password"}
            placeholder={tracker === "mlflow" ? "http://localhost:5000" : "Paste LangSmith API key for this run"}
            value={secretValue}
            onChange={(event) => setSecretValue(event.target.value)}
            autoComplete="off"
          />
        </div>
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
            {item.configured ? "Configured" : "Not configured"}
          </span>
          {stored?.find((secret) => secret.name === item.secret_name)?.fingerprint && (
            <div className="stored-secret-row">
              <span>Stored key: {stored.find((secret) => secret.name === item.secret_name).fingerprint}</span>
              <button className="danger-link" onClick={() => onDeleteSecret(stored.find((secret) => secret.name === item.secret_name))}>Delete</button>
            </div>
          )}
          <code>{item.env.join(", ")}</code>
        </div>
      ))}
    </div>
  );
}

function RunResults({ run, approval, comparison, comparingModels, onApprove, onCompare }) {
  const accepted = run.accepted_versions || [];
  const rejected = run.rejected_versions || [];
  const approved = approval?.status === "approved";
  const [showArtifact, setShowArtifact] = useState(true);
  return (
    <>
      <div className="result-grid">
        <div><strong>{run.best_version.score}</strong><span>Best score</span></div>
        <div><strong>{accepted.length}</strong><span>Accepted versions</span></div>
        <div><strong>{rejected.length}</strong><span>Rejected candidates</span></div>
        <div><strong>{run.artifact_id}</strong><span>Artifact ID</span></div>
      </div>
      {run.insights && <Insights insights={run.insights} />}
      {run.evaluation_report && <EvaluationReport report={run.evaluation_report} />}
      <div className="approval-flow">
        <div className="approval-step complete">
          <Check size={16} />
          Run optimizer
        </div>
        <div className={`approval-step ${run.best_version.score > 0 ? "complete" : ""}`}>
          <Check size={16} />
          View latest best version
        </div>
        <button className={approved ? "approval-step complete" : "approval-step"} onClick={onApprove} disabled={approved}>
          <Check size={16} />
          {approved ? "Approved best version" : "Approve best version"}
        </button>
      </div>
      <div className="inline-actions result-actions">
        <button className="secondary" onClick={() => setShowArtifact(!showArtifact)}>
          {showArtifact ? "Hide prompt/skill" : "Open prompt/skill"}
        </button>
        <button className="secondary" onClick={() => navigator.clipboard?.writeText(run.best_version.content)}>
          Copy prompt/skill
        </button>
        <button className="secondary" onClick={onCompare} disabled={comparingModels}>
          {comparingModels ? "Comparing models" : "Compare models with same prompt"}
        </button>
      </div>
      {comparison && <ModelComparison comparison={comparison} />}
      <div className="decision-list">
        {accepted.map((version) => (
          <div key={version.id} className="decision accepted">
            Accepted v{version.version}: score {version.score} - {version.change_summary}
          </div>
        ))}
        {rejected.map((version) => (
          <div key={version.id} className="decision rejected">
            Rejected candidate: score {version.score} did not beat parent version.
          </div>
        ))}
      </div>
      {approval?.message && <div className="integration">{approval.message}</div>}
      {showArtifact && <pre>{run.best_version.content}</pre>}
    </>
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
            <span>Score {result.score}</span>
            <span>{result.recommendation}</span>
            <ul>
              {result.risks.map((risk) => <li key={risk}>{risk}</li>)}
            </ul>
          </div>
        ))}
      </div>
      <div className="integration">{comparison.summary}</div>
    </div>
  );
}

function RunDashboard({ runHistory, loading, error }) {
  if (loading) return <div className="empty">Loading previous runs...</div>;
  if (error) return <div className="error">{error}</div>;
  const names = Object.keys(runHistory || {});
  if (!names.length) {
    return <div className="empty">No previous runs yet. Completed optimizations will appear here grouped by name.</div>;
  }
  return (
    <div className="history-grid">
      {names.map((name) => (
        <div className="history-group" key={name}>
          <div className="history-title">{name}</div>
          {runHistory[name].slice(0, 4).map((item) => (
            <div className="history-run" key={item.id}>
              <strong>Score {item.best_score ?? "n/a"}</strong>
              <span>{item.accepted_versions} accepted - {item.rejected_versions} rejected</span>
              <span>{item.model || "unknown model"} - {new Date(item.created_at).toLocaleString()}</span>
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

function categoryToPolicy(category) {
  const id = category.toLowerCase().replaceAll(" ", "_");
  return { id, text: `Optimize for ${category.toLowerCase()}.`, weight: 1 };
}

function equalizeWeights(rules) {
  if (!rules.length) return rules;
  const weight = Number((1 / rules.length).toFixed(2));
  return rules.map((rule) => ({ ...rule, weight }));
}

function modelForProvider(provider) {
  if (provider === "openai") return "gpt-4.1";
  if (provider === "anthropic") return "claude-3-5-sonnet-latest";
  if (provider === "azure_openai") return "gpt-4.1";
  if (provider === "aws_bedrock") return "anthropic.claude-3-5-sonnet-20240620-v1:0";
  if (provider === "litellm") return "anthropic/claude-3-5-sonnet";
  return "gpt-4.1";
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
  const secretName = tracker === "mlflow" ? "MLFLOW_TRACKING_URI" : "LANGSMITH_API_KEY";
  return setup.tracking.find((item) => item.secret_name === secretName || item.env?.includes(secretName)) || null;
}

function buildValidationIssues({
  backendStatus,
  rawData,
  policies,
  optimizerProvider,
  optimizerModel,
  targetProvider,
  targetModel,
  tracker,
  mlflowUri,
  langsmithProject,
  reuseTrackingCredential,
  trackingSecret,
  enableGit,
  createPr,
  gitRemote,
  gitBranch,
  prBase,
  reuseOptimizerCredential,
  reuseTargetCredential,
  optimizerKey,
  targetKey,
  integrationStatus,
}) {
  const issues = [];
  if (backendStatus !== "online") issues.push("Backend is offline or still starting.");
  if (rawData.trim().length < 20) issues.push("Add enough raw data to create a useful prompt or skill.");
  if (!policies.length) issues.push("Add at least one policy or priority.");
  if (policies.some((policy) => !policy.id.trim() || !policy.text.trim())) issues.push("Every policy needs an id and text.");
  if (!optimizerProvider || !optimizerModel.trim()) issues.push("Choose an optimizer provider and model.");
  if (!targetProvider || !targetModel.trim()) issues.push("Choose the provider/model where the prompt will be used.");
  if (tracker === "noop") issues.push("Choose MLflow or LangSmith tracking.");
  const trackerConfigured = integrationStatus?.tracking?.[tracker]?.configured;
  const canReuseTracking = trackerConfigured !== false;
  if (tracker === "mlflow" && reuseTrackingCredential && !mlflowUri.trim() && !canReuseTracking) {
    issues.push("Enter an MLflow tracking URI or save one as a tracking credential.");
  }
  if (tracker === "mlflow" && (!reuseTrackingCredential || !canReuseTracking) && !trackingSecret.trim() && !mlflowUri.trim()) {
    issues.push("Paste an MLflow tracking URI for this run.");
  }
  if (tracker === "langsmith" && !langsmithProject.trim()) issues.push("Enter a LangSmith project name.");
  if (tracker === "langsmith" && reuseTrackingCredential && !canReuseTracking) {
    issues.push("Save or paste a LangSmith API key.");
  }
  if (tracker === "langsmith" && (!reuseTrackingCredential || !canReuseTracking) && !trackingSecret.trim()) {
    issues.push("Paste a LangSmith API key for this run.");
  }
  const optimizerConfigured = integrationStatus?.providers?.[optimizerProvider]?.configured;
  const targetConfigured = integrationStatus?.providers?.[targetProvider]?.configured;
  const canReuseOptimizer = optimizerConfigured !== false;
  const canReuseTarget = targetConfigured !== false;
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
