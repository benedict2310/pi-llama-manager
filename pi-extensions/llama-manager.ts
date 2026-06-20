import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import {
  LLAMA_SERVER_PROFILES,
  buildLlamaArgs as buildManagedLlamaArgs,
  derivePiModelIdFromPath,
  findModelPathForPiModelId,
  formatPiModelBaseUrl,
  loadPiModelsRegistry,
  normalizeLlamaProfileName,
  parseLlamaServerLogMetrics,
  profileSyncDefaults,
  assessLaunchMetrics,
  createLaunchRecord,
  savePiModelsRegistry,
  syncPiModelsRegistryData,
} from "../lib/llama-manager-lib.js";
function shellQuote(value: string): string {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function parsePiLlamaModels(raw: string): Array<{ id: string; name?: string }> {
  try {
    const parsed = JSON.parse(raw) as any;
    const models = parsed?.providers?.["llama-cpp"]?.models;
    if (!Array.isArray(models)) return [];
    return models
      .map((model: any) => ({
        id: typeof model?.id === "string" ? model.id : "",
        name: typeof model?.name === "string" ? model.name : undefined,
      }))
      .filter((model: { id: string }) => model.id);
  } catch {
    return [];
  }
}

function tokenizeCommand(command: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let quote: string | null = null;

  for (let i = 0; i < command.length; i += 1) {
    const ch = command[i]!;

    if (quote) {
      if (ch === quote) quote = null;
      else current += ch;
      continue;
    }

    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }

    if (/\s/.test(ch)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }

    current += ch;
  }

  if (current) tokens.push(current);
  return tokens;
}

function parseArgValue(parts: string[], flag: string): string | undefined {
  const idx = parts.indexOf(flag);
  if (idx < 0 || idx + 1 >= parts.length) return undefined;
  return parts[idx + 1];
}

function extractModelPathFromCommand(command: string): string | undefined {
  const parts = tokenizeCommand(command);
  return parseArgValue(parts, "-m") ?? parseArgValue(parts, "--model");
}

function extractPortFromCommand(command: string, fallback = 8080): number {
  const parts = tokenizeCommand(command);
  const value = parseArgValue(parts, "--port") ?? parseArgValue(parts, "-p");
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parsePsLlamaLines(output: string): Array<{ pid: number; command: string; modelPath?: string; port: number }> {
  return String(output)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(\d+)\s+(.+)$/);
      if (!match) return null;
      const pid = Number.parseInt(match[1]!, 10);
      const command = match[2]!;
      return {
        pid,
        command,
        modelPath: extractModelPathFromCommand(command),
        port: extractPortFromCommand(command),
      };
    })
    .filter((row): row is { pid: number; command: string; modelPath?: string; port: number } => Boolean(row));
}

function buildLlamaArgs(config: LlamaManagerConfig, modelPath: string): string[] {
  return buildManagedLlamaArgs(config, modelPath);
}

function modelMatchesRequest(runningModelPath?: string, requestedModelPath?: string): boolean {
  if (!runningModelPath || !requestedModelPath) return false;
  if (runningModelPath === requestedModelPath) return true;

  const runningBase = String(runningModelPath).split(/[\\/]/).pop();
  const requestedBase = String(requestedModelPath).split(/[\\/]/).pop();
  return Boolean(runningBase && requestedBase && runningBase === requestedBase);
}

function isLikelyMainModelPath(modelPath: string): boolean {
  const base = String(modelPath).split(/[\\/]/).pop()?.toLowerCase() || "";
  return base.endsWith(".gguf") && !base.startsWith("mmproj-") && !base.includes("-mmproj-");
}

type LlamaDefaults = {
  jinja: boolean;
  reasoning: "off" | "on" | "auto";
  chatTemplateKwargs: Record<string, unknown>;
  temp: number;
  topP: number;
};

type PiModelSyncDefaults = {
  contextWindow: number;
  maxTokens: number;
  reasoning: boolean;
};

type LlamaManagerConfig = {
  host: string;
  port: number;
  modelsRoots: string[];
  defaultModelPath?: string;
  downloadDir?: string;
  logFile: string;
  launchHistoryFile: string;
  stableToolCalling: boolean;
  autoSyncPiModels: boolean;
  autoSelectAfterStart: boolean;
  autoSwitchOnModelSelect: boolean;
  serverProfile: "fast" | "code" | "deep" | "wide";
  syncDefaults: PiModelSyncDefaults;
  defaultArgs: LlamaDefaults;
  extraArgs: string[];
};
type RunningServer = {
  pid: number;
  command: string;
  modelPath?: string;
  port?: number;
};

type ModelOption = {
  label: string;
  value: string;
  description: string;
};

const CONFIG_PATH = path.join(os.homedir(), ".pi", "agent", "llama-manager.json");
const PI_MODELS_PATH = path.join(os.homedir(), ".pi", "agent", "models.json");
const STATUS_KEY = "llama-manager";

type DownloadState = {
  child: ChildProcessWithoutNullStreams;
  fileName: string;
  targetPath: string;
  totalBytes: number | null;
  downloadedBytes: number;
  lastSampleAt: number;
  lastSampleBytes: number;
  bytesPerSecond: number;
  status: "running" | "done" | "error" | "aborted";
  error?: string;
  pollTimer?: NodeJS.Timeout;
};

let activeDownload: DownloadState | null = null;

function expandHome(input: string): string {
  if (input === "~") return os.homedir();
  if (input.startsWith("~/")) return path.join(os.homedir(), input.slice(2));
  return input;
}

function defaultConfig(): LlamaManagerConfig {
  return {
    host: "127.0.0.1",
    port: 8080,
    modelsRoots: [path.join(os.homedir(), ".pi", "models")],
    defaultModelPath: "",
    downloadDir: path.join(os.homedir(), ".pi", "models"),
    logFile: path.join(os.homedir(), ".pi", "agent", "llama-server.log"),
    launchHistoryFile: path.join(os.homedir(), ".pi", "agent", "llama-launch-history.jsonl"),
    stableToolCalling: true,
    autoSyncPiModels: true,
    autoSelectAfterStart: false,
    autoSwitchOnModelSelect: false,
    serverProfile: "code",
    syncDefaults: profileSyncDefaults("code"),
    defaultArgs: {
      jinja: true,
      reasoning: "off",
      chatTemplateKwargs: { enable_thinking: false },
      temp: 0.2,
      topP: 0.9,
    },
    extraArgs: [],
  };
}

async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.stat(p);
    return true;
  } catch {
    return false;
  }
}

async function loadConfig(): Promise<LlamaManagerConfig> {
  const defaults = defaultConfig();
  try {
    const raw = await fs.readFile(CONFIG_PATH, "utf8");
    const parsed = JSON.parse(raw) as Partial<LlamaManagerConfig>;
    const serverProfile = normalizeLlamaProfileName((parsed as any).serverProfile) as "fast" | "code" | "deep" | "wide";
    const parsedSyncDefaults = parsed.syncDefaults ?? profileSyncDefaults(serverProfile);
    return {
      host: parsed.host || defaults.host,
      port: typeof parsed.port === "number" ? parsed.port : defaults.port,
      modelsRoots: Array.isArray(parsed.modelsRoots) && parsed.modelsRoots.length > 0
        ? parsed.modelsRoots.map((v) => expandHome(String(v)))
        : defaults.modelsRoots,
      defaultModelPath: parsed.defaultModelPath ? expandHome(String(parsed.defaultModelPath)) : defaults.defaultModelPath,
      downloadDir: parsed.downloadDir ? expandHome(String(parsed.downloadDir)) : defaults.downloadDir,
      logFile: parsed.logFile ? expandHome(String(parsed.logFile)) : defaults.logFile,
      launchHistoryFile: (parsed as any).launchHistoryFile ? expandHome(String((parsed as any).launchHistoryFile)) : defaults.launchHistoryFile,
      stableToolCalling: parsed.stableToolCalling ?? defaults.stableToolCalling,
      autoSyncPiModels: parsed.autoSyncPiModels ?? defaults.autoSyncPiModels,
      autoSelectAfterStart: parsed.autoSelectAfterStart ?? defaults.autoSelectAfterStart,
      autoSwitchOnModelSelect: (parsed as any).autoSwitchOnModelSelect ?? defaults.autoSwitchOnModelSelect,
      serverProfile,
      syncDefaults: {
        contextWindow: typeof parsedSyncDefaults.contextWindow === "number"
          ? parsedSyncDefaults.contextWindow
          : defaults.syncDefaults.contextWindow,
        maxTokens: typeof parsedSyncDefaults.maxTokens === "number"
          ? parsedSyncDefaults.maxTokens
          : defaults.syncDefaults.maxTokens,
        reasoning: typeof parsedSyncDefaults.reasoning === "boolean"
          ? parsedSyncDefaults.reasoning
          : defaults.syncDefaults.reasoning,
      },
      defaultArgs: {
        ...defaults.defaultArgs,
        ...(parsed.defaultArgs ?? {}),
      },
      extraArgs: Array.isArray(parsed.extraArgs) ? parsed.extraArgs.map((v) => String(v)) : defaults.extraArgs,
    };
  } catch {
    await fs.mkdir(path.dirname(CONFIG_PATH), { recursive: true });
    await fs.writeFile(CONFIG_PATH, JSON.stringify(defaults, null, 2) + "\n", "utf8");
    return defaults;
  }
}

async function saveConfig(config: LlamaManagerConfig): Promise<void> {
  await fs.mkdir(path.dirname(CONFIG_PATH), { recursive: true });
  await fs.writeFile(CONFIG_PATH, JSON.stringify(config, null, 2) + "\n", "utf8");
}

function getPiModelBaseUrl(config: LlamaManagerConfig): string {
  return formatPiModelBaseUrl(config.host, config.port);
}

function getServerBaseUrl(config: LlamaManagerConfig): string {
  return getPiModelBaseUrl(config).replace(/\/v1$/, "");
}

function getSyncDefaults(config: LlamaManagerConfig): PiModelSyncDefaults {
  const profileDefaults = profileSyncDefaults(config.serverProfile);
  return {
    contextWindow: config.syncDefaults?.contextWindow ?? profileDefaults.contextWindow,
    maxTokens: config.syncDefaults?.maxTokens ?? profileDefaults.maxTokens,
    reasoning: config.syncDefaults?.reasoning ?? false,
  };
}

function formatSyncError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
}

async function syncPiRegistryForPaths(
  ctx: ExtensionContext,
  config: LlamaManagerConfig,
  modelPaths: string[],
  options?: { force?: boolean; notifyPerModel?: boolean },
): Promise<ReturnType<typeof syncPiModelsRegistryData> | null> {
  if ((!config.autoSyncPiModels && !options?.force) || modelPaths.length === 0) {
    return null;
  }

  try {
    const registry = await loadPiModelsRegistry(PI_MODELS_PATH);
    const result = syncPiModelsRegistryData(registry, {
      modelPaths,
      baseUrl: getPiModelBaseUrl(config),
      defaults: getSyncDefaults(config),
    });
    await savePiModelsRegistry(PI_MODELS_PATH, result.registry);

    if (options?.notifyPerModel !== false) {
      for (const model of result.syncedModels) {
        ctx.ui.notify(`Synced model to Pi registry: ${model.id}`, "info");
      }
    }

    return result;
  } catch (error) {
    ctx.ui.notify(`Pi models registry sync failed: ${formatSyncError(error)}`, "warning");
    return null;
  }
}

async function syncStartedModelToPiRegistry(
  ctx: ExtensionContext,
  pi: ExtensionAPI,
  config: LlamaManagerConfig,
  modelPath: string,
): Promise<void> {
  if (!isLikelyMainModelPath(modelPath)) return;

  const syncResult = await syncPiRegistryForPaths(ctx, config, [modelPath]);
  if (!config.autoSelectAfterStart) return;

  const setModel = (pi as any)?.setModel;
  if (typeof setModel !== "function") {
    ctx.ui.notify("Auto-select after start is unavailable in this Pi runtime", "warning");
    return;
  }

  const synced = syncResult?.syncedModels[0];
  const defaults = getSyncDefaults(config);
  const modelId = synced?.id || derivePiModelIdFromPath(modelPath);
  const modelName = synced?.name || `${modelId} (llama.cpp)`;

  try {
    await setModel.call(pi, {
      id: modelId,
      name: modelName,
      api: "openai-completions",
      provider: "llama-cpp",
      baseUrl: getPiModelBaseUrl(config),
      reasoning: defaults.reasoning,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: defaults.contextWindow,
      maxTokens: defaults.maxTokens,
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
        supportsUsageInStreaming: false,
        maxTokensField: "max_tokens",
      },
    });
    ctx.ui.notify(`Selected Pi model: ${modelId}`, "info");
  } catch (error) {
    ctx.ui.notify(`Could not auto-select Pi model: ${formatSyncError(error)}`, "warning");
  }
}

async function collectMainModelPaths(config: LlamaManagerConfig): Promise<string[]> {
  const discovered = new Set<string>();
  for (const root of config.modelsRoots) {
    const files = await listGgufFiles(root);
    for (const file of files) {
      const resolved = path.resolve(file);
      if (isLikelyMainModelPath(resolved)) discovered.add(resolved);
    }
  }
  return [...discovered].sort();
}

async function syncAllDiscoveredModels(
  ctx: ExtensionContext,
  config: LlamaManagerConfig,
): Promise<ReturnType<typeof syncPiModelsRegistryData> | null> {
  return syncPiRegistryForPaths(ctx, config, await collectMainModelPaths(config), {
    force: true,
    notifyPerModel: false,
  });
}

async function switchServerForSelectedPiModel(
  event: { model?: { provider?: string; id?: string }; source?: string },
  ctx: ExtensionContext,
  pi: ExtensionAPI,
): Promise<void> {
  if (event.model?.provider !== "llama-cpp") return;

  const modelId = event.model.id || "";
  const config = await loadConfig();
  if (!config.autoSwitchOnModelSelect) {
    ctx.ui.notify(`llama.cpp auto-switch is disabled. Use /llama start or /llama restart for ${modelId}.`, "info");
    return;
  }
  const modelPath = findModelPathForPiModelId(modelId, await collectMainModelPaths(config));
  if (!modelPath) {
    ctx.ui.notify(`No local GGUF path found for llama.cpp model: ${modelId}`, "warning");
    return;
  }

  const running = await getRunningServers(pi);
  if (running.length > 0 && !running.some((server) => modelMatchesRequest(server.modelPath, modelPath))) {
    ctx.ui.notify(`Switching llama-server to ${path.basename(modelPath)}…`, "info");
    await stopServers(pi, running);
  }

  const result = await startServer(ctx, pi, config, modelPath);
  ctx.ui.notify(result.message, result.ok ? "info" : "error");
  await refreshStatusWidget(ctx, pi);
}

async function listGgufFiles(root: string, maxDepth = 8): Promise<string[]> {
  const out: string[] = [];
  const start = expandHome(root);

  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > maxDepth) return;
    let entries: Array<import("node:fs").Dirent> = [];
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full, depth + 1);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".gguf")) {
        out.push(full);
      }
    }
  }

  await walk(start, 0);
  return out;
}

async function readPiLlamaModelIds(): Promise<Array<{ id: string; name?: string }>> {
  try {
    const raw = await fs.readFile(PI_MODELS_PATH, "utf8");
    return parsePiLlamaModels(raw);
  } catch {
    return [];
  }
}

async function collectModelOptions(config: LlamaManagerConfig): Promise<ModelOption[]> {
  const discovered = new Set<string>();
  for (const root of config.modelsRoots) {
    const files = await listGgufFiles(root);
    for (const file of files) {
      const resolved = path.resolve(file);
      if (!isLikelyMainModelPath(resolved)) continue;
      discovered.add(resolved);
    }
  }

  const options: ModelOption[] = [];
  const seen = new Set<string>();

  for (const abs of [...discovered].sort()) {
    if (seen.has(abs)) continue;
    seen.add(abs);
    options.push({
      label: path.basename(abs),
      value: abs,
      description: abs,
    });
  }

  const ids = await readPiLlamaModelIds();
  for (const model of ids) {
    const id = model.id;
    if (!id) continue;

    if (path.isAbsolute(id) && (await fileExists(id))) {
      const abs = path.resolve(id);
      if (!seen.has(abs)) {
        seen.add(abs);
        options.push({
          label: model.name ? `${model.name}` : path.basename(abs),
          value: abs,
          description: `models.json → ${abs}`,
        });
      }
      continue;
    }

    if (id.toLowerCase().endsWith(".gguf")) {
      // If it is only an id (not absolute path), we only expose it when we can
      // resolve it to an actual local file in discovered roots.
      const match = [...seen].find((p) => path.basename(p) === id);
      if (!match) continue;
    }
  }

  return options.sort((a, b) => a.label.localeCompare(b.label));
}

async function getRunningServers(pi: ExtensionAPI): Promise<RunningServer[]> {
  const { stdout } = await pi.exec("bash", [
    "-lc",
    "ps -ax -o pid=,command= | grep '[l]lama-server' || true",
  ]);

  return parsePsLlamaLines(stdout).map((row) => ({
    pid: row.pid,
    command: row.command,
    modelPath: extractModelPathFromCommand(row.command),
    port: row.port,
  }));
}

async function stopServers(pi: ExtensionAPI, servers: RunningServer[]): Promise<void> {
  if (servers.length === 0) return;
  const pids = servers.map((s) => String(s.pid));
  await pi.exec("kill", pids);

  for (let i = 0; i < 20; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    const running = await getRunningServers(pi);
    const remaining = running.filter((s) => pids.includes(String(s.pid)));
    if (remaining.length === 0) return;
  }

  await pi.exec("kill", ["-9", ...pids]);
}

function resolveUserModelPath(input: string, cwd?: string): string {
  const expanded = expandHome(input);
  if (path.isAbsolute(expanded)) return path.resolve(expanded);
  return path.resolve(cwd || process.cwd(), expanded);
}

function parseContentLengthFromHeaders(headersText: string): number | null {
  const lines = String(headersText || "").split(/\r?\n/);
  let contentLength: number | null = null;
  for (const line of lines) {
    const m = line.match(/^\s*content-length\s*:\s*(\d+)\s*$/i);
    if (!m) continue;
    const value = Number.parseInt(m[1]!, 10);
    if (Number.isFinite(value)) contentLength = value;
  }
  return contentLength;
}

function formatBytes(bytes: number): string {
  const n = Math.max(0, Number(bytes) || 0);
  if (n < 1024) return `${Math.round(n)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = n / 1024;
  let unit = units[0]!;
  for (let i = 1; i < units.length && value >= 1024; i += 1) {
    value /= 1024;
    unit = units[i]!;
  }
  return `${value.toFixed(1)} ${unit}`;
}

function formatPercent(done: number, total: number | null): string {
  const d = Math.max(0, Number(done) || 0);
  const t = Number(total) || 0;
  if (!Number.isFinite(t) || t <= 0) return "--";
  const pct = Math.min(100, (d / t) * 100);
  return `${pct.toFixed(1)}%`;
}

function normalizeDownloadUrl(url: string): string {
  const trimmed = url.trim();
  if (!trimmed) return "";

  try {
    const parsed = new URL(trimmed);
    if (parsed.hostname === "huggingface.co" && parsed.pathname.includes("/blob/")) {
      parsed.pathname = parsed.pathname.replace("/blob/", "/resolve/");
      parsed.search = "";
    }
    return parsed.toString();
  } catch {
    return trimmed;
  }
}

function inferFilenameFromUrl(url: string): string {
  try {
    const parsed = new URL(url);
    const base = parsed.pathname.split("/").pop() || "";
    return decodeURIComponent(base);
  } catch {
    return "";
  }
}

function buildCurlDownloadCommand(url: string, targetPath: string): string {
  return `curl -L --fail --progress-bar -C - -o ${shellQuote(targetPath)} ${shellQuote(url)}`;
}

async function fetchRemoteContentLength(pi: ExtensionAPI, url: string): Promise<number | null> {
  const cmd = `curl -sIL ${shellQuote(url)}`;
  const result = await pi.exec("bash", ["-lc", cmd]);
  if (result.code !== 0) return null;
  return parseContentLengthFromHeaders(result.stdout || "");
}

async function updateDownloadProgress(ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
  if (!activeDownload || activeDownload.status !== "running") {
    await refreshStatusWidget(ctx, pi);
    return;
  }

  try {
    const st = await fs.stat(activeDownload.targetPath);
    const now = Date.now();
    const elapsedSec = Math.max(0.001, (now - activeDownload.lastSampleAt) / 1000);
    const delta = Math.max(0, st.size - activeDownload.lastSampleBytes);

    activeDownload.downloadedBytes = st.size;
    activeDownload.bytesPerSecond = delta / elapsedSec;
    activeDownload.lastSampleAt = now;
    activeDownload.lastSampleBytes = st.size;
  } catch {
    // file may not exist yet at the very beginning
  }

  await refreshStatusWidget(ctx, pi);
}

function clearDownloadTimer() {
  if (activeDownload?.pollTimer) {
    clearInterval(activeDownload.pollTimer);
    activeDownload.pollTimer = undefined;
  }
}

async function abortActiveDownload(ctx: ExtensionContext, pi: ExtensionAPI): Promise<{ ok: boolean; message: string }> {
  if (!activeDownload || activeDownload.status !== "running") {
    return { ok: false, message: "No active download to abort" };
  }

  activeDownload.status = "aborted";
  try {
    activeDownload.child.kill("SIGINT");
  } catch {
    // ignore
  }
  clearDownloadTimer();
  await refreshStatusWidget(ctx, pi);
  return { ok: true, message: `Aborted download: ${activeDownload.fileName}` };
}

async function downloadModelFromUrl(
  ctx: ExtensionContext,
  pi: ExtensionAPI,
  config: LlamaManagerConfig,
  url: string,
  destinationDir?: string,
): Promise<{ ok: boolean; message: string; targetPath?: string }> {
  if (activeDownload && activeDownload.status === "running") {
    return { ok: false, message: `Another download is running: ${activeDownload.fileName}` };
  }

  const cleanedUrl = normalizeDownloadUrl(url);
  if (!cleanedUrl) return { ok: false, message: "Download URL is required" };

  const inferred = inferFilenameFromUrl(cleanedUrl);
  const fileName = inferred || "model.gguf";
  const resolvedDir = resolveUserModelPath(destinationDir || config.downloadDir || config.modelsRoots[0] || "~/.pi/models", ctx.cwd);
  const targetPath = path.join(resolvedDir, fileName);

  await fs.mkdir(resolvedDir, { recursive: true });

  if (await fileExists(targetPath)) {
    return { ok: false, message: `File already exists: ${targetPath}` };
  }

  const totalBytes = await fetchRemoteContentLength(pi, cleanedUrl);
  const command = buildCurlDownloadCommand(cleanedUrl, targetPath);

  const child = spawn("bash", ["-lc", command], {
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });

  activeDownload = {
    child,
    fileName,
    targetPath,
    totalBytes,
    downloadedBytes: 0,
    lastSampleAt: Date.now(),
    lastSampleBytes: 0,
    bytesPerSecond: 0,
    status: "running",
  };

  activeDownload.pollTimer = setInterval(() => {
    void updateDownloadProgress(ctx, pi);
  }, 1000);

  child.on("error", async (error) => {
    if (!activeDownload) return;
    activeDownload.status = "error";
    activeDownload.error = error.message;
    clearDownloadTimer();
    ctx.ui.notify(`Download failed: ${error.message}`, "error");
    await refreshStatusWidget(ctx, pi);
  });

  child.on("exit", async (code, signal) => {
    if (!activeDownload) return;

    const wasRunning = activeDownload.status === "running";
    if (activeDownload.status === "aborted") {
      ctx.ui.notify(`Download aborted: ${fileName}`, "warning");
    } else if (code === 0) {
      activeDownload.status = "done";
      ctx.ui.notify(`Download complete: ${targetPath}`, "info");
      await syncPiRegistryForPaths(ctx, config, [targetPath]);
    } else if (wasRunning) {
      activeDownload.status = "error";
      const reason = signal ? `signal ${signal}` : `exit code ${code}`;
      ctx.ui.notify(`Download failed: ${reason}`, "error");
    }

    clearDownloadTimer();
    await updateDownloadProgress(ctx, pi);

    setTimeout(async () => {
      if (activeDownload && activeDownload.status !== "running") {
        activeDownload = null;
        await refreshStatusWidget(ctx, pi);
      }
    }, 5000);
  });

  ctx.ui.notify(`Download started in background: ${fileName}`, "info");
  await refreshStatusWidget(ctx, pi);
  return { ok: true, message: `Download started: ${targetPath}`, targetPath };
}

function buildStartCommand(config: LlamaManagerConfig, modelPath: string, logFile = config.logFile): string {
  const args = buildLlamaArgs(config, modelPath);
  const cmd = ["nohup", "llama-server", ...args]
    .map((part) => shellQuote(part))
    .join(" ");
  return `${cmd} >> ${shellQuote(logFile)} 2>&1 & echo $!`;
}

function timestampForFile(date = new Date()): string {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function buildLaunchLogPath(config: LlamaManagerConfig, date = new Date()): string {
  const directory = path.join(path.dirname(config.logFile), "llama-server-runs");
  return path.join(directory, `llama-server-${timestampForFile(date)}.log`);
}

async function pointCurrentLogAtLaunch(config: LlamaManagerConfig, launchLogFile: string): Promise<void> {
  await fs.mkdir(path.dirname(launchLogFile), { recursive: true });
  await fs.mkdir(path.dirname(config.logFile), { recursive: true });
  try {
    const stat = await fs.lstat(config.logFile);
    if (stat.isSymbolicLink()) {
      await fs.unlink(config.logFile);
    } else if (stat.isFile()) {
      await fs.rename(config.logFile, `${config.logFile}.legacy-${timestampForFile()}`);
    }
  } catch (error) {
    if ((error as any)?.code !== "ENOENT") throw error;
  }
  await fs.symlink(launchLogFile, config.logFile);
}

async function appendLaunchHistory(config: LlamaManagerConfig, record: Record<string, unknown>): Promise<void> {
  await fs.mkdir(path.dirname(config.launchHistoryFile), { recursive: true });
  await fs.appendFile(config.launchHistoryFile, `${JSON.stringify(record)}\n`, "utf8");
}

async function readLastLaunchRecord(config: LlamaManagerConfig): Promise<any | null> {
  try {
    const raw = await fs.readFile(config.launchHistoryFile, "utf8");
    const line = raw.trim().split(/\r?\n/).filter(Boolean).at(-1);
    return line ? JSON.parse(line) : null;
  } catch {
    return null;
  }
}

async function readLaunchMetrics(logFile: string): Promise<Record<string, unknown>> {
  try {
    return parseLlamaServerLogMetrics(await fs.readFile(logFile, "utf8"));
  } catch {
    return {};
  }
}

async function writeLaunchHeader(logFile: string, record: Record<string, unknown>): Promise<void> {
  await fs.writeFile(logFile, [
    `# pi-llama-manager launch ${record.startedAt}`,
    `# profile: ${record.profile}`,
    `# model: ${record.modelPath}`,
    `# args: ${((record.args as string[]) || []).join(" ")}`,
    "",
  ].join("\n"), "utf8");
}

async function startServer(
  ctx: ExtensionContext,
  pi: ExtensionAPI,
  config: LlamaManagerConfig,
  modelPath: string,
): Promise<{ ok: boolean; message: string }> {
  const running = await getRunningServers(pi);
  if (running.length > 0) {
    const currentModel = running[0]?.modelPath || "unknown model";
    if (modelMatchesRequest(currentModel, modelPath)) {
      await syncStartedModelToPiRegistry(ctx, pi, config, modelPath);
      return { ok: true, message: `llama-server already running requested model (${currentModel})` };
    }
    return { ok: false, message: `Refusing start: llama-server already running (${currentModel}). Use /llama restart to switch models.` };
  }

  if (!path.isAbsolute(modelPath)) {
    return { ok: false, message: `Model path is not absolute: ${modelPath}` };
  }

  if (!(await fileExists(modelPath))) {
    return { ok: false, message: `Model does not exist: ${modelPath}` };
  }

  const startedAt = new Date().toISOString();
  const launchLogFile = buildLaunchLogPath(config, new Date(startedAt));
  const args = buildLlamaArgs(config, modelPath);
  const baseRecord = createLaunchRecord({
    startedAt,
    profile: config.serverProfile,
    modelPath,
    args,
    logFile: launchLogFile,
    outcome: "starting",
  });

  await pointCurrentLogAtLaunch(config, launchLogFile);
  await writeLaunchHeader(launchLogFile, baseRecord as Record<string, unknown>);
  const cmd = buildStartCommand(config, modelPath, launchLogFile);
  const result = await pi.exec("bash", ["-lc", cmd]);
  if (result.code !== 0) {
    const metrics = await readLaunchMetrics(launchLogFile);
    const warnings = assessLaunchMetrics(metrics, profileSyncDefaults(config.serverProfile));
    await appendLaunchHistory(config, createLaunchRecord({
      ...baseRecord,
      finishedAt: new Date().toISOString(),
      outcome: "start-command-failed",
      message: result.stderr || result.stdout || "Failed to start llama-server",
      metrics,
      warnings,
    }));
    return { ok: false, message: result.stderr || result.stdout || "Failed to start llama-server" };
  }

  await new Promise((resolve) => setTimeout(resolve, 700));
  const nowRunning = await getRunningServers(pi);
  const metrics = await readLaunchMetrics(launchLogFile);
  const warnings = assessLaunchMetrics(metrics, profileSyncDefaults(config.serverProfile));
  if (nowRunning.length === 0) {
    await appendLaunchHistory(config, createLaunchRecord({
      ...baseRecord,
      finishedAt: new Date().toISOString(),
      outcome: "not-detected-after-start",
      message: `Start command ran but server not detected. Check logs: ${config.logFile}`,
      metrics,
      warnings,
    }));
    return { ok: false, message: `Start command ran but server not detected. Check logs: ${config.logFile}` };
  }

  await appendLaunchHistory(config, createLaunchRecord({
    ...baseRecord,
    finishedAt: new Date().toISOString(),
    pid: nowRunning[0].pid,
    outcome: "started",
    metrics,
    warnings,
  }));
  await syncStartedModelToPiRegistry(ctx, pi, config, nowRunning[0].modelPath || modelPath);
  return {
    ok: true,
    message: `Started llama-server pid ${nowRunning[0].pid} with ${nowRunning[0].modelPath || modelPath}`,
  };
}

function statusSummary(servers: RunningServer[]): string {
  if (servers.length === 0) return "llama.cpp: stopped";
  const first = servers[0];
  const model = first.modelPath ? path.basename(first.modelPath) : "unknown-model";
  const port = first.port ?? "?";
  return `llama.cpp: running (${model} @ :${port})`;
}

function downloadStatusSummary(download: DownloadState | null): string {
  if (!download) return "";

  if (download.status === "aborted") {
    return ` | download: aborted (${download.fileName})`;
  }
  if (download.status === "error") {
    return ` | download: error (${download.fileName})`;
  }
  if (download.status === "done") {
    return ` | download: done (${download.fileName})`;
  }

  const pct = formatPercent(download.downloadedBytes, download.totalBytes);
  const done = formatBytes(download.downloadedBytes);
  const total = download.totalBytes ? formatBytes(download.totalBytes) : "?";
  const speed = download.bytesPerSecond > 0 ? ` @ ${formatBytes(download.bytesPerSecond)}/s` : "";
  return ` | download: ${download.fileName} ${pct} (${done}/${total})${speed}`;
}

function setStatusWidget(ctx: ExtensionContext, servers: RunningServer[]): void {
  if (!ctx.hasUI) return;
  const text = `${statusSummary(servers)}${downloadStatusSummary(activeDownload)}`;
  ctx.ui.setStatus(STATUS_KEY, ctx.ui.theme.fg("dim", text));
}

async function refreshStatusWidget(ctx: ExtensionContext, pi: ExtensionAPI): Promise<RunningServer[]> {
  const servers = await getRunningServers(pi);
  setStatusWidget(ctx, servers);
  return servers;
}

async function showStatus(ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
  const servers = await refreshStatusWidget(ctx, pi);
  const summary = statusSummary(servers);
  if (servers.length === 0) {
    ctx.ui.notify(summary, "info");
    return;
  }

  const first = servers[0];
  ctx.ui.notify(`${summary} pid=${first.pid}`, "info");
}

async function fetchServerJson(config: LlamaManagerConfig, endpoint: string): Promise<any | null> {
  try {
    const response = await fetch(`${getServerBaseUrl(config)}${endpoint}`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function commandHasFlag(command: string, flag: string): boolean {
  return command.split(/\s+/).includes(flag);
}

async function showDoctor(ctx: ExtensionContext, pi: ExtensionAPI, config: LlamaManagerConfig): Promise<void> {
  const servers = await refreshStatusWidget(ctx, pi);
  const profileDefaults = profileSyncDefaults(config.serverProfile);
  const profile = (LLAMA_SERVER_PROFILES as any)[config.serverProfile];
  const props = servers.length > 0 ? await fetchServerJson(config, "/props") : null;
  const health = servers.length > 0 ? await fetchServerJson(config, "/health") : null;
  const runningCtx = props?.default_generation_settings?.n_ctx ?? props?.default_generation_settings?.params?.n_ctx;
  const command = servers[0]?.command || "";
  const lines = [
    `llama.cpp doctor`,
    `configured profile: ${config.serverProfile} (${profile?.label || "unknown"})`,
    `desired context/max tokens: ${profileDefaults.contextWindow}/${profileDefaults.maxTokens}`,
    `configured bind: ${config.host}:${config.port}`,
    `auto-switch on Pi model select: ${config.autoSwitchOnModelSelect ? "enabled" : "disabled"}`,
    `server: ${servers.length > 0 ? "running" : "stopped"}`,
  ];

  if (servers.length > 0) {
    lines.push(`running model: ${props?.model_alias || path.basename(servers[0].modelPath || "unknown")}`);
    lines.push(`running context: ${runningCtx || "unknown"}`);
    lines.push(`health: ${health?.status || "unknown"}`);
    lines.push(`--no-context-shift: ${commandHasFlag(command, "--no-context-shift") ? "present" : "missing"}`);
    lines.push(`--n-predict: ${commandHasFlag(command, "--n-predict") || commandHasFlag(command, "--predict") ? "present" : "missing"}`);
    lines.push(`note: doctor never starts or restarts llama-server`);
  }

  const logMetrics = await readLaunchMetrics(config.logFile);
  const metricWarnings = assessLaunchMetrics(logMetrics, profileDefaults);
  const lastLaunch = await readLastLaunchRecord(config);
  if (Object.keys(logMetrics).length > 0) {
    lines.push(`last log metrics: ctx=${(logMetrics as any).contextSize || "?"}, kv=${(logMetrics as any).metalKvCacheMiB || "?"} MiB, projected=${(logMetrics as any).deviceMemoryProjectedMiB || "?"}/${(logMetrics as any).deviceMemoryFreeMiB || "?"} MiB`);
  }
  if (lastLaunch) {
    lines.push(`last launch: ${lastLaunch.outcome || "unknown"} profile=${lastLaunch.profile || "unknown"} log=${lastLaunch.logFile || config.logFile}`);
  }
  for (const warning of metricWarnings) {
    lines.push(`${warning.level === "warning" ? "warning" : "note"}: ${warning.message}`);
  }

  ctx.ui.notify(lines.join("\n"), metricWarnings.some((warning: any) => warning.level === "warning") ? "warning" : "info");
}

async function setServerProfile(ctx: ExtensionContext, config: LlamaManagerConfig, profileName: string): Promise<void> {
  const normalized = normalizeLlamaProfileName(profileName) as "fast" | "code" | "deep" | "wide";
  config.serverProfile = normalized;
  config.syncDefaults = profileSyncDefaults(normalized);
  await saveConfig(config);
  const profile = (LLAMA_SERVER_PROFILES as any)[normalized];
  ctx.ui.notify(`llama profile set to ${normalized} (${profile?.contextWindow} ctx, max ${profile?.maxTokens}). Restart manually when ready.`, "info");
}

async function pickModel(ctx: ExtensionContext, config: LlamaManagerConfig): Promise<string | undefined> {
  const options = await collectModelOptions(config);
  if (options.length === 0) {
    ctx.ui.notify("No .gguf files found. Update ~/.pi/agent/llama-manager.json modelsRoots.", "warning");
    return undefined;
  }

  const labels = options.map((o) => `${o.label} — ${o.description}`);
  const selected = await ctx.ui.select("Select GGUF model", labels);
  if (!selected) return undefined;

  const idx = labels.indexOf(selected);
  if (idx < 0) return undefined;
  const value = options[idx]?.value;
  return path.isAbsolute(value) ? value : undefined;
}

async function handleInteractive(ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
  let config = await loadConfig();

  while (true) {
    const choice = await ctx.ui.select("llama.cpp manager", [
      "Status",
      "Doctor",
      "Set server profile…",
      "Start (default model)",
      "Stop",
      "Restart",
      "Start with model…",
      "Set default model…",
      "Download model from URL…",
      "Sync Pi model registry",
      "Abort active download",
      "Open config path",
      "Tail logs",
    ]);

    if (!choice) return;

    if (choice === "Status") {
      await showStatus(ctx, pi);
      continue;
    }

    if (choice === "Doctor") {
      await showDoctor(ctx, pi, config);
      continue;
    }

    if (choice === "Set server profile…") {
      const selected = await ctx.ui.select("Select llama-server profile", [
        "fast — 8k context, safe/fast",
        "code — 16k context, default coding profile",
        "deep — 32k context, monitored deep investigation",
        "wide — 128k context, high-performance profile with 1GiB prompt cache",
      ]);
      if (selected) {
        await setServerProfile(ctx, config, selected.split(" ")[0] || "code");
        config = await loadConfig();
      }
      continue;
    }

    if (choice === "Start (default model)") {
      const model = config.defaultModelPath ? resolveUserModelPath(config.defaultModelPath, ctx.cwd) : "";
      if (!model) {
        ctx.ui.notify("No default model set. Choose 'Set default model…' first.", "warning");
        continue;
      }
      const result = await startServer(ctx, pi, config, model);
      ctx.ui.notify(result.message, result.ok ? "info" : "error");
      await refreshStatusWidget(ctx, pi);
      continue;
    }

    if (choice === "Stop") {
      const running = await getRunningServers(pi);
      if (running.length === 0) {
        ctx.ui.notify("llama-server already stopped", "info");
        await refreshStatusWidget(ctx, pi);
        continue;
      }
      await stopServers(pi, running);
      ctx.ui.notify("Stopped llama-server", "info");
      await refreshStatusWidget(ctx, pi);
      continue;
    }

    if (choice === "Restart") {
      const model = config.defaultModelPath ? resolveUserModelPath(config.defaultModelPath, ctx.cwd) : "";
      if (!model) {
        ctx.ui.notify("No default model set. Choose 'Set default model…' first.", "warning");
        continue;
      }

      const running = await getRunningServers(pi);
      if (running.length > 0) {
        await stopServers(pi, running);
      }
      const result = await startServer(ctx, pi, config, model);
      ctx.ui.notify(result.message, result.ok ? "info" : "error");
      await refreshStatusWidget(ctx, pi);
      continue;
    }

    if (choice === "Start with model…") {
      const model = await pickModel(ctx, config);
      if (!model) continue;
      const result = await startServer(ctx, pi, config, model);
      ctx.ui.notify(result.message, result.ok ? "info" : "error");
      await refreshStatusWidget(ctx, pi);
      continue;
    }

    if (choice === "Set default model…") {
      const model = await pickModel(ctx, config);
      if (!model) continue;
      config.defaultModelPath = model;
      await saveConfig(config);
      ctx.ui.notify(`Default model set: ${model}`, "info");
      continue;
    }

    if (choice === "Download model from URL…") {
      const url = await ctx.ui.input("Model URL (.gguf)", "");
      if (!url?.trim()) continue;

      const destination = await ctx.ui.input(
        "Destination directory",
        config.downloadDir || config.modelsRoots[0] || "~/.pi/models",
      );
      const destinationDir = destination?.trim() || config.downloadDir || config.modelsRoots[0] || "~/.pi/models";

      const result = await downloadModelFromUrl(ctx, pi, config, url, destinationDir);
      ctx.ui.notify(result.message, result.ok ? "info" : "error");

      if (result.ok && destinationDir.trim()) {
        config.downloadDir = destinationDir.trim();
        await saveConfig(config);
      }
      continue;
    }

    if (choice === "Sync Pi model registry") {
      const result = await syncAllDiscoveredModels(ctx, config);
      if (result) {
        ctx.ui.notify(
          `Pi registry sync complete: ${result.counts.added} added, ${result.counts.updated} updated, ${result.counts.skipped} skipped`,
          "info",
        );
      }
      continue;
    }

    if (choice === "Abort active download") {
      const result = await abortActiveDownload(ctx, pi);
      ctx.ui.notify(result.message, result.ok ? "info" : "warning");
      continue;
    }

    if (choice === "Open config path") {
      ctx.ui.notify(CONFIG_PATH, "info");
      continue;
    }

    if (choice === "Tail logs") {
      const { stdout } = await pi.exec("bash", ["-lc", `tail -n 60 ${shellQuote(config.logFile)} 2>/dev/null || true`]);
      if (!stdout.trim()) {
        ctx.ui.notify(`No log output at ${config.logFile}`, "info");
      } else {
        const lastLine = stdout.trim().split("\n").slice(-1)[0] ?? "";
        ctx.ui.notify(`Last log line: ${lastLine.slice(0, 180)}`, "info");
      }
      continue;
    }
  }
}

async function handleArgs(args: string, ctx: ExtensionContext, pi: ExtensionAPI): Promise<boolean> {
  const config = await loadConfig();
  const [cmd, ...rest] = args.trim().split(/\s+/).filter(Boolean);
  if (!cmd) return false;

  if (cmd === "status") {
    await showStatus(ctx, pi);
    return true;
  }

  if (cmd === "doctor") {
    await showDoctor(ctx, pi, config);
    return true;
  }

  if (cmd === "profile") {
    const requested = rest[0]?.trim();
    if (!requested) {
      const profile = (LLAMA_SERVER_PROFILES as any)[config.serverProfile];
      ctx.ui.notify(`current llama profile: ${config.serverProfile} (${profile?.contextWindow} ctx, max ${profile?.maxTokens})`, "info");
      return true;
    }
    if (!(requested in (LLAMA_SERVER_PROFILES as any))) {
      ctx.ui.notify("Usage: /llama profile fast|code|deep|wide", "warning");
      return true;
    }
    await setServerProfile(ctx, config, requested);
    return true;
  }

  if (cmd === "stop") {
    const running = await getRunningServers(pi);
    await stopServers(pi, running);
    ctx.ui.notify(running.length > 0 ? "Stopped llama-server" : "llama-server already stopped", "info");
    await refreshStatusWidget(ctx, pi);
    return true;
  }

  if (cmd === "start") {
    const modelArg = rest.join(" ").trim();
    const selected = modelArg || config.defaultModelPath || "";
    const model = selected ? resolveUserModelPath(selected, ctx.cwd) : "";
    if (!model) {
      ctx.ui.notify("No model provided and no default model configured", "warning");
      return true;
    }
    const result = await startServer(ctx, pi, config, model);
    ctx.ui.notify(result.message, result.ok ? "info" : "error");
    await refreshStatusWidget(ctx, pi);
    return true;
  }

  if (cmd === "restart") {
    const modelArg = rest.join(" ").trim();
    const selected = modelArg || config.defaultModelPath || "";
    const model = selected ? resolveUserModelPath(selected, ctx.cwd) : "";
    if (!model) {
      ctx.ui.notify("No model provided and no default model configured", "warning");
      return true;
    }
    const running = await getRunningServers(pi);
    await stopServers(pi, running);
    const result = await startServer(ctx, pi, config, model);
    ctx.ui.notify(result.message, result.ok ? "info" : "error");
    await refreshStatusWidget(ctx, pi);
    return true;
  }

  if (cmd === "download") {
    if (rest[0]?.trim() === "abort") {
      const aborted = await abortActiveDownload(ctx, pi);
      ctx.ui.notify(aborted.message, aborted.ok ? "info" : "warning");
      return true;
    }

    const url = rest[0]?.trim();
    if (!url) {
      ctx.ui.notify("Usage: /llama download <url> [destination-dir] | /llama download abort", "warning");
      return true;
    }
    const destination = rest.length > 1 ? rest.slice(1).join(" ").trim() : undefined;
    const result = await downloadModelFromUrl(ctx, pi, config, url, destination);
    ctx.ui.notify(result.message, result.ok ? "info" : "error");

    if (result.ok && destination) {
      config.downloadDir = destination;
      await saveConfig(config);
    }
    return true;
  }

  if (cmd === "download-abort") {
    const aborted = await abortActiveDownload(ctx, pi);
    ctx.ui.notify(aborted.message, aborted.ok ? "info" : "warning");
    return true;
  }

  if (cmd === "sync-models") {
    const result = await syncAllDiscoveredModels(ctx, config);
    if (result) {
      ctx.ui.notify(
        `Pi registry sync complete: ${result.counts.added} added, ${result.counts.updated} updated, ${result.counts.skipped} skipped`,
        "info",
      );
    }
    return true;
  }

  return false;
}

export default function llamaManagerExtension(pi: ExtensionAPI): void {
  pi.on("session_start", async (_event, ctx) => {
    await refreshStatusWidget(ctx, pi);
  });

  pi.on("session_switch", async (_event, ctx) => {
    await refreshStatusWidget(ctx, pi);
  });

  pi.on("model_select", async (event, ctx) => {
    await switchServerForSelectedPiModel(event, ctx, pi);
  });

  pi.on("turn_end", async (_event, ctx) => {
    await refreshStatusWidget(ctx, pi);
  });

  pi.registerCommand("llama", {
    description: "Manage local llama.cpp server and GGUF models",
    handler: async (args, ctx) => {
      const handled = await handleArgs(args, ctx, pi);
      if (handled) return;
      await handleInteractive(ctx, pi);
    },
  });
}
