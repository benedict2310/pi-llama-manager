import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
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
  const args = [
    "-m",
    modelPath,
    "--host",
    String(config.host ?? "127.0.0.1"),
    "--port",
    String(config.port ?? 8080),
  ];

  if (config?.stableToolCalling && config?.defaultArgs?.jinja) args.push("--jinja");
  if (config?.stableToolCalling && config?.defaultArgs?.reasoning) {
    args.push("--reasoning", String(config.defaultArgs.reasoning));
  }
  if (config?.stableToolCalling && config?.defaultArgs?.chatTemplateKwargs) {
    args.push("--chat-template-kwargs", JSON.stringify(config.defaultArgs.chatTemplateKwargs));
  }
  if (config?.stableToolCalling && typeof config?.defaultArgs?.temp === "number") {
    args.push("--temp", String(config.defaultArgs.temp));
  }
  if (config?.stableToolCalling && typeof config?.defaultArgs?.topP === "number") {
    args.push("--top-p", String(config.defaultArgs.topP));
  }
  if (Array.isArray(config?.extraArgs)) args.push(...config.extraArgs.map((part) => String(part)));

  return args;
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

type LlamaManagerConfig = {
  host: string;
  port: number;
  modelsRoots: string[];
  defaultModelPath?: string;
  logFile: string;
  stableToolCalling: boolean;
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

function expandHome(input: string): string {
  if (input === "~") return os.homedir();
  if (input.startsWith("~/")) return path.join(os.homedir(), input.slice(2));
  return input;
}

function defaultConfig(): LlamaManagerConfig {
  return {
    host: "0.0.0.0",
    port: 8080,
    modelsRoots: [path.join(os.homedir(), "models")],
    defaultModelPath: "",
    logFile: path.join(os.homedir(), ".pi", "agent", "llama-server.log"),
    stableToolCalling: true,
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
    return {
      host: parsed.host || defaults.host,
      port: typeof parsed.port === "number" ? parsed.port : defaults.port,
      modelsRoots: Array.isArray(parsed.modelsRoots) && parsed.modelsRoots.length > 0
        ? parsed.modelsRoots.map((v) => expandHome(String(v)))
        : defaults.modelsRoots,
      defaultModelPath: parsed.defaultModelPath ? expandHome(String(parsed.defaultModelPath)) : defaults.defaultModelPath,
      logFile: parsed.logFile ? expandHome(String(parsed.logFile)) : defaults.logFile,
      stableToolCalling: parsed.stableToolCalling ?? defaults.stableToolCalling,
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

function buildStartCommand(config: LlamaManagerConfig, modelPath: string): string {
  const args = buildLlamaArgs(config, modelPath);
  const cmd = ["nohup", "llama-server", ...args]
    .map((part) => shellQuote(part))
    .join(" ");
  return `${cmd} > ${shellQuote(config.logFile)} 2>&1 & echo $!`;
}

async function startServer(pi: ExtensionAPI, config: LlamaManagerConfig, modelPath: string): Promise<{ ok: boolean; message: string }> {
  const running = await getRunningServers(pi);
  if (running.length > 0) {
    const currentModel = running[0]?.modelPath || "unknown model";
    if (modelMatchesRequest(currentModel, modelPath)) {
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

  await fs.mkdir(path.dirname(config.logFile), { recursive: true });
  const cmd = buildStartCommand(config, modelPath);
  const result = await pi.exec("bash", ["-lc", cmd]);
  if (result.code !== 0) {
    return { ok: false, message: result.stderr || result.stdout || "Failed to start llama-server" };
  }

  await new Promise((resolve) => setTimeout(resolve, 700));
  const nowRunning = await getRunningServers(pi);
  if (nowRunning.length === 0) {
    return { ok: false, message: `Start command ran but server not detected. Check logs: ${config.logFile}` };
  }

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

function setStatusWidget(ctx: ExtensionContext, servers: RunningServer[]): void {
  if (!ctx.hasUI) return;
  ctx.ui.setStatus(STATUS_KEY, ctx.ui.theme.fg("dim", statusSummary(servers)));
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
      "Start (default model)",
      "Stop",
      "Restart",
      "Start with model…",
      "Set default model…",
      "Open config path",
      "Tail logs",
    ]);

    if (!choice) return;

    if (choice === "Status") {
      await showStatus(ctx, pi);
      continue;
    }

    if (choice === "Start (default model)") {
      const model = config.defaultModelPath ? resolveUserModelPath(config.defaultModelPath, ctx.cwd) : "";
      if (!model) {
        ctx.ui.notify("No default model set. Choose 'Set default model…' first.", "warning");
        continue;
      }
      const result = await startServer(pi, config, model);
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
      const result = await startServer(pi, config, model);
      ctx.ui.notify(result.message, result.ok ? "info" : "error");
      await refreshStatusWidget(ctx, pi);
      continue;
    }

    if (choice === "Start with model…") {
      const model = await pickModel(ctx, config);
      if (!model) continue;
      const result = await startServer(pi, config, model);
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
    const result = await startServer(pi, config, model);
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
    const result = await startServer(pi, config, model);
    ctx.ui.notify(result.message, result.ok ? "info" : "error");
    await refreshStatusWidget(ctx, pi);
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
