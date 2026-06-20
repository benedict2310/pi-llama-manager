import fs from 'node:fs/promises';
import path from 'node:path';

const DEFAULT_LLAMA_PROVIDER_COMPAT = {
  supportsDeveloperRole: false,
  supportsReasoningEffort: false,
  supportsUsageInStreaming: false,
  maxTokensField: 'max_tokens',
};

export const LLAMA_SERVER_PROFILES = {
  fast: {
    label: 'Fast / safe',
    contextWindow: 8192,
    maxTokens: 2048,
    args: ['--ctx-size', '8192', '--parallel', '1', '--batch-size', '512', '--ubatch-size', '128', '--flash-attn', 'auto', '--no-context-shift', '--n-predict', '2048'],
  },
  code: {
    label: 'Coding default',
    contextWindow: 16384,
    maxTokens: 2048,
    args: ['--ctx-size', '16384', '--parallel', '1', '--batch-size', '512', '--ubatch-size', '128', '--flash-attn', 'auto', '--no-context-shift', '--n-predict', '2048'],
  },
  deep: {
    label: 'Deep investigation',
    contextWindow: 32768,
    maxTokens: 4096,
    args: ['--ctx-size', '32768', '--parallel', '1', '--batch-size', '512', '--ubatch-size', '128', '--flash-attn', 'auto', '--no-context-shift', '--n-predict', '4096'],
  },
  wide: {
    label: 'Wide context / high performance',
    contextWindow: 131072,
    maxTokens: 4096,
    args: ['--ctx-size', '131072', '--parallel', '1', '--batch-size', '512', '--ubatch-size', '128', '--cache-ram', '1024', '--flash-attn', 'auto', '--no-context-shift', '--n-predict', '4096'],
  },
};

const LLAMA_PROFILE_DEFAULT = 'code';
const FLAGS_WITH_VALUES = new Set([
  '-m',
  '--model',
  '--host',
  '--port',
  '--ctx-size',
  '--parallel',
  '--batch-size',
  '--ubatch-size',
  '--flash-attn',
  '--n-predict',
  '--predict',
  '--reasoning',
  '--reasoning-format',
  '--chat-template-kwargs',
  '--temp',
  '--temperature',
  '--top-p',
  '--cache-type-k',
  '--cache-type-v',
  '--cache-ram',
]);

function zeroCost() {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
}

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function defaultLlamaCppProvider(baseUrl = 'http://127.0.0.1:8080/v1') {
  return {
    baseUrl,
    apiKey: 'local',
    api: 'openai-completions',
    compat: { ...DEFAULT_LLAMA_PROVIDER_COMPAT },
    models: [],
  };
}

function buildPiFriendlyModelName(modelId) {
  const friendly = String(modelId ?? '')
    .replace(/\.gguf$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/([a-z])(\d)/g, '$1 $2')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\s+/g, ' ')
    .trim();
  return friendly ? `${friendly} (llama.cpp)` : 'llama.cpp model';
}

export function normalizeLlamaProfileName(profileName) {
  const name = String(profileName || LLAMA_PROFILE_DEFAULT).trim();
  return Object.prototype.hasOwnProperty.call(LLAMA_SERVER_PROFILES, name) ? name : LLAMA_PROFILE_DEFAULT;
}

export function buildLlamaProfileArgs(profileName) {
  return [...LLAMA_SERVER_PROFILES[normalizeLlamaProfileName(profileName)].args];
}

export function profileSyncDefaults(profileName) {
  const profile = LLAMA_SERVER_PROFILES[normalizeLlamaProfileName(profileName)];
  return {
    contextWindow: profile.contextWindow,
    maxTokens: profile.maxTokens,
    reasoning: false,
  };
}

function flagKey(flag) {
  if (flag === '-c') return '--ctx-size';
  if (flag === '-n') return '--n-predict';
  if (flag === '-np') return '--parallel';
  if (flag === '-b') return '--batch-size';
  if (flag === '-ub') return '--ubatch-size';
  if (flag === '-fa') return '--flash-attn';
  if (flag === '-ctk') return '--cache-type-k';
  if (flag === '-ctv') return '--cache-type-v';
  return flag;
}

function flagKeysPresent(args) {
  const keys = new Set();
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (!String(token).startsWith('-')) continue;
    keys.add(flagKey(token));
    if (FLAGS_WITH_VALUES.has(token)) index += 1;
  }
  return keys;
}

export function mergeProfileAndExtraArgs(profileArgs, extraArgs = []) {
  const extra = Array.isArray(extraArgs) ? extraArgs.map((part) => String(part)) : [];
  const extraKeys = flagKeysPresent(extra);
  const merged = [];
  for (let index = 0; index < profileArgs.length; index += 1) {
    const token = String(profileArgs[index]);
    if (!token.startsWith('-')) {
      merged.push(token);
      continue;
    }

    const key = flagKey(token);
    const hasValue = FLAGS_WITH_VALUES.has(token);
    if (extraKeys.has(key)) {
      if (hasValue) index += 1;
      continue;
    }
    merged.push(token);
    if (hasValue && index + 1 < profileArgs.length) {
      index += 1;
      merged.push(String(profileArgs[index]));
    }
  }
  return [...merged, ...extra];
}

function buildLlamaModelEntry(entry) {
  return {
    id: String(entry.id),
    name: String(entry.name || buildPiFriendlyModelName(entry.id)),
    reasoning: Boolean(entry.reasoning),
    input: ['text'],
    cost: zeroCost(),
    contextWindow: Number(entry.contextWindow) || 32768,
    maxTokens: Number(entry.maxTokens) || 8192,
  };
}

export function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

export function parsePiLlamaModels(raw) {
  try {
    const parsed = JSON.parse(raw);
    const models = parsed?.providers?.['llama-cpp']?.models;
    if (!Array.isArray(models)) return [];
    return models
      .map((model) => ({
        id: typeof model?.id === 'string' ? model.id : '',
        name: typeof model?.name === 'string' ? model.name : undefined,
      }))
      .filter((model) => model.id);
  } catch {
    return [];
  }
}

function parseArgValue(parts, flag) {
  const index = parts.indexOf(flag);
  if (index < 0 || index + 1 >= parts.length) return undefined;
  return parts[index + 1];
}

function tokenizeCommand(command) {
  const tokens = [];
  let current = '';
  let quote = null;

  for (let i = 0; i < command.length; i += 1) {
    const ch = command[i];

    if (quote) {
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }

    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }

    if (/\s/.test(ch)) {
      if (current) {
        tokens.push(current);
        current = '';
      }
      continue;
    }

    current += ch;
  }

  if (current) tokens.push(current);
  return tokens;
}

export function extractModelPathFromCommand(command) {
  const parts = tokenizeCommand(command);
  return parseArgValue(parts, '-m') ?? parseArgValue(parts, '--model');
}

export function extractPortFromCommand(command, fallback = 8080) {
  const parts = tokenizeCommand(command);
  const value = parseArgValue(parts, '--port') ?? parseArgValue(parts, '-p');
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function isLikelyMainModelPath(modelPath) {
  const base = String(modelPath).split(/[\\/]/).pop()?.toLowerCase() || '';
  return base.endsWith('.gguf') && !base.startsWith('mmproj-') && !base.includes('-mmproj-');
}

export function derivePiModelIdFromPath(modelPath) {
  const base = path.basename(String(modelPath || ''));
  return base.replace(/\.gguf$/i, '');
}

export function formatPiModelBaseUrl(host = '127.0.0.1', port = 8080) {
  const rawHost = String(host || '127.0.0.1');
  const normalizedHost = rawHost === '0.0.0.0' || rawHost === '::' ? '127.0.0.1' : rawHost;
  const safeHost = normalizedHost.includes(':') && !normalizedHost.startsWith('[')
    ? `[${normalizedHost}]`
    : normalizedHost;
  return `http://${safeHost}:${Number(port) || 8080}/v1`;
}

export function findModelPathForPiModelId(modelId, modelPaths) {
  const id = String(modelId || '');
  if (!id || !Array.isArray(modelPaths)) return undefined;

  return modelPaths.find((modelPath) => {
    if (!isLikelyMainModelPath(modelPath)) return false;
    const base = path.basename(String(modelPath));
    return base === id || derivePiModelIdFromPath(modelPath) === id;
  });
}

export function ensureLlamaCppProvider(registry, options = {}) {
  const data = isObject(registry) ? registry : {};
  if (!isObject(data.providers)) data.providers = {};

  const existing = isObject(data.providers['llama-cpp']) ? data.providers['llama-cpp'] : {};
  const provider = {
    ...defaultLlamaCppProvider(options.baseUrl),
    ...existing,
    baseUrl: options.baseUrl || existing.baseUrl || 'http://127.0.0.1:8080/v1',
    compat: {
      ...DEFAULT_LLAMA_PROVIDER_COMPAT,
      ...(isObject(existing.compat) ? existing.compat : {}),
    },
    models: Array.isArray(existing.models) ? existing.models : [],
  };

  data.providers['llama-cpp'] = provider;
  return provider;
}

export function upsertLlamaModelEntry(registry, entry, options = {}) {
  const provider = ensureLlamaCppProvider(registry, options);
  const desired = buildLlamaModelEntry(entry);
  const index = provider.models.findIndex((model) => model?.id === desired.id);

  if (index < 0) {
    provider.models.push(desired);
    return { action: 'added', model: desired };
  }

  const existing = provider.models[index] ?? {};
  const merged = {
    ...existing,
    ...desired,
    name: typeof existing.name === 'string' && existing.name.trim() ? existing.name : desired.name,
    reasoning: typeof existing.reasoning === 'boolean' ? existing.reasoning : desired.reasoning,
    input: Array.isArray(existing.input) && existing.input.length > 0 ? existing.input : desired.input,
    cost: isObject(existing.cost) ? existing.cost : desired.cost,
    contextWindow: Number.isFinite(existing.contextWindow) && existing.contextWindow > 0
      ? existing.contextWindow
      : desired.contextWindow,
    maxTokens: Number.isFinite(existing.maxTokens) && existing.maxTokens > 0
      ? existing.maxTokens
      : desired.maxTokens,
  };
  provider.models[index] = merged;

  return {
    action: JSON.stringify(existing) === JSON.stringify(merged) ? 'skipped' : 'updated',
    model: merged,
  };
}

export function syncPiModelsRegistryData(registry, options = {}) {
  const data = isObject(registry) ? registry : {};
  const modelPaths = Array.isArray(options.modelPaths) ? options.modelPaths : [];
  const defaults = {
    contextWindow: Number(options.defaults?.contextWindow) || 32768,
    maxTokens: Number(options.defaults?.maxTokens) || 8192,
    reasoning: Boolean(options.defaults?.reasoning),
  };

  ensureLlamaCppProvider(data, { baseUrl: options.baseUrl });

  const counts = { added: 0, updated: 0, skipped: 0 };
  const syncedModels = [];
  const seenIds = new Set();

  for (const modelPath of modelPaths) {
    if (!isLikelyMainModelPath(modelPath)) {
      counts.skipped += 1;
      continue;
    }

    const id = derivePiModelIdFromPath(modelPath);
    if (!id || seenIds.has(id)) {
      counts.skipped += 1;
      continue;
    }
    seenIds.add(id);

    const result = upsertLlamaModelEntry(data, {
      id,
      name: buildPiFriendlyModelName(id),
      contextWindow: defaults.contextWindow,
      maxTokens: defaults.maxTokens,
      reasoning: defaults.reasoning,
    }, { baseUrl: options.baseUrl });

    counts[result.action] += 1;
    syncedModels.push({
      id: result.model.id,
      name: result.model.name,
      path: modelPath,
      action: result.action,
    });
  }

  return { registry: data, counts, syncedModels };
}

export async function loadPiModelsRegistry(registryPath) {
  try {
    const raw = await fs.readFile(registryPath, 'utf8');
    const parsed = JSON.parse(raw);
    return isObject(parsed) ? parsed : {};
  } catch (error) {
    if (error?.code === 'ENOENT') return {};
    if (error instanceof SyntaxError) {
      throw new Error(`Invalid Pi models registry JSON: ${registryPath}`);
    }
    throw error;
  }
}

export async function savePiModelsRegistry(registryPath, data) {
  const serialized = `${JSON.stringify(data, null, 2)}\n`;
  const directory = path.dirname(registryPath);
  const tempPath = path.join(directory, `.models.json.tmp-${process.pid}-${Date.now()}`);

  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(tempPath, serialized, 'utf8');
  await fs.rename(tempPath, registryPath);
}

export function modelMatchesRequest(runningModelPath, requestedModelPath) {
  if (!runningModelPath || !requestedModelPath) return false;
  if (runningModelPath === requestedModelPath) return true;

  const runningBase = String(runningModelPath).split(/[\\/]/).pop();
  const requestedBase = String(requestedModelPath).split(/[\\/]/).pop();
  return Boolean(runningBase && requestedBase && runningBase === requestedBase);
}

export function parsePsLlamaLines(output) {
  return String(output)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(\d+)\s+(.+)$/);
      if (!match) return null;
      const pid = Number.parseInt(match[1], 10);
      const command = match[2];
      return {
        pid,
        command,
        modelPath: extractModelPathFromCommand(command),
        port: extractPortFromCommand(command),
      };
    })
    .filter(Boolean);
}

export function normalizeDownloadUrl(url) {
  const trimmed = String(url ?? '').trim();
  if (!trimmed) return '';

  try {
    const parsed = new URL(trimmed);
    if (parsed.hostname === 'huggingface.co' && parsed.pathname.includes('/blob/')) {
      parsed.pathname = parsed.pathname.replace('/blob/', '/resolve/');
      parsed.search = '';
    }
    return parsed.toString();
  } catch {
    return trimmed;
  }
}

export function inferFilenameFromUrl(url) {
  try {
    const parsed = new URL(url);
    const base = parsed.pathname.split('/').pop() || '';
    return decodeURIComponent(base);
  } catch {
    return '';
  }
}

export function buildCurlDownloadCommand(url, targetPath) {
  return `curl -L --fail --progress-bar -C - -o ${shellQuote(targetPath)} ${shellQuote(url)}`;
}

export function parseContentLengthFromHeaders(headersText) {
  const lines = String(headersText || '').split(/\r?\n/);
  let contentLength = null;
  for (const line of lines) {
    const m = line.match(/^\s*content-length\s*:\s*(\d+)\s*$/i);
    if (!m) continue;
    const value = Number.parseInt(m[1], 10);
    if (Number.isFinite(value)) contentLength = value;
  }
  return contentLength;
}

export function formatBytes(bytes) {
  const n = Math.max(0, Number(bytes) || 0);
  if (n < 1024) return `${Math.round(n)} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = n / 1024;
  let unit = units[0];
  for (let i = 1; i < units.length && value >= 1024; i += 1) {
    value /= 1024;
    unit = units[i];
  }
  return `${value.toFixed(1)} ${unit}`;
}

export function formatPercent(done, total) {
  const d = Math.max(0, Number(done) || 0);
  const t = Number(total) || 0;
  if (!Number.isFinite(t) || t <= 0) return '--';
  const pct = Math.min(100, (d / t) * 100);
  return `${pct.toFixed(1)}%`;
}

function parseMiB(value) {
  const parsed = Number.parseFloat(String(value ?? '').replace(/,/g, ''));
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseInteger(value) {
  const parsed = Number.parseInt(String(value ?? '').replace(/,/g, ''), 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function assignIfNumber(target, key, value) {
  if (typeof value === 'number') target[key] = value;
}

export function parseLlamaServerLogMetrics(logText) {
  const text = String(logText || '');
  const metrics = {};

  for (const line of text.split(/\r?\n/)) {
    let match = line.match(/projected to use\s+([0-9.]+)\s+MiB of device memory vs\.\s+([0-9.]+)\s+MiB of free device memory/i);
    if (match) {
      assignIfNumber(metrics, 'deviceMemoryProjectedMiB', parseMiB(match[1]));
      assignIfNumber(metrics, 'deviceMemoryFreeMiB', parseMiB(match[2]));
      continue;
    }

    match = line.match(/context size reduced from\s+(\d+)\s+to\s+(\d+)/i);
    if (match) {
      assignIfNumber(metrics, 'contextSizeReducedFrom', parseInteger(match[1]));
      assignIfNumber(metrics, 'contextSizeReducedTo', parseInteger(match[2]));
      continue;
    }

    match = line.match(/CPU_Mapped model buffer size\s*=\s*([0-9.]+)\s+MiB/i);
    if (match) {
      assignIfNumber(metrics, 'cpuMappedModelMiB', parseMiB(match[1]));
      continue;
    }

    match = line.match(/MTL\d+_Mapped model buffer size\s*=\s*([0-9.]+)\s+MiB/i);
    if (match) {
      assignIfNumber(metrics, 'metalMappedModelMiB', parseMiB(match[1]));
      continue;
    }

    match = line.match(/llama_context:\s+n_ctx\s*=\s*(\d+)/i);
    if (match) {
      assignIfNumber(metrics, 'contextSize', parseInteger(match[1]));
      continue;
    }

    match = line.match(/llama_context:\s+n_batch\s*=\s*(\d+)/i);
    if (match) {
      assignIfNumber(metrics, 'batchSize', parseInteger(match[1]));
      continue;
    }

    match = line.match(/llama_context:\s+n_ubatch\s*=\s*(\d+)/i);
    if (match) {
      assignIfNumber(metrics, 'ubatchSize', parseInteger(match[1]));
      continue;
    }

    match = line.match(/MTL\d+ KV buffer size\s*=\s*([0-9.]+)\s+MiB/i);
    if (match) {
      assignIfNumber(metrics, 'metalKvCacheMiB', parseMiB(match[1]));
      continue;
    }

    match = line.match(/MTL\d+ compute buffer size\s*=\s*([0-9.]+)\s+MiB/i);
    if (match) {
      assignIfNumber(metrics, 'metalComputeBufferMiB', parseMiB(match[1]));
      continue;
    }

    match = line.match(/CPU compute buffer size\s*=\s*([0-9.]+)\s+MiB/i);
    if (match) {
      assignIfNumber(metrics, 'cpuComputeBufferMiB', parseMiB(match[1]));
      continue;
    }

    match = line.match(/n_slots\s*=\s*(\d+)/i);
    if (match) {
      assignIfNumber(metrics, 'slots', parseInteger(match[1]));
      continue;
    }

    match = line.match(/prompt cache is enabled, size limit:\s*([0-9.]+)\s+MiB/i);
    if (match) {
      assignIfNumber(metrics, 'promptCacheLimitMiB', parseMiB(match[1]));
      continue;
    }

    match = line.match(/server is listening on\s+(https?:\/\/\S+)/i);
    if (match) {
      metrics.listeningUrl = match[1];
    }
  }

  return metrics;
}

export function assessLaunchMetrics(metrics = {}, profileDefaults = {}, options = {}) {
  const warnings = [];
  const metalHeadroomRatio = Number(options.metalHeadroomRatio ?? 0.9);
  const largePromptCacheMiB = Number(options.largePromptCacheMiB ?? 4096);

  if (metrics.deviceMemoryProjectedMiB && metrics.deviceMemoryFreeMiB) {
    const ratio = metrics.deviceMemoryProjectedMiB / metrics.deviceMemoryFreeMiB;
    if (ratio >= metalHeadroomRatio) {
      warnings.push({
        level: 'warning',
        code: 'low-metal-headroom',
        message: `projected Metal memory uses ${(ratio * 100).toFixed(1)}% of reported free device memory`,
      });
    }
  }

  if (metrics.contextSize && profileDefaults.contextWindow && metrics.contextSize > profileDefaults.contextWindow) {
    warnings.push({
      level: 'warning',
      code: 'context-exceeds-profile',
      message: `running context ${metrics.contextSize} exceeds profile target ${profileDefaults.contextWindow}`,
    });
  }

  if (metrics.promptCacheLimitMiB && metrics.promptCacheLimitMiB >= largePromptCacheMiB) {
    warnings.push({
      level: 'warning',
      code: 'large-prompt-cache',
      message: `prompt cache limit is ${metrics.promptCacheLimitMiB} MiB`,
    });
  }

  if (metrics.contextSizeReducedFrom && metrics.contextSizeReducedTo) {
    warnings.push({
      level: 'info',
      code: 'context-auto-reduced',
      message: `llama.cpp auto-reduced context from ${metrics.contextSizeReducedFrom} to ${metrics.contextSizeReducedTo}`,
    });
  }

  return warnings;
}

export function createLaunchRecord(input = {}) {
  const modelPath = String(input.modelPath || '');
  return {
    startedAt: input.startedAt || new Date().toISOString(),
    finishedAt: input.finishedAt,
    profile: input.profile || 'unknown',
    modelPath,
    modelName: modelPath ? path.basename(modelPath) : 'unknown-model',
    args: Array.isArray(input.args) ? input.args.map((arg) => String(arg)) : [],
    pid: input.pid,
    logFile: input.logFile,
    outcome: input.outcome || 'unknown',
    message: input.message,
    metrics: isObject(input.metrics) ? input.metrics : {},
    warnings: Array.isArray(input.warnings) ? input.warnings : [],
  };
}

export function buildLlamaArgs(config, modelPath) {
  const args = [
    '-m',
    modelPath,
    '--host',
    String(config.host ?? '127.0.0.1'),
    '--port',
    String(config.port ?? 8080),
  ];

  const managedArgs = buildLlamaProfileArgs(config?.serverProfile);

  if (config?.stableToolCalling && config?.defaultArgs?.jinja) {
    managedArgs.push('--jinja');
  }

  if (config?.stableToolCalling && config?.defaultArgs?.reasoning) {
    managedArgs.push('--reasoning', String(config.defaultArgs.reasoning));
  }

  if (config?.stableToolCalling && config?.defaultArgs?.chatTemplateKwargs) {
    managedArgs.push('--chat-template-kwargs', JSON.stringify(config.defaultArgs.chatTemplateKwargs));
  }

  if (config?.stableToolCalling && typeof config?.defaultArgs?.temp === 'number') {
    managedArgs.push('--temp', String(config.defaultArgs.temp));
  }

  if (config?.stableToolCalling && typeof config?.defaultArgs?.topP === 'number') {
    managedArgs.push('--top-p', String(config.defaultArgs.topP));
  }

  args.push(...mergeProfileAndExtraArgs(managedArgs, config?.extraArgs));

  return args;
}
