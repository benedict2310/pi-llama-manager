export function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

export function parsePiLlamaModels(raw) {
  try {
    const parsed = JSON.parse(raw);
    const models = parsed?.providers?.["llama-cpp"]?.models;
    if (!Array.isArray(models)) return [];
    return models
      .map((model) => ({
        id: typeof model?.id === "string" ? model.id : "",
        name: typeof model?.name === "string" ? model.name : undefined,
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
  let current = "";
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
        current = "";
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
  return parseArgValue(parts, "-m") ?? parseArgValue(parts, "--model");
}

export function extractPortFromCommand(command, fallback = 8080) {
  const parts = tokenizeCommand(command);
  const value = parseArgValue(parts, "--port") ?? parseArgValue(parts, "-p");
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function isLikelyMainModelPath(modelPath) {
  const base = String(modelPath).split(/[\\/]/).pop()?.toLowerCase() || "";
  return base.endsWith(".gguf") && !base.startsWith("mmproj-") && !base.includes("-mmproj-");
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

export function buildLlamaArgs(config, modelPath) {
  const args = [
    "-m",
    modelPath,
    "--host",
    String(config.host ?? "127.0.0.1"),
    "--port",
    String(config.port ?? 8080),
  ];

  if (config?.stableToolCalling && config?.defaultArgs?.jinja) {
    args.push("--jinja");
  }

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

  if (Array.isArray(config?.extraArgs)) {
    args.push(...config.extraArgs.map((part) => String(part)));
  }

  return args;
}
