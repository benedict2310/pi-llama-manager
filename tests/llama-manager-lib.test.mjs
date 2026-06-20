import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import fs from "node:fs/promises";

import {
  buildCurlDownloadCommand,
  buildLlamaArgs,
  buildLlamaProfileArgs,
  derivePiModelIdFromPath,
  ensureLlamaCppProvider,
  findModelPathForPiModelId,
  formatPiModelBaseUrl,
  extractModelPathFromCommand,
  formatBytes,
  formatPercent,
  inferFilenameFromUrl,
  isLikelyMainModelPath,
  loadPiModelsRegistry,
  normalizeDownloadUrl,
  modelMatchesRequest,
  parseContentLengthFromHeaders,
  parseLlamaServerLogMetrics,
  parsePiLlamaModels,
  parsePsLlamaLines,
  savePiModelsRegistry,
  profileSyncDefaults,
  assessLaunchMetrics,
  createLaunchRecord,
  shellQuote,
  syncPiModelsRegistryData,
  upsertLlamaModelEntry,
} from "../lib/llama-manager-lib.js";

test("parsePiLlamaModels extracts llama-cpp models", () => {
  const raw = JSON.stringify({
    providers: {
      "llama-cpp": {
        models: [
          { id: "Qwen3.6-27B.gguf", name: "Qwen" },
          { id: "Gemma-4-31B.gguf", name: "Gemma" },
        ],
      },
      other: { models: [{ id: "ignore-me" }] },
    },
  });

  const models = parsePiLlamaModels(raw);
  assert.deepEqual(models, [
    { id: "Qwen3.6-27B.gguf", name: "Qwen" },
    { id: "Gemma-4-31B.gguf", name: "Gemma" },
  ]);
});

test("extractModelPathFromCommand extracts -m argument", () => {
  const cmd = "llama-server -m /home/tester/models/qwen.gguf --host 0.0.0.0 --port 8080";
  assert.equal(extractModelPathFromCommand(cmd), "/home/tester/models/qwen.gguf");
});

test("parsePsLlamaLines parses pid/command pairs", () => {
  const ps = "11481 llama-server -m /home/tester/models/qwen.gguf --host 0.0.0.0 --port 8080\n";
  const rows = parsePsLlamaLines(ps);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].pid, 11481);
  assert.equal(rows[0].modelPath, "/home/tester/models/qwen.gguf");
  assert.equal(rows[0].port, 8080);
});

test("buildLlamaArgs emits stable coding-profile flags", () => {
  const args = buildLlamaArgs(
    {
      host: "127.0.0.1",
      port: 8080,
      logFile: "/tmp/llama.log",
      serverProfile: "code",
      extraArgs: [],
      stableToolCalling: true,
      defaultArgs: {
        jinja: true,
        reasoning: "off",
        chatTemplateKwargs: { enable_thinking: false },
        temp: 0.2,
        topP: 0.9,
      },
    },
    "/home/tester/models/qwen.gguf",
  );

  assert.deepEqual(args, [
    "-m",
    "/home/tester/models/qwen.gguf",
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
    "--ctx-size",
    "16384",
    "--parallel",
    "1",
    "--batch-size",
    "512",
    "--ubatch-size",
    "128",
    "--flash-attn",
    "auto",
    "--no-context-shift",
    "--n-predict",
    "2048",
    "--jinja",
    "--reasoning",
    "off",
    "--chat-template-kwargs",
    '{"enable_thinking":false}',
    "--temp",
    "0.2",
    "--top-p",
    "0.9",
  ]);
});

test("buildLlamaArgs lets explicit extraArgs override profile flags without duplicates", () => {
  const args = buildLlamaArgs(
    {
      host: "127.0.0.1",
      port: 8080,
      serverProfile: "wide",
      extraArgs: ["--ctx-size", "8192", "--n-predict", "1024", "--cache-ram", "512"],
      stableToolCalling: true,
      defaultArgs: { jinja: true, reasoning: "off", chatTemplateKwargs: { enable_thinking: false }, temp: 0.2, topP: 0.9 },
    },
    "/home/tester/models/qwen.gguf",
  );

  assert.equal(args.filter((arg) => arg === "--ctx-size").length, 1);
  assert.equal(args[args.indexOf("--ctx-size") + 1], "8192");
  assert.equal(args.filter((arg) => arg === "--n-predict").length, 1);
  assert.equal(args[args.indexOf("--n-predict") + 1], "1024");
  assert.equal(args.filter((arg) => arg === "--cache-ram").length, 1);
  assert.equal(args[args.indexOf("--cache-ram") + 1], "512");
  assert(args.includes("--no-context-shift"));
});

test("buildLlamaArgs lets explicit extraArgs override stable tool defaults", () => {
  const args = buildLlamaArgs(
    {
      host: "127.0.0.1",
      port: 8080,
      serverProfile: "wide",
      extraArgs: ["--temp", "0.7", "--top-p", "0.5", "--reasoning", "auto"],
      stableToolCalling: true,
      defaultArgs: { jinja: true, reasoning: "off", chatTemplateKwargs: { enable_thinking: false }, temp: 0.2, topP: 0.9 },
    },
    "/home/tester/models/qwen.gguf",
  );

  assert.equal(args.filter((arg) => arg === "--temp").length, 1);
  assert.equal(args[args.indexOf("--temp") + 1], "0.7");
  assert.equal(args.filter((arg) => arg === "--top-p").length, 1);
  assert.equal(args[args.indexOf("--top-p") + 1], "0.5");
  assert.equal(args.filter((arg) => arg === "--reasoning").length, 1);
  assert.equal(args[args.indexOf("--reasoning") + 1], "auto");
});

test("profile helpers describe fast code deep and wide server profiles", () => {
  assert.deepEqual(buildLlamaProfileArgs("fast").slice(0, 2), ["--ctx-size", "8192"]);
  assert.deepEqual(buildLlamaProfileArgs("code").slice(0, 2), ["--ctx-size", "16384"]);
  assert.deepEqual(buildLlamaProfileArgs("deep").slice(0, 2), ["--ctx-size", "32768"]);
  assert.deepEqual(buildLlamaProfileArgs("wide").slice(0, 2), ["--ctx-size", "131072"]);
  assert(buildLlamaProfileArgs("wide").includes("--cache-ram"));
  assert.equal(buildLlamaProfileArgs("wide")[buildLlamaProfileArgs("wide").indexOf("--cache-ram") + 1], "1024");
  assert.deepEqual(profileSyncDefaults("code"), { contextWindow: 16384, maxTokens: 2048, reasoning: false });
  assert.deepEqual(profileSyncDefaults("wide"), { contextWindow: 131072, maxTokens: 4096, reasoning: false });
});

test("shellQuote escapes single quotes", () => {
  assert.equal(shellQuote("O'Reilly"), "'O'\\''Reilly'");
});

test("isLikelyMainModelPath filters mmproj files", () => {
  assert.equal(isLikelyMainModelPath("/home/tester/models/Qwen3.6-27B.gguf"), true);
  assert.equal(isLikelyMainModelPath("/home/tester/models/mmproj-Qwen3.6-27B-f16.gguf"), false);
});

test("normalizeDownloadUrl rewrites huggingface blob links", () => {
  assert.equal(
    normalizeDownloadUrl("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"),
    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
  );
  assert.equal(
    normalizeDownloadUrl("https://example.com/models/model-Q5.gguf"),
    "https://example.com/models/model-Q5.gguf",
  );
});

test("inferFilenameFromUrl parses normal and query URLs", () => {
  assert.equal(
    inferFilenameFromUrl("https://huggingface.co/foo/bar/resolve/main/model-Q5.gguf?download=true"),
    "model-Q5.gguf",
  );
  assert.equal(
    inferFilenameFromUrl("https://example.com/models/model-f16.gguf"),
    "model-f16.gguf",
  );
  assert.equal(inferFilenameFromUrl("https://example.com/models/"), "");
});

test("buildCurlDownloadCommand produces resumable curl command", () => {
  const cmd = buildCurlDownloadCommand(
    "https://example.com/models/model-Q5.gguf",
    "/home/tester/models/model-Q5.gguf",
  );
  assert.equal(
    cmd,
    "curl -L --fail --progress-bar -C - -o '/home/tester/models/model-Q5.gguf' 'https://example.com/models/model-Q5.gguf'",
  );
});

test("parseContentLengthFromHeaders handles common header formats", () => {
  const headers = [
    "HTTP/2 302",
    "content-length: 0",
    "",
    "HTTP/2 200",
    "content-length: 123456789",
    "",
  ].join("\n");

  assert.equal(parseContentLengthFromHeaders(headers), 123456789);
  assert.equal(parseContentLengthFromHeaders("HTTP/2 200\ncontent-type: application/octet-stream\n"), null);
});

test("formatBytes and formatPercent are user friendly", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(formatBytes(1024 * 1024), "1.0 MB");
  assert.equal(formatPercent(0, 100), "0.0%");
  assert.equal(formatPercent(50, 100), "50.0%");
  assert.equal(formatPercent(150, 100), "100.0%");
  assert.equal(formatPercent(0, 0), "--");
});

test("modelMatchesRequest matches absolute and basename forms", () => {
  assert.equal(
    modelMatchesRequest(
      "/home/tester/models/qwen3.6-27b-uncensored-hauhaucs-aggressive/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
      "/home/tester/models/qwen3.6-27b-uncensored-hauhaucs-aggressive/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
    ),
    true,
  );

  assert.equal(
    modelMatchesRequest(
      "/home/tester/models/qwen3.6-27b-uncensored-hauhaucs-aggressive/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
      "Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
    ),
    true,
  );

  assert.equal(
    modelMatchesRequest(
      "/home/tester/models/a.gguf",
      "/home/tester/models/b.gguf",
    ),
    false,
  );
});

test("derivePiModelIdFromPath strips the .gguf suffix", () => {
  assert.equal(
    derivePiModelIdFromPath("/Users/tester/.pi/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"),
    "Qwen3.6-35B-A3B-UD-Q8_K_XL",
  );
});

test("ensureLlamaCppProvider creates provider and preserves unrelated providers", () => {
  const registry = {
    providers: {
      anthropic: { baseUrl: "https://api.anthropic.com" },
    },
  };

  const provider = ensureLlamaCppProvider(registry, { baseUrl: "http://127.0.0.1:9090/v1" });

  assert.deepEqual(registry.providers.anthropic, { baseUrl: "https://api.anthropic.com" });
  assert.equal(provider.baseUrl, "http://127.0.0.1:9090/v1");
  assert.equal(provider.apiKey, "local");
  assert.equal(provider.api, "openai-completions");
  assert.deepEqual(provider.models, []);
});

test("upsertLlamaModelEntry is idempotent and preserves existing metadata", () => {
  const registry = {
    providers: {
      "llama-cpp": {
        baseUrl: "http://127.0.0.1:8080/v1",
        apiKey: "local",
        api: "openai-completions",
        compat: { supportsDeveloperRole: false },
        models: [
          {
            id: "Qwen3.6-35B-A3B-UD-Q8_K_XL",
            name: "Curated Qwen Name",
            reasoning: true,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 65536,
            maxTokens: 16384,
            customField: "keep-me",
          },
        ],
      },
    },
  };

  const first = upsertLlamaModelEntry(registry, {
    id: "Qwen3.6-35B-A3B-UD-Q8_K_XL",
    name: "Qwen 3.6 35B A3B UD Q8_K_XL (llama.cpp)",
    contextWindow: 32768,
    maxTokens: 8192,
    reasoning: false,
  });
  const second = upsertLlamaModelEntry(registry, {
    id: "Qwen3.6-35B-A3B-UD-Q8_K_XL",
    name: "Qwen 3.6 35B A3B UD Q8_K_XL (llama.cpp)",
    contextWindow: 32768,
    maxTokens: 8192,
    reasoning: false,
  });

  assert.equal(first.action, "skipped");
  assert.equal(second.action, "skipped");
  assert.equal(registry.providers["llama-cpp"].models[0].name, "Curated Qwen Name");
  assert.equal(registry.providers["llama-cpp"].models[0].reasoning, true);
  assert.equal(registry.providers["llama-cpp"].models[0].contextWindow, 65536);
  assert.equal(registry.providers["llama-cpp"].models[0].maxTokens, 16384);
  assert.equal(registry.providers["llama-cpp"].models[0].customField, "keep-me");
});


test("formatPiModelBaseUrl handles wildcard and IPv6 hosts", () => {
  assert.equal(formatPiModelBaseUrl("0.0.0.0", 8080), "http://127.0.0.1:8080/v1");
  assert.equal(formatPiModelBaseUrl("::1", 8081), "http://[::1]:8081/v1");
});

test("findModelPathForPiModelId maps Pi picker ids back to main GGUF files", () => {
  const paths = [
    "/models/google_gemma-4-31B-it-Q5_K_M.gguf",
    "/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
    "/models/mmproj-Qwen3.6-35B-A3B-f16.gguf",
  ];

  assert.equal(
    findModelPathForPiModelId("Qwen3.6-35B-A3B-UD-Q8_K_XL", paths),
    "/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
  );
  assert.equal(
    findModelPathForPiModelId("google_gemma-4-31B-it-Q5_K_M.gguf", paths),
    "/models/google_gemma-4-31B-it-Q5_K_M.gguf",
  );
  assert.equal(findModelPathForPiModelId("mmproj-Qwen3.6-35B-A3B-f16", paths), undefined);
});

test("syncPiModelsRegistryData ignores mmproj files and reports add/update/skip counts", () => {
  const registry = {
    providers: {
      "llama-cpp": {
        baseUrl: "http://127.0.0.1:8080/v1",
        apiKey: "local",
        api: "openai-completions",
        compat: { supportsDeveloperRole: false },
        models: [
          {
            id: "existing-model",
            name: "Existing Model (curated)",
            reasoning: true,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 8192,
            maxTokens: 2048,
          },
        ],
      },
    },
  };

  const result = syncPiModelsRegistryData(registry, {
    modelPaths: [
      "/models/existing-model.gguf",
      "/models/new-model.gguf",
      "/models/mmproj-new-model.gguf",
      "/models/some-mmproj-helper.gguf",
    ],
    baseUrl: "http://127.0.0.1:8080/v1",
    defaults: { contextWindow: 4096, maxTokens: 1024, reasoning: false },
  });

  assert.deepEqual(result.counts, { added: 1, updated: 0, skipped: 3 });
  assert.deepEqual(
    result.syncedModels.map((model) => model.id),
    ["existing-model", "new-model"],
  );
  assert.equal(result.registry.providers["llama-cpp"].models.some((model) => model.id === "new-model"), true);
  assert.equal(result.registry.providers["llama-cpp"].models[0].name, "Existing Model (curated)");
  assert.equal(result.registry.providers["llama-cpp"].models[0].contextWindow, 8192);
});

test("savePiModelsRegistry and loadPiModelsRegistry round-trip data", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-llama-manager-"));
  const registryPath = path.join(tempDir, "models.json");
  const original = {
    providers: {
      "llama-cpp": {
        baseUrl: "http://127.0.0.1:8080/v1",
        apiKey: "local",
        api: "openai-completions",
        compat: { supportsDeveloperRole: false },
        models: [],
      },
    },
  };

  await savePiModelsRegistry(registryPath, original);
  const loaded = await loadPiModelsRegistry(registryPath);

  assert.deepEqual(loaded, original);
});

test("parseLlamaServerLogMetrics extracts Apple Metal memory and context metrics", () => {
  const metrics = parseLlamaServerLogMetrics(`
llama_params_fit_impl: projected to use 36845 MiB of device memory vs. 37739 MiB of free device memory
llama_params_fit_impl: context size reduced from 262144 to 260096 -> need 130 MiB less memory in total
load_tensors:   CPU_Mapped model buffer size =   833.59 MiB
load_tensors:  MTL0_Mapped model buffer size = 19838.29 MiB
llama_context: n_ctx         = 260096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_kv_cache:       MTL0 KV buffer size = 16256.00 MiB
sched_reserve:       MTL0 compute buffer size =   852.49 MiB
sched_reserve:        CPU compute buffer size =   528.02 MiB
srv    load_model: initializing slots, n_slots = 4
srv    load_model: prompt cache is enabled, size limit: 8192 MiB
main: server is listening on http://127.0.0.1:8080
`);

  assert.equal(metrics.contextSize, 260096);
  assert.equal(metrics.batchSize, 2048);
  assert.equal(metrics.ubatchSize, 512);
  assert.equal(metrics.slots, 4);
  assert.equal(metrics.deviceMemoryProjectedMiB, 36845);
  assert.equal(metrics.deviceMemoryFreeMiB, 37739);
  assert.equal(metrics.contextSizeReducedFrom, 262144);
  assert.equal(metrics.contextSizeReducedTo, 260096);
  assert.equal(metrics.cpuMappedModelMiB, 833.59);
  assert.equal(metrics.metalMappedModelMiB, 19838.29);
  assert.equal(metrics.metalKvCacheMiB, 16256);
  assert.equal(metrics.metalComputeBufferMiB, 852.49);
  assert.equal(metrics.cpuComputeBufferMiB, 528.02);
  assert.equal(metrics.promptCacheLimitMiB, 8192);
  assert.equal(metrics.listeningUrl, "http://127.0.0.1:8080");
});

test("assessLaunchMetrics flags tight headroom without requiring conservative profile", () => {
  const warnings = assessLaunchMetrics(
    {
      contextSize: 260096,
      deviceMemoryProjectedMiB: 36845,
      deviceMemoryFreeMiB: 37739,
      promptCacheLimitMiB: 8192,
      contextSizeReducedFrom: 262144,
      contextSizeReducedTo: 260096,
    },
    { contextWindow: 32768, maxTokens: 4096, reasoning: false },
  );

  assert.deepEqual(
    warnings.map((warning) => warning.code),
    ["low-metal-headroom", "context-exceeds-profile", "large-prompt-cache", "context-auto-reduced"],
  );
  assert.equal(warnings[0].level, "warning");
});

test("createLaunchRecord captures exact args and parsed metrics", () => {
  const record = createLaunchRecord({
    startedAt: "2026-06-20T07:45:00.000Z",
    profile: "deep",
    modelPath: "/home/tester/.pi/models/qwen.gguf",
    args: ["-m", "/home/tester/.pi/models/qwen.gguf", "--ctx-size", "32768"],
    pid: 1234,
    logFile: "/home/tester/.pi/agent/logs/llama-server-20260620-074500.log",
    outcome: "started",
    metrics: { contextSize: 32768, deviceMemoryProjectedMiB: 24000, deviceMemoryFreeMiB: 37739 },
    warnings: [],
  });

  assert.equal(record.modelName, "qwen.gguf");
  assert.equal(record.profile, "deep");
  assert.deepEqual(record.args, ["-m", "/home/tester/.pi/models/qwen.gguf", "--ctx-size", "32768"]);
  assert.equal(record.metrics.contextSize, 32768);
});
