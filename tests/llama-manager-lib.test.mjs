import test from "node:test";
import assert from "node:assert/strict";

import {
  buildCurlDownloadCommand,
  buildLlamaArgs,
  extractModelPathFromCommand,
  inferFilenameFromUrl,
  isLikelyMainModelPath,
  normalizeDownloadUrl,
  modelMatchesRequest,
  parsePiLlamaModels,
  parsePsLlamaLines,
  shellQuote,
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

test("buildLlamaArgs emits stable tool-calling flags", () => {
  const args = buildLlamaArgs(
    {
      host: "0.0.0.0",
      port: 8080,
      logFile: "/tmp/llama.log",
      extraArgs: ["--ctx-size", "32768"],
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
    "0.0.0.0",
    "--port",
    "8080",
    "--jinja",
    "--reasoning",
    "off",
    "--chat-template-kwargs",
    '{"enable_thinking":false}',
    "--temp",
    "0.2",
    "--top-p",
    "0.9",
    "--ctx-size",
    "32768",
  ]);
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
