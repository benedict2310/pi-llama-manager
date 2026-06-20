# pi-llama-manager

Pi extension for managing a local `llama-server` process and selecting GGUF models.

## Features

- `/llama` interactive menu
- `/llama status|doctor|profile|start|stop|restart|sync-models` command variants
- Model picker from local GGUF files
- Automatic sync of started/downloaded GGUF models into `~/.pi/agent/models.json`
- Optional `llama-server` switch when a `llama-cpp` model is selected in Pi's model picker
- Optional auto-select of the started llama.cpp model in the current Pi session
- Download additional models directly from URL (resume support, background)
- Live download progress in Pi status widget
- Abort running downloads from menu or command
- Prevents accidental second `llama-server` instance
- Filters out `mmproj-*.gguf` projector files from model list
- Timestamped per-launch logs plus JSONL launch history
- `/llama doctor` parses startup memory/context metrics and flags tight Metal headroom
- Stable tool-calling defaults for Qwen/llama.cpp workflows

## Screenshots

![screen1](./assets/screen1.png)

![screen2](./assets/screen2.png)

## Requirements

- Pi coding agent with extension support
- `llama-server` (from `llama.cpp`) available in `PATH`
- At least one local GGUF model file
- A writable `~/.pi/agent/` directory

## Install

### From GitHub

```bash
pi install https://github.com/benedict2310/pi-llama-manager
```

### Or via settings

```json
{
  "packages": [
    "https://github.com/benedict2310/pi-llama-manager"
  ]
}
```

## Usage

```text
/llama
/llama status
/llama doctor
/llama profile fast|code|deep|wide
/llama start
/llama stop
/llama restart
/llama sync-models
/llama start /absolute/path/to/model.gguf
/llama download <url-to-model.gguf> [destination-dir]
/llama download abort
/llama download-abort
```

After `/llama start ...` or a successful download, the extension syncs the main GGUF model into Pi's model registry at `~/.pi/agent/models.json` (unless disabled). Use `/reload` if your current Pi UI does not refresh the model picker immediately.

When `autoSwitchOnModelSelect` is enabled and you select a `llama-cpp` entry in Pi's model picker, the extension scans `modelsRoots`, maps the selected model id back to the matching main `.gguf`, and starts/restarts `llama-server` with that file.

## Configuration

On first use, the extension creates:

- `~/.pi/agent/llama-manager.json`

Example:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "modelsRoots": ["~/.pi/models"],
  "defaultModelPath": "",
  "downloadDir": "~/.pi/models",
  "logFile": "/Users/you/.pi/agent/llama-server.log",
  "launchHistoryFile": "/Users/you/.pi/agent/llama-launch-history.jsonl",
  "stableToolCalling": true,
  "autoSyncPiModels": true,
  "autoSelectAfterStart": false,
  "autoSwitchOnModelSelect": false,
  "serverProfile": "code",
  "syncDefaults": {
    "contextWindow": 16384,
    "maxTokens": 2048,
    "reasoning": false
  },
  "defaultArgs": {
    "jinja": true,
    "reasoning": "off",
    "chatTemplateKwargs": { "enable_thinking": false },
    "temp": 0.2,
    "topP": 0.9
  },
  "extraArgs": []
}
```

Built-in server profiles:

- `fast`: `--ctx-size 8192 --parallel 1 --batch-size 512 --ubatch-size 128 --flash-attn auto --no-context-shift --n-predict 2048`
- `code`: `--ctx-size 16384 --parallel 1 --batch-size 512 --ubatch-size 128 --flash-attn auto --no-context-shift --n-predict 2048`
- `deep`: `--ctx-size 32768 --parallel 1 --batch-size 512 --ubatch-size 128 --flash-attn auto --no-context-shift --n-predict 4096`
- `wide`: `--ctx-size 131072 --parallel 1 --batch-size 512 --ubatch-size 128 --cache-ram 1024 --flash-attn auto --no-context-shift --n-predict 4096`

Use `/llama profile fast|code|deep|wide` to change the profile for the next manual start/restart. The extension does not restart the server when changing profiles. `wide` is the current high-performance target from local benchmarking: 128k context with prompt cache capped at 1GiB.

`/llama doctor` compares the configured profile with the already-running server, parses the latest launch log, reports context/memory metrics, and warns about tight Metal memory headroom, unexpectedly large context, large prompt cache, or llama.cpp context auto-reduction. It never starts or restarts `llama-server`.

Each `/llama start` writes a timestamped log under `~/.pi/agent/llama-server-runs/`, updates `logFile` as the current-log symlink, and appends a launch record to `launchHistoryFile` with exact args, profile, model, outcome, parsed metrics, and warnings.

## Test

```bash
npm test
```

## Notes

- If the requested model is already running, start returns success/info instead of error and still attempts Pi registry sync.
- `/llama sync-models` scans `modelsRoots` and reports added/updated/skipped counts.
- Selecting a `llama-cpp` model in Pi's picker requires that a matching GGUF exists under `modelsRoots`.
- Existing non-llama providers in `~/.pi/agent/models.json` are preserved.
- `mmproj-*.gguf` projector files are ignored during sync.
- If a different model is running, use restart to switch models.
