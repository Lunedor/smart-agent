[![Buy me a coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/lunedor)

# Smart Agent

Smart Agent is a Windows PowerShell agent CLI that turns natural-language requests into safe-to-review PowerShell commands.
It outputs a strict JSON object with `explanation` and `command`, then asks for confirmation before executing the command.

It supports multiple AI providers (cloud and local), persists configuration to a local JSON file, and includes simple persistent-memory slash commands.

## Why this exists

If you often think “I know what I want to do, but I don’t want to look up the exact PowerShell syntax,” Smart Agent helps you:

- Describe the task in plain English
- Get a concrete PowerShell command + a short explanation
- Review it
- Execute only if you confirm

## Features

- Natural language → PowerShell command generation (JSON-only output: `explanation`, `command`)
- Confirmation prompt before execution
- Interactive setup wizard that saves config locally
- Multi-provider support:
  - Google Gemini
  - OpenAI
  - Anthropic Claude
  - OpenRouter
  - LM Studio (local OpenAI-compatible server)
- Persistent “facts” memory:
  - `remember <fact>`
  - `forget <fact>`
  - `mem` (show stored facts)
- Stores files next to the script:
  - `agentconfig.json` (provider/model/API config)
  - `agentmemory.json` (persistent memory facts)

## Requirements

- Windows PowerShell available (Smart Agent executes via `powershell -NoProfile -Command ...`)
- Python 3.x
- Provider SDKs depending on what you choose:
  - Gemini: `google-genai`
  - OpenAI / OpenRouter / LM Studio: `openai`
  - Anthropic: `anthropic`

> Tip: If you only use one provider, you only need that provider’s library installed.

## Install

Clone this repository (or copy `smart-agent.py` into a folder), then install the dependency for the provider you’ll use.

OpenAI / OpenRouter / LM Studio:

```powershell
pip install openai
```

Gemini:

```powershell
pip install google-genai
```

Anthropic:

```powershell
pip install anthropic
```

## First-time setup

Run once with `--config` (or run normally if no config exists). You’ll be guided through:

- Provider selection
- API key validation
- Model selection (where supported)
- LM Studio base URL (for local server)

This creates/updates `agentconfig.json` next to the script.

## Usage

### One-shot mode

```powershell
python .\smart-agent.py "List running processes and sort by CPU"
```

### Interactive mode (REPL)

```powershell
python .\smart-agent.py
```

In interactive mode:

- Type `exit` or `quit` to leave
- Use the slash commands below to manage persistent memory

## Slash commands (persistent memory)

These are handled locally and stored in `agentmemory.json`.

- `remember <fact>`
- `forget <fact>`
- `mem`

Examples:

```text
remember I prefer using ripgrep (rg) for searching
mem
forget I prefer using ripgrep (rg) for searching
```

## Safety notes

- Always review the generated command before executing.
- The tool runs commands with the same permissions as your current PowerShell session.
- Treat API keys as secrets; do not commit `agentconfig.json` to a public repo.

## Suggested .gitignore

```gitignore
agentconfig.json
agentmemory.json
__pycache__/
```

## License

MIT License — see [LICENSE](LICENSE).
