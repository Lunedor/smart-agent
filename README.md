[![If you are a good person...](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/lunedor)
# Smart Agent

Smart Agent is a Windows PowerShell assistant that converts natural-language requests into a PowerShell command, returning a strict JSON response with `explanation` and `command`, and then asks for confirmation before running it.  
It supports multiple AI providers (cloud and local), persists configuration to a local JSON file, and includes simple “persistent memory” slash commands for facts you want the agent to always remember.

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
  - `agentconfig.json` (provider/model/api config)
  - `agentmemory.json` (persistent memory facts)

## Requirements

- Windows PowerShell available (the script runs commands via `powershell -NoProfile -Command ...`)
- Python 3.x
- Provider SDKs depending on what you choose:
  - Gemini: `google-genai`
  - OpenAI/OpenRouter/LM Studio: `openai`
  - Anthropic: `anthropic`

> Tip: If you only use one provider, you only need that provider’s library installed.

## Install

Clone/copy the script into a folder, then install the dependencies you need.

Example (OpenAI/OpenRouter/LM Studio):
```powershell
pip install openai
