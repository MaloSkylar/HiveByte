# HiveByte

HiveByte is a browser-based AI workspace at [https://hivebyte.net](https://hivebyte.net).

It gives you a clean chat interface for working with local AI models through Ollama, with optional local tools through MCP. The site is hosted publicly, but your model runs on your own machine.

## How It Works

- `hivebyte.net` serves the web app
- Ollama runs locally on your computer
- Optional MCP tools also run locally on your computer
- Your browser connects the site to your local services

This means you can use HiveByte from a public website while still keeping model inference and local tools on your own machine.

## Quick Start

### 1. Install Ollama

Download Ollama here:

[https://ollama.com/download](https://ollama.com/download)

Then install at least one model:

```bash
ollama pull llama3.2
```

### 2. Allow HiveByte To Connect To Ollama

HiveByte needs permission to reach your local Ollama server from `https://hivebyte.net`.

### Windows PowerShell

```powershell
$env:OLLAMA_ORIGINS = "https://hivebyte.net"
ollama serve
```

### macOS / Linux

```bash
OLLAMA_ORIGINS=https://hivebyte.net ollama serve
```

If Ollama was already running, restart it after setting the environment variable.

### 3. Open HiveByte

Go to:

[https://hivebyte.net](https://hivebyte.net)

If your browser asks for permission to connect to local devices or the local network, allow it.

### 4. Start Chatting

Once Ollama is running locally, you can start chatting in HiveByte right away.

## Optional: Local Tools With MCP

HiveByte can also connect to a local MCP server for tools such as file actions or other local capabilities.

If you want to use MCP tools, allow HiveByte in your local MCP server too.

### Windows PowerShell

```powershell
$env:MCP_ALLOWED_ORIGINS = "https://hivebyte.net"
python mcp_server.py
```

### macOS / Linux

```bash
MCP_ALLOWED_ORIGINS=https://hivebyte.net python mcp_server.py
```

If you are only testing chat with Ollama, you do not need MCP.

## Troubleshooting

### Could not reach `http://localhost:11434`

This usually means one of these:

- Ollama is not running
- Ollama was started without `OLLAMA_ORIGINS=https://hivebyte.net`
- Ollama needs to be restarted after setting the variable
- Your browser blocked the local connection request

### The page loads, but chat does not respond

Check that:

- Ollama is running
- You have at least one model installed
- The selected model exists locally

Example:

```bash
ollama pull llama3.2
```

### MCP tools are unavailable

Check that:

- Your local MCP server is running
- `MCP_ALLOWED_ORIGINS` includes `https://hivebyte.net`
- The browser was allowed to access local services

### It worked before, then stopped

Restart Ollama, reload the page, and try again.

## Notes

- HiveByte works best in modern browsers like Chrome, Edge, and Firefox
- Local browser permission prompts may appear the first time you connect
- Your local model and local tools are not hosted on HiveByte itself

## Feedback

If you test HiveByte, useful feedback includes:

- Operating system
- Browser
- Ollama model used
- Whether MCP was enabled
- Any confusing setup steps or errors
