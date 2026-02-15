# Setting Up Ollama

This guide helps you install and run Ollama locally so `edyant` benchmarks can connect to a local model endpoint.

## Install

Download and install Ollama for your OS:
- [ollama.com](https://ollama.com)

Follow the installer prompts for your platform.

## Start the server

After installation, start the Ollama server:

```bash
ollama serve
```

By default, Ollama listens on `http://localhost:11434`.

## Pull a model

Pull a model you want to benchmark:

```bash
ollama pull qwen2.5:3b
```

## Quick sanity check

Run a quick prompt to confirm Ollama is working:

```bash
ollama run qwen2.5:3b "Say hello in one sentence."
```

If this returns a response, Ollama is ready for the benchmark runner.
