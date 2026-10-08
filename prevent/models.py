"""Model adapters. Anything with complete(prompt) -> str works; these are the common doors.

Keys come from the environment only. Nothing here prints or stores a key.
"""
from __future__ import annotations

import json
import os
import urllib.request


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 120.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class OllamaModel:
    """A local model through Ollama. Ajax or any other model Hamdi installs is one `ollama pull` away."""

    def __init__(self, name: str = "qwen2.5:7b", host: str | None = None, temperature: float = 0.7):
        self.name = name
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        out = _post(f"{self.host}/api/generate", {"model": self.name, "prompt": prompt, "stream": False,
                                                 "options": {"temperature": self.temperature}})
        return str(out.get("response", ""))


class OpenAICompatibleModel:
    """Any /v1/chat/completions endpoint: OpenAI, Groq, OpenRouter, Cerebras, NVIDIA NIM, Alpha's router."""

    def __init__(self, model: str, base_url: str, key_env: str = "OPENAI_API_KEY", temperature: float = 0.7):
        self.model, self.base_url, self.key_env, self.temperature = model, base_url.rstrip("/"), key_env, temperature

    def complete(self, prompt: str) -> str:
        key = os.environ.get(self.key_env, "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        out = _post(f"{self.base_url}/chat/completions", {"model": self.model, "temperature": self.temperature,
                                                         "messages": [{"role": "user", "content": prompt}]}, headers)
        try:
            return str(out["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            return ""


class ScriptedModel:
    """Deterministic model for tests and dry runs: answers in order, then repeats the last answer."""

    def __init__(self, answers: list[str]):
        self.answers, self.i = list(answers), 0

    def complete(self, prompt: str) -> str:
        a = self.answers[min(self.i, len(self.answers) - 1)]
        self.i += 1
        return a


def from_spec(spec: str):
    """'ollama:qwen2.5:7b', 'openai:gpt-4o-mini', 'groq:llama-3.3-70b-versatile', 'url:MODEL@https://host/v1'."""
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaModel(rest or "qwen2.5:7b")
    if kind == "openai":
        return OpenAICompatibleModel(rest, "https://api.openai.com/v1", "OPENAI_API_KEY")
    if kind == "groq":
        return OpenAICompatibleModel(rest, "https://api.groq.com/openai/v1", "GROQ_API_KEY")
    if kind == "openrouter":
        return OpenAICompatibleModel(rest, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    if kind == "url":
        model, _, base = rest.partition("@")
        return OpenAICompatibleModel(model, base, os.environ.get("PREVENT_KEY_ENV", "OPENAI_API_KEY"))
    raise ValueError(f"unknown model spec: {spec}")
