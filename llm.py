"""The "policy": anything that maps a prompt to a completion.

Two backends:
  - openai:    any OpenAI-compatible server, e.g. vLLM serving Qwen3
               (`vllm serve Qwen/Qwen3-8B --port 8000`)
  - anthropic: the Claude API (reads ANTHROPIC_API_KEY or an `ant auth login` profile)
"""
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are an expert in computational geometry and numerical optimization. "
    "You improve Python programs through precise, minimal code edits."
)

# Models that accept the server-side refusal fallback ("default" form).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAICompatPolicy:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 temperature: float = 0.8, max_tokens: int = 8192):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def generate(self, prompt: str) -> Completion:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        u = r.usage
        return Completion(r.choices[0].message.content or "",
                          u.prompt_tokens if u else 0, u.completion_tokens if u else 0)


class AnthropicPolicy:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def generate(self, prompt: str) -> Completion:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if self.model in _FALLBACK_MODELS:
            r = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            r = self.client.messages.create(**kwargs)
        if r.stop_reason == "refusal":
            return Completion("", r.usage.input_tokens, r.usage.output_tokens)
        text = "".join(b.text for b in r.content if b.type == "text")
        return Completion(text, r.usage.input_tokens, r.usage.output_tokens)


def make_policy(backend: str, model: str | None = None, base_url: str | None = None):
    if backend == "anthropic":
        return AnthropicPolicy(model or "claude-opus-5-5")
    if backend == "openai":
        return OpenAICompatPolicy(model or "Qwen/Qwen3-8B", base_url or "http://localhost:8000/v1")
    raise ValueError(f"unknown backend {backend!r}")
