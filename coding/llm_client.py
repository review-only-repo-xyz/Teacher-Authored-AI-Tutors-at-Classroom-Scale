"""
Minimal provider-agnostic JSON completion helper for the coding scripts.

Model names starting with "claude" are sent to the Anthropic API
(ANTHROPIC_API_KEY); any other model name is sent to the OpenAI API
(OPENAI_API_KEY). Both are called at temperature 0 and expected to return a
single JSON object.
"""
import json
import os
import re


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1] if start >= 0 else text)


class JsonLLM:
    def __init__(self, model):
        _load_env()
        self.model = model
        if model.lower().startswith("claude"):
            import anthropic
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic()
            self._provider = "anthropic"
        else:
            import openai
            if not os.getenv("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY not set.")
            self._client = openai.OpenAI()
            self._provider = "openai"

    def complete_json(self, system, user, max_tokens=4096):
        if self._provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=0,
                system=system, messages=[{"role": "user", "content": user}])
            return _extract_json("".join(b.text for b in resp.content if getattr(b, "type", "") == "text"))
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return json.loads(resp.choices[0].message.content)
