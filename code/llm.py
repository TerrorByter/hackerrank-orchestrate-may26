"""
LLM wrapper — thin OpenAI-compatible client for APIYI (or any provider).

Handles structured JSON output, temperature control, and error retry.
"""

import os
import json
import time
from typing import Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Thin wrapper around OpenAI-compatible API.
    Works with APIYI, OpenRouter, or native OpenAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
    ):
        self.api_key = api_key or os.getenv("APIYI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.base_url = base_url or os.getenv("APIYI_BASE_URL", "https://api.apiyi.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "No API key found. Set APIYI_API_KEY or OPENAI_API_KEY in .env"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        retries: int = 3,
    ) -> str:
        """
        Generate a completion using the LLM.

        Args:
            system_prompt: System message with instructions
            user_prompt: User message with the actual query
            max_tokens: Maximum tokens in the response
            retries: Number of retry attempts on failure

        Returns:
            The model's response text
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    seed=42,  # Determinism
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"  LLM call failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """
        Generate a structured JSON response.
        Parses the response and extracts JSON, handling markdown code fences.

        Returns:
            Parsed JSON as a dictionary
        """
        raw = self.generate(system_prompt, user_prompt, max_tokens)

        # Try to extract JSON from the response
        return self._parse_json_response(raw)

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        """
        Extract JSON from a model response, handling common formats:
        - Pure JSON
        - JSON wrapped in ```json ... ``` code fences
        - JSON embedded in other text
        """
        text = text.strip()

        # Try parsing as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code fences
        import re
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding the first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: return an error dict
        print(f"  Warning: Could not parse JSON from LLM response")
        return {
            "status": "escalated",
            "product_area": "unknown",
            "response": "Unable to process this ticket. Escalating to a human agent.",
            "justification": "LLM response was not valid JSON; escalating as a safety measure.",
            "request_type": "product_issue",
        }
