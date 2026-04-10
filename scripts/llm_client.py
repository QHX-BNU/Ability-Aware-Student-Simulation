import openai
from openai import OpenAI
# from config import OPENAI_API_KEY, SIM_PARAMS,BASE_URL

import time
import random
from typing import List

class LLMClient:
    """
    LLM API wrapper module.
    Provides a unified interface for calling models such as GPT / DeepSeek.
    """

    def __init__(self, base_url=None, api_key=None, model=None,
                 max_retries=5, base_delay=1.0):
        openai.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        self.max_retries = max_retries
        self.base_delay = base_delay

    def _sleep_with_backoff(self, retry_idx: int):
        """
        Exponential backoff + jitter.
        """
        delay = self.base_delay * (2 ** retry_idx)
        delay += random.uniform(0, 0.3)
        time.sleep(delay)

    def call(self, messages: List[dict]) -> str:
        """
        Call the model for chat completion (with retries).
        """
        last_exception = None

        for retry in range(self.max_retries):
            try:
                print(f"Attempt {retry + 1}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()

                # Retry only for recoverable errors
                if any(k in err_msg for k in [
                    "rate limit",
                    "429",
                    "timeout",
                    "connection",
                    "network",
                    "temporarily unavailable"
                ]):
                    if retry < self.max_retries - 1:
                        self._sleep_with_backoff(retry)
                        continue
                break  # Non-recoverable error: exit retry loop

        # ---------- fallback: streaming ----------
        for retry in range(self.max_retries):
            try:
                print(f"Streaming attempt {retry + 1}")
                full_text = ""
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body={"enable_thinking": False},
                    stream=True,
                    temperature=0,
                    max_tokens=512
                )

                for chunk in completion:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        full_text += delta.content

                if full_text.strip():
                    return full_text.strip()

            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()

                if any(k in err_msg for k in [
                    "rate limit",
                    "429",
                    "timeout",
                    "connection",
                    "network"
                ]):
                    if retry < self.max_retries - 1:
                        self._sleep_with_backoff(retry)
                        continue
                break

        return f"ERROR after {self.max_retries} retries: {last_exception}"


