"""
Shared LLM-as-judge used by every evaluation metric.

Judge model : Groq, set with JUDGE_MODEL (default openai/gpt-oss-20b)

Every verdict is cached in evaluation/judge_cache/<model>.jsonl, keyed by the
exact prompt. If a run stops (rate limit, daily token cap, Ctrl+C), just run
the same command again: cached prompts are skipped and only new ones hit the API.
"""
import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "judge_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


class JudgeStopped(SystemExit):
    """Stops the whole run (not just one row). Subclasses SystemExit so the
    metric scripts' `except Exception` blocks can't turn it into a 0 score."""


class _Response:
    def __init__(self, content):
        self.content = content


class Judge:
    def __init__(self, model):
        self.model = model
        self.llm = ChatGroq(model=model, temperature=0, max_retries=0, reasoning_effort="low")
        self.cache_path = os.path.join(_CACHE_DIR, model.replace("/", "_") + ".jsonl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["content"]

    def invoke(self, prompt):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key in self.cache:
            return _Response(self.cache[key])

        for attempt in range(8):
            try:
                content = self.llm.invoke(prompt).content.strip()
                break
            except Exception as e:
                msg = str(e)
                if "429" not in msg and "rate" not in msg.lower():
                    if attempt == 7:
                        print(f"\n*** Judge error: {msg[:300]} ***")
                        raise JudgeStopped(1)
                    time.sleep(5)
                    continue
                if "per day" in msg.lower() or "TPD" in msg or "RPD" in msg:
                    print("\n*** Groq daily limit reached. Progress is saved — run the same command again later. ***")
                    raise JudgeStopped(1)
                wait = re.search(r"try again in ([\d.]+)(m?s)", msg)
                secs = float(wait.group(1)) / (1000 if wait and wait.group(2) == "ms" else 1) if wait else 20
                time.sleep(min(secs + 1, 120))
        else:
            print("\n*** Judge kept hitting rate limits. Progress is saved — run again. ***")
            raise JudgeStopped(1)

        self.cache[key] = content
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "content": content}) + "\n")
        return _Response(content)


llm = Judge(JUDGE_MODEL)
