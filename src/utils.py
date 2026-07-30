"""Cost tracking for API calls."""

import time
from dataclasses import dataclass


@dataclass
class CostTracker:
    transcription_yuan: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_yuan: float = 0.0
    llm_cost_known: bool = False

    def add_transcription(self, cost: float) -> None:
        self.transcription_yuan += cost

    def add_llm_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_yuan: float | None = None,
    ) -> None:
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        if cost_yuan is not None:
            self.llm_cost_yuan += cost_yuan
            self.llm_cost_known = True

    @property
    def total_yuan(self) -> float:
        return self.transcription_yuan + self.llm_cost_yuan

    def summary(self) -> str:
        llm_cost = f"{self.llm_cost_yuan:.4f} 元" if self.llm_cost_known else "以供应商账单为准"
        return (
            f"转录: {self.transcription_yuan:.4f} 元, "
            f"LLM: {llm_cost} "
            f"(输入 {self.llm_input_tokens} / 输出 {self.llm_output_tokens} tokens), "
            f"已知费用: {self.total_yuan:.4f} 元"
        )


def safe_filename(value: str, max_length: int = 80) -> str:
    """Return a Windows-safe filename stem."""
    translation = str.maketrans({char: "_" for char in '<>:"/\\|?*'})
    cleaned = value.translate(translation).strip().rstrip(".")
    return cleaned[:max_length] or "podcast"


class Timer:
    """Simple timer context manager."""
    def __init__(self):
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}分{s}秒"
