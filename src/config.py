import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_BASE_URL = "https://voltapi.ai/v1"
DEFAULT_LLM_PROVIDER = "qwen"
LLM_MODELS = {
    "qwen": "qwen3.7-plus",
    "deepseek": "deepseek-v4-flash",
}


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    provider: str
    model: str


def get_xunfei_config() -> dict:
    return {
        "app_id": os.getenv("XUNFEI_APP_ID", ""),
        "api_key": os.getenv("XUNFEI_API_KEY", ""),
        "api_secret": os.getenv("XUNFEI_API_SECRET", ""),
    }


def get_llm_config(
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
) -> LLMConfig:
    if provider not in LLM_MODELS:
        supported = ", ".join(sorted(LLM_MODELS))
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose from: {supported}")
    return LLMConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/"),
        provider=provider,
        model=model or LLM_MODELS[provider],
    )


def check_config() -> list[str]:
    missing = []
    if not get_xunfei_config()["app_id"]:
        missing.append("XUNFEI_APP_ID")
    if not get_xunfei_config()["api_key"]:
        missing.append("XUNFEI_API_KEY")
    if not get_xunfei_config()["api_secret"]:
        missing.append("XUNFEI_API_SECRET")
    if not os.getenv("LLM_API_KEY"):
        missing.append("LLM_API_KEY")
    return missing
