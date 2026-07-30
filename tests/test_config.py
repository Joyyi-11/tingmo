"""Tests for provider-neutral LLM configuration."""

import pytest

from src.config import get_llm_config


def test_qwen_defaults(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
    config = get_llm_config("qwen")
    assert config.api_key == "test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "qwen3.7-plus"


def test_deepseek_defaults(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert get_llm_config("deepseek").model == "deepseek-v4-flash"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_config("unknown")
