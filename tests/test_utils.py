"""Tests for the cost tracker utility."""

from src.utils import CostTracker, safe_filename


class TestCostTracker:
    def test_empty(self):
        t = CostTracker()
        assert t.total_yuan == 0.0

    def test_transcription_cost(self):
        t = CostTracker()
        t.add_transcription(0.66)
        assert t.transcription_yuan == 0.66
        assert t.total_yuan == 0.66

    def test_llm_cost(self):
        t = CostTracker()
        t.add_llm_usage(20000, 30000)
        assert t.llm_input_tokens == 20000
        assert t.llm_output_tokens == 30000
        assert not t.llm_cost_known
        assert "供应商账单" in t.summary()

    def test_total(self):
        t = CostTracker()
        t.add_transcription(0.66)
        t.add_llm_usage(20000, 30000, cost_yuan=0.07)
        assert abs(t.total_yuan - 0.73) < 0.001

    def test_safe_filename(self):
        assert safe_filename('节目: "AI/未来"?') == "节目_ _AI_未来__"
