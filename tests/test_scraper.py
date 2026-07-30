"""Tests for the scraper module."""

import pytest

from src.scraper.xiaoyuzhou import parse_episode_id


class TestParseEpisodeId:
    def test_standard_url(self):
        eid = parse_episode_id("https://www.xiaoyuzhoufm.com/episode/60b030ef3104c523cd6d7eef")
        assert eid == "60b030ef3104c523cd6d7eef"

    def test_without_www(self):
        eid = parse_episode_id("https://xiaoyuzhoufm.com/episode/abc123")
        assert eid == "abc123"

    def test_invalid_url(self):
        with pytest.raises(ValueError):
            parse_episode_id("https://example.com/not-a-podcast")
