"""讯飞录音文件识别 API 实现.

Pricing: 0.33 元/小时 (按小时计费，不足1小时按1小时算)
Docs: https://www.xfyun.cn/doc/asr/audio-revision/API.html
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path

import requests

from src.models.schemas import TranscriptResult
from src.transcriber.base import Transcriber

logger = logging.getLogger(__name__)

UPLOAD_URL = "https://raasr.xfyun.cn/v2/api/upload"
RESULT_URL = "https://raasr.xfyun.cn/v2/api/getResult"
POLL_INTERVAL = 3  #秒
MAX_POLL_SEC = 600  # 最长等10分钟


class XunfeiTranscriber(Transcriber):
    """讯飞录音文件识别."""

    def __init__(self, app_id: str, api_key: str, api_secret: str):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret

    def _signature(self, ts: str) -> str:
        """Generate HMAC-SHA1 signature."""
        raw = (self.app_id + ts).encode("utf-8")
        digest = hmac.new(
            self.api_secret.encode("utf-8"), raw, hashlib.sha1
        ).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _upload(self, audio_path: Path) -> str:
        """Upload audio file and return task_id."""
        ts = str(int(time.time()))
        sig = self._signature(ts)

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {
                "appId": self.app_id,
                "signature": sig,
                "ts": ts,
            }
            logger.info("Uploading %s to 讯飞 ASR...", audio_path.name)
            resp = requests.post(UPLOAD_URL, data=data, files=files, timeout=120)
            resp.encoding = "utf-8"
            result = resp.json()

        if result.get("code") != "0":
            raise RuntimeError(f"讯飞上传失败: code={result.get('code')}, desc={result.get('desc')}")

        task_id = result["data"]["taskId"]
        logger.info("Upload success, task_id=%s", task_id)
        return task_id

    def _poll_result(self, task_id: str) -> list[dict]:
        """Poll for transcription result until complete."""
        deadline = time.time() + MAX_POLL_SEC
        last_log = 0

        while time.time() < deadline:
            ts = str(int(time.time()))
            sig = self._signature(ts)
            data = {
                "appId": self.app_id,
                "signature": sig,
                "ts": ts,
                "taskId": task_id,
            }
            resp = requests.post(RESULT_URL, data=data, timeout=30)
            resp.encoding = "utf-8"
            result = resp.json()

            code = result.get("code", "")

            if code == "0":
                # done — parse result
                raw = result.get("data", {}).get("data", "[]")
                if isinstance(raw, str):
                    segments = json.loads(raw)
                else:
                    segments = raw
                logger.info("Transcription complete: %d segments", len(segments))
                return segments

            elif code == "1":
                # still processing
                elapsed = int(time.time() - deadline + MAX_POLL_SEC)
                if elapsed - last_log >= 15:
                    logger.info("Transcribing... (%d sec elapsed)", elapsed)
                    last_log = elapsed
                time.sleep(POLL_INTERVAL)
                continue

            else:
                raise RuntimeError(f"讯飞查询失败: code={code}, desc={result.get('desc')}")

        raise TimeoutError(f"讯飞转录超时（{MAX_POLL_SEC}秒）")

    def transcribe(self, audio_path: Path, duration_sec: float | None = None) -> TranscriptResult:
        """Transcribe audio file using 讯飞 ASR."""
        task_id = self._upload(audio_path)
        segments = self._poll_result(task_id)

        # Build full text
        full_text = "\n".join(seg["text"] for seg in segments if seg.get("text"))

        # Calculate cost: 0.33元/小时, rounded up to nearest hour
        if duration_sec:
            hours = max(1, int((duration_sec + 3599) / 3600))
        else:
            hours = 1
        cost = hours * 0.33

        return TranscriptResult(
            raw_text=full_text,
            segments=segments,
            duration_sec=duration_sec or 0,
            cost_yuan=cost,
        )
