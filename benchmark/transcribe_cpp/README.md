# transcribe.cpp 中文播客基准

## 目标

以一份人工校订、主动删减后的参考文稿为基准，比较：

1. 现有 `faster-whisper Medium + 旧 LLM 后处理` 文稿。
2. `transcribe.cpp v0.1.3 + Qwen3-ASR 0.6B Q8_0`。
3. `transcribe.cpp v0.1.3 + Qwen3-ASR 1.7B Q5_K_M`。

没有单独下载 `transcribe.cpp` 的 Whisper Medium GGUF：现有基线已经使用同一 Whisper Medium 模型族；本轮先验证新模型族是否带来准确率收益，以减少下载和重复推理成本。

## 评价原则

- 参考稿删掉的内容不算候选模型错误。
- 只评价参考稿保留的播客摘录能否在候选转录中被完整、准确地找到。
- 自动指标用于定位差异，不把参考稿的人工润色误称为逐字真值。
- 最终结论同时依据保留内容匹配、漏段、专名/数字错误和人工流畅度审阅。

## 目录

- `assets/`：本地 CLI、GGUF 模型和音频，不进入 Git。
- `results/`：各模型原始转录和中间指标，不进入 Git。
- `evaluate.py`：删减型参考对齐与指标计算；运行时需要显式传入参考文稿和当前输出。
- `report.md`：完成测试后写入结论与代表性差异。

## 当前结论

暂不切换。Qwen3-ASR 0.6B 和 1.7B 在 E236 样本上都低于当前 Whisper Medium + LLM 成稿，且 Qwen provider 没有时间戳、存在固定 256 token 生成上限。详细数据见 `report.md`。

```powershell
python benchmark/transcribe_cpp/evaluate.py --published path/to/reference.md --current path/to/current-output.md
```
