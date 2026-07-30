# transcribe.cpp 中文播客测试报告

测试日期：2026-07-20

## 结论

暂不将项目切换到 `transcribe.cpp`。继续使用现有的 Whisper Medium 转录链路，再优化 LLM 校订和专名词表。

`transcribe.cpp 0.1.3` 在 Docker Linux CPU 环境中可以稳定运行，但两个 Qwen3-ASR 模型在 E236 中文播客样本上都没有超过现有成稿。1.7B 在专名密集片段上也没有优于 0.6B，因此没有继续跑完整 20 分钟或下载更多量化版本。

## 测试范围

- 节目：E236《99%的作业都是 AI 写的：当代名校生眼里，大学还剩下什么？》
- 完整样本：`36:00-56:00`，20 分钟，40 个 30 秒分块
- 重点样本：`46:30-51:30`，5 分钟，包含大量中英文产品名和技术术语
- 参考：人工保留并校订的段落
- 原则：参考稿主动删除的内容不计为候选方案漏转

## 结果

| 方案 | 样本 | 三字组覆盖率 | 专名命中 | CPU 推理 | 时间戳 |
|---|---:|---:|---:|---:|---|
| 现有 Whisper Medium + LLM 成稿 | 20 分钟 | 95.2% | - | 未重跑 | 有说话人结构 |
| Qwen3-ASR 0.6B Q8_0 | 20 分钟 | 85.7% | - | 326.5 秒 | 无 |
| 现有 Whisper Medium + LLM 成稿 | 5 分钟 | 93.5% | 26/29 | 未重跑 | 有说话人结构 |
| Qwen3-ASR 0.6B Q8_0 | 5 分钟 | 81.1% | 20/29 | 约 82 秒 | 无 |
| Qwen3-ASR 1.7B Q5_K_M | 5 分钟 | 80.8% | 19/29 | 162.5 秒 | 无 |

三字组覆盖率只比较参考稿保留内容，忽略标点、空格、Markdown 和说话人前缀。现有成稿是参考稿的上游来源，因此它的分数天然偏高；该指标用于辅助定位差异，不能视为严格 CER/WER。

## 代表性差异

| 参考稿／正确形式 | 现有成稿 | Qwen3-ASR 0.6B | Qwen3-ASR 1.7B |
|---|---|---|---|
| 鉴赏能力 | 鉴赏能力 | 渐闪能力 | 未覆盖该片段 |
| Claude | Claude | Cloud | Cloud |
| Vibe Coding | Vibe Coding | 文本 coding | Web Coding |
| Dify | Dify | Diffie | Diffy |
| Raycast | Raycast | Recast | Recast |
| OpenClaw | OpenAI | Open cloud／OpenCL | open cloud／OpenCL |
| Manus | Manus | minus／Manage | manas／Minus |
| Claude Code | Claude Code | cloud code | cloud code |
| Claude Cowork | Claude Code | Clow Co work | Cloud Co Work |

Qwen 的中文标点、简体一致性和普通句子连贯性尚可，但英文专名错误密集。1.7B 没有表现出可见改善，反而在 Typeless 等词上出现了新的错误。30 秒切块还会在句中切断文本；后处理可以修复排版，但不能可靠恢复所有听错的产品名。

现有方案整体更顺畅，且保留了说话人结构，不过也不是人工真值：例如把 `OpenClaw` 识别成 `OpenAI`、把 `Claude Cowork` 识别成 `Claude Code`。参考稿中的人工修订仍然明显提升了准确率。

## 工程表现

- Windows 官方 provider 在 `v0.1.2` 和 `v0.1.3` 的 `transcribe_init_backends()` 均发生 access violation；CPU、Vulkan 和短路径都复现。
- Docker Linux CPU 后端稳定。0.6B 约占 2.0 GiB，1.7B 约占 2.9 GiB。
- Qwen provider 单次生成上限固定为 256 token，API 没有暴露 `max_tokens`。60 秒分块在语速快的片段仍会截断，30 秒才稳定。
- Qwen provider 返回 `timestamp_kind=none`，无法直接与项目现有说话人分离结果按时间对齐。
- 0.6B 的 20 分钟推理实时系数约为 0.27；1.7B 的 5 分钟实时系数约为 0.54。

## 建议

1. 保留 Whisper Medium，不切换 Qwen3-ASR 0.6B 或 1.7B。
2. 下一轮把成本投入到同一份 Whisper 原始转录的 LLM A/B：`qwen3.7-plus` 对比 `deepseek-v4-flash`，重点测忠实校订而不是摘要能力。
3. 在 LLM 校订前生成节目级专名词表，至少包含 Show Notes、标题、嘉宾名、产品名和英文术语；要求模型只在上下文证据充分时纠正。
4. `transcribe.cpp` 先保留为实验分支。等 Windows 崩溃、Qwen 时间戳和生成上限解决后再复测，不进入主流程。

## 复现

模型文件和结果位于被 Git 忽略的 `assets/`、`results/`。在模型和结果仍存在时运行：

```powershell
python benchmark/transcribe_cpp/evaluate.py
```
