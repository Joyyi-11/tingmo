# 听墨（Tingmo）

面向小宇宙播客的低成本结构化文稿生成工具。输入单集链接，自动完成节目抓取、音频下载、本地转写、说话人区分和 LLM 内容整理，最终输出适合阅读、编辑和引用的 Markdown 文稿。

## 为什么做

现有播客转录服务通常采用按月订阅或有限免费额度。对于只需要偶尔处理单期节目的人，这意味着持续付费；只调用 Whisper 又只能得到原始逐字稿，仍要人工处理错字、口语、专名、说话人和内容结构。

听墨把计算量最大的语音转写放在本地完成，只把文本校订和内容提炼交给 LLM。这样既避免持续订阅，也把单期 API 支出控制在较低水平，同时保留可回查的原始转录。

## 功能

- 爬取小宇宙播客节目信息、Show Notes
- 下载音频并转写为文字（本地 faster-whisper，免费）
- 千问或 DeepSeek 分块校订：错字修正、口语清理、语义分段、专名纠错
- 独立生成内容提要、人物简介和闪光语句，避免长文输出截断
- 永久保留 Whisper 原始转录，已完成的校订分块支持失败后续跑
- 按模型和来源 hash 隔离缓存，避免失败重跑时重复消耗 LLM token
- 记录转写耗时、实时系数和 LLM token 用量
- 输出结构化 Markdown：Show Notes → 内容提要 → 闪光语句 → 人物简介 → 全文转录

## 完整链路

从链接到文稿共经过六个阶段：

```text
小宇宙单集链接
  -> 抓取节目元信息、Show Notes 和音频地址
  -> 下载音频并转换为 WAV
  -> faster-whisper 本地转写，保留原始逐字稿
  -> 说话人识别，为不同声音添加标签（可选）
  -> LLM 分块校订全文，再生成内容提要、人物简介和闪光语句
  -> 组装为结构化 Markdown 文稿
```

## 各部分分工

| 部分 | 实际职责 | 不负责什么 |
|------|----------|------------|
| 小宇宙页面抓取 | 获取标题、节目名、发布日期、Show Notes 和音频地址 | Show Notes 来自节目原页面，不由 LLM 生成 |
| 音频处理 | 下载音频，通过 ffmpeg 转换为本地转写所需格式 | 不改变节目内容 |
| faster-whisper | 在本地把音频转成带时间信息的原始文字 | 不负责内容提要、人物判断和文稿结构 |
| 说话人识别 | 根据声音特征区分说话人，生成 `SPEAKER_00` 等标签 | 只区分声音，不直接确认真实姓名和身份 |
| LLM 全文校订 | 分块修正错字、口语、专名和分段，保留原意 | 不重新创作或扩写播客观点 |
| LLM 内容提炼 | 结合 Show Notes 和校订全文，生成内容提要、人物简介、闪光语句，并在证据充分时映射说话人身份 | 信息不足时不猜测人物身份 |
| Markdown 组装 | 按固定结构写入 Show Notes、提要、闪光语句、人物简介和全文 | 不参与语义判断 |

## 快速开始

```powershell
# 安装依赖
pip install -r requirements.txt
pip install -e .

# Windows：将 API Key 保存为用户环境变量，重开终端后生效
[Environment]::SetEnvironmentVariable("LLM_API_KEY", "your_key", "User")
[Environment]::SetEnvironmentVariable("LLM_BASE_URL", "https://voltapi.ai/v1", "User")

# 转录播客
tingmo https://www.xiaoyuzhoufm.com/episode/xxxxx

# 切换到 DeepSeek V4 Flash
tingmo https://www.xiaoyuzhoufm.com/episode/xxxxx --llm-provider deepseek
```

## 前置依赖

- Python 3.10+
- ffmpeg（用于音频格式转换）

## 配置

LLM 配置从操作系统环境变量读取：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://voltapi.ai/v1
```

默认使用 `qwen3.7-plus`；`--llm-provider deepseek` 使用 `deepseek-v4-flash`。
也可以通过 `--llm-model` 临时覆盖模型名称。项目 `.env` 仅保留其他本地配置，API Key 不必写入其中。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `url` | 必填 | 小宇宙播客单集链接 |
| `-o` | `output/` | 输出目录 |
| `--model` | `medium` | Whisper 模型大小（tiny/base/small/medium/large-v3） |
| `--llm-provider` | `qwen` | 后处理模型提供方（qwen/deepseek） |
| `--llm-model` | 提供方默认值 | 覆盖默认模型名称 |
| `--speakers` | 自动检测 | 明确指定说话人数 |
| `--no-llm` | 否 | 仅转写，不需要 LLM API Key |
| `--no-diarization` | 否 | 跳过说话人识别 |

## 输出格式

```
# 节目标题
> 来源：播客名称 | 发布日期

# Show Notes
（完整保留的节目介绍）

## 内容提要
- **要点名称**：支撑证据

## 闪光语句
- 核心洞见原句

## 人物简介
> **身份姓名**：简介

## 全文转录
（校订后的分角色完整文稿）
```

## 成本

- 转写：本地免费
- LLM 后处理费用取决于模型、节目长度和 VoltAPI 实际账单
- 个人使用 DeepSeek 校订的历史实测中，典型单期 API 支出低于 0.05 元
- 程序记录输入／输出 token，不再使用过期的固定单价估算

`0.05 元/期` 是特定模型、节目长度和实际账单下的历史结果，不是固定报价或程序保证；更换模型、API 服务或处理更长节目时应以实际账单为准。

## 中间文件

- `output/<节目名>_raw.txt`：未经 LLM 修改的 Whisper 原始转录
- `output/.work/<节目名>/`：按模型隔离的校订分块缓存
- `output/<节目名>.md`：最终文稿

模型请求失败或返回被截断时，程序会停止写入最终文稿。再次运行相同节目和模型时，来源一致的已完成分块会直接复用。

## Benchmark

`benchmark/` 记录了两类工程实验：

- `transcribe_cpp/`：对比 `transcribe.cpp` 与现有 Whisper Medium 链路，当前结论是不切换。
- `llm_ab/`：在同一份 Whisper 原始转录上比较 `qwen3.7-plus` 与 `deepseek-v4-flash` 的忠实校订表现。

这些实验不进入主流程，但保留了模型选择和工程取舍依据。

## 限制

- 当前主要支持小宇宙单集页面；网页结构变化可能影响抓取。
- LLM 后处理需要兼容 OpenAI Chat Completions 的 API。
- 说话人识别是辅助能力，不确定时应保留标签或人工复核。

## 项目结构

```
src/
├── main.py                 # CLI 入口
├── config.py               # 配置加载
├── audio.py                # 音频下载与格式转换
├── utils.py                # 计时、计费等工具
├── models/schemas.py       # 数据模型
├── scraper/xiaoyuzhou.py   # 小宇宙页面爬取
├── transcriber/local.py    # faster-whisper 本地转写
└── processor/
    ├── prompt.py           # 忠实校订与提要生成提示词
    └── llm_processor.py    # 分块校订、校验、缓存与文稿整理
```

## 作者

连漪（Lianyi）

## 许可证

MIT
