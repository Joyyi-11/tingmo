# Whisper transcript LLM A/B benchmark

This benchmark compares faithful transcript cleaning with identical inputs:

- `qwen3.7-plus`
- `deepseek-v4-flash`

It uses the E236 `36:00-56:00` audio slice. The LLM stage only runs the
cleaning prompt; it does not generate summaries, key points, quotes, or speaker
introductions. Published text is reserved for evaluation and is never included
in either model request.

Generated raw transcripts and API results are stored in `results/`, which is
excluded from Git.

```powershell
python benchmark/llm_ab/run_ab.py transcribe

$env:LLM_API_KEY = [Environment]::GetEnvironmentVariable('LLM_API_KEY', 'User')
$env:LLM_BASE_URL = [Environment]::GetEnvironmentVariable('LLM_BASE_URL', 'User')
python benchmark/llm_ab/run_ab.py clean qwen --source-output path/to/current-output.md
python benchmark/llm_ab/run_ab.py clean deepseek --source-output path/to/current-output.md
```
