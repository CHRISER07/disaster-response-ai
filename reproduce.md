# Reproduction Instructions

## Environment
```
git clone https://github.com/CHRISER07/disaster-response-ai.git
cd disaster_response_rag
pip install -r requirements.txt
cp .env.example .env  # add GROQ_API_KEY
```

## Commit Hash
```
591e9657be4f8fda9624d7c17cd35ccaae93ce01
```

## Run Order
```powershell
$env:PYTHONUTF8=1

# Phase 2: Dev set evaluation (8 configs x 50 queries x 3 seeds)
python evaluation/run_dev_eval_v2.py

# Phase 3: Heldout evaluation
python evaluation/run_heldout_eval_v2.py

# Phase 4: Judge scoring + kappa
python evaluation/run_judge_v2.py

# Phase 4b: Human label sample
python evaluation/build_human_sample.py

# Phase 5: Safety evaluation
python evaluation/run_safety_eval_v2.py

# Phase 6: TGI temporal conflict
python evaluation/run_tgi_eval_v2.py

# Phase 9: Summary reports
python evaluation/build_summary_reports.py
```

## Models
| Role | Model | Provider | Temperature |
|------|-------|----------|-------------|
| Primary agent | qwen/qwen3.8-27b | Groq | 0.1 |
| Fallback 1 | openai/gpt-oss-120b | Groq | 0.1 |
| Fallback 2 | openai/gpt-oss-20b | Groq | 0.1 |
| Judge A | qwen/qwen3.8-27b | Groq | 0.0 |
| Judge B | openai/gpt-oss-120b | Groq | 0.0 |

## Seeds
Dev/heldout runs: seeds [42, 7, 99] per config  
Safety runs: seeds [42, 7, 99] per config  
TGI runs: seeds [42, 7, 99] per config  

## Notes
- Retrieval: SQLite FTS only (dense vector broken — PyTorch not installed)
- recursion_limit: 18
- Rate limit throttle: 0.8s between requests + exponential backoff for 429
- All files UTF-8 (PYTHONUTF8=1)
