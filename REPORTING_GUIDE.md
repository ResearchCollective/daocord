# Event Collection and Reporting Guide

This guide shows how to search X (Twitter), bundle an event around a tweet, and generate an LLM report.

## Prerequisites

- Python venv activated
- Dependencies installed from `requirements.txt`
- Credentials:
  - X API bearer token available to the current shell.
    - Configurable env var name in `config.yaml` at `twitter.bearer_token_env` (default: `X_BEARER_TOKEN`).
    - Example (PowerShell):
      ```powershell
      $env:X_BEARER_TOKEN = "<YOUR_TWITTER_BEARER_TOKEN>"
      ```
  - Gemini API key set in `config.yaml` under `providers.google.api_key` or via env var `GOOGLE_API_KEY`.

## 1) Search X (Twitter)

Use `tools/x_search.py` to search recent posts and rank locally by engagement.

Examples:
```powershell
# Top 5, English, last 7 days, JSON output, with cache
python tools/x_search.py search "longevity dogs" --limit 5 --lang en --since-days 7 --use-cache --json
```

- Cache location and TTL configured in `config.yaml` at `twitter.cache.search_dir` (e.g., `data/x/searches`).
- Results include `score = likes + retweets + replies + quotes` (computed locally).

## 2) Build an Event Bundle for a Tweet

Bundle collects the original tweet plus related context and stores it under `twitter.cache.events_dir` (e.g., `data/x/events/<tweet_id>.json`).

```powershell
# Heuristic mode: if the tweet looks like a thread ("thread" or "1/"), fetch self-replies; otherwise fetch conversation replies
python tools/x_search.py bundle <TWEET_ID> --since-days 14 --save-bundle --json

# Force general conversation replies (exclude the author), cap to 3
python tools/x_search.py bundle <TWEET_ID> --include-replies --replies-limit 3 --since-days 14 --save-bundle --json

# Force self-replies (thread mode), disable heuristic
python tools/x_search.py bundle <TWEET_ID> --no-include-replies --no-thread-heuristic --self-limit 10 --since-days 14 --save-bundle --json
```

Bundle JSON shape (X):
- `original`: the target tweet
- `self_replies`: replies by the same author within the conversation
- `quote_tweets`: quotes referencing the tweet
- `replies`: conversation replies excluding the author

## 3) Generate a Gemini Event Report

Use `tools/event_report.py` to synthesize a structured report from a saved bundle JSON.

```powershell
# Generate a Markdown report next to repo-level reports directory
python tools/event_report.py --input data/x/events/<TWEET_ID>.json --config config.yaml

# Write to a specific output path
python tools/event_report.py --input data/x/events/<TWEET_ID>.json --out reports/event_<TWEET_ID>.md

# Get JSON-only output (printed path will still be the markdown file if --out provided)
python tools/event_report.py --input data/x/events/<TWEET_ID>.json --json-only
```

Prompt/Model:
- The tool builds a compact prompt that lists the original, quotes, self-replies, and replies.
- Output JSON schema includes `companies`, `people`, `interventions`, `developments_summary`, and `notable_posts`.
- Uses `providers.google.api_key` or `GOOGLE_API_KEY`.

## 4) Logs and Caching

- X calls log to `data/logs/x_calls.jsonl`.
- Search cache: `data/x/searches/*.json`.
- Event bundles: `data/x/events/*.json`.
- Reports (default): `reports/event_<id>.md`.

## Troubleshooting

- Missing X bearer token:
  - Verify process-level visibility:
    ```powershell
    python -c "import os; print('seen' if os.getenv('X_BEARER_TOKEN') else 'missing')"
    ```
  - If missing, set for current shell:
    ```powershell
    $env:X_BEARER_TOKEN = "<YOUR_TWITTER_BEARER_TOKEN>"
    ```
  - Ensure `config.yaml` has `twitter.bearer_token_env: "X_BEARER_TOKEN"` (or change accordingly).

- Gemini key not found:
  - Set `providers.google.api_key` in `config.yaml` or export `GOOGLE_API_KEY`.

- Replies vs self-replies:
  - Default heuristic picks self-replies if the original looks like a thread ("thread" or "1/").
  - Override with `--include-replies` or `--no-include-replies` and tune with `--replies-limit` / `--self-limit`.
