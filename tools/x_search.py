import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

# Twitter (X) API endpoints
RECENT_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
TWEET_LOOKUP_URL = "https://api.twitter.com/2/tweets"


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config if present, else return empty dict."""
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_env_tokens(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Resolve token environment variable names from config, then read env.

    Expected config section (optional):

    twitter:
      bearer_token_env: TWITTER_BEARER_TOKEN

    Falls back to TWITTER_BEARER_TOKEN if not provided.
    """
    tw_cfg = (cfg or {}).get("twitter", {}) or {}
    bearer_env_name = tw_cfg.get("bearer_token_env", "TWITTER_BEARER_TOKEN")

    return {
        "bearer_token": os.getenv(bearer_env_name),
        "bearer_env_name": bearer_env_name,
    }


def _cfg_cache_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tw = (cfg or {}).get("twitter", {}) or {}
    cache = tw.get("cache", {}) or {}
    logs = (tw.get("logs") or {}) if isinstance(tw.get("logs"), dict) else {}
    return {
        "search_dir": cache.get("search_dir", "data/searches"),
        "events_dir": cache.get("events_dir", "data/events"),
        "ttl_hours": int(cache.get("ttl_hours", 24)),
        "calls_log": logs.get("calls_file", "data/logs/x_calls.jsonl"),
    }


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _log_call(log_path: str, record: Dict[str, Any]) -> None:
    try:
        _ensure_dir(str(Path(log_path).parent))
        record = {**record, "ts": datetime.now(timezone.utc).isoformat()}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def build_query(query: str, lang: Optional[str], exclude_replies: bool = True, exclude_retweets: bool = True) -> str:
    parts = [query.strip()]
    if exclude_replies:
        parts.append("-is:reply")
    if exclude_retweets:
        parts.append("-is:retweet")
    if lang:
        parts.append(f"lang:{lang}")
    return " ".join([p for p in parts if p])


def fetch_recent(client: httpx.Client, bearer: str, query: str, max_results: int = 100, start_time: Optional[datetime] = None) -> Dict[str, Any]:
    """Call recent search once. We purposely do a single request to minimize API usage."""
    headers = {"Authorization": f"Bearer {bearer}"}

    params = {
        "query": query,
        "max_results": max(10, min(max_results, 100)),  # API bounds 10..100
        "tweet.fields": "author_id,created_at,lang,public_metrics,conversation_id",
        "expansions": "author_id",
        "user.fields": "username,name,verified,created_at,public_metrics",
        # Do not set sort_order; default is recency. We'll rank locally.
    }

    if start_time is not None:
        # RFC3339 format
        params["start_time"] = start_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    resp = client.get(RECENT_SEARCH_URL, headers=headers, params=params, timeout=20)
    if resp.status_code == 429:
        raise RuntimeError("Twitter API rate limit (HTTP 429). Try again later or reduce calls.")
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Twitter API error {resp.status_code}: {detail}")
    payload = resp.json()
    # Lightweight call logging
    try:
        cfg = load_config()
        c = _cfg_cache_settings(cfg)
        _log_call(c["calls_log"], {
            "endpoint": "search_recent",
            "status": resp.status_code,
            "query": query,
            "max_results": max_results,
            "start_time": start_time.isoformat() if start_time else None,
            "returned": len(payload.get("data", []) or []),
        })
    except Exception:
        pass
    return payload


def get_tweet(client: httpx.Client, bearer: str, tweet_id: str) -> Dict[str, Any]:
    """Fetch a single tweet with author expansion."""
    headers = {"Authorization": f"Bearer {bearer}"}
    url = f"{TWEET_LOOKUP_URL}/{tweet_id}"
    params = {
        "tweet.fields": "author_id,created_at,lang,public_metrics,conversation_id",
        "expansions": "author_id",
        "user.fields": "username,name,verified,created_at,public_metrics",
    }
    resp = client.get(url, headers=headers, params=params, timeout=20)
    if resp.status_code == 429:
        raise RuntimeError("Twitter API rate limit (HTTP 429) on tweet lookup.")
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Twitter API error {resp.status_code}: {detail}")
    payload = resp.json()
    # Lightweight call logging
    try:
        cfg = load_config()
        c = _cfg_cache_settings(cfg)
        _log_call(c["calls_log"], {
            "endpoint": "tweet_lookup",
            "status": resp.status_code,
            "tweet_id": tweet_id,
            "returned": 1 if payload.get("data") else 0,
        })
    except Exception:
        pass
    return payload


def _map_user_includes(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {u.get("id"): u for u in (payload.get("includes", {}) or {}).get("users", [])}


def search_self_replies(
    client: httpx.Client,
    bearer: str,
    conversation_id: str,
    author_id: str,
    limit: int,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Find self-replies by the same author within the conversation.

    Uses one recent search call, filtered to from:author and conversation_id.
    """
    q = f"conversation_id:{conversation_id} from:{author_id} -is:retweet"
    return fetch_recent(client, bearer, q, max_results=limit, start_time=since)


def search_conversation_replies(
    client: httpx.Client,
    bearer: str,
    conversation_id: str,
    author_id: str,
    limit: int,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Find replies in the conversation excluding the original author.

    Uses one recent search call constrained to the conversation and excluding from:author.
    """
    q = f"conversation_id:{conversation_id} -from:{author_id} -is:retweet"
    return fetch_recent(client, bearer, q, max_results=limit, start_time=since)


def search_quote_tweets(
    client: httpx.Client,
    bearer: str,
    author_username: str,
    tweet_id: str,
    limit: int,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Find quote-tweets referencing the target tweet using url operator.

    We use a URL match and is:quote to constrain results within one call.
    """
    url_match = f"\"https://twitter.com/{author_username}/status/{tweet_id}\""
    q = f"url:{url_match} is:quote -is:retweet"
    return fetch_recent(client, bearer, q, max_results=limit, start_time=since)


def collect_event_bundle(
    tweet_id: str,
    self_replies_limit: int = 10,
    quotes_limit: int = 10,
    since_days: int = 14,
    config_path: str = "config.yaml",
    save_bundle: bool = False,
    out_dir: Optional[str] = None,
    include_replies: Optional[bool] = None,
    replies_limit: int = 50,
    thread_heuristic: bool = True,
) -> Dict[str, Any]:
    """Collect an event bundle: original tweet, self-replies, and quote-tweets.

    All components are fetched with a single call each (max), honoring limits
    to conserve read quota.
    """
    cfg = load_config(config_path)
    tokens = resolve_env_tokens(cfg)
    bearer = tokens.get("bearer_token")
    if not bearer:
        env_name = tokens.get("bearer_env_name") or "TWITTER_BEARER_TOKEN"
        raise RuntimeError(
            f"Missing bearer token. Set environment variable {env_name} with your X API Bearer Token or specify it in config via twitter.bearer_token_env."
        )

    since = datetime.now(timezone.utc) - timedelta(days=max(1, since_days))

    with httpx.Client() as client:
        original_payload = get_tweet(client, bearer, tweet_id)
        original = original_payload.get("data") or {}
        users_map = _map_user_includes(original_payload)
        author = users_map.get(original.get("author_id")) if original else None

        # Defaults if username missing; quotes search needs it for URL query
        author_username = (author or {}).get("username", "i")
        conversation_id = (original or {}).get("conversation_id", tweet_id)
        author_id = (original or {}).get("author_id")

        self_replies_payload: Dict[str, Any] = {}
        quotes_payload: Dict[str, Any] = {}
        replies_payload: Dict[str, Any] = {}

        # Decide whether to prioritize self-replies (thread) vs general replies
        mode = "self"
        if include_replies is True:
            mode = "replies"
        elif include_replies is False:
            mode = "self"
        else:
            # Heuristic: if original text looks like a thread starter, prefer self-replies
            text0 = (original or {}).get("text") or ""
            looks_thread = ("thread" in text0.lower()) or ("1/" in text0)
            mode = "self" if (thread_heuristic and looks_thread) else "replies"

        if author_username and tweet_id:
            quotes_payload = search_quote_tweets(
                client, bearer, author_username, tweet_id, limit=max(10, min(quotes_limit, 100)), since=since
            )

        if author_id:
            if mode == "self":
                self_replies_payload = search_self_replies(
                    client, bearer, conversation_id, author_id, limit=max(10, min(self_replies_limit, 100)), since=since
                )
            else:
                replies_payload = search_conversation_replies(
                    client, bearer, conversation_id, author_id, limit=max(3, min(replies_limit, 100)), since=since
                )

    # Shape bundle for consumers, including URLs and basic author fields
    def normalize_single(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not payload or not payload.get("data"):
            return None
        data = payload["data"]
        users = _map_user_includes(payload)
        user = users.get(data.get("author_id")) if isinstance(users, dict) else None
        created_at = data.get("created_at")
        try:
            created_at_dt = datetime.fromisoformat((created_at or "").replace("Z", "+00:00")) if created_at else None
        except Exception:
            created_at_dt = None
        return {
            "id": data.get("id"),
            "text": data.get("text"),
            "created_at": created_at_dt or created_at,
            "lang": data.get("lang"),
            "metrics": data.get("public_metrics", {}),
            "author": {
                "id": user.get("id") if user else None,
                "username": user.get("username") if user else None,
                "name": user.get("name") if user else None,
                "verified": user.get("verified") if user else None,
            },
            "url": f"https://twitter.com/{(user or {}).get('username', 'i')}/status/{data.get('id')}",
        }

    def normalize_many(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not payload:
            return []
        since_dt = since
        return rank_tweets(payload, since_dt)

    bundle = {
        "original": normalize_single(original_payload),
        "self_replies": normalize_many(self_replies_payload),
        "quote_tweets": normalize_many(quotes_payload),
        "replies": normalize_many(replies_payload),
    }
    # Optional persistence
    if save_bundle:
        c = _cfg_cache_settings(cfg)
        base_dir = out_dir or c.get("events_dir", "data/events")
        _ensure_dir(base_dir)
        path = Path(base_dir) / f"{tweet_id}.json"
        def _ser(o):
            if isinstance(o, datetime):
                return o.isoformat()
            return o
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, ensure_ascii=False, indent=2, default=_ser)
        except Exception:
            pass
    return bundle


def rank_tweets(payload: Dict[str, Any], since: datetime) -> List[Dict[str, Any]]:
    """Filter to tweets since 'since' and rank by engagement (likes+retweets+replies+quotes)."""
    data = payload.get("data", []) or []
    users = {u.get("id"): u for u in (payload.get("includes", {}) or {}).get("users", [])}

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    enriched = []
    for t in data:
        try:
            created_at = parse_dt(t.get("created_at"))
        except Exception:
            continue
        if created_at < since:
            continue
        metrics = (t.get("public_metrics") or {})
        score = (
            int(metrics.get("like_count", 0))
            + int(metrics.get("retweet_count", 0))
            + int(metrics.get("reply_count", 0))
            + int(metrics.get("quote_count", 0))
        )
        user = users.get(t.get("author_id"))
        enriched.append({
            "id": t.get("id"),
            "text": t.get("text"),
            "created_at": created_at,
            "lang": t.get("lang"),
            "metrics": metrics,
            "score": score,
            "author": {
                "id": user.get("id") if user else None,
                "username": user.get("username") if user else None,
                "name": user.get("name") if user else None,
                "verified": user.get("verified") if user else None,
            },
            "url": f"https://twitter.com/{user.get('username') if user else 'i'}/status/{t.get('id')}"
        })

    enriched.sort(key=lambda x: (x["score"], x["metrics"].get("like_count", 0), x["created_at"]), reverse=True)
    return enriched


def _search_cache_key(params: Dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_search_cache(dir_path: str, key: str, ttl_hours: int) -> Optional[List[Dict[str, Any]]]:
    p = Path(dir_path) / f"{key}.json"
    if not p.is_file():
        return None
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if datetime.now(timezone.utc) - mtime > timedelta(hours=max(1, ttl_hours)):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_search_cache(dir_path: str, key: str, results: List[Dict[str, Any]]) -> None:
    try:
        _ensure_dir(dir_path)
        p = Path(dir_path) / f"{key}.json"
        def _ser(o):
            if isinstance(o, datetime):
                return o.isoformat()
            return o
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_ser)
    except Exception:
        pass


def search_x(
    query: str,
    limit: int = 10,
    lang: Optional[str] = None,
    since_days: int = 7,
    config_path: str = "config.yaml",
    use_cache: bool = False,
) -> List[Dict[str, Any]]:
    cfg = load_config(config_path)
    tokens = resolve_env_tokens(cfg)
    bearer = tokens.get("bearer_token")
    if not bearer:
        env_name = tokens.get("bearer_env_name") or "TWITTER_BEARER_TOKEN"
        raise RuntimeError(
            f"Missing bearer token. Set environment variable {env_name} with your X API Bearer Token or specify it in config via twitter.bearer_token_env."
        )

    # Build query and timeframe
    since = datetime.now(timezone.utc) - timedelta(days=max(1, since_days))
    built_query = build_query(query, lang)

    # Optional cache check
    cache_settings = _cfg_cache_settings(cfg)
    cache_key = _search_cache_key({
        "query": built_query,
        "limit": int(limit),
        "since_days": int(since_days),
        "lang": lang or "",
    })
    if use_cache:
        cached = _read_search_cache(cache_settings["search_dir"], cache_key, cache_settings["ttl_hours"])
        if cached is not None:
            return cached[:limit]

    # One API call to minimize usage; request up to 100 then rank locally
    with httpx.Client() as client:
        # To conserve read quota, request only as many tweets as the caller needs (bounded 10..100 by API)
        payload = fetch_recent(
            client,
            bearer,
            built_query,
            max_results=limit,
            start_time=since,
        )

    ranked = rank_tweets(payload, since)
    topn = ranked[:limit]
    if use_cache:
        _write_search_cache(cache_settings["search_dir"], cache_key, topn)
    return topn


def format_compact(results: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        m = r["metrics"]
        uname = r["author"].get("username") if r.get("author") else None
        # Precompute values to avoid backslashes inside f-string expressions
        created_at = r.get("created_at")
        if isinstance(created_at, datetime):
            date_str = created_at.strftime('%Y-%m-%d')
        else:
            date_str = str(created_at)
        text_single_line = (r.get('text') or '').replace('\n', ' ')
        lines.append(
            f"{i:02d}. @{uname or 'unknown'} | {m.get('like_count', 0)} {m.get('retweet_count', 0)} {m.get('reply_count', 0)} | {date_str}\n    {text_single_line}\n    {r['url']}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X (Twitter) helper: search and event bundles.")
    subparsers = parser.add_subparsers(dest="cmd")

    # search subcommand (default behavior)
    p_search = subparsers.add_parser("search", help="Search recent tweets and rank locally")
    p_search.add_argument("query", help="Search topic or query string")
    p_search.add_argument("--limit", type=int, default=10, help="Number of posts to return (default: 10)")
    p_search.add_argument("--lang", type=str, default=None, help="Language filter, e.g., en")
    p_search.add_argument("--since-days", type=int, default=7, help="Days back to include (default: 7)")
    p_search.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    p_search.add_argument("--json", action="store_true", help="Output JSON instead of text")
    p_search.add_argument("--use-cache", action="store_true", help="Use and populate on-disk cache for this query")

    # bundle subcommand
    p_bundle = subparsers.add_parser("bundle", help="Collect an event bundle around a tweet")
    p_bundle.add_argument("tweet_id", help="Target tweet ID")
    p_bundle.add_argument("--self-limit", type=int, default=10, help="Max self-replies to fetch (default: 10)")
    p_bundle.add_argument("--quotes-limit", type=int, default=10, help="Max quote-tweets to fetch (default: 10)")
    p_bundle.add_argument("--since-days", type=int, default=7, help="Days back to include (default: 7)")
    p_bundle.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    p_bundle.add_argument("--json", action="store_true", help="Output JSON (recommended)")
    p_bundle.add_argument("--save-bundle", action="store_true", help="Persist bundle JSON to events cache directory")
    p_bundle.add_argument("--out-dir", type=str, default=None, help="Override events directory when saving bundle")
    # Replies control
    p_bundle.add_argument("--include-replies", dest="include_replies", action="store_true", help="Force including conversation replies (excludes self-replies)")
    p_bundle.add_argument("--no-include-replies", dest="include_replies", action="store_false", help="Force including self-replies (thread mode)")
    p_bundle.set_defaults(include_replies=None)  # use heuristic by default
    p_bundle.add_argument("--replies-limit", type=int, default=50, help="Max conversation replies to fetch (default: 50)")
    p_bundle.add_argument("--no-thread-heuristic", dest="thread_heuristic", action="store_false", help="Disable thread heuristic; use include-replies setting only")
    p_bundle.set_defaults(thread_heuristic=True)

    # If no subcommand provided, assume 'search' for backward compatibility
    if argv is None:
        argv = sys.argv[1:]
    if not argv or (argv and argv[0] not in {"search", "bundle"}):
        argv = ["search", *argv]

    args = parser.parse_args(argv)

    try:
        if args.cmd == "bundle":
            bundle = collect_event_bundle(
                tweet_id=args.tweet_id,
                self_replies_limit=args.self_limit,
                quotes_limit=args.quotes_limit,
                since_days=args.since_days,
                config_path=args.config,
                save_bundle=args.save_bundle,
                out_dir=args.out_dir,
                include_replies=args.include_replies,
                replies_limit=args.replies_limit,
                thread_heuristic=args.thread_heuristic,
            )
            if args.json:
                def _ser(o):
                    if isinstance(o, datetime):
                        return o.isoformat()
                    return o
                print(json.dumps(bundle, default=_ser, ensure_ascii=False, indent=2))
            else:
                orig = bundle.get("original")
                print("Original:")
                print(format_compact([orig]) if orig else "(missing)")
                print("\nSelf-replies:")
                print(format_compact(bundle.get("self_replies", [])) or "(none)")
                print("\nQuote-tweets:")
                print(format_compact(bundle.get("quote_tweets", [])) or "(none)")
                print("\nReplies:")
                print(format_compact(bundle.get("replies", [])) or "(none)")
        else:
            results = search_x(
                query=args.query,
                limit=args.limit,
                lang=args.lang,
                since_days=args.since_days,
                config_path=args.config,
                use_cache=args.use_cache,
            )
            if args.json:
                def _ser(o):
                    if isinstance(o, datetime):
                        return o.isoformat()
                    return o
                print(json.dumps(results, default=_ser, ensure_ascii=False, indent=2))
            else:
                print(format_compact(results))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
