import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import yaml
import httpx

try:
    # Reuse helpers from x_search where possible
    from .x_search import load_config  # type: ignore
    from .x_search import _ensure_dir as ensure_dir  # type: ignore
except Exception:  # noqa: BLE001
    from x_search import load_config  # type: ignore
    from x_search import _ensure_dir as ensure_dir  # type: ignore

try:
    import praw  # Reddit API
    import prawcore
except Exception as e:  # noqa: BLE001
    praw = None  # type: ignore
    prawcore = None  # type: ignore


def _cfg_cache_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rd = (cfg or {}).get("reddit", {}) or {}
    cache = rd.get("cache", {}) or {}
    logs = (rd.get("logs") or {}) if isinstance(rd.get("logs"), dict) else {}
    return {
        "search_dir": cache.get("search_dir", "data/reddit/searches"),
        "events_dir": cache.get("events_dir", "data/reddit/events"),
        "ttl_hours": int(cache.get("ttl_hours", 24)),
        "calls_log": logs.get("calls_file", "data/logs/reddit_calls.jsonl"),
    }


def _public_search(
    sub: str,
    query: str,
    sort: str = "top",
    time_filter: str = "week",
    limit: int = 25,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    # Use Reddit's public JSON endpoint (no OAuth). Rate-limited; for light usage only.
    # Endpoint: /r/{sub}/search.json or /search.json when sub == 'all'
    base = "https://www.reddit.com"
    path = f"/search.json" if sub.lower() == "all" else f"/r/{sub}/search.json"
    params = {
        "q": query,
        "sort": sort,
        "t": time_filter,
        "limit": max(1, min(int(limit), 100)),
        "restrict_sr": "on" if sub.lower() != "all" else None,
        "include_over_18": "on",
    }
    # Drop None params
    params = {k: v for k, v in params.items() if v is not None}
    headers = {"User-Agent": "daocord/0.1 (+https://github.com)"}
    try:
        with httpx.Client(timeout=15.0, headers=headers) as client:
            resp = client.get(base + path, params=params)
            if verbose:
                print(f"[reddit_search] public GET {resp.request.url} status={resp.status_code}", file=sys.stderr)
            resp.raise_for_status()
            data = resp.json()
            children = (((data or {}).get("data") or {}).get("children") or [])
            out: List[Dict[str, Any]] = []
            for ch in children:
                d = (ch or {}).get("data") or {}
                try:
                    created_utc = datetime.fromtimestamp(float(d.get("created_utc", 0)), tz=timezone.utc)
                except Exception:
                    created_utc = datetime.fromtimestamp(0, tz=timezone.utc)
                out.append({
                    "id": d.get("id"),
                    "subreddit": d.get("subreddit"),
                    "author": d.get("author"),
                    "created_utc": created_utc,
                    "title": d.get("title"),
                    "selftext": d.get("selftext"),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "url": d.get("url_overridden_by_dest") or d.get("url"),
                    "permalink": ("https://www.reddit.com" + d.get("permalink", "")) if d.get("permalink") else None,
                })
            return out
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[reddit_search] public search error: {e}", file=sys.stderr)
    return []

def _log_call(log_path: str, record: Dict[str, Any]) -> None:
    try:
        ensure_dir(str(Path(log_path).parent))
        record = {**record, "ts": datetime.now(timezone.utc).isoformat()}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def resolve_reddit_creds(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    rd_cfg = (cfg or {}).get("reddit", {}) or {}
    client_id = rd_cfg.get("client_id") or os.getenv("REDDIT_CLIENT_ID")
    client_secret = rd_cfg.get("client_secret") or os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = rd_cfg.get("user_agent") or os.getenv("REDDIT_USER_AGENT") or "daocord/0.1 by yourusername"
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": user_agent,
    }


def _client(creds: Dict[str, Optional[str]]):
    if praw is None:
        raise RuntimeError("praw is not installed. Please add 'praw' to requirements and install dependencies.")
    if not creds.get("client_id") or not creds.get("client_secret"):
        raise RuntimeError("Missing Reddit API credentials. Set reddit.client_id and reddit.client_secret in config or REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET env vars.")
    return praw.Reddit(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        user_agent=creds["user_agent"] or "daocord/0.1",
        ratelimit_seconds=5,
    )


def _search_cache_key(params: Dict[str, Any]) -> str:
    # Simple deterministic key
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False)
    import hashlib
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
        ensure_dir(dir_path)
        p = Path(dir_path) / f"{key}.json"
        def _ser(o):
            if isinstance(o, datetime):
                return o.isoformat()
            return o
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_ser)
    except Exception:
        pass


def _norm_submission(s) -> Dict[str, Any]:  # s: praw.models.Submission
    return {
        "id": s.id,
        "subreddit": str(s.subreddit.display_name) if getattr(s, "subreddit", None) else None,
        "author": str(s.author) if getattr(s, "author", None) else None,
        "created_utc": datetime.fromtimestamp(getattr(s, "created_utc", 0), tz=timezone.utc),
        "title": getattr(s, "title", None),
        "selftext": getattr(s, "selftext", None),
        "score": getattr(s, "score", 0),
        "num_comments": getattr(s, "num_comments", 0),
        "url": getattr(s, "url", None),
        "permalink": f"https://www.reddit.com{s.permalink}" if getattr(s, "permalink", None) else None,
    }


def search_reddit(
    subreddits: List[str],
    limit_per_sub: int = 25,
    since_hours: int = 24,
    keywords: Optional[List[str]] = None,
    query: Optional[str] = None,
    sort: str = "top",
    time_filter: str = "week",
    config_path: str = "config.yaml",
    use_cache: bool = False,
    force_public: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    cfg = load_config(config_path)
    creds = resolve_reddit_creds(cfg)
    cache = _cfg_cache_settings(cfg)
    if verbose:
        print(
            (
                f"[reddit_search] cfg: search_dir={cache.get('search_dir')} | events_dir={cache.get('events_dir')} | ttl_hours={cache.get('ttl_hours')}\n"
                f"[reddit_search] creds: client_id_set={bool(creds.get('client_id'))} | client_secret_set={bool(creds.get('client_secret'))} | ua={creds.get('user_agent')}"
            ),
            file=sys.stderr,
        )

    since_dt = datetime.now(timezone.utc) - timedelta(hours=max(1, since_hours))
    if verbose:
        print(
            f"[reddit_search] since >= {since_dt.isoformat()} | subs={subreddits} | limit_per_sub={limit_per_sub} | keywords={keywords}",
            file=sys.stderr,
        )

    # Cache key
    cache_key = _search_cache_key({
        "subs": sorted(subreddits),
        "limit": int(limit_per_sub),
        "since_hours": int(since_hours),
        "keywords": sorted([k.lower() for k in (keywords or [])]),
        "query": query or "",
        "sort": sort,
        "time_filter": time_filter,
    })
    if use_cache:
        cached = _read_search_cache(cache["search_dir"], cache_key, cache["ttl_hours"])
        if verbose:
            print(
                f"[reddit_search] cache lookup: dir={cache['search_dir']} | key={cache_key[:8]}... | hit={cached is not None}",
                file=sys.stderr,
            )
        if cached is not None:
            return cached

    reddit = None
    if not (query and force_public):
        reddit = _client(creds)

    results: List[Dict[str, Any]] = []
    for sub in subreddits:
        try:
            scanned = 0
            matched = 0
            sr = reddit.subreddit(sub) if reddit is not None else None
            if query and force_public:
                if verbose:
                    print(f"[reddit_search] r/{sub}: public search query='{query}' sort={sort} time_filter={time_filter} limit={limit_per_sub}", file=sys.stderr)
                pub = _public_search(sub, query, sort, time_filter, limit_per_sub, verbose=verbose)
                results.extend(pub)
                matched += len(pub)
                scanned += len(pub)
                iterator = []
            elif query:
                # Use Reddit's search API first; on auth errors, fall back to public JSON endpoint
                if verbose:
                    print(f"[reddit_search] r/{sub}: search query='{query}' sort={sort} time_filter={time_filter} limit={limit_per_sub}", file=sys.stderr)
                iterator = None
                try:
                    if sr is None:
                        raise RuntimeError("PRAW client not initialized")
                    iterator = sr.search(query=query, sort=sort, time_filter=time_filter, limit=limit_per_sub)
                except Exception as e:  # noqa: BLE001
                    if verbose:
                        print(f"[reddit_search] r/{sub}: praw search error={e} -> falling back to public JSON", file=sys.stderr)
                    pub = _public_search(sub, query, sort, time_filter, limit_per_sub, verbose=verbose)
                    results.extend(pub)
                    matched += len(pub)
                    scanned += len(pub)
                    iterator = []  # Skip PRAW loop below
            else:
                # Fallback to scanning 'new'
                if sr is None:
                    iterator = []
                else:
                    iterator = sr.new(limit=max(10, min(limit_per_sub * 2, 200)))
            try:
                for s in iterator:
                    scanned += 1
                    item = _norm_submission(s)
                    if not query:
                        # Apply local time and keyword filters only in 'new' mode
                        created = item.get("created_utc")
                        if isinstance(created, datetime) and created < since_dt:
                            continue
                        if keywords:
                            text = f"{item.get('title') or ''}\n{item.get('selftext') or ''}".lower()
                            if not any(k.lower() in text for k in keywords):
                                continue
                    matched += 1
                    results.append(item)
            except Exception as e:  # noqa: BLE001
                # If iteration failed (e.g., 401 mid-stream), try public fallback for query mode
                if query:
                    if verbose:
                        print(f"[reddit_search] r/{sub}: iterator error={e} -> falling back to public JSON", file=sys.stderr)
                    pub = _public_search(sub, query, sort, time_filter, limit_per_sub, verbose=verbose)
                    results.extend(pub)
                    matched += len(pub)
                    scanned += len(pub)
            if verbose:
                print(f"[reddit_search] r/{sub}: scanned={scanned} matched={matched}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            endpoint = "subreddit.search" if query else "subreddit.new"
            _log_call(cache["calls_log"], {"endpoint": endpoint, "sub": sub, "error": str(e)})
            if verbose:
                print(f"[reddit_search] r/{sub}: error={e}", file=sys.stderr)
            continue

    # Rank by score and recency
    if query:
        # Assume Reddit returns already sorted by 'sort', but apply a light stable sort by score as a tie-breaker
        results.sort(key=lambda r: (r.get("score", 0), r.get("created_utc") or datetime.fromtimestamp(0, tz=timezone.utc)), reverse=True)
    else:
        results.sort(key=lambda r: (r.get("score", 0), r.get("created_utc") or datetime.fromtimestamp(0, tz=timezone.utc)), reverse=True)
    topn = results[: max(1, limit_per_sub) * max(1, len(subreddits))]

    if use_cache:
        _write_search_cache(cache["search_dir"], cache_key, topn)
        if verbose:
            print(
                f"[reddit_search] cache write: dir={cache['search_dir']} | key={cache_key[:8]}... | n={len(topn)}",
                file=sys.stderr,
            )
    return topn


def _flatten_comments(submission, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        submission.comments.replace_more(limit=0)
    except Exception:
        pass
    out: List[Dict[str, Any]] = []
    count = 0
    for c in submission.comments.list():
        out.append({
            "id": c.id,
            "author": str(c.author) if getattr(c, "author", None) else None,
            "body": getattr(c, "body", None),
            "score": getattr(c, "score", 0),
            "created_utc": datetime.fromtimestamp(getattr(c, "created_utc", 0), tz=timezone.utc),
            "permalink": f"https://www.reddit.com{c.permalink}" if getattr(c, "permalink", None) else None,
        })
        count += 1
        if limit and count >= limit:
            break
    return out


def _fetch_crossposts(reddit, submission) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        for x in submission.crossposts():  # type: ignore[attr-defined]
            items.append(_norm_submission(x))
    except Exception:
        # Fallback: search by url may find crossposts
        try:
            q = f"url:'{submission.url}'"
            for s in reddit.subreddit("all").search(q, sort="new", limit=50):
                items.append(_norm_submission(s))
        except Exception:
            pass
    return items


def collect_event_bundle(
    submission_id: str,
    comments_limit: int = 200,
    include_crossposts: bool = True,
    author_history: bool = False,
    author_limit: int = 50,
    config_path: str = "config.yaml",
    save_bundle: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    creds = resolve_reddit_creds(cfg)
    reddit = _client(creds)

    s = reddit.submission(id=submission_id)
    s_data = _norm_submission(s)

    comments = _flatten_comments(s, limit=comments_limit)
    crossposts = _fetch_crossposts(reddit, s) if include_crossposts else []

    author_recent = None
    if author_history and getattr(s, "author", None):
        try:
            user = reddit.redditor(str(s.author))
            recent_posts = []
            for ps in user.submissions.new(limit=author_limit):
                recent_posts.append(_norm_submission(ps))
            recent_comments = []
            for pc in user.comments.new(limit=author_limit):
                recent_comments.append({
                    "id": pc.id,
                    "subreddit": str(pc.subreddit.display_name) if getattr(pc, "subreddit", None) else None,
                    "body": getattr(pc, "body", None),
                    "score": getattr(pc, "score", 0),
                    "created_utc": datetime.fromtimestamp(getattr(pc, "created_utc", 0), tz=timezone.utc),
                    "permalink": f"https://www.reddit.com{pc.permalink}" if getattr(pc, "permalink", None) else None,
                })
            author_recent = {"submissions": recent_posts, "comments": recent_comments}
        except Exception:
            author_recent = None

    bundle: Dict[str, Any] = {
        "submission": s_data,
        "comments": comments,
        "crossposts": crossposts,
        "author_recent": author_recent,
    }

    if save_bundle:
        c = _cfg_cache_settings(cfg)
        base_dir = out_dir or c.get("events_dir", "data/reddit/events")
        ensure_dir(base_dir)
        path = Path(base_dir) / f"{submission_id}.json"
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reddit helper: search and event bundles.")
    subparsers = parser.add_subparsers(dest="cmd")

    p_search = subparsers.add_parser("search", help="Fetch recent posts from subreddits and rank locally")
    p_search.add_argument("subs", help="Comma-separated subreddits, e.g., health,science")
    p_search.add_argument("--limit-per-sub", type=int, default=25, help="Max results per subreddit (default: 25)")
    p_search.add_argument("--since-hours", type=int, default=24, help="Hours back to include (default: 24)")
    p_search.add_argument("--keywords", type=str, default=None, help="Comma-separated keywords for filtering")
    p_search.add_argument("--query", type=str, default=None, help="Use Reddit search API with this query (bypasses --keywords filtering)")
    p_search.add_argument("--sort", type=str, default="top", help="Search sort: relevance, hot, top, new, comments (default: top)")
    p_search.add_argument("--time-filter", type=str, default="week", help="Search time filter: hour, day, week, month, year, all (default: week)")
    p_search.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    p_search.add_argument("--json", action="store_true", help="Output JSON instead of text")
    p_search.add_argument("--use-cache", action="store_true", help="Use and populate on-disk cache for this query")
    p_search.add_argument("--out", type=str, default=None, help="Write JSON results to this file path (ensures directory)")
    p_search.add_argument("--verbose", action="store_true", help="Print diagnostic info to stderr")

    p_bundle = subparsers.add_parser("bundle", help="Collect a Reddit event bundle around a submission")
    p_bundle.add_argument("submission_id", help="Target submission ID")
    p_bundle.add_argument("--comments-limit", type=int, default=200, help="Max comments to collect (default: 200)")
    p_bundle.add_argument("--include-crossposts", action="store_true", help="Include crossposts where possible")
    p_bundle.add_argument("--author-history", action="store_true", help="Include author's recent submissions and comments")
    p_bundle.add_argument("--author-limit", type=int, default=50, help="Max items for author history lists")
    p_bundle.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    p_bundle.add_argument("--json", action="store_true", help="Output JSON (recommended)")
    p_bundle.add_argument("--save-bundle", action="store_true", help="Persist bundle JSON to events cache directory")
    p_bundle.add_argument("--out-dir", type=str, default=None, help="Override events directory when saving bundle")

    if argv is None:
        argv = sys.argv[1:]
    if not argv or (argv and argv[0] not in {"search", "bundle"}):
        argv = ["search", *argv]

    args = parser.parse_args(argv)

    try:
        if args.cmd == "bundle":
            bundle = collect_event_bundle(
                submission_id=args.submission_id,
                comments_limit=args.comments_limit,
                include_crossposts=args.include_crossposts,
                author_history=args.author_history,
                author_limit=args.author_limit,
                config_path=args.config,
                save_bundle=args.save_bundle,
                out_dir=args.out_dir,
            )
            if args.json:
                def _ser(o):
                    if isinstance(o, datetime):
                        return o.isoformat()
                    return o
                print(json.dumps(bundle, default=_ser, ensure_ascii=False, indent=2))
            else:
                s = bundle.get("submission")
                print(f"{s.get('subreddit')}: {s.get('title')}\n{s.get('permalink')}")
                print(f"Comments: {len(bundle.get('comments', []))}, Crossposts: {len(bundle.get('crossposts', []))}")
        else:
            subs = [s.strip() for s in (args.subs or "").split(",") if s.strip()]
            keywords = [k.strip() for k in (args.keywords or "").split(",") if k.strip()] if args.keywords else None
            if args.verbose:
                print(
                    (
                        f"[reddit_search] subs={args.subs} | limit_per_sub={args.limit_per_sub} | "
                        f"since_hours={args.since_hours} | keywords={keywords} | use_cache={args.use_cache}"
                    ),
                    file=sys.stderr,
                )
            results = search_reddit(
                subreddits=subs,
                limit_per_sub=args.limit_per_sub,
                since_hours=args.since_hours,
                keywords=keywords,
                query=args.query,
                sort=args.sort,
                time_filter=args["time_filter"] if isinstance(args, dict) else args.time_filter,
                config_path=args.config,
                use_cache=args.use_cache,
                verbose=args.verbose,
            )
            if args.verbose:
                print(f"[reddit_search] total_results={len(results)}", file=sys.stderr)
            if args.out:
                try:
                    out_path = Path(args.out)
                    ensure_dir(str(out_path.parent))
                    def _ser(o):
                        if isinstance(o, datetime):
                            return o.isoformat()
                        return o
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, default=_ser, ensure_ascii=False, indent=2)
                    print(str(out_path))
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    return 1
            elif args.json:
                def _ser(o):
                    if isinstance(o, datetime):
                        return o.isoformat()
                    return o
                print(json.dumps(results, default=_ser, ensure_ascii=False, indent=2))
            else:
                for i, r in enumerate(results, 1):
                    dt = r.get("created_utc")
                    ts = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else str(dt)
                    print(f"{i:02d}. r/{r.get('subreddit')} | ⬆️ {r.get('score', 0)} | 💬 {r.get('num_comments', 0)} | {ts}\n    {r.get('title')}\n    {r.get('permalink')}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
