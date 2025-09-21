import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    # When executed as a package module: python -m tools.reddit_monitor
    from .reddit_search import (
        search_reddit,
        collect_event_bundle,
        _cfg_cache_settings as cfg_cache_settings,
        ensure_dir,
    )
    from .x_search import load_config  # reuse same loader
except Exception:  # noqa: BLE001
    # When executed as a script: python tools/reddit_monitor.py
    from reddit_search import (
        search_reddit,
        collect_event_bundle,
        _cfg_cache_settings as cfg_cache_settings,  # type: ignore
        ensure_dir,  # type: ignore
    )
    from x_search import load_config  # type: ignore


# Simple keyword sentiment heuristic (placeholder)
POS_WORDS = {"good", "great", "excellent", "love", "amazing", "helpful", "win", "best"}
NEG_WORDS = {"bad", "terrible", "awful", "hate", "worse", "broken", "bug", "down"}


def _text_sentiment_score(text: str) -> int:
    if not text:
        return 0
    t = text.lower()
    score = 0
    for w in POS_WORDS:
        if w in t:
            score += 1
    for w in NEG_WORDS:
        if w in t:
            score -= 1
    return score


def _passes_filters(post: Dict[str, Any], filt: Dict[str, Any]) -> bool:
    score = int(post.get("score", 0))
    n_comments = int(post.get("num_comments", 0))
    text = f"{post.get('title') or ''}\n{post.get('selftext') or ''}".lower()

    if filt.get("min_score", 0) and score < int(filt.get("min_score", 0)):
        return False
    if filt.get("min_comments", 0) and n_comments < int(filt.get("min_comments", 0)):
        return False

    include_any = [s.lower() for s in filt.get("include_any", [])]
    exclude_any = [s.lower() for s in filt.get("exclude_any", [])]

    if include_any and not any(s in text for s in include_any):
        return False
    if exclude_any and any(s in text for s in exclude_any):
        return False

    return True


def analyze_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    sub = bundle.get("submission") or {}
    comments = bundle.get("comments", [])
    crossposts = bundle.get("crossposts", [])

    # Sentiment over title + selftext + top-level comments bodies (sample)
    texts: List[str] = []
    texts.append((sub.get("title") or "") + "\n" + (sub.get("selftext") or ""))
    for c in comments[:200]:
        texts.append(c.get("body") or "")
    scores = [(_text_sentiment_score(t)) for t in texts if t]
    total = sum(scores) if scores else 0
    avg = (total / len(scores)) if scores else 0.0
    label = "neutral"
    if avg > 0.75:
        label = "positive"
    elif avg < -0.75:
        label = "negative"

    # Top commenters
    author_counts: Dict[str, int] = {}
    for c in comments:
        a = c.get("author") or ""
        if a:
            author_counts[a] = author_counts.get(a, 0) + 1
    top_commenters = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "submission_id": sub.get("id"),
        "subreddit": sub.get("subreddit"),
        "author": sub.get("author"),
        "created_utc": sub.get("created_utc"),
        "counts": {
            "comments": len(comments),
            "crossposts": len(crossposts),
        },
        "score": sub.get("score", 0),
        "sentiment": {"total": total, "avg": avg, "label": label},
        "top_commenters": top_commenters,
        "permalink": sub.get("permalink"),
        "url": sub.get("url"),
    }


def run_monitor(
    monitor: Dict[str, Any],
    config_path: str = "config.yaml",
    dry_run: bool = False,
    max_events: Optional[int] = None,
    use_cache: bool = False,
    save_events: bool = False,
    events_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single monitor definition and return a report."""
    name = monitor.get("name") or "reddit-monitor"
    subs = monitor.get("subreddits") or monitor.get("subs") or []
    if isinstance(subs, str):
        subs = [s.strip() for s in subs.split(",") if s.strip()]
    since_hours = int(monitor.get("since_hours", 24))
    limit_per_sub = int(monitor.get("limit_per_sub", 25))
    keywords = monitor.get("keywords")  # list[str] optional
    filters = monitor.get("filters", {}) or {}

    # Get config for exclude_keywords
    cfg = load_config(config_path)
    rd_cfg = (cfg or {}).get("reddit", {}) or {}
    exclude_keywords = rd_cfg.get("exclude_keywords", [])

    # Search recent posts per subreddit
    results = search_reddit(
        subreddits=subs,
        limit_per_sub=limit_per_sub,
        since_hours=since_hours,
        keywords=keywords,
        config_path=config_path,
        use_cache=use_cache,
    )

    # Apply exclude_keywords filtering if specified
    if exclude_keywords:
        filtered_results = []
        for result in results:
            text = f"{result.get('title') or ''}\n{result.get('selftext') or ''}".lower()
            if not any(exclude_kw.lower() in text for exclude_kw in exclude_keywords):
                filtered_results.append(result)
        results = filtered_results

    # Filter and cap
    passed: List[Dict[str, Any]] = [r for r in results if _passes_filters(r, filters)]
    if max_events is not None:
        passed = passed[:max_events]

    # Build bundles
    bundles: List[Dict[str, Any]] = []
    if not dry_run:
        for r in passed:
            b = collect_event_bundle(
                submission_id=r.get("id"),
                comments_limit=int(monitor.get("comments_limit", 200)),
                include_crossposts=bool(monitor.get("include_crossposts", True)),
                author_history=bool(monitor.get("author_history", False)),
                author_limit=int(monitor.get("author_limit", 50)),
                config_path=config_path,
                save_bundle=save_events,
                out_dir=events_dir,
            )
            bundles.append(b)

    analyses = [analyze_bundle(b) for b in bundles]

    report = {
        "monitor": name,
        "subreddits": subs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "search_returned": len(results),
        "passed_filter": len(passed),
        "events": analyses,
        "aggregates": {
            "total_events": len(bundles),
            "avg_score": (sum((a.get("score") or 0) for a in analyses) / len(analyses)) if analyses else 0,
        },
    }
    return report


def load_monitors_from_config(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    cfg = load_config(config_path)
    rd = (cfg or {}).get("reddit", {}) or {}
    monitors = rd.get("monitors", []) or []
    return monitors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Configurable Reddit monitors: fetch, filter, bundle, analyze.")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run one monitor by name or all from config")
    p_run.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p_run.add_argument("--name", help="Monitor name to run (if omitted and --all not set, list monitors)")
    p_run.add_argument("--all", action="store_true", help="Run all monitors defined in config")
    p_run.add_argument("--dry-run", action="store_true", help="Do not build bundles; just show counts")
    p_run.add_argument("--max-events", type=int, default=None, help="Cap number of events (bundles) to build")
    p_run.add_argument("--json-out", help="Write JSON report to this file path")
    p_run.add_argument("--use-cache", action="store_true", help="Use on-disk cache for searches (reduces reads)")
    p_run.add_argument("--save-events", action="store_true", help="Persist each built event bundle to events dir")
    p_run.add_argument("--events-dir", type=str, default=None, help="Override events directory for saved bundles")

    args = parser.parse_args(argv)

    if args.cmd != "run":
        parser.print_help()
        return 0

    monitors = load_monitors_from_config(args.config)
    cfg = load_config(args.config)
    cache_cfg = cfg_cache_settings(cfg)
    if not monitors:
        print("No monitors defined under reddit.monitors in config.")
        return 1

    if args.all:
        to_run = monitors
    else:
        if not args.name:
            print("Available monitors:")
            for m in monitors:
                subs = m.get("subreddits") or m.get("subs")
                print(f"- {m.get('name')}: {subs}")
            return 0
        found = [m for m in monitors if m.get("name") == args.name]
        if not found:
            print(f"Monitor not found: {args.name}")
            return 1
        to_run = found

    reports: List[Dict[str, Any]] = []
    for m in to_run:
        ev_dir = args.events_dir or cache_cfg.get("events_dir", "data/reddit/events")
        if args.save_events:
            ensure_dir(ev_dir)
        report = run_monitor(
            m,
            config_path=args.config,
            dry_run=args.dry_run,
            max_events=args.max_events,
            use_cache=args.use_cache,
            save_events=args.save_events,
            events_dir=ev_dir,
        )
        reports.append(report)

    # Output
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(reports if len(reports) > 1 else reports[0], f, ensure_ascii=False, indent=2, default=str)
            print(f"Wrote report to {args.json_out}")
        except Exception as e:
            print(f"Failed to write report: {e}", file=sys.stderr)
            return 1
    else:
        monitor_out_dir = (cfg.get("reddit", {}) or {}).get("monitor", {}) or {}
        default_reports_dir = monitor_out_dir.get("out_dir", "reports")
        try:
            ensure_dir(default_reports_dir)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            fname = f"reddit_monitor_{ts}.json"
            out_path = os.path.join(default_reports_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(reports if len(reports) > 1 else reports[0], f, ensure_ascii=False, indent=2, default=str)
            print(json.dumps({"wrote": out_path}, ensure_ascii=False))
        except Exception:
            print(json.dumps(reports if len(reports) > 1 else reports[0], ensure_ascii=False, indent=2, default=str))

    return 0


def monitor_reddit(
    config: Dict[str, Any],
    save_events: bool = True,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Monitor Reddit based on simple configuration.

    Args:
        config: Configuration dictionary
        save_events: Whether to save event bundles to disk
        use_cache: Whether to use search cache

    Returns:
        List of monitor reports
    """
    # Get simple configuration
    rd_cfg = (config or {}).get("reddit", {}) or {}

    # Build monitor config from simple configuration
    subreddits = rd_cfg.get("subreddits", ["all"])
    keywords = rd_cfg.get("keywords", [])
    exclude_keywords = rd_cfg.get("exclude_keywords", [])

    # Create a simple monitor config
    monitor_config = {
        "name": "reddit-monitor",
        "subreddits": subreddits,
        "keywords": keywords,
        "lang": "en",
        "search_limit": rd_cfg.get("search_limit", 15),
        "since_days": rd_cfg.get("since_days", 7),
        "filters": rd_cfg.get("filters", {}),
        "bundle": rd_cfg.get("bundle", {})
    }

    # Run the monitor
    try:
        report = run_monitor(
            monitor_config,
            config_path="",  # Use the passed config
            dry_run=False,
            max_events=None,
            use_cache=use_cache,
            save_events=save_events,
            events_dir="data/reddit/events",
        )
        return [report]
    except Exception as e:
        print(f"Error running Reddit monitor: {e}")
        return []
