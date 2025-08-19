import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    # When executed as a package module: python -m tools.x_monitor
    from .x_search import (
        load_config,
        resolve_env_tokens,
        search_x,
        collect_event_bundle,
    )
    from .x_search import _cfg_cache_settings as cfg_cache_settings  # reuse config helpers
    from .x_search import _ensure_dir as ensure_dir
except Exception:  # noqa: BLE001
    # When executed as a script: python tools/x_monitor.py
    from x_search import (
        load_config,
        resolve_env_tokens,
        search_x,
        collect_event_bundle,
    )
    from x_search import _cfg_cache_settings as cfg_cache_settings  # type: ignore
    from x_search import _ensure_dir as ensure_dir  # type: ignore

# Simple heuristic sentiment lexicon (placeholder)
POS_WORDS = {
    "good", "great", "excellent", "positive", "beneficial", "win", "success", "love", "like",
    "amazing", "awesome", "fantastic", "improve", "improved", "best", "progress", "support"
}
NEG_WORDS = {
    "bad", "terrible", "awful", "negative", "harmful", "fail", "worse", "hate", "dislike",
    "problem", "issue", "concern", "risk", "bug", "broken", "delay", "down"
}


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


def _bundle_sentiment(bundle: Dict[str, Any]) -> Dict[str, Any]:
    texts: List[str] = []
    if bundle.get("original"):
        texts.append(bundle["original"].get("text") or "")
    for r in bundle.get("self_replies", []) + bundle.get("quote_tweets", []):
        texts.append(r.get("text") or "")
    scores = [(_text_sentiment_score(t)) for t in texts if t]
    total = sum(scores) if scores else 0
    avg = (total / len(scores)) if scores else 0.0
    label = "neutral"
    if avg > 0.75:
        label = "positive"
    elif avg < -0.75:
        label = "negative"
    return {"total": total, "avg": avg, "label": label}


def _engagement(m: Dict[str, Any]) -> int:
    return int(m.get("like_count", 0)) + int(m.get("retweet_count", 0)) + int(m.get("reply_count", 0)) + int(m.get("quote_count", 0))


def _passes_filters(tweet: Dict[str, Any], filt: Dict[str, Any]) -> bool:
    metrics = tweet.get("metrics", {})
    text = (tweet.get("text") or "").lower()

    if filt.get("min_like", 0) and metrics.get("like_count", 0) < filt.get("min_like", 0):
        return False
    if filt.get("min_retweet", 0) and metrics.get("retweet_count", 0) < filt.get("min_retweet", 0):
        return False
    if filt.get("min_reply", 0) and metrics.get("reply_count", 0) < filt.get("min_reply", 0):
        return False
    if filt.get("min_quote", 0) and metrics.get("quote_count", 0) < filt.get("min_quote", 0):
        return False

    include_any = [s.lower() for s in filt.get("include_any", [])]
    exclude_any = [s.lower() for s in filt.get("exclude_any", [])]

    if include_any:
        if not any(s in text for s in include_any):
            return False
    if exclude_any:
        if any(s in text for s in exclude_any):
            return False

    return True


def analyze_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    original = bundle.get("original") or {}
    self_replies = bundle.get("self_replies", [])
    quotes = bundle.get("quote_tweets", [])

    # Engagements
    original_eng = _engagement(original.get("metrics", {})) if original else 0
    self_total = sum(_engagement(r.get("metrics", {})) for r in self_replies)
    quotes_total = sum(_engagement(r.get("metrics", {})) for r in quotes)

    # Sentiment
    sentiment = _bundle_sentiment(bundle)

    # Top contributors (by username) among quotes
    author_counts: Dict[str, int] = {}
    for r in quotes:
        u = (r.get("author") or {}).get("username") or ""
        if u:
            author_counts[u] = author_counts.get(u, 0) + 1
    top_quote_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "original_id": original.get("id"),
        "original_author": (original.get("author") or {}).get("username"),
        "created_at": original.get("created_at"),
        "counts": {
            "self_replies": len(self_replies),
            "quote_tweets": len(quotes),
        },
        "engagements": {
            "original": original_eng,
            "self_replies_total": self_total,
            "quotes_total": quotes_total,
        },
        "sentiment": sentiment,
        "top_quote_authors": top_quote_authors,
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
    name = monitor.get("name") or "monitor"
    query = monitor.get("query") or ""
    lang = monitor.get("lang")
    search_limit = int(monitor.get("search_limit", 10))
    filters = monitor.get("filters", {}) or {}
    bundle_limits = monitor.get("bundle", {}) or {}
    self_limit = int(bundle_limits.get("self_limit", 10))
    quotes_limit = int(bundle_limits.get("quotes_limit", 10))
    since_days = int(monitor.get("since_days", 7))

    # Search
    results = search_x(
        query=query,
        limit=search_limit,
        lang=lang,
        since_days=since_days,
        config_path=config_path,
        use_cache=use_cache,
    )

    # Filter and optionally cap the number of events we build
    passed: List[Dict[str, Any]] = [r for r in results if _passes_filters(r, filters)]
    if max_events is not None:
        passed = passed[:max_events]

    # Build bundles
    bundles: List[Dict[str, Any]] = []
    if not dry_run:
        for r in passed:
            bundle = collect_event_bundle(
                tweet_id=r.get("id"),
                self_replies_limit=self_limit,
                quotes_limit=quotes_limit,
                since_days=max(since_days, 7),
                config_path=config_path,
                save_bundle=save_events,
                out_dir=events_dir,
            )
            bundles.append(bundle)

    # Analyze
    analyses = [analyze_bundle(b) for b in bundles]

    # Summary aggregates
    total_events = len(bundles)
    total_original_eng = sum(a["engagements"]["original"] for a in analyses)
    avg_original_eng = (total_original_eng / total_events) if total_events else 0

    report = {
        "monitor": name,
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "search_returned": len(results),
        "passed_filter": len(passed),
        "events": analyses,
        "aggregates": {
            "total_events": total_events,
            "avg_original_engagement": avg_original_eng,
        },
    }
    return report


def load_monitors_from_config(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    cfg = load_config(config_path)
    tw = (cfg or {}).get("twitter", {}) or {}
    monitors = tw.get("monitors", []) or []
    # Sample structure:
    # twitter:
    #   monitors:
    #     - name: longevity-dogs
    #       query: "longevity dogs -is:retweet -is:reply lang:en"
    #       lang: en
    #       search_limit: 10
    #       since_days: 7
    #       filters:
    #         min_like: 5
    #         include_any: ["trial", "study", "paper"]
    #         exclude_any: ["giveaway"]
    #       bundle:
    #         self_limit: 10
    #         quotes_limit: 10
    return monitors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Configurable X (Twitter) monitors: search, filter, event bundle, analyze.")
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
        print("No monitors defined under twitter.monitors in config.")
        return 1

    reports: List[Dict[str, Any]] = []

    if args.all:
        to_run = monitors
    else:
        if not args.name:
            print("Available monitors:")
            for m in monitors:
                print(f"- {m.get('name')}: {m.get('query')}")
            return 0
        found = [m for m in monitors if m.get("name") == args.name]
        if not found:
            print(f"Monitor not found: {args.name}")
            return 1
        to_run = found

    for m in to_run:
        # Determine events directory
        ev_dir = args.events_dir or cache_cfg.get("events_dir", "data/events")
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
        # If no output path provided, write to default reports directory with timestamped filename
        monitor_out_dir = (cfg.get("twitter", {}) or {}).get("monitor", {}) or {}
        default_reports_dir = monitor_out_dir.get("out_dir", "reports")
        try:
            ensure_dir(default_reports_dir)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            fname = f"monitor_{ts}.json"
            out_path = os.path.join(default_reports_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(reports if len(reports) > 1 else reports[0], f, ensure_ascii=False, indent=2, default=str)
            print(json.dumps({"wrote": out_path}, ensure_ascii=False))
        except Exception:
            # fallback to stdout
            print(json.dumps(reports if len(reports) > 1 else reports[0], ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
