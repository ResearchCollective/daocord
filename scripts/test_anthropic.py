#!/usr/bin/env python3
"""
Quick connectivity test for Anthropic API.

Usage:
  python scripts/test_anthropic.py \
    --model claude-3-5-sonnet-20241022 \
    --message "Say hello"

Environment variables respected:
  - ANTHROPIC_API_KEY (preferred if not provided via config)

Optional: you can also run without args; it will use a default model and message.
This script prints diagnostics for common TLS/SSL and network errors.
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
from typing import Optional

# Third-party
try:
    import anthropic
except Exception as e:  # pragma: no cover
    print("[error] Failed to import anthropic. Did you install requirements? pip install anthropic", file=sys.stderr)
    raise

try:
    import httpx
except Exception:
    httpx = None  # only for nicer error typing


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')


def resolve_api_key() -> Optional[str]:
    # Prefer env var; if your project stores it in config.yaml, set env prior to running
    return os.getenv("ANTHROPIC_API_KEY")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]

    parser = argparse.ArgumentParser(description="Test Anthropic connectivity")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Anthropic model to call")
    parser.add_argument("--message", default="Say hello in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    api_key = resolve_api_key()
    if not api_key:
        logging.warning("ANTHROPIC_API_KEY is not set. Set it in your environment before running.")

    logging.info("Testing Anthropic: model=%s, has_api_key=%s", args.model, bool(api_key))

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        # Minimal test message
        resp = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            system=(
                "You are a simple connectivity test. Reply with a short confirmation."
            ),
            messages=[
                {"role": "user", "content": args.message}
            ],
        )

        # Try to extract plain text
        try:
            text = "".join([b.text for b in resp.content])
        except Exception:
            text = str(resp)

        logging.info("SUCCESS: Received response (%d chars)", len(text))
        print("\n--- Anthropic Response ---\n" + text + "\n--------------------------\n")
        return 0

    except anthropic.APIConnectionError as e:
        logging.error("APIConnectionError: %s", e)
        logging.error("This is often a TLS/SSL or proxy problem on the local machine.")
        logging.error("Hints: (1) `pip install --upgrade certifi`, (2) ensure system time is correct, (3) check VPN/Proxy/AV.")
        return 2
    except anthropic.APIStatusError as e:
        logging.error("APIStatusError: %s", e)
        if hasattr(e, 'response'):
            logging.error("Status code: %s", getattr(e.response, 'status_code', '?'))
            logging.error("Body: %s", getattr(e.response, 'text', ''))
        return 3
    except Exception as e:
        etype = type(e).__name__
        logging.exception("Unexpected error (%s)", etype)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
