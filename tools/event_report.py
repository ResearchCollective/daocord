import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# Optional: google-generativeai (Gemini)
try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return _expand(raw)


def read_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_source(bundle: Dict[str, Any], explicit: Optional[str]) -> str:
    if explicit:
        return explicit.lower()
    # Heuristic: X bundle keys vs Reddit structure
    # X bundles we create have keys: original, self_replies, quote_tweets, replies
    if set(bundle.keys()) & {"original", "self_replies", "quote_tweets", "replies"}:
        return "x"
    # Reddit path might contain submissions/comments fields if implemented later
    return "unknown"


def extract_text_blocks(bundle: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if source == "x":
        def _author(d: Optional[Dict[str, Any]]):
            a = (d or {}).get("author") or {}
            return {
                "id": a.get("id"),
                "username": a.get("username"),
                "name": a.get("name"),
            }
        # original
        if bundle.get("original"):
            b = bundle["original"]
            blocks.append({
                "type": "original",
                "text": b.get("text") or "",
                "author": _author(b),
                "url": b.get("url"),
            })
        # quotes
        for q in bundle.get("quote_tweets", []) or []:
            blocks.append({
                "type": "quote",
                "text": q.get("text") or "",
                "author": _author(q),
                "url": q.get("url"),
            })
        # self replies
        for s in bundle.get("self_replies", []) or []:
            blocks.append({
                "type": "self_reply",
                "text": s.get("text") or "",
                "author": _author(s),
                "url": s.get("url"),
            })
        # general replies
        for r in bundle.get("replies", []) or []:
            blocks.append({
                "type": "reply",
                "text": r.get("text") or "",
                "author": _author(r),
                "url": r.get("url"),
            })
    else:
        # Generic fallback: walk common containers
        for key in ("original", "quotes", "comments", "replies", "items"):
            val = bundle.get(key)
            if isinstance(val, dict):
                txt = (val or {}).get("text") or ""
                if txt:
                    blocks.append({"type": key, "text": txt, "author": (val or {}).get("author")})
            elif isinstance(val, list):
                for it in val:
                    txt = (it or {}).get("text") or ""
                    if txt:
                        blocks.append({"type": key[:-1] if key.endswith('s') else key, "text": txt, "author": (it or {}).get("author")})
    return blocks


def build_prompt(blocks: List[Dict[str, Any]], source: str) -> str:
    # Truncate total characters to keep prompt manageable (~14k chars)
    max_chars = 14000
    def shorten(s: str) -> str:
        return s if len(s) <= 800 else s[:780] + " …"

    lines: List[str] = []
    lines.append("You are an analyst creating a concise research note about developments in canine longevity.")
    lines.append("Goal: extract companies, people (and roles), interventions/treatments tried, and summarize interesting developments.")
    lines.append("Source platform: %s" % source.upper())
    lines.append("")
    lines.append("Input posts (type | author | text):")

    used = 0
    for b in blocks:
        author = b.get("author") or {}
        label = f"{b.get('type','post')} | @{author.get('username') or author.get('name') or author.get('id') or 'unknown'}"
        text = shorten(b.get("text") or "")
        line = f"- {label}: {text}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)

    lines.append("")
    lines.append("Produce JSON with this schema:")
    lines.append("{"
                 "\n  \"companies\": [ { \"name\": str, \"notes\": str } ],"
                 "\n  \"people\": [ { \"name\": str, \"role\": str, \"affiliation\": str } ],"
                 "\n  \"interventions\": [ { \"name\": str, \"type\": str, \"notes\": str } ],"
                 "\n  \"developments_summary\": str,"
                 "\n  \"notable_posts\": [ { \"type\": str, \"author\": str, \"url\": str } ]\n}")
    lines.append("Rules: be factual, cite only from input; if unknown, leave fields empty or omit. Keep summary under 200 words.")

    return "\n".join(lines)


def call_gemini(cfg: Dict[str, Any], prompt: str, model_name: Optional[str] = None) -> str:
    providers = (cfg or {}).get("providers", {}) or {}
    google_cfg = providers.get("google", {}) if isinstance(providers.get("google"), dict) else {}
    api_key = google_cfg.get("api_key") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Google Gemini API key missing. Set providers.google.api_key in config.yaml or GOOGLE_API_KEY env var.")
    if genai is None:
        raise RuntimeError("google-generativeai not installed. Please install dependencies from requirements.txt.")

    genai.configure(api_key=api_key)
    model_name = model_name or "gemini-2.5-pro"
    model = genai.GenerativeModel(model_name)

    resp = model.generate_content(prompt)
    # Prefer text candidates
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # Fallback to first candidate plain text
    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        return json.dumps({"error": "No text response from model"})


def call_anthropic(cfg: Dict[str, Any], prompt: str, model_name: Optional[str] = None) -> str:
    providers = (cfg or {}).get("providers", {}) or {}
    anth_cfg = providers.get("anthropic", {}) if isinstance(providers.get("anthropic"), dict) else {}
    api_key = anth_cfg.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Anthropic API key missing. Set providers.anthropic.api_key or ANTHROPIC_API_KEY.")
    if anthropic is None:
        raise RuntimeError("anthropic package not installed. Please install dependencies from requirements.txt.")

    client = anthropic.Anthropic(api_key=api_key)
    model_name = model_name or "claude-3.5-sonnet"
    resp = client.messages.create(
        model=model_name,
        max_tokens=768,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        texts = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                texts.append(block.text)
        return "\n".join(texts) if texts else json.dumps({"error": "No text response from model"})
    except Exception:
        return json.dumps({"error": "Failed to parse Anthropic response"})


def call_openai_compatible(cfg: Dict[str, Any], provider: str, prompt: str, model_name: Optional[str] = None) -> str:
    providers = (cfg or {}).get("providers", {}) or {}
    p_cfg = providers.get(provider, {}) if isinstance(providers.get(provider), dict) else {}
    base_url = p_cfg.get("base_url")
    api_key = p_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not base_url:
        raise RuntimeError(f"Provider '{provider}' missing base_url in config.yaml under providers.{provider}.")
    if not api_key:
        raise RuntimeError(f"Provider '{provider}' missing api_key (or OPENAI_API_KEY).")
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please install dependencies from requirements.txt.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    model_name = model_name or "gpt-4o-mini"
    resp = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
    try:
        return resp.choices[0].message.content or json.dumps({"error": "No text response from model"})
    except Exception:
        return json.dumps({"error": "Failed to parse OpenAI-compatible response"})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an event report from an event bundle JSON (provider selected via config).")
    parser.add_argument("--input", required=True, help="Path to bundle JSON (e.g., data/x/events/<id>.json)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default=None, help="Override model name (otherwise uses model from the first entry in config.models)")
    parser.add_argument("--source", default=None, choices=["x", "reddit"], help="Override inferred source platform")
    parser.add_argument("--out", default=None, help="Output path for report (.md). If not set, write next to input.")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only (no markdown wrapper)")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        cfg = load_config(args.config)
        bundle = read_bundle(args.input)
        source = infer_source(bundle, args.source)
        blocks = extract_text_blocks(bundle, source)
        prompt = build_prompt(blocks, source)
        # Strictly use the first configured model key to determine provider/model unless --model overrides the model name
        model_key = next(iter(cfg.get("models", {}) or {"openai/gpt-4o-mini": {}}))
        provider, conf_model = model_key.split("/", 1) if "/" in model_key else ("openai", model_key)
        sel_model = args.model or conf_model

        prov_lower = provider.lower()
        if prov_lower in ("gemini", "google"):
            result_text = call_gemini(cfg, prompt, model_name=sel_model)
        elif prov_lower == "anthropic":
            result_text = call_anthropic(cfg, prompt, model_name=sel_model)
        else:
            result_text = call_openai_compatible(cfg, prov_lower, prompt, model_name=sel_model)

        # If the model returned JSON text inside code fences, try to strip
        def _strip_code_fences(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                tmp = s[3:]
                # Drop optional language label
                if "\n" in tmp:
                    tmp = tmp.split("\n", 1)[1]
                else:
                    return tmp.strip()
                # Remove trailing fence if present
                if tmp.endswith("```"):
                    tmp = tmp[:-3]
                return tmp.strip()
            return s

        cleaned = _strip_code_fences(result_text)

        # Compose markdown unless JSON only requested
        if args.json_only:
            output_md = cleaned
        else:
            output_md = "\n".join([
                "# Canine Longevity Event Report",
                f"Source: {source.upper()}",
                "",
                "## Extracted Findings (JSON)",
                "",
                "```json",
                cleaned,
                "```",
            ])

        out_path = args.out
        if not out_path:
            base = Path(args.input)
            out_dir = base.parent.parent / ".." / ".." / "reports"
            out_dir = out_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"event_{base.stem}.md"
        else:
            out_dir = Path(out_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_md)

        print(str(out_path))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
