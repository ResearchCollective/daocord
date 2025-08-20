import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional
import io

import discord
from discord.app_commands import Choice
from discord.ext import commands
import httpx
from openai import AsyncOpenAI
import google.generativeai as genai
try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None
import yaml
from tools.dao_docs import DAODocsTool
from tools.gdocs_cache import GDocsCache
from aiohttp import client_exceptions
import os
from pathlib import Path
import socket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Reduce noisy network stack traces and gateway chatter
try:
    class NetworkNoiseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.exc_info:
                exc = record.exc_info[1]
                if isinstance(exc, (client_exceptions.ClientConnectorDNSError, socket.gaierror)):
                    record.msg = "Network issue: %s. Auto-reconnect will occur." % (exc.__class__.__name__)
                    record.args = ()
                    record.levelno = logging.WARNING
                    record.levelname = 'WARNING'
            return True

    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        h.addFilter(NetworkNoiseFilter())

    for name in ("aiohttp.client", "aiohttp.connector", "discord.gateway"):
        logging.getLogger(name).setLevel(logging.WARNING)
except Exception:
    # Non-fatal if logging tweak fails
    pass

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " "
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config and expand environment variables in all string values.
    Supports $VAR and ${VAR} patterns anywhere in the string.
    """
    with open(filename, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            # Expand $VARS inside strings; leaves unknown vars unchanged
            return os.path.expandvars(obj)
        return obj

    return _expand(raw)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()
dao_docs = DAODocsTool()
# Initialize optional Google Docs cache (no-op if disabled in config)
try:
    gdocs_cache = GDocsCache(config)
except Exception:
    logging.exception("[gdocs] failed to initialize; continuing without gdocs")

# --- Masked env diagnostics ---
def _mask_info(name: str, val: Any) -> None:
    try:
        if val is None:
            status = "MISSING"
        else:
            s = str(val)
            if s.startswith("$") or ("${" in s):
                status = "UNSET_PLACEHOLDER"
            elif (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                # Likely inline JSON (e.g., expanded from $VAR containing JSON)
                status = f"INLINE_JSON(len={len(s)})"
            else:
                prefix = s[:2]
                status = f"SET({prefix}..., len={len(s)})"
        logging.info("[env] %s: %s", name, status)
    except Exception:
        logging.exception("[env] failed to log %s", name)
    
def _cfg_get(path: str, default: Any = None) -> Any:
    """Fetch nested value from global config by dot path."""
    try:
        cur: Any = config
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur
    except Exception:
        return default

# Log key config/env resolution statuses at startup
_mask_info("bot_token", _cfg_get("bot_token"))
_mask_info("client_id", _cfg_get("client_id"))
_mask_info("providers.anthropic.api_key", _cfg_get("providers.anthropic.api_key"))
_mask_info("providers.google.api_key", _cfg_get("providers.google.api_key"))
_mask_info("gdocs.google_application_credentials", _cfg_get("gdocs.google_application_credentials"))
_mask_info("gdocs.folder_id", _cfg_get("gdocs.folder_id"))

class ReportsCache:
    """Lightweight indexer over markdown reports in a directory."""
    def __init__(self, reports_dir: Path | str = "reports") -> None:
        self.dir = Path(reports_dir)
        self.enabled = self.dir.exists()
        self.entries: list[tuple[Path, str, float]] = []  # (path, text, mtime)

    def refresh(self) -> None:
        try:
            self.enabled = self.dir.exists()
            self.entries = []
            if not self.enabled:
                return
            for p in sorted(self.dir.glob("**/*.md")):
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    mtime = p.stat().st_mtime
                    self.entries.append((p, txt, mtime))
                except Exception:
                    logging.exception("[reports] failed reading %s", p)
        except Exception:
            logging.exception("[reports] refresh failed")

    def top_with_scores(self, query: str, k: int = 10) -> list[tuple[str, str, float]]:
        if not self.enabled or not self.entries:
            return []
        q = query.lower()
        terms = [t for t in (''.join(ch if ch.isalnum() else ' ' for ch in q)).split() if len(t) > 2]
        results: list[tuple[str, str, float]] = []
        now = datetime.now().timestamp()
        for path, text, mtime in self.entries:
            tl = text.lower()
            term_score = sum(tl.count(t) for t in terms) or 0.0
            # Recency boost (up to +2.0 within ~14 days)
            recency_days = max(0.0, (now - mtime) / 86400.0)
            recency = max(0.0, 2.0 - (recency_days / 7.0))
            score = term_score + recency
            if score > 0:
                # Extract a short snippet around the first strongest term
                idx = -1
                for t in terms:
                    idx = tl.find(t)
                    if idx != -1:
                        break
                start = max(0, idx - 200) if idx != -1 else 0
                end = min(len(text), (idx + 200)) if idx != -1 else min(len(text), 400)
                snippet = text[start:end].strip()
                results.append((str(path), snippet, float(score)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

# Initialize reports cache
reports_dir = Path(config.get("reports_dir", "reports"))
reports_cache = ReportsCache(reports_dir)


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.event
async def on_ready():
    try:
        # Kick off an initial refresh so the cache populates at startup
        if getattr(gdocs_cache, "enabled", False):
            logging.info("[gdocs] initial refresh on startup…")
            await asyncio.to_thread(gdocs_cache.refresh)
            logging.info("[gdocs] initial refresh complete")
        # Refresh reports cache as well
        logging.info("[reports] initial refresh on startup…")
        await asyncio.to_thread(reports_cache.refresh)
        logging.info("[reports] initial refresh complete")
    except Exception:
        logging.exception("[gdocs] initial refresh failed")


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()][:24]
    choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []

    return choices


@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_disconnect():
    # Cleaner message when machine sleeps or network drops
    logging.warning("Discord gateway disconnected. Auto-reconnect will occur.")


@discord_bot.event
async def on_resumed():
    logging.info("Discord gateway connection resumed.")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    # Simple echo test mode: reply with ALL-CAPS text and skip LLM
    if config.get("test_echo_mode", False):
        try:
            echo_text = new_msg.content.removeprefix(discord_bot.user.mention).strip()
            if echo_text:
                await new_msg.reply(echo_text.upper(), suppress_embeds=True, silent=True)
        except Exception:
            logging.exception("Echo mode failed")
        return

    # Minimal DAO docs commands/triggers
    content_stripped = new_msg.content.strip()
    lower = content_stripped.lower()
    # Treat @mentions as docs-triggered queries as well
    mentioned = False
    try:
        mentioned = discord_bot.user is not None and discord_bot.user.mentioned_in(new_msg)
    except Exception:
        mentioned = False

    dao_trigger = lower.startswith("dao ") or lower.startswith("!dao ") or mentioned
    if lower.startswith("dao "):
        user_query = content_stripped[4:].strip()
    elif lower.startswith("!dao "):
        user_query = content_stripped[5:].strip()
    elif mentioned:
        # Remove the bot mention from the start if present
        mention_text = getattr(discord_bot.user, "mention", "") or ""
        user_query = content_stripped.replace(mention_text, "").strip()
    else:
        user_query = content_stripped

    # Refresh command: !dao refresh
    if lower.startswith("!dao refresh"):
        try:
            await asyncio.to_thread(dao_docs.refresh)
            await asyncio.to_thread(reports_cache.refresh)
            await new_msg.reply("DAO docs and reports refreshed.", suppress_embeds=True, silent=True)
        except Exception as e:
            await new_msg.reply(f"Failed to refresh docs: {e}", suppress_embeds=True, silent=True)
        return

    # Query trigger: "dao <question>" or "!dao <question>"
    # dao_trigger = False
    # if lower.startswith("dao "):
    #     dao_trigger = True
    #     user_query = content_stripped.split(" ", 1)[1].strip()
    # elif lower.startswith("!dao "):
    #     dao_trigger = True
    #     user_query = content_stripped.split(" ", 1)[1].strip()
    # else:
    #     user_query = content_stripped

    # Always use the first model from config at message time (config is reloaded above)
    provider_slash_model = next(iter(config.get("models", {}) or {"openai/gpt-4o-mini": {}}))
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config.get("providers", {}).get(provider, {})
    base_url = provider_config.get("base_url", None)
    api_key = provider_config.get("api_key", None)

    # Configure Gemini (if present); model selection happens later
    gem_cfg = (config.get("providers", {}).get("gemini", {}) or {})
    goo_cfg = (config.get("providers", {}).get("google", {}) or {})
    gemini_api_key = gem_cfg.get("api_key") or goo_cfg.get("api_key") or os.getenv("GEMINI_API_KEY")
    anth_cfg = (config.get("providers", {}).get("anthropic", {}) or {})
    anthropic_api_key = anth_cfg.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
        except Exception:
            logging.exception("Failed to configure Gemini API key")
    api_key_present = gemini_api_key is not None
    logging.info(f"api_key_present={api_key_present} (source={'GEMINI_API_KEY' if gemini_api_key == os.getenv('GEMINI_API_KEY') else 'providers.gemini' if gemini_api_key == gem_cfg.get('api_key') else 'providers.google'})")

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers", None)
    extra_query = provider_config.get("extra_query", None)
    extra_body = (provider_config.get("extra_body", None) or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config["system_prompt"]:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Optional global system prompt from config (avoid duplicates if already present)
    sys_prompt = (config.get("system_prompt") or "").strip()
    if sys_prompt:
        already_has_sys = any((m.get("role") == "system" and m.get("content") == sys_prompt) for m in messages)
        if not already_has_sys:
            logging.info("Adding system_prompt from config (chars=%d)", len(sys_prompt))
            messages.append(dict(role="system", content=sys_prompt))

    # Inject DAO + optional Google Docs context when triggered
    if dao_trigger:
        try:
            # Build focused context from top-N relevant chunks across sources
            dao_top = await asyncio.to_thread(dao_docs.top_with_scores, user_query, 8)
            merged = [("dao", p, c, s) for (p, c, s) in dao_top]
            rep_top = await asyncio.to_thread(reports_cache.top_with_scores, user_query, 8)
            merged.extend(("report", p, c, s) for (p, c, s) in rep_top)
            if getattr(gdocs_cache, "enabled", False):
                # Refresh gdocs cache if needed and fetch top
                await asyncio.to_thread(gdocs_cache.refresh)
                g_top = await asyncio.to_thread(gdocs_cache.top_with_scores, user_query, 8)
                merged.extend(("gdocs", p, c, s) for (p, c, s) in g_top)

            # Sort by score desc and take top 12 overall
            merged.sort(key=lambda t: t[3], reverse=True)
            merged = merged[:12]

            sources = []
            parts = []
            for src, path, chunk, score in merged:
                try:
                    if src == "dao":
                        rel = os.path.relpath(path, "docs")
                    else:
                        rel = os.path.basename(path)
                except Exception:
                    rel = os.path.basename(path)
                label = f"{src}:{rel}"
                if label not in sources:
                    sources.append(label)
                parts.append(f"### Source: {label} (score {score})\n{chunk}")
            docs_context = "\n\n".join(parts)
            logging.info(
                "Context built from %d chunks (%d unique sources, reports=%s, gdocs=%s), length=%d chars",
                len(merged), len(sources), "T" if getattr(reports_cache, "enabled", False) else "F",
                "T" if getattr(gdocs_cache, "enabled", False) else "F", len(docs_context)
            )

            context_prompt = (
                "SYSTEM INSTRUCTIONS: Answer ONLY using the DAO documentation context below.\n"
                "- If the answer is not present, say it is not covered. Do not speculate.\n"
                "- Be concise: at most 4 short bullet points (max 1 sentence each).\n"
                "- You MUST include a final line exactly: Sources: <comma-separated filenames>.\n\n"
                f"USER QUESTION: {user_query}\n\n"
                f"DAO CONTEXT (top relevant snippets):\n{docs_context}"
            )
            messages.append(dict(role="system", content=context_prompt))
        except Exception:
            logging.exception("Error generating DAO docs context")

    # Determine provider/model strictly from config
    active_provider = provider.lower()
    use_gemini = (active_provider in ("gemini", "google")) and bool(gemini_api_key)
    use_anthropic = (active_provider == "anthropic") and bool(anthropic_api_key)
    gemini_model_name = model

    # Generate and send response message(s)
    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    use_plain_responses = config.get("use_plain_responses", False)
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    # Fast relevance prefilter: if docs/reports don't match the query well, skip LLM
    if dao_trigger:
        try:
            scored_docs = await asyncio.to_thread(dao_docs.top_with_scores, user_query, 3)
            scored_reports = await asyncio.to_thread(reports_cache.top_with_scores, user_query, 3)
            best_score = max((s for _, _, s in (scored_docs + scored_reports)), default=0)
            # Threshold tuned conservatively; adjust as needed
            if best_score < 35:
                not_covered = (
                    "This question does not appear to be covered in the DAO documentation provided.\n"
                    "Please rephrase or ask about topics present in the docs.\n\n"
                    "Sources: none"
                )
                if use_plain_responses:
                    response_msg = await new_msg.reply(content=not_covered[:max_message_length], suppress_embeds=True)
                    response_msgs.append(response_msg)
                else:
                    embed_nc = discord.Embed(description=not_covered[:max_message_length], color=EMBED_COLOR_COMPLETE)
                    response_msg = await new_msg.reply(embed=embed_nc, silent=True)
                    response_msgs.append(response_msg)

                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()
                return
        except Exception:
            logging.exception("Relevance prefilter failed")

    try:
        async with new_msg.channel.typing():
            if use_gemini:
                # Non-streaming path via Gemini; combine messages into a single prompt
                ordered = messages[::-1]
                prompt_text = "\n\n".join([m.get("content", "") for m in ordered])

                # Optional debug: log exact prompt to terminal
                if (config.get("debug_prompt") is True) and prompt_text:
                    logging.info("===== DEBUG PROMPT BEGIN (chars=%d) =====", len(prompt_text))
                    # Print full prompt to terminal
                    print(prompt_text)
                    logging.info("===== DEBUG PROMPT END =====")

                def _gen_gemini():
                    gmodel = genai.GenerativeModel(gemini_model_name)
                    safety_settings = None
                    if config.get("disable_safety") is True:
                        safety_settings = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ]
                    return gmodel.generate_content(
                        prompt_text,
                        generation_config={
                            "temperature": 0.2,
                            "max_output_tokens": 768,
                        },
                        safety_settings=safety_settings,
                    )
                resp = await asyncio.to_thread(_gen_gemini)
                # Safely extract text from Gemini response; handle blocked/empty candidates
                def _extract_text(r):
                    try:
                        for cand in getattr(r, "candidates", []) or []:
                            content = getattr(cand, "content", None)
                            parts = getattr(content, "parts", []) if content else []
                            texts = []
                            for p in parts:
                                t = getattr(p, "text", None)
                                if t:
                                    texts.append(t)
                            if texts:
                                return "\n".join(texts), getattr(cand, "finish_reason", None)
                    except Exception:
                        logging.exception("Failed to parse Gemini response text")
                    return "", None

                text, finish = _extract_text(resp)
                if not text:
                    logging.warning("Gemini returned no text (finish_reason=%s). Retrying with reduced context...", str(finish))
                    # Retry once with fewer chunks and conservative settings
                    try:
                        rescored = await asyncio.to_thread(dao_docs.top_with_scores, user_query, 6)
                        r_sources = []
                        r_parts = []
                        for path, chunk, score in rescored:
                            try:
                                rel = os.path.relpath(path, "docs")
                            except Exception:
                                rel = os.path.basename(path)
                            if rel not in r_sources:
                                r_sources.append(rel)
                            r_parts.append(f"### Source: {rel} (score {score})\n{chunk}")
                        r_docs_context = "\n\n".join(r_parts)
                        r_prompt = (
                            (sys_prompt + "\n\n") if (config.get("system_prompt") and sys_prompt) else ""
                        ) + (
                            "SYSTEM INSTRUCTIONS: Answer ONLY using the DAO documentation context below.\n"
                            "- If the answer is not present, say it is not covered. Do not speculate.\n"
                            "- Be concise: at most 4 short bullet points (max 1 sentence each).\n"
                            "- You MUST include a final line exactly: Sources: <comma-separated filenames>.\n\n"
                            f"USER QUESTION: {user_query}\n\n"
                            f"DAO CONTEXT (top relevant snippets):\n{r_docs_context}"
                        )
                        logging.info("Retry prompt length: %d chars, sources=%s", len(r_prompt), ", ".join(r_sources[:5]))
                        def _gen_gemini_retry():
                            gmodel = genai.GenerativeModel(gemini_model_name)
                            safety_settings = None
                            if config.get("disable_safety") is True:
                                safety_settings = [
                                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                                ]
                            return gmodel.generate_content(
                                r_prompt,
                                generation_config={
                                    "temperature": 0.15,
                                    "max_output_tokens": 512,
                                },
                                safety_settings=safety_settings,
                            )
                        r_resp = await asyncio.to_thread(_gen_gemini_retry)
                        r_text, r_finish = _extract_text(r_resp)
                        if r_text:
                            text = r_text
                            finish = r_finish
                        else:
                            logging.warning("Retry also returned no text (finish_reason=%s). Using friendly notice.", str(r_finish))
                            text = (
                                "The model returned no textual output (possibly due to safety or token limits). "
                                "If this should be answerable, please rephrase or narrow the question."
                            )
                    except Exception:
                        logging.exception("Retry generation failed")
                        text = (
                            "The model returned no textual output (possibly due to safety or token limits). "
                            "If this should be answerable, please rephrase or narrow the question."
                        )
                # Compute sources line from earlier selection (fallback to recompute if empty)
                try:
                    if dao_trigger:
                        # Reuse last computed 'sources' if available in scope; otherwise recompute
                        if 'sources' not in locals() or not sources:
                            rescored = await asyncio.to_thread(dao_docs.top_with_scores, user_query, 5)
                            sources = []
                            for path, _, _ in rescored:
                                try:
                                    rel = os.path.relpath(path, "docs")
                                except Exception:
                                    rel = os.path.basename(path)
                                if rel not in sources:
                                    sources.append(rel)
                    else:
                        sources = []
                except Exception:
                    logging.exception("Failed to gather sources for footer")
                    sources = []

                # Do not append a computed Sources footer; rely on the model output per prompt instructions
                logging.info("Generated response length: %d", len(text or ""))
                # Respect plain vs embed output
                try:
                    text = text[:max_message_length]
                except Exception:
                    pass
                logging.info("Generated response length: %d", len(text or ""))
                # Respect plain vs embed output
                try:
                    if use_plain_responses:
                        response_msg = await new_msg.reply(content=(text or "(no content)"), suppress_embeds=True)
                        response_msgs.append(response_msg)
                        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                        await msg_nodes[response_msg.id].lock.acquire()
                    else:
                        embed.description = text or "(no content)"
                        embed.color = EMBED_COLOR_COMPLETE
                        response_msg = await new_msg.reply(embed=embed, silent=True)
                        response_msgs.append(response_msg)
                        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                        await msg_nodes[response_msg.id].lock.acquire()
                except Exception:
                    logging.exception("Failed to send Discord response message")
            elif use_anthropic:
                # Non-streaming path via Anthropic
                if anthropic is None:
                    raise RuntimeError("anthropic package not installed. Please install dependencies from requirements.txt.")

                # Prepare system + messages for Anthropic
                ordered = messages[::-1]
                system_msgs = [m.get("content", "") for m in ordered if m.get("role") == "system"]
                system_text = "\n\n".join([str(s) for s in system_msgs if s]) or None

                def _to_text_parts(content: Any) -> list[dict[str, str]]:
                    try:
                        if isinstance(content, str):
                            return [{"type": "text", "text": content}]
                        if isinstance(content, list):
                            parts: list[dict[str, str]] = []
                            for p in content:
                                if isinstance(p, dict) and p.get("type") == "text" and p.get("text"):
                                    parts.append({"type": "text", "text": str(p["text"])})
                                elif isinstance(p, dict) and p.get("type") == "image_url":
                                    # Skip images for now in Anthropic path (could be extended to use input_images)
                                    continue
                                elif isinstance(p, str):
                                    parts.append({"type": "text", "text": p})
                            return parts or [{"type": "text", "text": ""}]
                    except Exception:
                        logging.exception("Failed to convert content to Anthropic text parts")
                    return [{"type": "text", "text": str(content) if content is not None else ""}]

                anth_messages = [
                    {"role": m.get("role"), "content": _to_text_parts(m.get("content", ""))}
                    for m in ordered
                    if m.get("role") in ("user", "assistant")
                ]

                def _gen_anthropic():
                    client = anthropic.Anthropic(api_key=anthropic_api_key)
                    return client.messages.create(
                        model=model,
                        max_tokens=768,
                        temperature=0.2,
                        system=system_text,
                        messages=anth_messages,
                    )

                # Call Anthropic with retry/backoff on 429 rate limits
                resp = None
                max_retries = 3
                base_delay = 3  # seconds
                for attempt in range(max_retries):
                    try:
                        resp = await asyncio.to_thread(_gen_anthropic)
                        break
                    except Exception as e:
                        # Handle Anthropic SDK RateLimitError gracefully
                        is_rate_limit = False
                        try:
                            is_rate_limit = (anthropic is not None) and isinstance(e, getattr(anthropic, "RateLimitError", Exception))
                        except Exception:
                            is_rate_limit = False
                        if is_rate_limit and attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            msg = f"Rate limited by Anthropic. Retrying in {delay}s…"
                            try:
                                if use_plain_responses:
                                    notice_msg = await new_msg.reply(content=msg, suppress_embeds=True)
                                    response_msgs.append(notice_msg)
                                    msg_nodes[notice_msg.id] = MsgNode(parent_msg=new_msg)
                                    await msg_nodes[notice_msg.id].lock.acquire()
                                else:
                                    embed_rl = discord.Embed(description=msg, color=EMBED_COLOR_INCOMPLETE)
                                    notice_msg = await new_msg.reply(embed=embed_rl, silent=True)
                                    response_msgs.append(notice_msg)
                                    msg_nodes[notice_msg.id] = MsgNode(parent_msg=new_msg)
                                    await msg_nodes[notice_msg.id].lock.acquire()
                            except Exception:
                                logging.exception("Failed to send rate limit notice (Anthropic)")
                            await asyncio.sleep(delay)
                            continue
                        # Not a rate limit or out of retries -> re-raise
                        raise

                # If still no response after retries, notify user and stop this path
                if resp is None:
                    final_msg = (
                        "Still rate limited by Anthropic after multiple retries. "
                        "Please wait a bit and try again."
                    )
                    try:
                        if use_plain_responses:
                            notice_msg = await new_msg.reply(content=final_msg, suppress_embeds=True)
                            response_msgs.append(notice_msg)
                            msg_nodes[notice_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[notice_msg.id].lock.acquire()
                        else:
                            embed_rl = discord.Embed(description=final_msg, color=EMBED_COLOR_COMPLETE)
                            notice_msg = await new_msg.reply(embed=embed_rl, silent=True)
                            response_msgs.append(notice_msg)
                            msg_nodes[notice_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[notice_msg.id].lock.acquire()
                    except Exception:
                        logging.exception("Failed to send final rate limit notice (Anthropic)")
                    return

                # Extract text from content blocks
                a_text = ""
                try:
                    for block in getattr(resp, "content", []) or []:
                        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                            a_text += (block.text or "")
                except Exception:
                    logging.exception("Failed to parse Anthropic response text")
                text = a_text or ""

                try:
                    text = text[:max_message_length]
                except Exception:
                    pass

                try:
                    if use_plain_responses:
                        response_msg = await new_msg.reply(content=(text or "(no content)"), suppress_embeds=True)
                        response_msgs.append(response_msg)
                        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                        await msg_nodes[response_msg.id].lock.acquire()
                    else:
                        embed.description = text or "(no content)"
                        embed.color = EMBED_COLOR_COMPLETE
                        response_msg = await new_msg.reply(embed=embed, silent=True)
                        response_msgs.append(response_msg)
                        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                        await msg_nodes[response_msg.id].lock.acquire()
                except Exception:
                    logging.exception("Failed to send Discord response message (Anthropic)")
            else:
                # OpenAI-compatible streaming path
                # Optional debug: log exact prompt to terminal (OpenAI path)
                if config.get("debug_prompt") is True:
                    ordered = messages[::-1]
                    prompt_text = "\n\n".join([m.get("content", "") for m in ordered])
                    if prompt_text:
                        logging.info("===== DEBUG PROMPT BEGIN (chars=%d) =====", len(prompt_text))
                        print(prompt_text)
                        logging.info("===== DEBUG PROMPT END =====")
                if not api_key:
                    # Friendly error if OpenAI is selected but no key configured
                    err_msg = (
                        "OpenAI provider is selected but no API key is configured. "
                        "Set providers.openai.api_key or OPENAI_API_KEY, or switch models to a Gemini entry."
                    )
                    if use_plain_responses:
                        response_msg = await new_msg.reply(content=err_msg[:max_message_length], suppress_embeds=True)
                        response_msgs.append(response_msg)
                    else:
                        embed_err = discord.Embed(description=err_msg[:max_message_length], color=EMBED_COLOR_COMPLETE)
                        response_msg = await new_msg.reply(embed=embed_err, silent=True)
                        response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()
                    return

                openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
                kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
                async for chunk in await openai_client.chat.completions.create(**kwargs):
                    if finish_reason != None:
                        break

                    if not (choice := chunk.choices[0] if chunk.choices else None):
                        continue

                    finish_reason = choice.finish_reason

                    prev_content = curr_content or ""
                    curr_content = choice.delta.content or ""

                    new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        ready_to_edit = (edit_task == None or edit_task.done()) and datetime.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason != None or msg_split_incoming
                        is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                        if start_next_msg or ready_to_edit or is_final_edit:
                            if edit_task != None:
                                await edit_task

                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            if start_next_msg:
                                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                                response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                                response_msgs.append(response_msg)

                                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                await msg_nodes[response_msg.id].lock.acquire()
                            else:
                                edit_task = asyncio.create_task(response_msg.edit(embed=embed))

                            last_task_time = datetime.now().timestamp()

                if use_plain_responses:
                    for content in response_contents:
                        reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                        response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                        response_msgs.append(response_msg)

                        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                        await msg_nodes[response_msg.id].lock.acquire()

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    # Startup diagnostics: backend/model selection and modes
    model_key_startup = next(iter(config.get("models", {}) or {"openai/gpt-4o-mini": {}}))
    provider_startup = model_key_startup.split("/", 1)[0].lower()
    if provider_startup in ("gemini", "google"):
        backend = "Gemini"
        model_name = model_key_startup.split("/", 1)[1] if "/" in model_key_startup else model_key_startup
        gem_cfg = (config.get("providers", {}).get("gemini", {}) or {})
        goo_cfg = (config.get("providers", {}).get("google", {}) or {})
        key_gem = gem_cfg.get("api_key")
        key_goo = goo_cfg.get("api_key")
        key_env = os.getenv("GEMINI_API_KEY")
        api_present = bool(key_gem or key_goo or key_env)
        source = "providers.gemini" if key_gem else ("providers.google" if key_goo else ("env:GEMINI_API_KEY" if key_env else "none"))
        logging.info("Startup: backend=%s, model=%s, api_key_present=%s, source=%s", backend, model_name, api_present, source)
    elif provider_startup == "openai":
        backend = "OpenAI"
        model_name = model_key_startup.split("/", 1)[1] if "/" in model_key_startup else model_key_startup
        base_url = (config.get("providers", {}).get("openai", {}) or {}).get("base_url")
        api_present = bool((config.get("providers", {}).get("openai", {}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY"))
        logging.info("Startup: backend=%s, model=%s, base_url=%s, api_key_present=%s", backend, model_name, base_url, api_present)
    elif provider_startup == "anthropic":
        backend = "Anthropic"
        model_name = model_key_startup.split("/", 1)[1] if "/" in model_key_startup else model_key_startup
        api_present = bool((config.get("providers", {}).get("anthropic", {}) or {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY"))
        logging.info("Startup: backend=%s, model=%s, api_key_present=%s", backend, model_name, api_present)
    else:
        logging.info("Startup: backend=%s (custom), model_key=%s", provider_startup, model_key_startup)

    if config.get("test_echo_mode", False):
        logging.info("Startup: test_echo_mode is ENABLED (LLM calls will be skipped)")

    asyncio.run(main())
except KeyboardInterrupt:
    pass
