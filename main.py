import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional

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
from llm_config import create_llm_config
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

# Discord streaming indicator
STREAMING_INDICATOR = " "
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

# Discord embed colors
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

# Add detailed Discord error logging
import discord.http
original_request = discord.http.HTTPClient.request

async def debug_request(self, route, **kwargs):
    """Debug version of Discord HTTP request to log full responses"""
    try:
        # logging.info(f"🔗 Discord API Request: {route.method} {route.path}")
        response = await original_request(self, route, **kwargs)

        # Log successful responses
        if hasattr(response, 'status'):
            logging.info(f"✅ Discord API Response: {response.status} {route.method} {route.path}")
        else:
            logging.info(f"✅ Discord API Response: Success {route.method} {route.path}")

        return response

    except discord.HTTPException as e:
        logging.error(f"❌ Discord API Error: {e.status} {e.code} - {e.text}")
        logging.error(f"   Method: {route.method}")
        logging.error(f"   Path: {route.path}")
        logging.error(f"   Full response: {e.response}")
        if hasattr(e, 'json') and e.json:
            logging.error(f"   JSON response: {e.json}")
        raise
    except Exception as e:
        logging.error(f"❌ Discord API Exception: {type(e).__name__}: {e}")
        logging.error(f"   Method: {route.method}")
        logging.error(f"   Path: {route.path}")
        raise

# Monkey patch the HTTP client
discord.http.HTTPClient.request = debug_request


def get_config(filename: str = "config.yaml") -> dict:
    """Load config from YAML file with environment variable expansion."""
    import os

    with open(filename, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            # Expand $VARS inside strings; leaves unknown vars unchanged
            expanded = os.path.expandvars(obj)
            return expanded
        return obj

    expanded_config = _expand(raw)

    # Debug: Show config loading details
    # logging.info("Config loading debug:")
    # logging.info(f"   Raw config keys: {list(raw.keys())}")
    # logging.info(f"   Environment variables available: {list(os.environ.keys())}")

    # Show bot token specifically
    bot_token_raw = raw.get("bot_token", "")
    bot_token_expanded = expanded_config.get("bot_token", "")
    # logging.info(f"   Bot token - Raw: {bot_token_raw}")
    # logging.info(f"   Bot token - Expanded: {bot_token_expanded[:20]}... (length: {len(bot_token_expanded)})")

    if bot_token_expanded.startswith('$'):
        logging.warning(f"Bot token still contains unexpanded variable: {bot_token_expanded}")
        logging.warning("Available env vars with 'TOKEN': " +
                       ", ".join([k for k in os.environ.keys() if 'TOKEN' in k.upper()]))

    return expanded_config


def load_system_prompt() -> str:
    """Load system prompt from multiple sources with priority order:
    1. Multi-line env var: SYSTEM_PROMPT_DAOCORD (Railway-friendly)
    2. Base64 encoded env var: SYSTEM_PROMPT_DAOCORD_B64 (fallback)
    3. File: config.get("system_prompt_file") (if specified and file exists)
    4. File: system_prompt.txt (default fallback)
    5. Raise error if nothing found
    """
    import base64

    config_obj = get_config()

    logging.info("System prompt loading debug:")
    logging.info(f"Checking SYSTEM_PROMPT_DAOCORD: {'✅ Set' if os.getenv('SYSTEM_PROMPT_DAOCORD') else '❌ Not set'}")
    logging.info(f"Checking SYSTEM_PROMPT_DAOCORD_B64: {'✅ Set' if os.getenv('SYSTEM_PROMPT_DAOCORD_B64') else '❌ Not set'}")

    # Try multi-line environment variable first (Railway-friendly)
    env_prompt = os.getenv("SYSTEM_PROMPT_DAOCORD", "").strip()
    if env_prompt:
        logging.info(f"Found system prompt via SYSTEM_PROMPT_DAOCORD ({len(env_prompt)} chars)")
        return env_prompt

    # Try base64 encoded environment variable (fallback)
    b64_prompt = os.getenv("SYSTEM_PROMPT_DAOCORD_B64", "").strip()
    if b64_prompt:
        try:
            decoded = base64.b64decode(b64_prompt).decode('utf-8')
            if decoded.strip():
                logging.info(f"Found system prompt via SYSTEM_PROMPT_DAOCORD_B64 ({len(decoded)} chars)")
                return decoded.strip()
        except Exception as e:
            logging.warning(f"Failed to decode base64 system prompt: {e}")

    # Try to load from configured file first
    prompt_file_path = config_obj.get("system_prompt_file")
    if prompt_file_path:
        prompt_file = Path(prompt_file_path)
        logging.info(f"Checking configured prompt file: {prompt_file_path} - {'✅ Exists' if prompt_file.exists() else '❌ Not found'}")
        if prompt_file.exists():
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    logging.info(f"Found system prompt in configured file ({len(content)} chars)")
                    return content
            except Exception as e:
                logging.warning(f"Failed to load system prompt from {prompt_file}: {e}")

    # Try to load from default file
    default_prompt_file = Path("system_prompt.txt")
    logging.info(f"Checking default prompt file: system_prompt.txt - {'✅ Exists' if default_prompt_file.exists() else '❌ Not found'}")
    if default_prompt_file.exists():
        try:
            with open(default_prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logging.info(f"Found system prompt in default file ({len(content)} chars)")
                return content
        except Exception as e:
            logging.warning(f"Failed to load system prompt from default file: {e}")

    # No fallback - require explicit configuration
    logging.error("No system prompt found!")
    logging.error("Available options:")
    logging.error("1. Set SYSTEM_PROMPT_DAOCORD environment variable")
    logging.error("2. Set SYSTEM_PROMPT_DAOCORD_B64 environment variable (base64)")
    logging.error("3. Create system_prompt.txt file")
    logging.error("4. Configure system_prompt_file in config.yaml")
    raise ValueError("No system prompt found. Set SYSTEM_PROMPT_DAOCORD environment variable or create system_prompt.txt file.")


config = get_config()
curr_model = next(iter(config["models"]))

# Load system prompt once at startup
try:
    SYSTEM_PROMPT = load_system_prompt()
    logging.info("[system] system prompt loaded (%d chars)", len(SYSTEM_PROMPT or ""))
except Exception:
    SYSTEM_PROMPT = (
        "You are a helpful assistant for a DAO community. "
        "Use organizational documentation context when relevant and cite files."
    )
    logging.warning("[system] failed to load system prompt; using fallback")

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

# Enable all Discord logging for maximum verbosity
discord_logger = logging.getLogger('discord')
discord_logger.setLevel(logging.INFO)

# Also enable HTTP client logging
http_logger = logging.getLogger('discord.http')
http_logger.setLevel(logging.INFO)

logging.info("Discord bot client created successfully")
logging.info(f"Intents: {intents}")
logging.info(f"Activity: {activity}")
logging.info(f"Command prefix: {discord_bot.command_prefix}")

httpx_client = httpx.AsyncClient()
dao_docs = DAODocsTool()
# Initialize optional Google Docs cache (no-op if disabled in config)
try:
    gdocs_cache = GDocsCache(config)
except Exception:
    logging.exception("[gdocs] failed to initialize; continuing without gdocs")


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
    # Only show invite URL if we actually need it (bot failed to authenticate)
    # This is just informational - doesn't mean bot isn't already in server
    if config.get("client_id"):
        logging.info(f"🔗 Bot Invite URL (if needed): https://discord.com/oauth2/authorize?client_id={config['client_id']}&permissions=412317191168&scope=bot")

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

    # Enhanced logging: always log receipt of message with key context
    try:
        author_name = getattr(new_msg.author, "display_name", str(new_msg.author))
        channel_id = getattr(new_msg.channel, "id", None)
        guild_id = getattr(getattr(new_msg, "guild", None), "id", None)
        logging.info(
            "[msg] received: id=%s author=%s(%s) guild=%s channel=%s is_dm=%s len=%s",
            getattr(new_msg, "id", None),
            author_name,
            getattr(new_msg.author, "id", None),
            guild_id,
            channel_id,
            is_dm,
            len((new_msg.content or ""))
        )
        if new_msg.content:
            logging.info("[msg] content: %s", new_msg.content[:500])
    except Exception:
        logging.exception("[msg] logging failed for received message")

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        try:
            logging.info(
                "[msg] ignoring early: is_dm=%s mentioned=%s is_bot=%s",
                is_dm,
                (discord_bot.user in new_msg.mentions) if discord_bot.user else False,
                new_msg.author.bot,
            )
        except Exception:
            pass
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
    try:
        logging.info(
            "[perm] user: admin=%s good=%s bad=%s roles=%s",
            user_is_admin,
            is_good_user,
            is_bad_user,
            list(role_ids)
        )
    except Exception:
        pass

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)
    try:
        logging.info(
            "[perm] channel: allow_dms=%s good=%s bad=%s channels=%s",
            allow_dms,
            is_good_channel,
            is_bad_channel,
            list(channel_ids)
        )
    except Exception:
        pass

    if is_bad_user or is_bad_channel:
        try:
            logging.info("[msg] blocked by permissions: user_bad=%s channel_bad=%s", is_bad_user, is_bad_channel)
        except Exception:
            pass
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

    content_stripped = (new_msg.content or "").strip()
    lower = content_stripped.lower()
    try:
        logging.info("[logic] lower='%s'", lower[:200])
    except Exception:
        pass

    mentioned = False
    try:
        mentioned = discord_bot.user is not None and discord_bot.user.mentioned_in(new_msg)
    except Exception:
        mentioned = False

    if is_dm or mentioned:
        # Prepare a simple query text (strip bot mention if present)
        query_text = content_stripped
        try:
            mention_text = getattr(discord_bot.user, "mention", "") or ""
            if mention_text and query_text.startswith(mention_text):
                query_text = query_text.replace(mention_text, "", 1).strip()
        except Exception:
            pass

        # Log and run a lightweight DAO docs search
        try:
            logging.info("[docs] querying dao_docs with: '%s'", query_text[:200])
            top = dao_docs.top_with_scores(query_text, limit=3)
        except Exception:
            top = []
            logging.exception("[docs] dao_docs.top_with_scores failed")

        if not top:
            try:
                await new_msg.reply(
                    "No relevant documentation found for your query.",
                    suppress_embeds=True,
                    silent=True,
                )
                logging.info("[docs] no results returned")
            except Exception:
                logging.exception("[docs] failed to send 'no results' reply")
            return

        # Relevance gating: if scores are too low, decline to answer
        try:
            min_relevance = int(config.get("min_relevance_score", 25))
        except Exception:
            min_relevance = 25
        try:
            max_score = max((score for _, _, score in top), default=0)
        except Exception:
            max_score = 0
        logging.info("[docs] relevance: max_score=%s threshold=%s", max_score, min_relevance)
        if max_score < min_relevance:
            try:
                await new_msg.reply(
                    "This looks outside the project's scope, so I'm not going to answer. "
                    "If you have questions about the DAO, governance, longevity, or dogs (and related docs), ask away!",
                    suppress_embeds=True,
                    silent=True,
                )
                logging.info("[docs] declined due to low relevance (max_score=%s)", max_score)
            except Exception:
                logging.exception("[docs] failed to send low-relevance decline message")
            return

        # Use LLM to answer using the top documentation chunks
        try:
            async with new_msg.channel.typing():
                # Build a context string capped to a safe size
                context_parts = []
                total = 0
                for path, chunk, score in top:
                    rel = os.path.relpath(path, start="docs") if isinstance(path, str) else str(path)
                    header = f"From {rel} (score {score}):\n"
                    body = (chunk or "").strip()
                    piece = header + body + "\n\n"
                    if total + len(piece) > 6000:
                        remaining = 6000 - total
                        if remaining > 0:
                            context_parts.append(piece[:remaining])
                            total += remaining
                        break
                    context_parts.append(piece)
                    total += len(piece)
                docs_context = "".join(context_parts)

                # Provider-agnostic LLM call via llm_config
                try:
                    answer_text = await llm_config.generate_answer(query_text, docs_context, SYSTEM_PROMPT)
                    if answer_text:
                        logging.info("[llm] answer generated via llm_config (%d chars)", len(answer_text))
                except Exception:
                    logging.exception("[llm] llm_config.generate_answer failed")
                    answer_text = None

            if not answer_text:
                # Fallback: compact docs summary
                lines = []
                for path, chunk, score in top:
                    rel = os.path.relpath(path, start="docs") if isinstance(path, str) else str(path)
                    excerpt = (chunk or "").strip().replace("\n", " ")[:240]
                    lines.append(f"• {rel} (score {score})\n  {excerpt}")
                summary = "Here are relevant excerpts from the documentation:\n\n" + "\n\n".join(lines)
                await new_msg.reply(summary[:1800], suppress_embeds=True, silent=True)
                logging.info("[docs] replied with %d result(s) [fallback]", len(top))
                return

            # Send the LLM answer, with a short list of sources
            try:
                sources = []
                for path, _, score in top:
                    rel = os.path.relpath(path, start="docs") if isinstance(path, str) else str(path)
                    sources.append(f"{rel} (score {score})")
                suffix = "\n\nSources:\n- " + "\n- ".join(sources[:5])
                await new_msg.reply((answer_text + suffix)[:1900], suppress_embeds=True, silent=True)
            except Exception:
                logging.exception("[llm] failed to send final answer")
        except Exception:
            logging.exception("[llm] pipeline crashed; no reply sent")
        return

    # Not DM and not mentioned: ignore (but log)
    try:
        logging.info("[msg] ignored (not dm and not mentioned)")
    except Exception:
        pass
    return


# Module-level startup code
try:
    # Initialize LLM configuration manager
    llm_config = create_llm_config(config)

    # Get provider information
    provider_info = llm_config.get_provider_info()
    provider = provider_info["provider"]
    model = provider_info["model"]
    model_name = provider_info["name"]

    # Get provider settings from the unified manager
    provider_settings = llm_config.get_provider_settings()

    # Extract provider settings (no provider-specific logic here)
    use_anthropic = provider_settings["use_anthropic"]
    use_gemini = provider_settings["use_gemini"]
    use_openrouter = provider_settings["use_openrouter"]
    use_openai = provider_settings["use_openai"]
    anthropic_api_key = provider_settings["anthropic_api_key"]
    gemini_api_key = provider_settings["gemini_api_key"]
    model_name = provider_settings["model_name"]  # Generic model name

    # Log startup diagnostics
    llm_config.log_startup_info()

    if config.get("test_echo_mode", False):
        logging.info("Startup: test_echo_mode is ENABLED (LLM calls will be skipped)")

    # --- Discord token sanitation and diagnostics (helps with Railway 401s) ---
    raw_token = str(config.get("bot_token") or "")
    token_clean = raw_token.strip().replace("\r", "").replace("\n", "")
    if token_clean.lower().startswith("bot "):
        logging.warning("[discord] Token had 'Bot ' prefix; removing per discord.py expectations")
        token_clean = token_clean[4:].strip()
    if token_clean.startswith("$"):
        logging.error("[discord] Bot token appears to be an unexpanded env var: %s", token_clean)
        # Attempt to resolve from environment (supports $VAR or ${VAR})
        env_name = token_clean[2:-1] if (token_clean.startswith("${") and token_clean.endswith("}")) else token_clean.lstrip("$")
        resolved = os.getenv(env_name, "")
        if resolved:
            token_clean = resolved.strip().replace("\r", "").replace("\n", "")
            logging.info("[discord] Resolved token from environment: %s -> len=%s", env_name, len(token_clean))
        else:
            logging.error("[discord] Env var %s is not set; cannot resolve bot token", env_name)
    # Basic shape check: Discord tokens usually contain dots
    looks_ok = "." in token_clean and len(token_clean) > 20
    masked = (token_clean[:4] + "…" + token_clean[-6:]) if token_clean else "<empty>"
    logging.info("[discord] Using bot token (sanitized): len=%s looks_ok=%s masked=%s", len(token_clean), looks_ok, masked)

    if not looks_ok:
        logging.error("[discord] Bot token does not look valid. Check Railway env var expansion and value.")

    discord_bot.run(token_clean)
except KeyboardInterrupt:
    pass
