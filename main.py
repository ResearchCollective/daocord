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
        # if hasattr(response, 'status'):
        #     logging.info(f"✅ Discord API Response: {response.status} {route.method} {route.path}")
        # else:
        #     logging.info(f"✅ Discord API Response: Success {route.method} {route.path}")

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
    # logging.info("📋 Config loading debug:")
    # logging.info(f"   Raw config keys: {list(raw.keys())}")
    # logging.info(f"   Environment variables available: {list(os.environ.keys())}")

    # Show bot token specifically
    bot_token_raw = raw.get("bot_token", "")
    bot_token_expanded = expanded_config.get("bot_token", "")
    # logging.info(f"   Bot token - Raw: {bot_token_raw}")
    # logging.info(f"   Bot token - Expanded: {bot_token_expanded[:20]}... (length: {len(bot_token_expanded)})")

    if bot_token_expanded.startswith('$'):
        logging.warning(f"⚠️  Bot token still contains unexpanded variable: {bot_token_expanded}")
        logging.warning("   Available env vars with 'TOKEN': " +
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

    logging.info("🔍 System prompt loading debug:")
    logging.info(f"   Checking SYSTEM_PROMPT_DAOCORD: {'✅ Set' if os.getenv('SYSTEM_PROMPT_DAOCORD') else '❌ Not set'}")
    logging.info(f"   Checking SYSTEM_PROMPT_DAOCORD_B64: {'✅ Set' if os.getenv('SYSTEM_PROMPT_DAOCORD_B64') else '❌ Not set'}")

    # Try multi-line environment variable first (Railway-friendly)
    env_prompt = os.getenv("SYSTEM_PROMPT_DAOCORD", "").strip()
    if env_prompt:
        logging.info(f"   ✅ Found system prompt via SYSTEM_PROMPT_DAOCORD ({len(env_prompt)} chars)")
        return env_prompt

    # Try base64 encoded environment variable (fallback)
    b64_prompt = os.getenv("SYSTEM_PROMPT_DAOCORD_B64", "").strip()
    if b64_prompt:
        try:
            decoded = base64.b64decode(b64_prompt).decode('utf-8')
            if decoded.strip():
                logging.info(f"   ✅ Found system prompt via SYSTEM_PROMPT_DAOCORD_B64 ({len(decoded)} chars)")
                return decoded.strip()
        except Exception as e:
            logging.warning(f"   ❌ Failed to decode base64 system prompt: {e}")

    # Try to load from configured file first
    prompt_file_path = config_obj.get("system_prompt_file")
    if prompt_file_path:
        prompt_file = Path(prompt_file_path)
        logging.info(f"   Checking configured prompt file: {prompt_file_path} - {'✅ Exists' if prompt_file.exists() else '❌ Not found'}")
        if prompt_file.exists():
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    logging.info(f"   ✅ Found system prompt in configured file ({len(content)} chars)")
                    return content
            except Exception as e:
                logging.warning(f"   ❌ Failed to load system prompt from {prompt_file}: {e}")

    # Try to load from default file
    default_prompt_file = Path("system_prompt.txt")
    logging.info(f"   Checking default prompt file: system_prompt.txt - {'✅ Exists' if default_prompt_file.exists() else '❌ Not found'}")
    if default_prompt_file.exists():
        try:
            with open(default_prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logging.info(f"   ✅ Found system prompt in default file ({len(content)} chars)")
                return content
        except Exception as e:
            logging.warning(f"   ❌ Failed to load system prompt from default file: {e}")

    # No fallback - require explicit configuration
    logging.error("❌ No system prompt found!")
    logging.error("   Available options:")
    logging.error("   1. Set SYSTEM_PROMPT_DAOCORD environment variable")
    logging.error("   2. Set SYSTEM_PROMPT_DAOCORD_B64 environment variable (base64)")
    logging.error("   3. Create system_prompt.txt file")
    logging.error("   4. Configure system_prompt_file in config.yaml")
    raise ValueError("No system prompt found. Set SYSTEM_PROMPT_DAOCORD environment variable or create system_prompt.txt file.")


config = get_config()
curr_model = next(iter(config["models"]))

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

logging.info("🤖 Discord bot client created successfully")
logging.info(f"   Intents: {intents}")
logging.info(f"   Activity: {activity}")
logging.info(f"   Command prefix: {discord_bot.command_prefix}")

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
            await new_msg.reply("DAO docs refreshed.", suppress_embeds=True, silent=True)
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

    # Use the NEW explicit primary provider configuration (preferred)
    primary_provider = config.get("primary_provider")
    primary_model = config.get("primary_model")

    if primary_provider and primary_model:
        logging.info("📋 Using NEW explicit primary provider configuration:")
        logging.info(f"   Primary Provider: {primary_provider}")
        logging.info(f"   Primary Model: {primary_model}")

        provider = primary_provider
        model = primary_model
        model_name = f"{primary_provider}/{primary_model}"
        first_model = {"provider": primary_provider, "model": primary_model}
    else:
        # Fall back to OLD model list format for compatibility
        logging.info("Using OLD model list format (deprecated):")
        first_model = next(iter(config.get("models", [])), None)
        if first_model is None:
            raise ValueError("No models configured in config.yaml")

        provider = first_model.get("provider", "openai")
        model = first_model.get("model", "gpt-4o-mini")
        model_name = first_model.get("name", f"{provider}/{model}")

        logging.info(f"   Provider: {provider}")
        logging.info(f"   Model: {model}")

    logging.info(f"FINAL: Using provider '{provider}' with model '{model}'")

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

    model_parameters = first_model

    extra_headers = provider_config.get("extra_headers", None)
    extra_query = provider_config.get("extra_query", None)
    extra_body = (provider_config.get("extra_body", None) or {}) | (model_parameters or {}) or None

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES) or active_provider in ("openai", "x-ai")

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
                user_warnings.add(f"Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    try:
        system_prompt = load_system_prompt()
    except ValueError as e:
        # No system prompt configured - send error message
        error_msg = f"Configuration Error: {str(e)}\n\nPlease set SYSTEM_PROMPT_DAOCORD environment variable in Railway."
        if use_plain_responses:
            response_msg = await new_msg.reply(content=error_msg, suppress_embeds=True)
        else:
            embed_err = discord.Embed(description=error_msg, color=0xff0000)
            response_msg = await new_msg.reply(embed=embed_err, silent=True)
        response_msgs.append(response_msg)
        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()
        return

    if system_prompt:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Optional global system prompt from config (avoid duplicates if already present)
    try:
        sys_prompt = load_system_prompt().strip()
    except ValueError:
        sys_prompt = ""
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
                "Context built from %d chunks (%d unique sources, gdocs=%s), length=%d chars",
                len(merged), len(sources), "T" if getattr(gdocs_cache, "enabled", False) else "F", len(docs_context)
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
    use_gemini = (active_provider == "google") and bool(gemini_api_key)
    use_anthropic = (active_provider == "anthropic") and bool(anthropic_api_key)
    use_openrouter = (active_provider == "openrouter") and bool(api_key)
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

    # Fast relevance prefilter: if docs don't match the query well, skip LLM
    if dao_trigger:
        try:
            scored_docs = await asyncio.to_thread(dao_docs.top_with_scores, user_query, 3)
            best_score = max((s for _, _, s in scored_docs), default=0)
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
                        # Try to get system prompt for retry
                        try:
                            retry_system_prompt = load_system_prompt()
                        except ValueError:
                            retry_system_prompt = ""

                        r_prompt = (
                            (retry_system_prompt + "\n\n") if retry_system_prompt else ""
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
            elif use_openrouter:
                # OpenRouter streaming path
                # Optional debug: log exact prompt to terminal (OpenRouter path)
                if config.get("debug_prompt") is True:
                    ordered = messages[::-1]
                    prompt_text = "\n\n".join([m.get("content", "") for m in ordered])
                    if prompt_text:
                        logging.info("===== DEBUG PROMPT BEGIN (chars=%d) =====", len(prompt_text))
                        print(prompt_text)
                        logging.info("===== DEBUG PROMPT END =====")
                if not api_key:
                    # Friendly error if OpenRouter is selected but no key configured
                    err_msg = (
                        "OpenRouter provider is selected but no API key is configured. "
                        "Set providers.openrouter.api_key or OPENROUTER_API_KEY, or switch models to a different provider."
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

                # Filter extra_body for OpenRouter compatibility
                openrouter_body = {}
                if extra_body:
                    # Only pass parameters that OpenRouter accepts
                    for key, value in extra_body.items():
                        if key in ('temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty'):
                            openrouter_body[key] = value

                logging.info(f"🔍 OpenRouter Request Debug:")
                logging.info(f"   Model: {model}")
                logging.info(f"   API Key present: {bool(api_key)}")
                logging.info(f"   Filtered extra_body: {openrouter_body}")

                openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
                kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, **openrouter_body)
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
    # Debug: Show what bot token we're trying to use
    bot_token = config["bot_token"]
    token_preview = bot_token[:20] + "..." if len(bot_token) > 20 else bot_token
    # logging.info(f"🔐 Attempting Discord login with token: {token_preview}")
    # logging.info(f"   Token length: {len(bot_token)} characters")
    # logging.info(f"   Token starts with: {bot_token[:10] if bot_token else 'None'}")
    # logging.info(f"   Token ends with: {bot_token[-10:] if bot_token else 'None'}")
    # logging.info(f"   Config loaded: {type(config)} with keys: {list(config.keys())}")

    if not bot_token or bot_token.startswith('$'):
        logging.error("❌ Bot token appears to be unset or using unexpanded environment variable!")
        logging.error("   Make sure to set KNOWLEDGE_BOT_DISCORD_TOKEN environment variable")
        logging.error("   In PowerShell: $env:KNOWLEDGE_BOT_DISCORD_TOKEN = 'YOUR_TOKEN'")
        raise ValueError("Discord bot token not properly configured")

    try:
        logging.info("🚀 Starting Discord bot...")
        await discord_bot.start(bot_token)
    except discord.LoginFailure as e:
        logging.error(f"❌ Discord Login Failure: {e}")
        logging.error(f"   Error type: {type(e).__name__}")
        logging.error(f"   Error message: {str(e)}")
        # Try to get more details from the underlying exception
        if hasattr(e, '__cause__') and e.__cause__:
            logging.error(f"   Underlying cause: {e.__cause__}")
            logging.error(f"   Underlying type: {type(e.__cause__).__name__}")
        raise
    except discord.HTTPException as e:
        logging.error(f"❌ Discord HTTP Exception: {e}")
        logging.error(f"   Status: {e.status}")
        logging.error(f"   Code: {e.code}")
        logging.error(f"   Text: {e.text}")
        if hasattr(e, 'response'):
            logging.error(f"   Response: {e.response}")
        if hasattr(e, 'json') and e.json:
            logging.error(f"   JSON: {e.json}")
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected Discord error: {type(e).__name__}: {e}")
        logging.error(f"   Full traceback: {e.__traceback__}")
        raise


try:
    # Use the NEW explicit primary provider configuration (preferred)
    startup_primary_provider = config.get("primary_provider")
    startup_primary_model = config.get("primary_model")

    if startup_primary_provider and startup_primary_model:
        logging.info("📋 Using NEW explicit primary provider configuration:")
        logging.info(f"   Primary Provider: {startup_primary_provider}")
        logging.info(f"   Primary Model: {startup_primary_model}")

        provider_startup = startup_primary_provider
        model_name = startup_primary_model
        startup_name = f"{startup_primary_provider}/{startup_primary_model}"
    else:
        # Fall back to OLD model list format for compatibility
        logging.info("📋 Using OLD model list format (deprecated):")
        first_model = next(iter(config.get("models", [])), None)
        if first_model:
            startup_provider = first_model.get("provider", "unknown")
            startup_model = first_model.get("model", "unknown")
            startup_name = first_model.get("name", f"{startup_provider}/{startup_model}")
        else:
            startup_provider = "none"
            startup_model = "none"
            startup_name = "none"

        provider_startup = startup_provider
        model_name = startup_model

    # Use the NEW explicit primary provider configuration (preferred)
    startup_primary_provider = config.get("primary_provider")
    startup_primary_model = config.get("primary_model")

    if startup_primary_provider and startup_primary_model:
        logging.info("📋 Using NEW explicit primary provider configuration:")
        logging.info(f"   Primary Provider: {startup_primary_provider}")
        logging.info(f"   Primary Model: {startup_primary_model}")

        provider_startup = startup_primary_provider
        model_name = startup_primary_model
        startup_name = f"{startup_primary_provider}/{startup_primary_model}"
    else:
        # Fall back to OLD model list format for compatibility
        logging.info("📋 Using OLD model list format (deprecated):")
        first_model = next(iter(config.get("models", [])), None)
        if first_model:
            startup_provider = first_model.get("provider", "unknown")
            startup_model = first_model.get("model", "unknown")
            startup_name = first_model.get("name", f"{startup_provider}/{startup_model}")
        else:
            startup_provider = "none"
            startup_model = "none"
            startup_name = "none"

        provider_startup = startup_provider
        model_name = startup_model

    # Always define startup_provider for debug logging
    if not startup_primary_provider:
        startup_provider = provider_startup
    else:
        startup_provider = startup_primary_provider

    # Now we have consistent variable names - log them
    logging.info(f"   startup_provider: '{startup_provider}'")
    logging.info(f"   provider_startup.lower(): '{provider_startup}'")
    logging.info(f"   Checking conditions:")
    logging.info(f"   - 'gemini' or 'google': {provider_startup in ('gemini', 'google')}")
    logging.info(f"   - 'openai': {provider_startup == 'openai'}")
    logging.info(f"   - 'openrouter': {provider_startup == 'openrouter'}")
    logging.info(f"   - 'anthropic': {provider_startup == 'anthropic'}")

    if provider_startup in ("gemini", "google"):
        backend = "Gemini"
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
        base_url = (config.get("providers", {}).get("openai", {}) or {}).get("base_url")
        api_present = bool((config.get("providers", {}).get("openai", {}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY"))
        logging.info("Startup: backend=%s, model=%s, base_url=%s, api_key_present=%s", backend, model_name, base_url, api_present)
    elif provider_startup == "openrouter":
        backend = "OpenRouter"
        base_url = (config.get("providers", {}).get("openrouter", {}) or {}).get("base_url")
        api_present = bool((config.get("providers", {}).get("openrouter", {}) or {}).get("api_key") or os.getenv("OPENROUTER_API_KEY"))
        logging.info("Startup: backend=%s, model=%s, base_url=%s, api_key_present=%s", backend, model_name, base_url, api_present)
    elif provider_startup == "anthropic":
        backend = "Anthropic"
        api_present = bool((config.get("providers", {}).get("anthropic", {}) or {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY"))
        logging.info("Startup: backend=%s, model=%s, api_key_present=%s", backend, model_name, api_present)
    else:
        backend = f"Unknown ({provider_startup})"
        logging.warning("Startup: Unknown provider %s for model %s", provider_startup, model_name)

    # Check if the selected provider actually has API access
    provider_config = config.get("providers", {}).get(provider_startup, {})
    provider_api_key = provider_config.get("api_key") or os.getenv(f"{provider_startup.upper()}_API_KEY".replace("OPENROUTER", "OPENROUTER").replace("GEMINI", "GEMINI"))
    if provider_api_key:
        logging.info("Startup: ✅ %s API key found and configured", provider_startup.upper())
    else:
        logging.warning("Startup: ⚠️  %s API key NOT found - check environment variables", provider_startup.upper())

    if config.get("test_echo_mode", False):
        logging.info("Startup: test_echo_mode is ENABLED (LLM calls will be skipped)")

    asyncio.run(main())
except KeyboardInterrupt:
    pass
