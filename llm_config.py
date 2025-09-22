"""
Configuration Manager for LLM Provider Setup
Handles provider detection, API key validation, and setup logic.
Also exposes a provider-agnostic answer generation method so callers don't
need to know provider specifics.
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional

# Provider SDKs
from openai import AsyncOpenAI
try:  # Anthropic is optional
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None


class LLMConfigManager:
    """Unified configuration manager for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_info = self._determine_provider()
        self.api_key_valid = self._validate_api_key()

    def _determine_provider(self) -> Dict[str, Any]:
        """Determine which provider and model to use"""
        # Use NEW explicit primary provider configuration (preferred)
        primary_provider = self.config.get("primary_provider")
        primary_model = self.config.get("primary_model")

        if primary_provider and primary_model:
            logging.info("Using NEW explicit primary provider configuration:")
            logging.info(f"   Primary Provider: {primary_provider}")
            logging.info(f"   Primary Model: {primary_model}")

            return {
                "provider": primary_provider,
                "model": primary_model,  # keep exactly as configured
                "name": f"{primary_model}",  # display name should not re-prefix provider
                "source": "primary_config"
            }
        else:
            # Fall back to OLD model list format for compatibility
            logging.info("Using OLD model list format (deprecated):")
            first_model = next(iter(self.config.get("models", [])), None)
            if first_model:
                provider = first_model.get("provider", "openai")
                model = first_model.get("model", "gpt-4o-mini")
                name = first_model.get("name", f"{provider}/{model}")

                logging.info(f"   Provider: {provider}")
                logging.info(f"   Model: {model}")

                return {
                    "provider": provider,
                    "model": model,
                    "name": name,
                    "source": "model_list"
                }
            else:
                raise ValueError("No models configured in config.yaml")

    @staticmethod
    def _sanitize_api_key(value: Optional[str], provider_name: str) -> Optional[str]:
        """Return a cleaned API key with newlines and surrounding whitespace removed.

        Logs a warning if sanitation changed the original value length (likely pasted with newlines).
        """
        if not value:
            return value
        original = value
        # Remove surrounding whitespace and any internal CR/LF characters
        cleaned = original.strip().replace("\r", "").replace("\n", "")
        if cleaned != original:
            logging.warning("%s API key contained whitespace/newlines; sanitized for use", provider_name.upper())
        return cleaned

    def _validate_api_key(self) -> bool:
        """Validate that the selected provider has a valid API key"""
        provider = self.provider_info["provider"]
        provider_config = self.config.get("providers", {}).get(provider, {})
        env_var = f"{provider.upper()}_API_KEY".replace("OPENROUTER", "OPENROUTER").replace("GEMINI", "GEMINI")
        raw_key = provider_config.get("api_key") or os.getenv(env_var)
        provider_api_key = self._sanitize_api_key(raw_key, provider)

        if provider_api_key:
            logging.info("%s API key found and configured", self.provider_info["provider"].upper())
            return True
        else:
            logging.warning("%s API key NOT found - check environment variables", self.provider_info["provider"].upper())
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the selected provider"""
        return self.provider_info.copy()

    def get_api_key_status(self) -> bool:
        """Get API key validation status"""
        return self.api_key_valid

    def log_startup_info(self):
        """Log startup configuration information"""
        info = self.get_startup_diagnostics()
        logging.info(f"✅ FINAL: Using provider '{self.provider_info['provider']}' with model '{self.provider_info['name']}'")
        logging.info("Startup: backend=%s, model=%s, api_key_present=%s",
                    info["backend"], self.provider_info["name"], info["api_present"])
        if info.get("base_url"):
            logging.info("Startup: backend=%s, model=%s, base_url=%s, api_key_present=%s",
                        info["backend"], self.provider_info["name"], info["base_url"], info["api_present"])

    def get_startup_diagnostics(self) -> Dict[str, Any]:
        """Get startup diagnostics information"""
        diagnostics = {
            "backend": "Unknown",
            "api_present": False,
            "base_url": None
        }

        provider = self.provider_info["provider"]

        if provider == "anthropic":
            diagnostics.update({
                "backend": "Anthropic",
                "api_present": bool((self.config.get("providers", {}).get("anthropic", {}) or {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY"))
            })
        elif provider == "openai":
            diagnostics.update({
                "backend": "OpenAI",
                "base_url": (self.config.get("providers", {}).get("openai", {}) or {}).get("base_url"),
                "api_present": bool((self.config.get("providers", {}).get("openai", {}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY"))
            })
        elif provider == "openrouter":
            diagnostics.update({
                "backend": "OpenRouter",
                "base_url": (self.config.get("providers", {}).get("openrouter", {}) or {}).get("base_url"),
                "api_present": bool((self.config.get("providers", {}).get("openrouter", {}) or {}).get("api_key") or os.getenv("OPENROUTER_API_KEY"))
            })
        elif provider in ("gemini", "google"):
            diagnostics.update({
                "backend": "Gemini",
                "api_present": bool((self.config.get("providers", {}).get("gemini", {}) or {}).get("api_key") or
                                 (self.config.get("providers", {}).get("google", {}) or {}).get("api_key") or
                                 os.getenv("GEMINI_API_KEY"))
            })

        return diagnostics

    def get_provider_settings(self) -> Dict[str, Any]:
        """Get provider-specific settings for use in main.py"""
        provider = self.provider_info["provider"]

        # Sanitize keys from config/env
        anth_key = self._sanitize_api_key(
            (self.config.get("providers", {}).get("anthropic", {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY")),
            "anthropic",
        )
        gemini_key = self._sanitize_api_key(
            (self.config.get("providers", {}).get("gemini", {}).get("api_key") or
             self.config.get("providers", {}).get("google", {}).get("api_key") or
             os.getenv("GEMINI_API_KEY")),
            "gemini",
        )

        return {
            "use_anthropic": provider == "anthropic",
            "use_gemini": provider == "google",
            "use_openrouter": provider == "openrouter",
            "use_openai": provider in ("openai", "x-ai"),
            "anthropic_api_key": anth_key,
            "gemini_api_key": gemini_key,
            "model_name": self.provider_info["model"]  # Generic name, not provider-specific
        }

    # -------------------- LLM Answer Generation --------------------
    @staticmethod
    def _normalize_model_for_provider(provider: str, model: str) -> str:
        prov = (provider or "").lower()
        m = model or ""
        if prov == "anthropic" and "/" in m:
            # Accept forms like "anthropic/claude-3-5-sonnet-20241022"
            return m.split("/", 1)[1]
        return m

    async def generate_answer(self, query_text: str, docs_context: str, system_prompt: str) -> Optional[str]:
        """Generate an answer using the selected provider. Returns None on failure.

        This centralizes all provider-specific logic (API keys, model names, SDK calls).
        """
        prov_info = self.get_provider_info()
        prov = prov_info.get("provider")
        model = prov_info.get("model")

        if not prov or not model:
            logging.error("[llm] provider or model missing from configuration")
            return None

        async def call_anthropic(model_name: str) -> Optional[str]:
            if anthropic is None:
                return None
            def _do() -> Optional[str]:
                try:
                    # Resolve Anthropic API key lazily only when needed
                    raw_key = (self.config.get("providers", {}).get("anthropic", {}).get("api_key")
                               or os.getenv("ANTHROPIC_API_KEY"))
                    anth_key = self._sanitize_api_key(raw_key, "anthropic")
                    client = anthropic.Anthropic(api_key=anth_key) if anth_key else anthropic.Anthropic()
                    msg = client.messages.create(
                        model=self._normalize_model_for_provider("anthropic", model_name),
                        max_tokens=700,
                        system=system_prompt,
                        messages=[{"role": "user", "content": f"User question:\n{query_text}\n\nDocumentation context:\n{docs_context}"}],
                    )
                    try:
                        return "".join([b.text for b in msg.content])
                    except Exception:
                        return str(msg)
                except Exception:
                    logging.exception("[llm] anthropic call failed in llm_config")
                    return None
            return await asyncio.to_thread(_do)

        async def call_openai(model_name: str) -> Optional[str]:
            try:
                oai_key = (self.config.get("providers", {}).get("openai", {}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY")
                oai = AsyncOpenAI(api_key=oai_key)
                resp = await oai.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"User question:\n{query_text}\n\nDocumentation context:\n{docs_context}"},
                    ],
                    temperature=0.3,
                    max_tokens=700,
                )
                return resp.choices[0].message.content
            except Exception:
                logging.exception("[llm] OpenAI call failed in llm_config")
                return None

        async def call_openrouter(model_name: str) -> Optional[str]:
            try:
                base_url = (self.config.get("providers", {}).get("openrouter", {}) or {}).get("base_url") or "https://openrouter.ai/api/v1"
                or_key = (self.config.get("providers", {}).get("openrouter", {}) or {}).get("api_key") or os.getenv("OPENROUTER_API_KEY")
                if not or_key:
                    logging.error("[llm] OPENROUTER_API_KEY missing")
                    return None
                oai = AsyncOpenAI(api_key=or_key, base_url=base_url)
                resp = await oai.chat.completions.create(
                    model=model_name,  # e.g., "anthropic/claude-sonnet-4"
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"User question:\n{query_text}\n\nDocumentation context:\n{docs_context}"},
                    ],
                    temperature=0.3,
                    max_tokens=700,
                )
                return resp.choices[0].message.content
            except Exception:
                logging.exception("[llm] OpenRouter call failed in llm_config")
                return None

        # Try primary provider first
        if prov == "anthropic":
            ans = await call_anthropic(model)
            if ans:
                return ans
        elif prov in ("openai", "x-ai"):
            ans = await call_openai(model)
            if ans:
                return ans
        elif prov == "openrouter":
            ans = await call_openrouter(model)
            if ans:
                return ans

        # Try configured fallback_models in order
        for fb in (self.config.get("fallback_models") or []):
            fb_prov = fb.get("provider")
            fb_model = fb.get("model")
            if not fb_prov or not fb_model:
                continue
            if fb_prov == "anthropic":
                ans = await call_anthropic(fb_model)
            elif fb_prov in ("openai", "x-ai"):
                ans = await call_openai(fb_model)
            elif fb_prov == "openrouter":
                ans = await call_openrouter(fb_model)
            else:
                ans = None
            if ans:
                return ans

        # Nothing worked
        return None


def create_llm_config(config: Dict[str, Any]) -> LLMConfigManager:
    """Factory function to create LLM configuration manager"""
    return LLMConfigManager(config)
