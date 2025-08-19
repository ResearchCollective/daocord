import os
import json
import time
import logging
from pathlib import Path
from typing import List, Tuple

from rapidfuzz import fuzz, process

# Google API imports (lazy import in methods to avoid hard dependency if disabled)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:  # pragma: no cover
    service_account = None
    build = None
    HttpError = Exception


TOP_N_RESULTS = 12
DEFAULT_EXPORT_MIME = "text/markdown"
DRIVE_DOC_MIME = "application/vnd.google-apps.document"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GDocsCache:
    """
    Syncs a Google Drive folder of Google Docs into a local cache directory as Markdown,
    builds text chunks, and exposes fuzzy top-N retrieval similar to DAODocsTool.

    Config expectations (already env-expanded by get_config):
      gdocs:
        enabled: true
        folder_id: "..."
        cache_dir: "gdocs_cache"
        sync_interval_minutes: 360
        max_files: 1000
        export_mime: "text/markdown"

    Credentials:
      - Preferred: provide raw JSON via env var. You can set gdocs.google_application_credentials to one of:
          * "ENV:VARNAME"  -> read JSON from env var VARNAME
          * "VARNAME"      -> read JSON from env var VARNAME
        If that is not provided, we also check standard envs:
          * GOOGLE_SERVICE_ACCOUNT_JSON (raw JSON)
          * GOOGLE_SERVICE_ACCOUNT_JSON_B64 (base64-encoded JSON)
          * GOOGLE_APPLICATION_CREDENTIALS (path to json file)
      - File path in config is still supported, but env JSON takes precedence.
      - The Drive folder must be shared with the service account email
    """

    def __init__(self, cfg: dict):
        gcfg = cfg.get("gdocs", {}) or {}
        self.enabled = bool(gcfg.get("enabled"))
        self.folder_id = gcfg.get("folder_id")
        self.cache_dir = Path(gcfg.get("cache_dir") or "gdocs_cache")
        self.sync_interval_s = int(gcfg.get("sync_interval_minutes") or 360) * 60
        self.max_files = int(gcfg.get("max_files") or 1000)
        self.export_mime = gcfg.get("export_mime") or DEFAULT_EXPORT_MIME
        # Optional credential path from config
        self.cred_path_cfg = gcfg.get("google_application_credentials")

        self.state_path = self.cache_dir / "state.json"
        self.state = {"last_sync": 0, "files": {}}  # fileId -> {modifiedTime, title, path}
        self.chunks: List[Tuple[str, str]] = []       # (path_or_title, chunk)

        if not self.enabled:
            logging.info("[gdocs] disabled in config; skipping init")
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    # ---------- Public API ----------
    def refresh(self) -> None:
        if not self.enabled:
            return
        if not self.folder_id:
            logging.warning("[gdocs] folder_id not set; skipping refresh")
            return
        try:
            creds = self._load_credentials()
        except Exception:
            logging.exception("[gdocs] Failed to load Google credentials; skipping refresh")
            return

        now = time.time()
        if now - self.state.get("last_sync", 0) < self.sync_interval_s and self.chunks:
            logging.info("[gdocs] using cached chunks (within sync interval)")
            return

        try:
            service = build("drive", "v3", credentials=creds, cache_discovery=False)
            files = self._list_folder_docs(service, self.folder_id, self.max_files)
            total = len(files)
            docs_like = sum(1 for f in files if f.get("mimeType") == DRIVE_DOC_MIME)
            if total == 0:
                logging.warning("[gdocs] folder has 0 items (check folder_id or sharing)")
            elif docs_like == 0:
                logging.warning("[gdocs] no Google Docs found in folder (found %d items, %d non-Docs skipped)", total, total)
            else:
                logging.info("[gdocs] listed %d items; %d Google Docs candidates", total, docs_like)
            synced = 0
            skipped_unchanged = 0
            for f in files:
                fid = f["id"]
                mt = f.get("modifiedTime")
                title = f.get("name")
                mime = f.get("mimeType")

                if mime != DRIVE_DOC_MIME:
                    continue  # skip non-Google-Docs for now

                prev = self.state["files"].get(fid)
                if prev and prev.get("modifiedTime") == mt:
                    skipped_unchanged += 1
                    continue

                content = self._export_doc(service, fid, self.export_mime)
                local_path = self._write_cache_file(title, content)
                self.state["files"][fid] = {"modifiedTime": mt, "title": title, "path": str(local_path)}
                synced += 1

            self.state["last_sync"] = int(now)
            self._save_state()
            logging.info("[gdocs] sync complete: %d updated, %d unchanged, %d tracked", synced, skipped_unchanged, len(self.state["files"]))
        except HttpError:
            logging.exception("[gdocs] Drive API error during refresh")
        except Exception:
            logging.exception("[gdocs] Unexpected error during refresh")

        # Rebuild chunks from cache files
        try:
            self._build_chunks()
            logging.info("[gdocs] built %d chunks from %d cached files", len(self.chunks), len(self.state["files"]))
        except Exception:
            logging.exception("[gdocs] Failed building chunks")

    def top_with_scores(self, query: str, limit: int = TOP_N_RESULTS):
        if not self.enabled or not self.chunks:
            return []
        scored = process.extract(
            query,
            [c[1] for c in self.chunks],
            scorer=fuzz.token_set_ratio,
            limit=limit,
        )
        out = []
        for _, score, idx in scored:
            path, chunk = self.chunks[idx]
            out.append((path, chunk, score))
        return out

    # ---------- Internals ----------
    def _load_credentials(self):
        # Priority: inline JSON in config (possibly via $VAR expansion), config-specified env JSON, standard env JSON,
        #           base64 env, env path, then config file path
        # 0) If config value itself looks like JSON (e.g., $VAR expanded to JSON), parse it directly
        if self.cred_path_cfg:
            cfg_val = str(self.cred_path_cfg)
            s = cfg_val.strip()
            # If value is quoted (e.g., ' {...} '), strip matching quotes first
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
            # Try whole-string JSON first
            try:
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    info = json.loads(s)
                    logging.info("[gdocs] using credentials from inline JSON in config (possibly expanded from $VAR)")
                    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                pass
            # If not a clean JSON string, try extracting the largest {...} region and parse that
            try:
                if "{" in s and "}" in s:
                    i = s.find("{")
                    j = s.rfind("}")
                    if i != -1 and j != -1 and j > i:
                        inner = s[i:j+1]
                        info = json.loads(inner)
                        logging.info("[gdocs] using credentials from embedded JSON in config value")
                        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                pass

        # 1) If config provides an env var name, read JSON from it
        if self.cred_path_cfg:
            env_name = None
            # Reuse s from above if available
            cfg_val = str(self.cred_path_cfg).strip()
            if cfg_val.upper().startswith("ENV:"):
                env_name = cfg_val.split(":", 1)[1].strip()
            else:
                # If it's not a path and contains no JSON braces, and matches an env var name, treat as env var name
                maybe_path = Path(cfg_val)
                if ("{" not in cfg_val and "}" not in cfg_val) and not (cfg_val.startswith("/") or ":\\" in cfg_val or maybe_path.exists()):
                    env_name = cfg_val
            if env_name:
                env_val = os.environ.get(env_name)
                if env_val:
                    try:
                        from json import loads
                        info = loads(env_val)
                        logging.info("[gdocs] using credentials JSON from env var '%s' (as specified in config)", env_name)
                        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
                    except Exception:
                        logging.exception("[gdocs] failed parsing JSON from env var '%s'", env_name)
                else:
                    short = (env_name[:32] + "…") if len(env_name) > 40 else env_name
                    logging.warning("[gdocs] env var '%s' specified in config is not set", short)

        # 2) Standard envs (JSON only; no file paths)
        raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if raw_json:
            try:
                from json import loads
                info = loads(raw_json)
                logging.info("[gdocs] using credentials from GOOGLE_SERVICE_ACCOUNT_JSON")
                return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                logging.exception("[gdocs] failed parsing GOOGLE_SERVICE_ACCOUNT_JSON")

        b64_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
        if b64_json:
            try:
                import base64, json as _json
                data = base64.b64decode(b64_json)
                info = _json.loads(data)
                logging.info("[gdocs] using credentials from GOOGLE_SERVICE_ACCOUNT_JSON_B64")
                return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                logging.exception("[gdocs] failed parsing GOOGLE_SERVICE_ACCOUNT_JSON_B64")

        cred_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_env:
            # Treat strictly as JSON; never as a file path
            s = cred_env.strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
            try:
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    info = json.loads(s)
                    logging.info("[gdocs] using credentials from inline JSON in GOOGLE_APPLICATION_CREDENTIALS")
                    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                pass
            try:
                if "{" in s and "}" in s:
                    i = s.find("{")
                    j = s.rfind("}")
                    if i != -1 and j != -1 and j > i:
                        inner = s[i:j+1]
                        info = json.loads(inner)
                        logging.info("[gdocs] using credentials from embedded JSON in GOOGLE_APPLICATION_CREDENTIALS")
                        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                pass
            short = (s[:32] + "…") if len(s) > 40 else s
            logging.warning("[gdocs] GOOGLE_APPLICATION_CREDENTIALS provided but not valid JSON; ignoring (val=%s)", short)

        # 3) Do NOT allow file paths from config; JSON-only policy
        if self.cred_path_cfg:
            s = str(self.cred_path_cfg).strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
            try:
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    info = json.loads(s)
                    logging.info("[gdocs] using credentials from inline JSON in config")
                    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                pass
            try:
                if "{" in s and "}" in s:
                    i = s.find("{")
                    j = s.rfind("}")
                    if i != -1 and j != -1 and j > i:
                        inner = s[i:j+1]
                        info = json.loads(inner)
                        logging.info("[gdocs] using credentials from embedded JSON in config value")
                        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except Exception:
                pass
            short = (s[:32] + "…") if len(s) > 40 else s
            logging.warning("[gdocs] config google_application_credentials is not valid JSON; file paths are disabled (val=%s)", short)

        raise RuntimeError("No Google credentials found. Provide JSON via env var only: set gdocs.google_application_credentials to ENV:YOUR_VAR or $YOUR_VAR, or set GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_SERVICE_ACCOUNT_JSON_B64 / GOOGLE_APPLICATION_CREDENTIALS to pure JSON.")

    def _list_folder_docs(self, service, folder_id: str, max_files: int):
        # Query to list files within folder (non-trashed), recurse by iterating children using 'q' on 'parents'
        q = f"'{folder_id}' in parents and trashed = false"
        fields = "nextPageToken, files(id, name, mimeType, modifiedTime)"
        files = []
        page_token = None
        while True:
            resp = service.files().list(q=q, spaces="drive", fields=fields, pageToken=page_token, pageSize=1000).execute()
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token or len(files) >= max_files:
                break
        return files[:max_files]

    def _export_doc(self, service, file_id: str, mime: str) -> str:
        try:
            data = service.files().export(fileId=file_id, mimeType=mime).execute()
            text = data.decode("utf-8", errors="ignore") if isinstance(data, (bytes, bytearray)) else str(data)
            if not text.strip() and mime != "text/plain":
                # fallback
                data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
                text = data.decode("utf-8", errors="ignore") if isinstance(data, (bytes, bytearray)) else str(data)
            return text
        except HttpError:
            logging.exception("[gdocs] export failed for %s", file_id)
            return ""

    def _sanitize(self, name: str) -> str:
        keep = [c if c.isalnum() or c in (" ", "-", "_", ".") else "_" for c in name]
        return "".join(keep).strip() or f"doc_{int(time.time())}"

    def _write_cache_file(self, title: str, content: str) -> Path:
        safe = self._sanitize(title) + ".md"
        p = self.cache_dir / safe
        p.write_text(content or "", encoding="utf-8")
        return p

    def _build_chunks(self):
        self.chunks.clear()
        for meta in self.state.get("files", {}).values():
            try:
                p = Path(meta.get("path", ""))
                if not p.exists():
                    continue
                text = p.read_text(encoding="utf-8")
                for ch in self._chunk_text(text):
                    self.chunks.append((str(p), ch))
            except Exception:
                logging.exception("[gdocs] failed reading %s", meta)

    def _chunk_text(self, text: str, max_chars: int = 800, overlap: int = 80):
        text = text.replace("\r\n", "\n")
        paras = [p.strip() for p in text.split("\n\n")]
        buf = []
        size = 0
        for p in paras:
            if not p:
                continue
            if size + len(p) + 2 <= max_chars:
                buf.append(p)
                size += len(p) + 2
            else:
                if buf:
                    yield "\n\n".join(buf)
                # start new buffer, optionally include overlap from end of previous
                if overlap and buf:
                    tail = "\n\n".join(buf)[-overlap:]
                else:
                    tail = ""
                buf = [tail + p if tail else p]
                size = len(buf[0])
        if buf:
            yield "\n\n".join(buf)

    def _load_state(self):
        try:
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("[gdocs] failed loading state.json; starting fresh")

    def _save_state(self):
        try:
            self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logging.exception("[gdocs] failed saving state.json")
