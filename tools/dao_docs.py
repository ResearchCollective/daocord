# tools/dao_docs.py
import os
import time
import subprocess
from rapidfuzz import process, fuzz

DOCS_DIR = "docs"  # this is the local docs storage level
REPO_URL = "https://github.com/ResearchCollective/Dog-Years-Exocortex"  # https or PAT format if private
# note: we should ignore all files that start with "." (e.g. .gitattributes)
CHUNK_SIZE_WORDS = 1500
TOP_N_RESULTS = 3

# Only update the repo at most once per this interval (to avoid rate limits / frequent pulls)
UPDATE_INTERVAL_SECONDS = 12 * 60 * 60  # 12 hours
_LAST_UPDATE_MARKER = os.path.join(DOCS_DIR, ".last_update")

class DAODocsTool:
    def __init__(self):
        self.chunks = []
        self._ensure_repo()
        self._load_chunks()

    def _ensure_repo(self):
        if not os.path.exists(DOCS_DIR):
            print(f"[dao_docs] cloning {REPO_URL}...")
            os.makedirs(DOCS_DIR, exist_ok=True)
            # Clone into DOCS_DIR if empty
            if not os.listdir(DOCS_DIR):
                subprocess.run(["git", "clone", REPO_URL, DOCS_DIR], check=True)
                # Write last update marker immediately after initial clone
                try:
                    with open(_LAST_UPDATE_MARKER, "w", encoding="utf-8") as fp:
                        fp.write(str(int(time.time())))
                except OSError:
                    pass
        else:
            # Only pull if this looks like a git repo and interval elapsed
            git_dir = os.path.join(DOCS_DIR, ".git")
            should_update = False
            if os.path.isdir(git_dir):
                try:
                    last = 0
                    if os.path.exists(_LAST_UPDATE_MARKER):
                        last = os.path.getmtime(_LAST_UPDATE_MARKER)
                    should_update = (time.time() - last) > UPDATE_INTERVAL_SECONDS
                except OSError:
                    should_update = True

                if should_update:
                    print("[dao_docs] pulling latest changes (interval reached)...")
                    subprocess.run(["git", "-C", DOCS_DIR, "pull"], check=True)
                    try:
                        with open(_LAST_UPDATE_MARKER, "w", encoding="utf-8") as fp:
                            fp.write(str(int(time.time())))
                    except OSError:
                        pass
            else:
                # Not a git repo; skip pulling and just use current files
                print("[dao_docs] DOCS_DIR exists but is not a git repo; using local files only.")

    def refresh(self):
        print("[dao_docs] refreshing repo + chunks...")
        git_dir = os.path.join(DOCS_DIR, ".git")
        if os.path.isdir(git_dir):
            subprocess.run(["git", "-C", DOCS_DIR, "fetch"], check=True)
            subprocess.run(["git", "-C", DOCS_DIR, "reset", "--hard", "origin/main"], check=True)
            try:
                with open(_LAST_UPDATE_MARKER, "w", encoding="utf-8") as fp:
                    fp.write(str(int(time.time())))
            except OSError:
                pass
        self._load_chunks()

    def _load_chunks(self):
        self.chunks.clear()
        for dirpath, dirnames, files in os.walk(DOCS_DIR):
            # prune hidden directories and .git
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '.git']
            for f in files:
                if f.startswith('.'):
                    continue
                if f.endswith(".md"):
                    path = os.path.join(dirpath, f)
                    with open(path, encoding="utf-8") as fp:
                        words = fp.read().split()
                    for i in range(0, len(words), CHUNK_SIZE_WORDS):
                        chunk_text = " ".join(words[i:i+CHUNK_SIZE_WORDS])
                        self.chunks.append((path, chunk_text))
        print(f"[dao_docs] loaded {len(self.chunks)} chunks from markdown files.")

    def search(self, query):
        scored = process.extract(
            query,
            [c[1] for c in self.chunks],
            scorer=fuzz.token_set_ratio,
            limit=TOP_N_RESULTS
        )
        return [(self.chunks[idx][0], self.chunks[idx][1]) for _, _, idx in scored]

    def top_with_scores(self, query, limit: int = TOP_N_RESULTS):
        """Return top (path, chunk, score) tuples to assess relevance."""
        if not self.chunks:
            return []
        scored = process.extract(
            query,
            [c[1] for c in self.chunks],
            scorer=fuzz.token_set_ratio,
            limit=limit,
        )
        # scored items are (matched_text, score, index)
        out = []
        for _, score, idx in scored:
            path, chunk = self.chunks[idx]
            out.append((path, chunk, score))
        return out

    def run(self, query: str) -> str:
        """Main tool interface for llmcord"""
        results = self.search(query)
        if not results:
            return "no relevant documentation found."
        context = "\n\n".join(
            [f"From {os.path.relpath(path, DOCS_DIR)}:\n{chunk}" for path, chunk in results]
        )
        return f"Here are relevant excerpts from the DAO documentation:\n\n{context}"

    def run_full(self, max_chars: int = 60000) -> str:
        """Return large combined context from all docs (MVP) with a safety cap."""
        if not self.chunks:
            return "no documentation loaded."
        combined = []
        total = 0
        for path, chunk in self.chunks:
            header = f"From {os.path.relpath(path, DOCS_DIR)}:\n"
            piece = header + chunk + "\n\n"
            if total + len(piece) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    combined.append(piece[:remaining])
                    total += remaining
                break
            combined.append(piece)
            total += len(piece)
        return "Here is the DAO documentation context (truncated):\n\n" + "".join(combined)
