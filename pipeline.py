#!/usr/bin/env python3
"""
LogCast - Build log to social posts pipeline
Reads build-log.md, drafts posts via local/cloud AI, serves approval UI

Requirements:
  pip install ollama pydantic
  (playwright only needed for X posting: pip install playwright && playwright install chromium)
"""

import json
import os
import re
import sys
import datetime
import http.server
import threading
import webbrowser
from pathlib import Path
import urllib.request
import time

# ─── DEPENDENCIES CHECK ───────────────────────────────────────────────────────

try:
    import ollama
    from pydantic import BaseModel, Field
    from typing import List
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    print("WARNING: 'ollama' and/or 'pydantic' not installed.")
    print("  Run: pip install ollama pydantic")
    print("  Falling back to raw HTTP API (less reliable for structured output).\n")

# ─── CONFIG ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Model provider: "ollama", "anthropic", "openai"
    "provider": "ollama",

    # Ollama settings
    "ollama_model": "qwen3.5:9b",
    "ollama_url": "http://localhost:11434",

    # Anthropic settings (if provider = "anthropic")
    "anthropic_model": "claude-sonnet-4-20250514",
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),

    # OpenAI settings (if provider = "openai")
    "openai_model": "gpt-4o-mini",
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),

    # Bluesky credentials (env vars only - never hardcode passwords)
    "bluesky_handle": os.environ.get("BLUESKY_HANDLE", ""),
    "bluesky_app_password": os.environ.get("BLUESKY_APP_PASSWORD", ""),

    # X credentials (env vars only)
    "x_username": os.environ.get("X_USERNAME", ""),
    "x_password": os.environ.get("X_PASSWORD", ""),

    # Paths
    "build_log": "build-log.md",
    "drafts_file": "drafts.json",
    "posted_log": "posted.json",

    # UI server port
    "ui_port": 7823,

    # Generation parameters (set from UI)
    "gen_params": {
        "tone": "balanced",
        "length": "medium",
        "post_count": 3,
        "vary_platforms": True,
    },
}

# Generation state (for UI polling)
GEN_STATE = {"running": False, "done": False, "error": None, "count": 0}

# ─── PYDANTIC SCHEMAS ─────────────────────────────────────────────────────────

if HAS_OLLAMA:
    class SocialPost(BaseModel):
        id: str = Field(description="Post number, e.g. '1', '2', '3'")
        bluesky: str = Field(description="Post text for Bluesky (max 300 characters)")
        x: str = Field(description="Post text for X/Twitter (max 280 characters)")
        source: str = Field(description="The build log line(s) this post came from")

    class PostDrafts(BaseModel):
        posts: List[SocialPost]

# ─── PROMPTS ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are a social media post writer for a solo developer and researcher.
Turn raw build log entries into authentic, non-cringe social posts.

RULES:
- Write like a real person, not a marketing bot
- No "excited to share" or "thrilled to announce" type language
- Show the actual work, the problem, or the insight - not feelings about the work
- First person, direct voice
- Bluesky max 300 chars. X max 280 chars.

{tone_instruction}
{length_instruction}
{platform_instruction}

Generate {post_count} posts per log entry. Pick the most interesting or specific moments.
Respond ONLY with valid JSON matching the provided schema. No extra text."""

TONE_INSTRUCTIONS = {
    "technical": "TONE: Very technical and precise. Use jargon freely. Write for developers who understand the stack. No hashtags, no emojis.",
    "dry": "TONE: Dry and matter-of-fact. Understated. No hashtags, no emojis. Like a terse commit message that happens to be interesting.",
    "balanced": "TONE: Direct and natural. No hashtags unless genuinely useful (max 1). No emojis unless they add meaning. Not too formal, not too casual.",
    "conversational": "TONE: Casual and approachable. Write like you're telling a friend what you built today. One emoji OK if natural. One hashtag max.",
    "punchy": "TONE: Bold and opinionated. Strong takes. Short sentences. Grab attention. Up to 2 hashtags OK. One emoji OK if it adds punch.",
}

LENGTH_INSTRUCTIONS = {
    "short": "LENGTH: Ultra-concise. One or two sentences max. Under 150 chars if possible. Every word earns its place.",
    "medium": "LENGTH: Medium length. 1-3 sentences. Use enough of the character limit to convey the point clearly, but don't pad.",
    "detailed": "LENGTH: Use most of the character limit. Include context, specifics, and technical details. Full sentences.",
}


def build_system_prompt() -> str:
    """Build the system prompt from current generation parameters."""
    params = CONFIG.get("gen_params", {})
    tone = params.get("tone", "balanced")
    length = params.get("length", "medium")
    post_count = params.get("post_count", 3)
    vary = params.get("vary_platforms", True)

    tone_inst = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["balanced"])
    length_inst = LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS["medium"])

    if vary:
        platform_inst = "PLATFORMS: Write DIFFERENT text for Bluesky and X. Bluesky can be slightly longer/more detailed. X should be punchier and tighter. They should NOT be identical."
    else:
        platform_inst = "PLATFORMS: Same text for both Bluesky and X is fine. Just respect the character limits."

    return SYSTEM_PROMPT_BASE.format(
        tone_instruction=tone_inst,
        length_instruction=length_inst,
        platform_instruction=platform_inst,
        post_count=post_count,
    )


# ─── MODEL CALLS ───────────────────────────────────────────────────────────────

def call_ollama(log_entry: str) -> list:
    """Call Ollama using the Python library with Pydantic structured output."""
    model_name = CONFIG["ollama_model"]
    print(f"  Calling Ollama ({model_name}) via ollama.chat()...")

    # Set Ollama host if non-default
    ollama_url = CONFIG.get("ollama_url", "http://localhost:11434")
    if ollama_url and ollama_url != "http://localhost:11434":
        # Strip /api/generate if someone has the old URL format
        base = ollama_url.replace("/api/generate", "").rstrip("/")
        os.environ["OLLAMA_HOST"] = base

    is_qwen3 = "qwen3" in model_name.lower()

    schema = PostDrafts.model_json_schema()
    post_count = CONFIG.get("gen_params", {}).get("post_count", 3)
    user_prompt = (
        f"Build log entry:\n{log_entry}\n\n"
        f"Draft {post_count} social media posts. Schema: {json.dumps(schema, indent=2)}"
    )

    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        "format": schema,
        "options": {"temperature": 0, "num_ctx": 8192},
    }

    # Disable thinking for Qwen3 models
    if is_qwen3:
        kwargs["think"] = False

    response = ollama.chat(**kwargs)
    raw = response.message.content.strip()
    print(f"  Raw response ({len(raw)} chars): {raw[:300]}...")

    # Validate with Pydantic
    drafts = PostDrafts.model_validate_json(raw)
    return [p.model_dump() for p in drafts.posts]


def call_ollama_http_fallback(log_entry: str) -> str:
    """Fallback: raw HTTP to Ollama /api/chat if ollama library not installed."""
    model_name = CONFIG["ollama_model"]
    print(f"  Calling Ollama ({model_name}) via HTTP fallback...")

    ollama_url = CONFIG.get("ollama_url", "http://localhost:11434")
    base = ollama_url.replace("/api/generate", "").rstrip("/")

    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": f"Build log entry:\n{log_entry}\n\nRespond with ONLY valid JSON:"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 1024},
    }

    # Try think=False for Qwen3
    if "qwen3" in model_name.lower():
        body["think"] = False

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
        msg = result.get("message", {})
        content = msg.get("content", "")

        # Qwen3 might put content in thinking
        if not content.strip() and result.get("thinking"):
            content = result["thinking"]

        return content


def call_anthropic(log_entry: str) -> str:
    """Call Anthropic API, returns raw text."""
    print(f"  Calling Anthropic ({CONFIG['anthropic_model']})...")
    payload = json.dumps({
        "model": CONFIG["anthropic_model"],
        "max_tokens": 1000,
        "system": build_system_prompt(),
        "messages": [{"role": "user", "content": f"Build log entry:\n{log_entry}"}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": CONFIG["anthropic_api_key"],
            "anthropic-version": "2023-06-01"
        }
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        return result["content"][0]["text"]


def call_openai(log_entry: str) -> str:
    """Call OpenAI API, returns raw text."""
    print(f"  Calling OpenAI ({CONFIG['openai_model']})...")
    payload = json.dumps({
        "model": CONFIG["openai_model"],
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": f"Build log entry:\n{log_entry}"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG['openai_api_key']}"
        }
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]


# ─── JSON EXTRACTION (for non-Ollama providers) ───────────────────────────────

def extract_json(raw: str) -> dict | None:
    """Extract JSON from raw text - handles fences, preamble, thinking tags."""
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()

    # Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Brace-counting extraction
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start:i+1])
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
    return None


def validate_posts(posts: list) -> list:
    """Ensure each post dict has required fields."""
    valid = []
    for p in posts:
        if isinstance(p, dict) and ("bluesky" in p or "x" in p):
            p.setdefault("bluesky", p.get("x", ""))
            p.setdefault("x", p.get("bluesky", ""))
            p.setdefault("source", "")
            p.setdefault("id", str(len(valid) + 1))
            valid.append(p)
    return valid


# ─── GENERATE POSTS ───────────────────────────────────────────────────────────

def generate_posts(log_entry: str, max_retries: int = 2) -> list:
    """Generate posts with retry logic."""
    provider = CONFIG["provider"]
    print(f"  Generating posts via {provider}...")

    for attempt in range(max_retries + 1):
        try:
            if provider == "ollama":
                if HAS_OLLAMA:
                    # Use Pydantic structured output - returns validated list directly
                    posts = call_ollama(log_entry)
                    if posts:
                        print(f"  Got {len(posts)} valid posts (structured output)")
                        return validate_posts(posts)
                else:
                    raw = call_ollama_http_fallback(log_entry)
                    data = extract_json(raw)
                    if data and "posts" in data:
                        posts = validate_posts(data["posts"])
                        if posts:
                            print(f"  Got {len(posts)} valid posts (HTTP fallback)")
                            return posts

            elif provider == "anthropic":
                raw = call_anthropic(log_entry)
                print(f"  Raw response ({len(raw)} chars): {raw[:300]}...")
                data = extract_json(raw)
                if data and "posts" in data:
                    posts = validate_posts(data["posts"])
                    if posts:
                        return posts

            elif provider == "openai":
                raw = call_openai(log_entry)
                print(f"  Raw response ({len(raw)} chars): {raw[:300]}...")
                data = extract_json(raw)
                if data and "posts" in data:
                    posts = validate_posts(data["posts"])
                    if posts:
                        return posts

            else:
                raise ValueError(f"Unknown provider: {provider}")

            print(f"  Attempt {attempt + 1}: no valid posts in response")

        except Exception as e:
            print(f"  Attempt {attempt + 1} error: {e}")

        if attempt < max_retries:
            print(f"  Retrying ({attempt + 2}/{max_retries + 1})...")
            time.sleep(2)

    print(f"  Failed to generate posts after {max_retries + 1} attempts")
    return []


# ─── BUILD LOG PARSING ─────────────────────────────────────────────────────────

def parse_log_entries(log_path: str) -> list:
    path = Path(log_path)
    if not path.exists():
        print(f"Build log not found: {log_path}")
        return []

    content = path.read_text(encoding="utf-8")
    sections = re.split(r'\n(?=## \d{4}-\d{2}-\d{2})', content)
    entries = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        match = re.match(r'## (\d{4}-\d{2}-\d{2})', section)
        if match:
            date = match.group(1)
            body = section[len(match.group(0)):].strip()
            if body:
                entries.append({"date": date, "content": body})

    return entries


def load_posted() -> set:
    path = Path(CONFIG["posted_log"])
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return set(data.get("posted_dates", []))
        except (json.JSONDecodeError, KeyError):
            return set()
    return set()


def save_posted(dates: set):
    path = Path(CONFIG["posted_log"])
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing["posted_dates"] = sorted(list(dates))
    path.write_text(json.dumps(existing, indent=2))


# ─── DRAFT GENERATION ──────────────────────────────────────────────────────────

def generate_drafts():
    global GEN_STATE
    GEN_STATE = {"running": True, "done": False, "error": None, "count": 0}

    try:
        print("Reading build log...")
        entries = parse_log_entries(CONFIG["build_log"])
        posted = load_posted()

        # Check existing drafts to avoid regenerating
        drafts_path = Path(CONFIG["drafts_file"])
        existing_drafts = []
        existing_dates = set()
        if drafts_path.exists():
            try:
                existing_drafts = json.loads(drafts_path.read_text())
                existing_dates = {d.get("date") for d in existing_drafts if d.get("date")}
            except json.JSONDecodeError:
                existing_drafts = []

        new_entries = [e for e in entries if e["date"] not in posted and e["date"] not in existing_dates]
        if not new_entries:
            print("No new entries to process.")
            GEN_STATE = {"running": False, "done": True, "error": None, "count": 0}
            return existing_drafts

        print(f"Found {len(new_entries)} new entries")
        all_drafts = []

        for entry in new_entries:
            print(f"Processing {entry['date']}...")
            posts = generate_posts(entry["content"])
            for post in posts:
                post["date"] = entry["date"]
                post["approved_bluesky"] = False
                post["approved_x"] = False
                post["posted_bluesky"] = False
                post["posted_x"] = False
                all_drafts.append(post)

        merged = existing_drafts + all_drafts
        drafts_path.write_text(json.dumps(merged, indent=2))
        print(f"Generated {len(all_drafts)} draft posts")

        GEN_STATE = {"running": False, "done": True, "error": None, "count": len(all_drafts)}
        return merged

    except Exception as e:
        print(f"Generation error: {e}")
        GEN_STATE = {"running": False, "done": True, "error": str(e), "count": 0}
        return []


# ─── POSTING ───────────────────────────────────────────────────────────────────

def post_to_bluesky(text: str) -> bool:
    handle = CONFIG["bluesky_handle"]
    password = CONFIG["bluesky_app_password"]

    if not handle or not password:
        print("  Bluesky credentials not set - set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD env vars")
        return False

    try:
        auth_payload = json.dumps({
            "identifier": handle,
            "password": password
        }).encode()
        req = urllib.request.Request(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            data=auth_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            session = json.loads(resp.read())
            token = session["accessJwt"]
            did = session["did"]

        post_payload = json.dumps({
            "repo": did,
            "collection": "app.bsky.feed.post",
            "record": {
                "$type": "app.bsky.feed.post",
                "text": text[:300],
                "createdAt": datetime.datetime.utcnow().isoformat() + "Z"
            }
        }).encode()
        req = urllib.request.Request(
            "https://bsky.social/xrpc/com.atproto.repo.createRecord",
            data=post_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
        )
        with urllib.request.urlopen(req) as resp:
            print("  Posted to Bluesky")
            return True
    except Exception as e:
        print(f"  Bluesky error: {e}")
        return False


def post_to_x_playwright(text: str) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright not installed. Run: pip install playwright && playwright install chromium")
        return False

    username = CONFIG["x_username"]
    password = CONFIG["x_password"]

    if not username or not password:
        print("  X credentials not set - set X_USERNAME and X_PASSWORD env vars")
        return False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto("https://x.com/login")
            page.wait_for_timeout(2000)

            page.fill('input[autocomplete="username"]', username)
            page.keyboard.press("Enter")
            page.wait_for_timeout(1500)

            page.fill('input[name="password"]', password)
            page.keyboard.press("Enter")
            page.wait_for_timeout(3000)

            page.goto("https://x.com/compose/post")
            page.wait_for_timeout(2000)
            page.fill('[data-testid="tweetTextarea_0"]', text[:280])
            page.wait_for_timeout(1000)
            page.click('[data-testid="tweetButton"]')
            page.wait_for_timeout(2000)

            browser.close()
            print("  Posted to X")
            return True
    except Exception as e:
        print(f"  X posting error: {e}")
        return False


def process_approved():
    drafts_path = Path(CONFIG["drafts_file"])
    if not drafts_path.exists():
        return

    drafts = json.loads(drafts_path.read_text())
    posted_dates = load_posted()
    changed = False

    for draft in drafts:
        if draft.get("approved_bluesky") and not draft.get("posted_bluesky"):
            if post_to_bluesky(draft["bluesky"]):
                draft["posted_bluesky"] = True
                changed = True

        if draft.get("approved_x") and not draft.get("posted_x"):
            if post_to_x_playwright(draft["x"]):
                draft["posted_x"] = True
                changed = True

        if draft.get("posted_bluesky") and draft.get("posted_x"):
            posted_dates.add(draft["date"])

    if changed:
        drafts_path.write_text(json.dumps(drafts, indent=2))
        save_posted(posted_dates)
        print("Done posting approved drafts.")


# ─── HTTP SERVER FOR APPROVAL UI ───────────────────────────────────────────────

class UIHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.serve_ui()
        elif self.path == "/drafts":
            self.serve_drafts()
        elif self.path == "/config":
            self.serve_config()
        elif self.path == "/gen-status":
            self.send_json(GEN_STATE)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if self.path == "/save":
            drafts_path = Path(CONFIG["drafts_file"])
            drafts_path.write_text(json.dumps(body, indent=2))
            self.send_json({"ok": True})

        elif self.path == "/post":
            process_approved()
            self.send_json({"ok": True})

        elif self.path == "/generate":
            if not GEN_STATE["running"]:
                threading.Thread(target=generate_drafts, daemon=True).start()
            self.send_json({"ok": True, "message": "Generating..."})

        elif self.path == "/clear-posted":
            # Reset posted tracking so regeneration picks up all dates
            posted_path = Path(CONFIG["posted_log"])
            if posted_path.exists():
                posted_path.write_text(json.dumps({"posted_dates": []}, indent=2))
            self.send_json({"ok": True})

        elif self.path == "/config":
            # Only allow safe config keys to be set from UI
            SAFE_CONFIG_KEYS = {
                "provider", "ollama_model", "ollama_url",
                "anthropic_model", "anthropic_api_key",
                "openai_model", "openai_api_key",
                "bluesky_handle", "bluesky_app_password",
                "x_username", "x_password",
                "build_log", "gen_params",
            }
            for k, v in body.items():
                if k in SAFE_CONFIG_KEYS and k in CONFIG:
                    CONFIG[k] = v
            self.send_json({"ok": True})

    def send_json(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def serve_drafts(self):
        drafts_path = Path(CONFIG["drafts_file"])
        data = []
        if drafts_path.exists():
            try:
                data = json.loads(drafts_path.read_text())
            except json.JSONDecodeError:
                data = []
        self.send_json(data)

    def serve_config(self):
        safe = {k: v for k, v in CONFIG.items()
                if "password" not in k.lower() and "key" not in k.lower() and "secret" not in k.lower()}
        self.send_json(safe)

    def serve_ui(self):
        html_path = Path(__file__).parent / "ui.html"
        if not html_path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"ui.html not found - place it next to pipeline.py")
            return
        content = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)


def start_server():
    port = CONFIG["ui_port"]
    server = http.server.HTTPServer(("localhost", port), UIHandler)
    url = f"http://localhost:{port}"
    print(f"\nLogCast UI running at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    server.serve_forever()


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ui"

    if cmd == "generate":
        generate_drafts()
    elif cmd == "post":
        process_approved()
    elif cmd == "ui":
        print("LogCast starting...")
        # Start generation in background - UI comes up immediately
        threading.Thread(target=generate_drafts, daemon=True).start()
        start_server()
    else:
        print("Usage: python pipeline.py [generate|post|ui]")
