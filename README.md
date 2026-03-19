
<img width="1271" height="959" alt="helloOK" src="https://github.com/user-attachments/assets/c6139fbd-3289-4c48-8691-b83ef3294e5d" />

# LogCast

Build log → social posts. No slop.

## What it does

1. You write a daily build log (`build-log.md`)
2. LogCast reads it, generates draft posts via local AI (Ollama) or cloud (Anthropic/OpenAI)
3. You review, edit, approve in a browser UI
4. One click posts to Bluesky + X

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally (for local generation)
- `pip install ollama pydantic` (for reliable structured output)
- Playwright (only needed for X posting): `pip install playwright && playwright install chromium`

## Setup

1. Put `pipeline.py` and `ui.html` in a folder

2. Create `build-log.md`:

```markdown
## 2026-03-18
Set up Codemagic CI/CD. Android and iOS builds green.
Uploaded AAB to Google Play internal testing.
IPA auto-uploaded to TestFlight via App Store Connect.
```

3. Set credentials as environment variables:

```bash
# Bluesky (get app password from Settings → Privacy → App Passwords)
export BLUESKY_HANDLE="you.bsky.social"
export BLUESKY_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"

# X (used for browser automation — no API needed)
export X_USERNAME="yourusername"
export X_PASSWORD="yourpassword"

# If using Anthropic instead of Ollama:
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Never hardcode passwords in the script.**

4. Run:

```bash
python pipeline.py
```

Opens the UI at http://localhost:7823

## Commands

```bash
python pipeline.py          # generate + open UI (default)
python pipeline.py generate # generate drafts only (no UI)
python pipeline.py post     # post all approved drafts
```

<img width="1587" height="1654" alt="settings" src="https://github.com/user-attachments/assets/87c4c68d-a7f8-49b3-b3ec-b095cd973de2" />


## AI Provider

Default: Ollama (local, free). Change in the UI config panel or edit `CONFIG["provider"]` in `pipeline.py`.

For small local models (under 14B), LogCast uses a simplified prompt and forces JSON output mode. Larger models and cloud APIs get the full prompt.

**Recommended local models** (by quality):
- `qwen2.5:32b` — best local quality, needs 24GB+ RAM
- `qwen2.5:14b` — good balance, needs 16GB+ RAM
- `qwen3.5:4b` — works but hit-or-miss on JSON output

<img width="2478" height="1492" alt="posts" src="https://github.com/user-attachments/assets/c1a11b02-c137-49d0-a2c4-4e4eed5f5a4a" />


## Build log format

Dated sections, no structure needed:

```markdown
## 2026-03-19
Whatever you actually did. Technical, casual, frustrated — doesn't matter.
The AI picks out the interesting bits.

## 2026-03-18
Previous day here.
```

## Windows Task Scheduler

Daily at 9am:
1. Task Scheduler → Create Basic Task
2. Trigger: Daily, 09:00
3. Action: Start program
   - Program: `python`
   - Arguments: `C:\logcast\pipeline.py generate`
   - Start in: `C:\logcast\`
