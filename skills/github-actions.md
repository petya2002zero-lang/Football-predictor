# Skill: GitHub Actions

## What This Skill Is For
This skill explains how our GitHub Actions automation works, how to debug failures,
and how to add or change the workflow. Read this before touching anything in `.github/`.

---

## How It Works

Every morning at 06:00 UTC, GitHub automatically:
1. Checks out this repo on a fresh Ubuntu machine
2. Installs Python and dependencies
3. Runs `python scripts/predict.py`
4. Commits the new JSON prediction files back to the repo
5. Pushes to GitHub → Vercel detects the push → website updates automatically

You can also trigger it manually any time (great for testing).

---

## The Workflow File

```yaml
# .github/workflows/daily-predictions.yml

name: Daily Football Predictions

on:
  schedule:
    - cron: '0 6 * * *'   # 06:00 UTC every day
  workflow_dispatch:        # Manual trigger from GitHub UI

jobs:
  predict:
    runs-on: ubuntu-latest
    timeout-minutes: 30     # Kill if it hangs

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'          # Cache pip downloads for faster runs

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run prediction pipeline
        env:
          API_FOOTBALL_KEY:   ${{ secrets.API_FOOTBALL_KEY }}
          ANTHROPIC_API_KEY:  ${{ secrets.ANTHROPIC_API_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID:   ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python scripts/predict.py

      - name: Commit and push predictions
        run: |
          git config user.name  "football-predictor-bot"
          git config user.email "bot@football-predictor.com"
          git add data/predictions/ data/accuracy/
          git diff --staged --quiet && echo "No changes to commit" || \
            (git commit -m "🤖 Predictions for $(date -u +%Y-%m-%d)" && git push)
```

---

## GitHub Secrets Setup

Go to: Your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets exactly (names are case-sensitive):

| Secret Name | Where to get it |
|---|---|
| `API_FOOTBALL_KEY` | RapidAPI dashboard → your app → API Key |
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys |
| `TELEGRAM_BOT_TOKEN` | Optional — talk to @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Optional — your Telegram user or group ID |

**Note:** `GITHUB_TOKEN` is automatic — GitHub provides it. You don't need to create it.

---

## How to Trigger Manually (For Testing)

1. Go to your GitHub repo
2. Click the **Actions** tab
3. Click **Daily Football Predictions** in the left sidebar
4. Click **Run workflow** button (top right of the table)
5. Click **Run workflow** in the dropdown
6. Watch the live logs as it runs

Use this to test before 6am, or after you make changes to the scripts.

---

## Reading the Logs When It Fails

When a run fails, click on it to see what went wrong:

```
❌ Run failed — click the red X
   → Click "predict" job
   → Expand the step that failed (red X on a step)
   → Read the error output
```

**Common failure reasons and fixes:**

| Error | Cause | Fix |
|---|---|---|
| `API_FOOTBALL_KEY not set` | Secret missing or wrong name | Re-check secret name in Settings |
| `ModuleNotFoundError: anthropic` | requirements.txt missing package | Add `anthropic` to requirements.txt |
| `rate limit exceeded` | Too many API calls | Reduce leagues or add caching |
| `JSONDecodeError` | Claude returned bad JSON | Check prediction-prompt.md, improve prompt |
| `git push` fails | Branch protection or auth | Check token has write permission |
| `timeout-minutes exceeded` | Script hung | Add timeouts to API calls |

---

## requirements.txt

```txt
anthropic>=0.40.0
requests>=2.31.0
python-dotenv>=1.0.0
```

That's it. Keep it minimal — fewer dependencies = fewer installation failures.

---

## Local `.env` File (For Running Locally)

Create a `.env` file in the project root (never commit this):

```bash
# .env  ← never commit this file
API_FOOTBALL_KEY=your_rapidapi_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
TELEGRAM_BOT_TOKEN=optional
TELEGRAM_CHAT_ID=optional
```

Load it in Python:
```python
from dotenv import load_dotenv
load_dotenv()  # Add this at top of predict.py for local testing
```

In GitHub Actions the `.env` file doesn't exist — it uses GitHub Secrets instead.

---

## Telegram Failure Alerts (Optional but Recommended)

This sends you a message if the pipeline crashes:

```python
# scripts/notify.py

import os
import requests

def send_failure_alert(message: str):
    """Send a Telegram message when the pipeline fails."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("Telegram not configured — skipping alert")
        return
    
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": f"⚠️ Football Predictor failed:\n{message}",
                "parse_mode": "HTML"
            },
            timeout=5
        )
    except Exception as e:
        print(f"Telegram alert failed: {e}")  # Don't crash over a notification failure
```

**Setup:**
1. Search `@BotFather` on Telegram → `/newbot` → get your token
2. Start a chat with your bot → send it a message
3. Get your chat ID from `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Add both to GitHub Secrets

---

## Checking if the Automation Is Working

After the first successful run, verify:
1. Check `data/predictions/` in your repo — a new JSON file should appear
2. Check your Vercel dashboard — a new deployment should have triggered
3. Visit your live site — new predictions should show

If predictions file exists but site didn't update, check Vercel deployment logs.

---

## Cron Schedule Reference

```
┌─────── minute (0-59)
│ ┌───── hour (0-23) UTC
│ │ ┌─── day of month (1-31)
│ │ │ ┌─ month (1-12)
│ │ │ │ ┌ day of week (0-6, 0=Sunday)
│ │ │ │ │
0 6 * * *   = Every day at 06:00 UTC
0 6 * * 1-5 = Weekdays only at 06:00 UTC
0 5,18 * * * = Twice a day at 05:00 and 18:00 UTC
```

06:00 UTC = 07:00 CET (Belgium/Netherlands) — good timing for same-day predictions.
