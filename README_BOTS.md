# Discord Bot Setup Guide

This project now consists of two separate Discord bots:

## 1. Main Bot (`main.py`)
**Purpose**: Handles user interactions and LLM conversations
**Features**:
- Responds to user queries using LLM providers (OpenAI, Anthropic, Google, etc.)
- Supports DAO documentation lookup
- Google Docs integration
- Vision model support for images

**Configuration**: `config.yaml`

## 2. Report Bot (`main_report.py`)
**Purpose**: Monitors X/Twitter and Reddit, generates reports using LLM
**Features**:
- Monitors social media for relevant content
- Generates research reports using LLM
- Posts reports to Discord channels
- Cron-based scheduling for automated reports

**Configuration**: `config_report.yaml`

## Setup Instructions

### 1. Environment Variables

We use YAML configuration files that reference environment variables.

Set the following environment variables in your system:

#### For Linux/Mac (add to ~/.bashrc or ~/.zshrc):
```bash
# Discord
export DISCORD_BOT_TOKEN=your_main_bot_token
export DISCORD_CLIENT_ID=your_main_bot_client_id

# LLM Providers (at least one required)
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export GEMINI_API_KEY=your_gemini_key
export OPENROUTER_API_KEY=your_openrouter_key

# System Prompt (optional)
export SYSTEM_PROMPT_DAOCORD="You are a helpful AI assistant..."

# Report Bot Variables:
export REPORT_BOT_DISCORD_TOKEN=your_report_bot_token
export REPORT_BOT_CLIENT_ID=your_report_bot_client_id

# Social Media APIs
export X_BEARER_TOKEN=your_twitter_bearer_token
export REDDIT_CLIENT_ID=your_reddit_client_id
export REDDIT_CLIENT_SECRET=your_reddit_client_secret

# LLM Providers (for report generation)
export ANTHROPIC_API_KEY=your_anthropic_key
export GEMINI_API_KEY=your_gemini_key
export OPENAI_API_KEY=your_openai_key
export GROQ_API_KEY=your_groq_key
export MISTRAL_API_KEY=your_mistral_key
export OPENROUTER_API_KEY=your_openrouter_key
export XAI_API_KEY=your_xai_key
```

#### For Windows (use System Properties or PowerShell):
```powershell
# Set environment variables
$env:DISCORD_BOT_TOKEN = "your_main_bot_token"
$env:DISCORD_CLIENT_ID = "your_main_bot_client_id"
# ... etc for other variables

# Or set permanently via System Properties
setx DISCORD_BOT_TOKEN "your_main_bot_token"
setx DISCORD_CLIENT_ID "your_main_bot_client_id"
```

**Note**: The YAML config files use `$VAR_NAME` syntax to reference these environment variables.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: You'll need to add `croniter` to requirements.txt for the report bot:
```
discord.py
PyYAML
openai
google-generativeai
anthropic
aiohttp
httpx
croniter  # For cron scheduling in report bot
```

### 3. Report Bot Configuration

#### Basic Setup
- **Discord Token**: Set `REPORT_BOT_DISCORD_TOKEN` environment variable
- **Channels**: Configure `report_channel_ids: ["123456789", "987654321"]`
- **Schedule**: Set `report_interval_cron: "0 0 0 * *"` (weekly at midnight)

#### Approval Workflow Configuration
The report bot now includes an approval workflow for content moderation:

```yaml
approval:
  enabled: true  # Enable approval workflow
  approvers_role_id: "123456789"  # Discord role ID for approvers
  quarantine_dir: "data/quarantine"  # Where reports go before approval
  approved_dir: "data/approved"  # Where reports go after approval
  reaction_emoji: "✅"  # Emoji for approval
  rejection_emoji: "❌"  # Emoji for rejection
  min_approvals: 1  # Minimum approvals needed
  auto_cleanup_days: 30  # Days to keep unapproved reports

reports:
  include_summary: true  # Include LLM-generated summary in Discord posts
  summary_max_length: 500  # Max length of summary in characters
  max_reports_per_post: 3  # Max reports to post at once
  embed_color_pending: 0xffa500  # Orange for pending approval
  embed_color_approved: 0x00ff00  # Green for approved
  embed_color_rejected: 0xff0000  # Red for rejected
```

#### Setting Up Approvers
1. Create a Discord role (e.g., "Report Approvers")
2. Set the role ID in `approvers_role_id` in config
3. Assign the role to users who should approve reports

### 4. Report Bot Commands

#### User Commands:
- `/report_status` - Check last run time and next scheduled run
- `/pending_reports` - List pending reports (approvers only)
- `/approved_reports` - List recently approved reports (approvers only)

#### Approval Process:
1. **Report Generation**: Bot monitors X/Twitter and Reddit, generates reports
2. **Discord Posting**: Reports posted with orange embed and approval/rejection reactions
3. **Approval**: Approvers react with ✅ to approve reports
4. **Rejection**: Approvers react with ❌ to reject reports
5. **Auto-Move**: Approved reports move from quarantine to approved folder
6. **Auto-Cleanup**: Unapproved reports older than 30 days are automatically deleted
7. **Knowledge Integration**: Approved reports can be used by the main bot

### 5. System Prompt Configuration (Railway-Friendly)

**Railway supports multi-line environment variables!** Just copy and paste your prompt text directly.

#### **Option A: Multi-line Environment Variable (Recommended)**
```bash
# In Railway environment variables:
# Key: SYSTEM_PROMPT_DAOCORD
# Value: (just paste your prompt text with line breaks)

SYSTEM_PROMPT_DAOCORD="You are Bjorn, a wise and elegant Borzoi
who serves as the knowledge keeper for Dog Years DAO.

You help users understand the latest developments in dog health..."

# Railway will preserve the line breaks!
```

#### **Option B: Base64 Encoded (Alternative)**
```bash
# If multi-line doesn't work, use base64:
./encode_system_prompt.sh "You are Bjorn, a wise Borzoi..."
# Key: SYSTEM_PROMPT_DAOCORD_B64
```

#### **Option C: File-based (Local Development)**
```bash
# Edit your prompt file
echo "You are Bjorn, a wise Borzoi..." > system_prompt.txt
```

### 6. Railway Deployment Setup

**Railway environment variables:**
```bash
# Required - just copy and paste your prompt text!
SYSTEM_PROMPT_DAOCORD="You are Bjorn, a wise and elegant Borzoi
who serves as the knowledge keeper for Dog Years DAO..."

# Other required variables
DISCORD_BOT_TOKEN=your_discord_token
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional (if you want base64 fallback)
SYSTEM_PROMPT_DAOCORD_B64=base64_encoded_prompt_here
```

**Priority Order (Railway-optimized):**
1. `SYSTEM_PROMPT_DAOCORD` (multi-line text - Railway supports this!)
2. `SYSTEM_PROMPT_DAOCORD_B64` (base64 encoded)
3. Files (if accessible)
4. **Error** - No hardcoded fallback (forces explicit configuration)

**Railway Setup Steps:**
1. Deploy your repository (no secrets in code)
2. Go to Railway dashboard → Project → Variables
3. Add environment variables by copying and pasting
4. For system prompt, just paste the multi-line text directly
5. Railway preserves line breaks in environment variables!

#### File Structure:
```
data/
├── quarantine/          # Pending approval
│   ├── report_001.json
│   └── report_002.json
├── approved/            # Approved for knowledge base
│   ├── report_001.json
│   └── report_002.json
└── last_report_run.json # Bot status tracking
```

#### Report States:
- **🟡 Pending**: In quarantine, awaiting approval
- **🟢 Approved**: Moved to approved folder, available for knowledge base
- **🔴 Rejected**: Marked as rejected, stays in quarantine

#### Integration with Main Bot:
Approved reports in `data/approved/` can be automatically ingested by the main knowledge keeper bot, making them available for user queries.

### 7. Configuration Files
- Remove X/Twitter and Reddit sections (already done)
- Configure LLM providers you want to use
- Set up Google Docs integration if needed
- Configure permissions and channels

#### Report Bot (`config_report.yaml`)
- Configure Discord bot token
- Set up X/Twitter monitoring rules:
  ```yaml
  twitter:
    keywords: ["longevity dogs", "aging research", "canine health"]
    users: ["dogagingproject", "steven_austad", "MattKaeberlein"]
    exclude_keywords: ["giveaway", "promo", "advertisement"]
    search_limit: 15
    since_days: 7
    filters:
      min_like: 1
      include_any: ["longevity", "aging", "lifespan", "research"]
      exclude_any: ["giveaway"]
  ```
- Set up Reddit monitoring rules:
  ```yaml
  reddit:
    subreddits: ["longevity", "aging", "biohacking", "dogs", "DogCare"]
    keywords: ["trial", "study", "research", "longevity", "aging", "lifespan"]
    exclude_keywords: ["giveaway", "promo", "advertisement", "vet bill"]
    search_limit: 15
    since_days: 7
    filters:
      min_score: 3
      include_any: ["trial", "study", "paper", "research"]
      exclude_any: ["giveaway", "promo", "advertisement"]
  ```
- Configure report channels: `report_channel_ids: ["123456789", "987654321"]`
- Set cron schedule: `report_interval_cron: "0 */6 * * *"` (every 6 hours)
- Configure LLM provider for report generation

### 8. Monitor Report Bot Status

#### Check via Discord Command:
```
/report_status
```

#### Check via File (for external monitoring):
```bash
cat data/last_report_run.json
```

Example output:
```json
{
  "last_run": "2025-01-20T14:30:00",
  "last_run_formatted": "2025-01-20 14:30:00 UTC",
  "next_run_estimate": "2025-01-20 20:30:00 UTC",
  "cron_schedule": "0 */6 * * *"
}
```

#### Check Logs:
```bash
tail -f logs/report_bot.log
```

### 9. Cron Schedule Examples

**Report Bot Cron Examples:**
- `0 */6 * * *` - Every 6 hours
- `0 9 * * 1` - Every Monday at 9 AM
- `0 0 * * 0` - Every Sunday at midnight (weekly reports)
- `0 0 1 * *` - First day of every month at midnight
- `0 0 0 * *` - Weekly (every Sunday at midnight)

### 10. Persistent Run Tracking

The report bot now tracks when it last ran successfully:

- **Storage**: `data/last_report_run.json`
- **Purpose**: Survives bot restarts, allows external monitoring
- **Contents**:
  - Last successful run timestamp
  - Next estimated run time
  - Current cron schedule
  - Human-readable timestamps

**Use cases:**
- Monitor bot health from external scripts
- Determine if reports are being generated on schedule
- Debug scheduling issues
- Track bot uptime and reliability

### 11. Option A: Run in Separate Terminals

**Terminal 1 - Main Bot:**
```bash
python main.py
```

**Terminal 2 - Report Bot:**
```bash
python main_report.py
```

#### Option B: Run in Background (Linux/Mac)
```bash
# Main Bot
nohup python main.py > logs/main_bot.log 2>&1 &

# Report Bot
nohup python main_report.py > logs/report_bot.log 2>&1 &
```

#### Option C: Using Process Manager (PM2)
```bash
# Install PM2
npm install -g pm2

# Start both bots
pm2 start main.py --name "main-bot"
pm2 start main_report.py --name "report-bot"

# Monitor
pm2 logs
pm2 status
```

### 12. Discord Bot Setup

1. **Create two separate Discord applications** at https://discord.com/developers/applications
2. **Generate bot tokens** for each application
3. **Add bots to your server** using the OAuth2 URL with appropriate permissions
4. **Set the bot tokens** via environment variables:
   - Main bot: `export DISCORD_BOT_TOKEN=your_token`
   - Report bot: `export REPORT_BOT_DISCORD_TOKEN=your_token`

### 13. API Keys Setup

- **OpenAI**: Get from https://platform.openai.com/api-keys
- **Anthropic**: Get from https://console.anthropic.com/
- **Google Gemini**: Get from https://makersuite.google.com/app/apikey
- **OpenRouter**: Get from https://openrouter.ai/keys
### 14. X/Twitter API Setup

**For X/Twitter monitoring to work, you need:**

1. **X Developer Account**: Apply at https://developer.twitter.com/en/apply-for-access
2. **Bearer Token**: Create a new app and get the Bearer Token from the app settings
3. **Environment Variable**: Set `export X_BEARER_TOKEN=your_bearer_token_here`

**Note**: The X API v2 only requires a Bearer Token (no API keys/secrets needed). The current implementation supports:
- Recent search (last 7 days)
- Tweet lookup with author expansion
- Conversation thread collection
- Quote tweet discovery

**Rate Limits**: ~300 requests per 15 minutes for search endpoints, ~900 for tweet lookup.

### 15. Monitoring and Logs

- Main bot logs: `logs/` directory
- Report bot logs: `logs/report_bot.log`
- Check bot status: `pm2 logs` or `pm2 status`

### 16. Cron Schedule Examples

**Report Bot Cron Examples:**
- `0 */6 * * *` - Every 6 hours
- `0 9 * * 1` - Every Monday at 9 AM
- `0 0 * * 0` - Every Sunday at midnight (weekly reports)
- `0 0 1 * *` - First day of every month at midnight

### 17. Troubleshooting

**Main Bot Issues:**
- Check `DISCORD_BOT_TOKEN` is correct
- Verify at least one LLM provider API key is set
- Check logs for specific error messages
- **System prompt missing**: Set `SYSTEM_PROMPT_DAOCORD` environment variable - no fallback is provided
- **Configuration error**: Ensure system prompt is properly configured in Railway environment variables

**Report Bot Issues:**
- Verify `REPORT_BOT_DISCORD_TOKEN` is different from main bot
- Check social media API credentials
- Ensure `report_channel_ids` contains valid Discord channel IDs
- Verify LLM provider for report generation is configured
- Check that all required environment variables are set
- **Approval System**: Ensure approvers have the correct Discord role
- **Rejection System**: Approvers can react with ❌ to reject reports
- **Auto-Cleanup**: Old unapproved reports are automatically deleted after 30 days
- **File Permissions**: Ensure bot has read/write access to data directories

**Common Issues:**
- "Invalid cron expression" - Fix cron syntax in `config_report.yaml`
- "Bot token invalid" - Double-check environment variables
- "API key missing" - Ensure required environment variables are set
- "Channel not found" - Verify channel IDs in `report_channel_ids`
- "Variable not found" - Check that environment variables are properly exported
- "Approvers can't react" - Check Discord role permissions and role ID
- "Reports not appearing" - Check if approval workflow is enabled and approvers role is set
- **"No system prompt found"** - Set `SYSTEM_PROMPT_DAOCORD` environment variable in Railway (required, no fallback)

### 18. Security Notes

- Never commit API keys or tokens to version control
- Use environment variables for all sensitive data (we reference them in YAML configs)
- Regularly rotate API keys
- Use separate Discord bots for different purposes
- Monitor API usage and costs
- The YAML config files use `$VAR_NAME` syntax to reference environment variables safely
