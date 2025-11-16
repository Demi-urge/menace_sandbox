import os
from pathlib import Path


def load_env(path: str | None = None) -> None:
    """Load environment variables from a file if provided."""
    if not path:
        return
    file_path = Path(path)
    if not file_path.exists():
        return
    for line in file_path.read_text().splitlines():
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip())


# Optionally load variables from MENACE_ENV_FILE if set
load_env(os.getenv('MENACE_ENV_FILE'))


# Convenience accessors for secrets
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')
ANTICAPTCHA_API_KEY = os.getenv('ANTICAPTCHA_API_KEY')

MENACE_MODE = os.getenv('MENACE_MODE', 'test')
MENACE_EMAIL = os.getenv('MENACE_EMAIL')
MENACE_PASSWORD = os.getenv('MENACE_PASSWORD')

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

FACEBOOK_APP_ID = os.getenv('FACEBOOK_APP_ID')
FACEBOOK_APP_SECRET = os.getenv('FACEBOOK_APP_SECRET')
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN')

IG_ACCESS_TOKEN = os.getenv('IG_ACCESS_TOKEN')
IG_PAGE_ID = os.getenv('IG_PAGE_ID')
IG_USERNAME = os.getenv('IG_USERNAME')
IG_PASSWORD = os.getenv('IG_PASSWORD')

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
YOUTUBE_CLIENT_ID = os.getenv('YOUTUBE_CLIENT_ID')
YOUTUBE_CLIENT_SECRET = os.getenv('YOUTUBE_CLIENT_SECRET')

SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = os.getenv('SMTP_PORT')
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')

DEFAULT_ACCOUNT_PASSWORD = os.getenv("DEFAULT_ACCOUNT_PASSWORD", "Password123!")

CHROMIUM_PATH = os.getenv('CHROMIUM_PATH', '/usr/bin/chromium-browser')
BOT_PERFORMANCE_DB = os.getenv('BOT_PERFORMANCE_DB', 'bot_performance_history.db')
MAINTENANCE_DB = os.getenv('MAINTENANCE_DB', 'maintenance.db')
# Optional PostgreSQL connection string for Maintenance logs
MAINTENANCE_DB_URL = os.getenv('MAINTENANCE_DB_URL')

# Shared queue configuration
SHARED_QUEUE_DIR = os.getenv("SHARED_QUEUE_DIR", "logs/queue")
Path(SHARED_QUEUE_DIR).mkdir(parents=True, exist_ok=True)
SYNC_INTERVAL = float(os.getenv("SYNC_INTERVAL", "10"))

# Cloud configuration ---------------------------------------------------------
# DATABASE_URL defines the persistent database connection string.  When unset
# Menace falls back to a local SQLite file for easier development.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///menace.db")
# Optional autoscaler endpoint used by ResourceAllocationOptimizer
AUTOSCALER_ENDPOINT = os.getenv("AUTOSCALER_ENDPOINT")
# Provider used by Autoscaler: local, kubernetes or swarm
AUTOSCALER_PROVIDER = os.getenv("AUTOSCALER_PROVIDER", "local")
# Optional upper bound for all autoscaled instances
BUDGET_MAX_INSTANCES = int(os.getenv("BUDGET_MAX_INSTANCES", "0") or 0)
# Autoscaler behaviour tuning
AUTOSCALE_TOLERANCE = float(os.getenv("AUTOSCALE_TOLERANCE", "0.1"))
SCALE_UP_THRESHOLD = float(os.getenv("SCALE_UP_THRESHOLD", "0.8"))
SCALE_DOWN_THRESHOLD = float(os.getenv("SCALE_DOWN_THRESHOLD", "0.2"))
# Local provider runtime options
LOCAL_PROVIDER_MAX_RESTARTS = int(os.getenv("LOCAL_PROVIDER_MAX_RESTARTS", "3"))
LOCAL_PROVIDER_LOG_MAX_BYTES = int(
    os.getenv("LOCAL_PROVIDER_LOG_MAX_BYTES", "10485760")
)
LOCAL_PROVIDER_LOG_BACKUP_COUNT = int(
    os.getenv("LOCAL_PROVIDER_LOG_BACKUP_COUNT", "3")
)
# Optional Terraform configuration directory for DeploymentBot
TERRAFORM_DIR = os.getenv("TERRAFORM_DIR")

# Tuning for PreExecutionROIBot energy scaling
PRE_ROI_SCALE = float(os.getenv("PRE_ROI_SCALE", "1.0"))
PRE_ROI_BIAS = float(os.getenv("PRE_ROI_BIAS", "0.0"))
PRE_ROI_CAP = float(os.getenv("PRE_ROI_CAP", "5.0"))


def validate_production_config() -> None:
    """Ensure mandatory variables are set when running in production."""
    if MENACE_MODE.lower() == "production" and DATABASE_URL.startswith("sqlite"):
        raise RuntimeError(
            "DATABASE_URL must point to a production database when MENACE_MODE=production"
        )


validate_production_config()
