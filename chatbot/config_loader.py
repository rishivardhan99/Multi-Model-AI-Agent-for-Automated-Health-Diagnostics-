# chatbot/config_loader.py
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
_cfg_path = BASE_DIR / "config" / "chatbot_config.yaml"
_templates_path = BASE_DIR / "config" / "prompt_templates.yaml"

# safe defaults
cfg = {}
templates = {}

try:
    cfg = yaml.safe_load(_cfg_path.read_text(encoding="utf-8"))
except Exception:
    cfg = {}

try:
    templates = yaml.safe_load(_templates_path.read_text(encoding="utf-8"))
except Exception:
    templates = {}
