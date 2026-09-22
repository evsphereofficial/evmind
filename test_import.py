import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import load_config
from src.tasks import build_tasks
cfg = load_config("configs/registry.yaml")
tasks = build_tasks(cfg.tasks)
print("loaded", len(tasks), "tasks")
print("names", [t.name for t in tasks])
