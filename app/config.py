from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_langchain_db"
CHECKPOINT_DB_PATH = PROJECT_ROOT / "chat_checkpoints.sqlite"