from pathlib import Path
from typing import List
from app.core.config import settings


class FileStorage:
    def __init__(self) -> None:
        self.base_dir = Path(settings.data_dir)

    def list_docs_files(self) -> List[Path]:
        docs_dir = Path(settings.docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        return list(docs_dir.glob("*.*"))

    def get_ads_csv_path(self) -> Path:
        ads_dir = Path(settings.ads_dir)
        ads_dir.mkdir(parents=True, exist_ok=True)
        return ads_dir / "ads_performance.csv"

    def get_news_csv_path(self) -> Path:
        news_dir = Path(settings.news_dir)
        news_dir.mkdir(parents=True, exist_ok=True)
        return news_dir / "news_sample.csv"


_storage: FileStorage | None = None


def get_storage() -> FileStorage:
    global _storage
    if _storage is None:
        _storage = FileStorage()
    return _storage
