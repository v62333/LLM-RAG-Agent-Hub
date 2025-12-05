from enum import Enum


class Domain(str, Enum):
    finance = "finance"
    ads = "ads"
    general = "general"


class CollectionName(str, Enum):
    docs = "docs"
    news = "news"
    custom = "custom"
