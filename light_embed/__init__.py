import importlib.metadata
from light_embed.text_embedding import TextEmbedding

try:
    version = importlib.metadata.version("light-embed")
except importlib.metadata.PackageNotFoundError as _:
    version = "0.1.0"

__version__ = version
__all__ = [
    "TextEmbedding",
]