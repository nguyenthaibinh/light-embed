import importlib.metadata
from light_embed.text_embedding import TextEmbedding

try:
    version = importlib.metadata.version("TextEmbedding")
except importlib.metadata.PackageNotFoundError as _:
    version = "1.0.1"

__version__ = version
__all__ = ["TextEmbedding"]